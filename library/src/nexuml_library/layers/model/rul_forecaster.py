"""Reusable temporal RUL backbones for supervised sequence regression.

Implementations are conventional, compact architectural baselines rather than
verbatim reproductions of specific leaderboard-winning paper variants.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


class _ResidualTCNBlock(nn.Module):
    """Length-preserving residual dilated temporal convolution."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=3,
                dilation=dilation, padding=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3,
                      dilation=dilation, padding=dilation),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply two padded convolutions and residual connection.

        Returns:
            Features with the same channel and temporal dimensions.
        """
        return self.activation(x + self.branch(x))


class _CausalTCNBlock(nn.Module):
    """Causal residual TCN block without future-time leakage."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Use left-only padding so output at t never sees future samples.

        Returns:
            Causal features with the original sequence length.
        """
        y = self.drop(self.act(self.conv1(nn.functional.pad(x, (2 * self.dilation, 0)))))
        y = self.drop(self.conv2(nn.functional.pad(y, (2 * self.dilation, 0))))
        return self.act(x + y)


class _MixerBlock(nn.Module):
    """TSMixer-style temporal and channel mixing with residuals."""

    def __init__(self, length: int, width: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(length, length), nn.GELU(), nn.Dropout(dropout)
        )
        self.channels = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        """Mix time and channels without adding learnable positional encoding.

        Returns:
            A [B,L,H] tensor of mixed sensor features.
        """
        x = self.norm1(x + self.temporal(x.transpose(1, 2)).transpose(1, 2))
        return self.norm2(x + self.channels(x))


@layer("TemporalRULForecaster")
class TemporalRULForecaster(LayerDefinition):
    """End-to-end regression models sharing the same causal [L,D] contract."""

    architecture: Literal["feature_mlp", "lstm", "tcn", "transformer", "tsmixer"] = "tcn"
    hidden_dim: int = Field(default=96, gt=0)
    depth: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    attention_heads: int = Field(default=4, gt=0)
    output_dim: int = Field(default=1, gt=0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Construct a typed NexuML temporal forecaster.

        Returns:
            Runtime model compatible with configured input/output keys.
        """
        return _TemporalRULRuntime(**context.runtime_kwargs(), **self.model_dump())


class _TemporalRULRuntime(PipelineLayer):
    """Model choices with a common [B,L,D] -> [B,output_dim] interface."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        architecture: str = "tcn",
        hidden_dim: int = 96,
        depth: int = 3,
        dropout: float = 0.1,
        attention_heads: int = 4,
        output_dim: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        length, channels = tuple(input_sizes[keys_in[0]])
        if length < 1 or channels < 1:
            raise ValueError("RUL history shape must be positive [L,D]")
        self.architecture = architecture
        self.length = length
        if architecture == "feature_mlp":
            self.extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(length * channels, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        elif architecture == "lstm":
            self.extractor = nn.LSTM(
                channels, hidden_dim, num_layers=depth,
                batch_first=True, dropout=dropout if depth > 1 else 0.0,
            )
        elif architecture == "tcn":
            self.input_projection = nn.Conv1d(channels, hidden_dim, kernel_size=1)
            self.extractor = nn.Sequential(
                *(_CausalTCNBlock(hidden_dim, 2**i, dropout) for i in range(depth))
            )
        elif architecture == "transformer":
            if hidden_dim % attention_heads:
                raise ValueError("hidden_dim must be divisible by attention_heads")
            self.input_projection = nn.Linear(channels, hidden_dim)
            self.position = nn.Parameter(torch.zeros(1, length, hidden_dim))
            self.extractor = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=attention_heads,
                    dim_feedforward=hidden_dim * 4, dropout=dropout,
                    batch_first=True, norm_first=True,
                ),
                num_layers=depth,
                enable_nested_tensor=False,
            )
        elif architecture == "tsmixer":
            self.input_projection = nn.Linear(channels, hidden_dim)
            self.extractor = nn.Sequential(
                *(_MixerBlock(length, hidden_dim, dropout) for _ in range(depth))
            )
        else:
            raise ValueError(f"Unknown RUL model architecture: {architecture}")
        self.regression = nn.Linear(hidden_dim, output_dim)

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Predict normalized remaining life from a causal sensor window.

        Returns:
            [B,output_dim] predictions, without target leakage.
        """
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("RUL inputs must be [B,L,D] with configured L")
        if self.architecture == "feature_mlp":
            latent = self.extractor(x)
        elif self.architecture == "lstm":
            _, (hidden, _) = cast(tuple[Tensor, tuple[Tensor, Tensor]], self.extractor(x))
            latent = hidden[-1]
        elif self.architecture == "tcn":
            latent = self.extractor(self.input_projection(x.transpose(1, 2)))[:, :, -1]
        elif self.architecture == "transformer":
            z = self.input_projection(x) + self.position
            mask = torch.triu(
                torch.ones(self.length, self.length, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            latent = self.extractor(z, mask=mask)[:, -1]
        else:
            latent = self.extractor(self.input_projection(x))[:, -1]
        return self.regression(latent)


@layer("RULRegressionObjective")
class RULRegressionObjective(LayerDefinition):
    """Supervised MAE/MSE/Huber or pinball loss over normalized RUL targets."""

    objective: Literal["mse", "mae", "huber", "quantile"] = "huber"
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Create regression objective with configurable quantile levels.

        Returns:
            RUL loss layer using the regular NexuML loss-key mechanism.
        """
        if self.objective == "quantile" and (
            not self.quantiles
            or any(not 0.0 < value < 1.0 for value in self.quantiles)
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("Quantiles must be unique ascending probabilities in (0,1)")
        return _RULObjectiveRuntime(**context.runtime_kwargs(), **self.model_dump())


class _RULObjectiveRuntime(PipelineLayer):
    """Per-batch loss that reads RUL targets only from the y TensorDict."""

    def __init__(
        self, input_sizes: dict[str, tuple], keys_in: list[str], keys_out: list[str],
        objective: str = "huber", quantiles: list[float] | None = None,
        label_key: str = "rul", **kwargs: object,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out,
            label_key=label_key, **kwargs,
        )
        self.objective = objective
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]

    def forward(
        self, x: TensorDict, y: TensorDict | None = None,
    ) -> tuple[TensorDict, TensorDict | None]:
        """Populate NexuML's scalar RUL loss from normalized targets.

        Returns:
            Unmodified labels and features with an additional loss key.
        """
        predicted = cast(Tensor, x[cast(list[str], self.keys_in)[0]])
        target = self.get_label(x, y)
        if target is None:
            loss = predicted.square().mean() * 0.0
        elif self.objective == "quantile":
            if predicted.shape[-1] != len(self.quantiles):
                raise ValueError("Quantile head/output size disagreement")
            error = target.float() - predicted
            q = predicted.new_tensor(self.quantiles)
            loss = torch.maximum(q * error, (q - 1.0) * error).mean()
        else:
            if predicted.shape != target.shape:
                raise ValueError("RUL prediction and target shape disagree")
            if self.objective == "mse":
                loss = nn.functional.mse_loss(predicted, target)
            elif self.objective == "mae":
                loss = nn.functional.l1_loss(predicted, target)
            else:
                loss = nn.functional.smooth_l1_loss(predicted, target)
        x[self.keys_out[0]] = loss.expand(x.batch_size)
        return x, y

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Unused: labels are supplied separately through NexuML TensorDicts."""
        raise NotImplementedError
