"""Sequence-aware time-conditioned vector fields for Flow Matching.

All backbones map [batch, sequence, channels] to an equally shaped velocity.
The CfC implementation uses the default closed-form gated interpolation cell,
with elapsed sequence time distinct from the flow interpolation time.
"""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


class _ConformerFeedForward(nn.Module):
    """Pre-normalized feed-forward module with channel expansion."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply position-wise transformation.

        Returns:
            A sequence with the same shape as the input.
        """
        return self.layers(x)


class _ConformerBlock(nn.Module):
    """Macaron feed-forward, attention and depthwise convolution sequence block."""

    def __init__(
        self, width: int, heads: int, kernel: int, dropout: float
    ) -> None:
        super().__init__()
        self.ff_in = _ConformerFeedForward(width, dropout)
        self.attn_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(width)
        self.conv_pointwise_in = nn.Conv1d(width, 2 * width, 1)
        self.conv_depthwise = nn.Conv1d(
            width, width, kernel, padding=kernel // 2, groups=width
        )
        self.conv_batchnorm = nn.BatchNorm1d(width)
        self.conv_pointwise_out = nn.Conv1d(width, width, 1)
        self.conv_dropout = nn.Dropout(dropout)
        self.ff_out = _ConformerFeedForward(width, dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        """Encode local and global temporal dependencies.

        Returns:
            Conformer-encoded sequence with unchanged shape.
        """
        x = x + 0.5 * self.ff_in(x)
        normalized = self.attn_norm(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + self.attn_dropout(attended)
        conv = self.conv_norm(x).transpose(1, 2)
        conv = nn.functional.glu(self.conv_pointwise_in(conv), dim=1)
        conv = self.conv_depthwise(conv)
        conv = nn.functional.silu(self.conv_batchnorm(conv))
        conv = self.conv_pointwise_out(conv).transpose(1, 2)
        x = x + self.conv_dropout(conv)
        return self.norm(x + 0.5 * self.ff_out(x))


class _CfCCell(nn.Module):
    """Fully connected CfC cell with continuous-time gated interpolation.

    For hidden state h and observation x, two candidate states are interpolated
    with sigmoid(time_a([x,h]) * dt + time_b([x,h])). No ODE solver is used.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU()
        )
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        self.time_a = nn.Linear(width, width)
        self.time_b = nn.Linear(width, width)

    def forward(self, x: Tensor, state: Tensor, dt: float) -> Tensor:
        """Advance a hidden state by one elapsed sequence interval.

        Returns:
            Continuously time-gated next hidden state.
        """
        joined = self.backbone(torch.cat((x, state), dim=-1))
        interpolation = torch.sigmoid(
            self.time_a(joined) * dt + self.time_b(joined)
        )
        return (
            torch.tanh(self.first(joined)) * (1.0 - interpolation)
            + torch.tanh(self.second(joined)) * interpolation
        )


class _CfCSequence(nn.Module):
    """Stacked CfC cells returning all per-position hidden states."""

    def __init__(self, width: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.cells = nn.ModuleList(_CfCCell(width) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        """Propagate contextual hidden states across sequence positions.

        Returns:
            Full-length per-position CfC representation [B,L,D].
        """
        elapsed = 1.0 / max(x.shape[1] - 1, 1)
        for cell in self.cells:
            state = x.new_zeros((x.shape[0], x.shape[-1]))
            outputs: list[Tensor] = []
            for item in x.unbind(dim=1):
                state = cell(item, state, elapsed)
                outputs.append(state)
            x = self.norm(x + self.dropout(torch.stack(outputs, dim=1)))
        return x


@layer("SequenceTimeConditionedVectorField")
class SequenceTimeConditionedVectorField(LayerDefinition):
    """Choose Transformer, Conformer or CfC per-position flow velocity."""

    architecture: Literal["transformer", "conformer", "cfc"] = "transformer"
    model_dim: int = Field(default=128, gt=0)
    depth: int = Field(default=2, gt=0)
    attention_heads: int = Field(default=4, gt=0)
    conv_kernel: int = Field(default=7, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build a registered NexuML vector field for 2-D sequence states.

        Returns:
            PipelineLayer preserving sequence length and feature width.

        Raises:
            ValueError: If the state or selected backbone parameters are invalid.
        """
        if len(context.keys_in) != 2 or len(context.keys_out) != 1:
            raise ValueError(
                "Sequence vector field requires state/time and velocity keys"
            )
        state_shape = context.input_sizes[context.keys_in[0]]
        if len(state_shape) != 2 or min(state_shape) < 1:
            raise ValueError("Sequence flow requires [length,channels] inputs")
        if self.architecture in ("transformer", "conformer"):
            if self.model_dim % self.attention_heads:
                raise ValueError("Attention heads must divide sequence model_dim")
        if self.architecture == "conformer" and self.conv_kernel % 2 == 0:
            raise ValueError("Conformer convolution kernel must be odd")
        return _SequenceVectorFieldRuntime(
            architecture=self.architecture,
            length=int(state_shape[0]),
            channels=int(state_shape[1]),
            model_dim=self.model_dim,
            depth=self.depth,
            heads=self.attention_heads,
            conv_kernel=self.conv_kernel,
            dropout=self.dropout,
            **context.runtime_kwargs(),
        )


class _SequenceVectorFieldRuntime(PipelineLayer):
    """Flow-time-conditioned sequential network with NexuML routing."""

    def __init__(
        self,
        architecture: Literal["transformer", "conformer", "cfc"],
        length: int,
        channels: int,
        model_dim: int,
        depth: int,
        heads: int,
        conv_kernel: int,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.length = length
        self.channels = channels
        self.input_projection = nn.Linear(channels, model_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(3, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, model_dim, 2, dtype=torch.float32)
            / model_dim
        )
        phases = positions * freqs.unsqueeze(0)
        encoding = torch.zeros(1, length, model_dim)
        encoding[..., 0::2] = phases.sin()
        encoding[..., 1::2] = phases.cos()[:, :encoding[..., 1::2].shape[-1]]
        self.register_buffer("position_encoding", encoding, persistent=False)
        if architecture == "transformer":
            block = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=heads,
                dim_feedforward=4 * model_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone: nn.Module = nn.TransformerEncoder(
                block, num_layers=depth, enable_nested_tensor=False
            )
        elif architecture == "conformer":
            self.backbone = nn.Sequential(
                *(
                    _ConformerBlock(model_dim, heads, conv_kernel, dropout)
                    for _ in range(depth)
                )
            )
        else:
            self.backbone = _CfCSequence(model_dim, depth, dropout)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, channels)

    def velocity(self, state: Tensor, time: Tensor) -> Tensor:
        """Predict the velocity for every sequence position.

        Returns:
            Flow velocity [batch,length,channels], equal to state shape.

        Raises:
            ValueError: If state shape or integration time is invalid.
        """
        if (
            state.ndim != 3
            or state.shape[1:] != (self.length, self.channels)
            or time.numel() != state.shape[0]
        ):
            raise ValueError("Sequence flow expects [B,L,C] and one time per sample")
        scalar = time.reshape(state.shape[0], 1)
        phase = 2.0 * math.pi * scalar
        conditioning = self.time_embedding(
            torch.cat((scalar, phase.sin(), phase.cos()), dim=-1)
        ).unsqueeze(1)
        position = cast(Tensor, self.position_encoding)
        encoded = (
            self.input_projection(state)
            + conditioning
            + position.to(dtype=state.dtype)
        )
        contextual = self.backbone(encoded)
        return self.output_projection(self.output_norm(contextual))

    def forward(
        self,
        x: TensorDict | Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Write the learned per-position velocity into the input TensorDict.

        Returns:
            TensorDict containing the predicted flow velocity and unchanged labels.

        Raises:
            TypeError: If the input is not a TensorDict.
        """
        if not self.check_update():
            return x, y
        if not isinstance(x, TensorDict):
            raise TypeError("Sequence vector field requires TensorDict inputs")
        keys = cast(list[str], self.keys_in)
        state = cast(Tensor, x[keys[0]])
        time = cast(Tensor, x[keys[1]])
        x[self.keys_out[0]] = self.velocity(state, time)
        return x, y

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Require flow-state and flow-time TensorDict routing.

        Raises:
            NotImplementedError: Direct tensor calls lack flow-time input.
        """
        raise NotImplementedError("Use velocity(state,time) or TensorDict forward")
