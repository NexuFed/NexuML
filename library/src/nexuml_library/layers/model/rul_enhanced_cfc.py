"""Configurable sensor-aware and degradation-aware CfC RUL estimator.

All optional components are repository-native architecture adaptations;
none is claimed to be an exact replication of the cited external models.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import functional as F

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml_library.data.maintenance.patchformer_flow import (
    _trusted_patchformer_weights,
    resolve_patchformer_checkpoint,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.evaluation.maintenance.rul_sequence_flow_checkpoint import (
    sequence_flow_run,
)
from nexuml_library.layers.generative.flow.sequence_vector_field import _CfCSequence
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel


class _SensorGraphAttention(nn.Module):
    """Learn complete sensor-to-sensor attention on the raw sensor dimension."""

    def __init__(self, sensors: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.values = nn.Linear(1, hidden_dim)
        self.identities = nn.Parameter(torch.zeros(1, sensors, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)
        self.residual_gain = nn.Parameter(torch.zeros(()))
        self.sensors = sensors

    def forward(self, samples: Tensor) -> Tensor:
        """Enrich each cycle with content-dependent cross-sensor interactions.

        Returns:
            Residual sensor values with the original [B,T,S] shape.

        Raises:
            ValueError: If the input sensor count differs from the configured count.
        """
        b, length, sensors = samples.shape
        if sensors != self.sensors:
            raise ValueError("Sensor graph expects the configured sensor count")
        x = samples.reshape(b * length, sensors, 1)
        embedded = self.values(x) + self.identities
        attended, _ = self.attention(embedded, embedded, embedded, need_weights=False)
        residual = self.output(attended).reshape(b, length, sensors)
        return samples + self.residual_gain.tanh() * residual


class _MultiscaleTemporalFeatures(nn.Module):
    """Fuse 3-, 7- and 15-cycle temporal receptive fields."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.filters = nn.ModuleList(
            nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width)
            for kernel in (3, 7, 15)
        )
        self.fuse = nn.Linear(3 * width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """Add learned multiscale temporal features.

        Returns:
            [B,30,width] residual representation.
        """
        transformed = tokens.transpose(1, 2)
        scales = torch.cat(
            [F.silu(conv(transformed)).transpose(1, 2) for conv in self.filters],
            dim=-1,
        )
        return tokens + self.dropout(self.fuse(scales))


class _AdaptiveSoftSegments(nn.Module):
    """Learn differentiable segment boundaries without hard test-tuned thresholds."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.boundary = nn.Linear(width, 1)
        self.merge = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Blend tokens within learned soft segments and preserve their order.

        Returns:
            Residually augmented time-aligned token sequence.
        """
        difference = tokens[:, 1:] - tokens[:, :-1]
        boundary = self.boundary(difference.abs()).sigmoid()
        state = tokens[:, 0]
        outputs = [state]
        for index in range(1, tokens.shape[1]):
            gate = boundary[:, index - 1]
            state = gate * tokens[:, index] + (1.0 - gate) * state
            outputs.append(state)
        segmented = torch.stack(outputs, dim=1)
        return tokens + self.merge(segmented)


@layer("EnhancedCfCRULHead")
class EnhancedCfCRULHead(LayerDefinition):
    """Ablate PatchFormer tuning, graph, trends, scales and soft segmentation."""

    subset: Literal["FD001", "FD003"] = "FD003"
    upstream_seed: int = 42
    initialization_seed: int = 42
    pretrained_encoder: bool = True
    train_encoder: bool = False
    sensor_graph: bool = False
    trend: bool = False
    multiscale: bool = False
    segmentation: bool = False
    trajectory: bool = False
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Create a preinitialized, optionally trainable PatchFormer-CfC model.

        Returns:
            NexuML registered runtime exposing current and prior RUL predictions.

        Raises:
            ValueError: If the raw sensor shapes or labels are incompatible.
        """
        sensors = 15 if self.subset == "FD003" else 14
        if (
            context.input_sizes.get("rul_history") != (30, sensors)
            or context.input_sizes.get("previous_rul_history") != (30, sensors)
            or context.input_sizes.get("pair_mask") != (1,)
            or context.keys_in != ["rul_history"]
            or context.keys_out != ["predicted_rul", "previous_predicted_rul"]
        ):
            raise ValueError("Enhanced CfC requires aligned 30-cycle sensor pairs")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            return _EnhancedCfCRuntime(
                subset=self.subset,
                upstream_seed=self.upstream_seed,
                initialization_seed=self.initialization_seed,
                pretrained_encoder=self.pretrained_encoder,
                train_encoder=self.train_encoder,
                sensor_graph=self.sensor_graph,
                trend=self.trend,
                multiscale=self.multiscale,
                segmentation=self.segmentation,
                trajectory=self.trajectory,
                dropout=self.dropout,
                **context.runtime_kwargs(),
            )


class _EnhancedCfCRuntime(PipelineLayer):
    """Raw-sensor PatchFormer and configurable CfC head with shared pair weights."""

    def __init__(
        self,
        *,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        subset: Literal["FD001", "FD003"],
        upstream_seed: int,
        initialization_seed: int,
        pretrained_encoder: bool,
        train_encoder: bool,
        sensor_graph: bool,
        trend: bool,
        multiscale: bool,
        segmentation: bool,
        trajectory: bool,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        sensors = 15 if subset == "FD003" else 14
        self.encoder = _PatchFormerModel(30, sensors, 64, 5, 2, 0.1)
        self.train_encoder = train_encoder
        self.trajectory = trajectory
        if not pretrained_encoder:
            raise ValueError("Matched CfC ablation requires pretrained PatchFormer")
        checkpoint = resolve_patchformer_checkpoint(subset, upstream_seed)
        encoder_hash = _trusted_patchformer_weights(self.encoder, checkpoint)
        # Normalization is fitted exclusively on the frozen upstream
        # training engine endpoints, never on validation or official test.
        endpoint = (
            f".experiments/rul/prepared/"
            f"{sequence_flow_run(subset, upstream_seed, 'cfc')}-endpoints.pt"
        )
        payload = read_prepared_artifact(endpoint)
        settings = cast(dict[str, Any], payload["settings"])
        if (
            settings.get("subset") != subset
            or settings.get("seed") != upstream_seed
            or settings.get("feature_normalization") != "train_endpoints_only"
            or settings.get("frozen_patchformer_checkpoint_sha256") != encoder_hash
            or settings.get("held_out_test_engine_labels_used") is not False
        ):
            raise ValueError("Enhanced CfC normalization has incompatible provenance")
        mean = torch.as_tensor(settings["feature_mean"], dtype=torch.float32)
        std = torch.as_tensor(settings["feature_std"], dtype=torch.float32)
        if mean.shape != (30, 128) or std.shape != (30, 128) or not bool((std > 0).all()):
            raise ValueError("Enhanced CfC expects normalized [30,128] DPAN features")
        self.reference_mean: Tensor
        self.reference_std: Tensor
        self.register_buffer("reference_mean", mean, persistent=True)
        self.register_buffer("reference_std", std, persistent=True)
        self.encoder.requires_grad_(train_encoder)
        # The pretrained FAN is unused; train only the DPAN when enabled.
        self.encoder.fan.requires_grad_(False)
        # Initialize the *common* head before any optional component.
        # Every matched head seed now has identical common starting weights.
        self.input_projection = nn.Linear(256, 64)
        self.positions = nn.Parameter(torch.zeros(1, 30, 64))
        self.temporal = _CfCSequence(64, depth=1, dropout=dropout)
        self.norm = nn.LayerNorm(64)
        self.token_scores = nn.Linear(64, 1)
        self.regressor = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        # Each optional branch has its own reproducible sub-seed so its
        # initialization agrees in single-on, all-on, and leave-one-out arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed + 1001)
            self.sensor_graph = _SensorGraphAttention(sensors) if sensor_graph else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed + 1002)
            self.trend_projection = nn.Linear(128, 64) if trend else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed + 1003)
            self.multiscale = _MultiscaleTemporalFeatures(64, dropout) if multiscale else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed + 1004)
            self.segmentation = _AdaptiveSoftSegments(64, dropout) if segmentation else None
        if not train_encoder:
            self.encoder.eval()

    def train(self, mode: bool = True) -> _EnhancedCfCRuntime:
        """Preserve frozen pretrained inference semantics during readout fitting.

        Returns:
            The same runtime with correct frozen encoder dropout behavior.
        """
        super().train(mode)
        if not self.train_encoder:
            self.encoder.eval()
        return self

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Predict capped RUL from a raw 30-cycle sensor sequence.

        Returns:
            [B,1] RUL prediction in 0..130 cycles.

        Raises:
            ValueError: If the sensor history dimensionality is incorrect.
        """
        if x.ndim != 3 or x.shape[1] != 30 or x.shape[2] != len(self.encoder.embedding):
            raise ValueError("Enhanced CfC requires [batch,30,sensors]")
        raw = self.sensor_graph(x) if self.sensor_graph is not None else x
        features = self.encoder.forward_features(raw)
        normalized = (features - self.reference_mean) / self.reference_std
        encoded = (
            self.input_projection(torch.cat((normalized, torch.zeros_like(normalized)), dim=-1))
            + self.positions
        )
        if self.trend_projection is not None:
            velocity = torch.cat(
                (torch.zeros_like(normalized[:, :1]), normalized[:, 1:] - normalized[:, :-1]),
                dim=1,
            )
            encoded = encoded + self.trend_projection(velocity)
        if self.multiscale is not None:
            encoded = self.multiscale(encoded)
        if self.segmentation is not None:
            encoded = self.segmentation(encoded)
        tokens = self.norm(self.temporal(encoded))
        attention = torch.softmax(self.token_scores(tokens), dim=1)
        pooled = (tokens * attention).sum(dim=1)
        return 130.0 * torch.sigmoid(self.regressor(pooled))

    def forward(
        self, x: TensorDict | Tensor, y: TensorDict | None = None
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Reuse the same RUL head for current and genuine previous windows.

        Returns:
            TensorDict with current and previous predictions.

        Raises:
            TypeError: If upstream NexuML routing omitted the TensorDict.
        """
        if not isinstance(x, TensorDict):
            raise TypeError("Enhanced CfC requires TensorDict input")
        current = cast(Tensor, x["rul_history"])
        x["predicted_rul"] = self.forward_tensor(current)
        previous = cast(Tensor, x["previous_rul_history"])
        if self.trajectory and self.training and y is not None:
            x["previous_predicted_rul"] = self.forward_tensor(previous)
        else:
            x["previous_predicted_rul"] = cast(Tensor, x["predicted_rul"]).detach()
        return x, y
