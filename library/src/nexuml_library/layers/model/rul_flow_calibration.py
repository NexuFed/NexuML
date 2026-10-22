"""Train-only calibrated RUL heads for frozen PatchFormer latent-flow coordinates.

The data source supplies deterministic flow progress, trajectory distance and
supervised PatchFormer baseline predictions, never RUL labels as model inputs.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("PatchFormerFlowRULHead")
class PatchFormerFlowRULHead(LayerDefinition):
    """Small MLP mapping flow time to RUL, optionally correcting PatchFormer."""

    mode: Literal["flow_time", "hybrid"] = "flow_time"
    hidden_dim: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    initialization_seed: int = 42

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Create a reproducibly initialized regression head using NexuML.

        Returns:
            Standard NexuML PipelineLayer for direct RUL prediction.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            return _PatchFormerFlowRULRuntime(
                **context.runtime_kwargs(),
                mode=self.mode,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )


class _PatchFormerFlowRULRuntime(PipelineLayer):
    """MLP whose only trainable parameters are the flow-to-RUL calibration."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        mode: Literal["flow_time", "hybrid"],
        hidden_dim: int,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs
        )
        if len(keys_in) != 1 or input_sizes[keys_in[0]] != (3,):
            raise ValueError("Flow RUL head expects [time, distance, baseline] triples")
        self.mode = mode
        input_width = 1 if mode == "flow_time" else 3
        self.regressor = nn.Sequential(
            nn.Linear(input_width, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Regress RUL from flow time alone or flow/forecast residual features.

        Returns:
            One prediction [batch,1] in physical remaining-cycle units.

        Raises:
            ValueError: If features have inconsistent channel dimensions.
        """
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError("Flow calibration expects [batch,3] features")
        if self.mode == "flow_time":
            return 130.0 * torch.sigmoid(self.regressor(x[:, :1]))
        residual = 30.0 * torch.tanh(self.regressor(x))
        return x[:, 2:3] + residual
