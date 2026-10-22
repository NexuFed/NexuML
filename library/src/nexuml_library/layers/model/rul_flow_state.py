"""Standalone RUL prediction from frozen PatchFormer embeddings and learned flow.

The original PatchFormer regression head is never called. Two matched heads
separate improvement from the encoder itself versus information introduced by
the pretrained healthy-to-degraded flow field.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import Field
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("PatchFormerFlowStateRULHead")
class PatchFormerFlowStateRULHead(LayerDefinition):
    """Predict bounded RUL using encoder/transport features, not baseline RUL."""

    hidden_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    initialization_seed: int = 42

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build a deterministic flow-state regression head.

        Returns:
            Native NexuML trainable flow-state RUL layer.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            return _PatchFormerFlowStateRuntime(
                **context.runtime_kwargs(),
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )


class _PatchFormerFlowStateRuntime(PipelineLayer):
    """Consume [encoder, flow displacement, time, distance] with no RUL baseline."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        hidden_dim: int,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs
        )
        if len(keys_in) != 1 or input_sizes[keys_in[0]] != (258,):
            raise ValueError("Flow-state-only RUL expects 258 frozen feature values")
        self.regressor = nn.Sequential(
            nn.Linear(258, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 or 1),
            nn.GELU(),
            nn.Linear(hidden_dim // 2 or 1, 1),
        )

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Infer RUL in cycles without using pretrained PatchFormer RUL output.

        Returns:
            Nonnegative capped RUL predictions of shape [batch,1].

        Raises:
            ValueError: If the feature tensor violates the expected shape.
        """
        if x.ndim != 2 or x.shape[-1] != 258:
            raise ValueError("Flow-state-only RUL features must be [batch,258]")
        return 130.0 * torch.sigmoid(self.regressor(x))
