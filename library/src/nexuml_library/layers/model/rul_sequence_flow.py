"""Temporal-attention RUL readout for complete frozen PatchFormer sequences.

Both the standalone sequence-flow and encoder-only control use this identical
trainable readout. No pretrained PatchFormer RUL prediction enters the model.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml_library.layers.generative.flow.sequence_vector_field import _CfCSequence
from nexuml_library.layers.model.rul_selective_ssm import SelectiveSSMBlock


@layer("PatchFormerSequenceFlowRULHead")
class PatchFormerSequenceFlowRULHead(LayerDefinition):
    """Learn RUL with global token attention and a learned attentive pooling."""

    architecture: Literal["transformer", "mamba", "cfc"] = "transformer"
    hidden_dim: int = Field(default=128, gt=0)
    attention_heads: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    initialization_seed: int = 42

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Initialize the same temporal readout for both frozen feature arms.

        Returns:
            Registered NexuML model with a [B,1] RUL prediction.

        Raises:
            ValueError: If token size or attention parameters are incompatible.
        """
        if self.architecture == "transformer" and self.hidden_dim % self.attention_heads:
            raise ValueError("RUL head attention heads must divide hidden dimension")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            return _SequenceFlowRULHeadRuntime(
                architecture=self.architecture,
                hidden_dim=self.hidden_dim,
                attention_heads=self.attention_heads,
                dropout=self.dropout,
                **context.runtime_kwargs(),
            )


class _SequenceFlowRULHeadRuntime(PipelineLayer):
    """Token self-attention and learned pooling; never temporal mean-pooling."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        architecture: Literal["transformer", "mamba", "cfc"],
        hidden_dim: int,
        attention_heads: int,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out,
            **kwargs,
        )
        if len(keys_in) != 1 or input_sizes[keys_in[0]] != (30, 256):
            raise ValueError("Sequence RUL readout expects [30,256] features")
        self.input_projection = nn.Linear(256, hidden_dim)
        self.positions = nn.Parameter(torch.zeros(1, 30, hidden_dim))
        if architecture == "transformer":
            self.temporal_attention: nn.Module = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=attention_heads,
                dim_feedforward=2 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        elif architecture == "mamba":
            self.temporal_attention = SelectiveSSMBlock(
                hidden_dim, state_dim=8, expand=2, conv_kernel=4, dropout=dropout
            )
        else:
            self.temporal_attention = _CfCSequence(
                hidden_dim, depth=1, dropout=dropout
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.token_scores = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 or 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 or 1, 1),
        )

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Predict RUL from a full token sequence using attentive aggregation.

        Returns:
            Capped nonnegative RUL cycles [batch,1].

        Raises:
            ValueError: If sequence dimensions or features are invalid.
        """
        if x.ndim != 3 or x.shape[1:] != (30, 256):
            raise ValueError("Sequence RUL head requires [batch,30,256]")
        encoded = self.input_projection(x) + self.positions
        tokens = self.norm(self.temporal_attention(encoded))
        weights = torch.softmax(self.token_scores(tokens), dim=1)
        context = (tokens * weights).sum(dim=1)
        return 130.0 * torch.sigmoid(self.regressor(context))
