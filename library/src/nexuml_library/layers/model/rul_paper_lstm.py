"""Goel et al. 2026 FD001 LSTM backbone reproduced from Algorithms 5 and 6."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import Field
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("Goel2026LSTMForecaster")
class Goel2026LSTMForecaster(LayerDefinition):
    """Single-layer LSTM and regression MLP from arXiv:2604.27234, Algorithm 5."""

    hidden_dim: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.5, ge=0, lt=1)
    initialization_seed: int = 42

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Instantiate the publication's recurrent model and three linear layers.

        Returns:
            NexuML model accepting the standard sensor-history TensorDict.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            return _Goel2026LSTMRuntime(
                **context.runtime_kwargs(),
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )


class _Goel2026LSTMRuntime(PipelineLayer):
    """LSTM(n,32,1), Dropout(0.5), Linear(32,8,8,1) from the paper."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        hidden_dim: int,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        sequence_length, channels = input_sizes[keys_in[0]]
        self.sequence_length = sequence_length
        self.recurrent = nn.LSTM(channels, hidden_dim, batch_first=True, num_layers=1)
        self.regression = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Regress RUL from the final hidden state.

        Returns:
            Scalar RUL prediction per input window.

        Raises:
            ValueError: If input does not have the declared sequence length.
        """
        if x.ndim != 3 or x.shape[1] != self.sequence_length:
            raise ValueError("Goel2026 LSTM expects sensor sequences [B,L,D]")
        _, (hidden, _) = self.recurrent(x)
        return self.regression(hidden[-1])
