"""Goel et al. 2026 FD001 1D CNN architecture from arXiv:2604.27234."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import Field
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("Goel2026CNNForecaster")
class Goel2026CNNForecaster(LayerDefinition):
    """Three-layer Conv1D and 128-unit head from Algorithm 4."""

    dropout: float = Field(default=0.3, ge=0, lt=1)
    initialization_seed: int = 42

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Construct the paper's time-convolutional RUL predictor.

        Returns:
            Native NexuML temporal CNN with the paper's kernel and layer layout.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            return _Goel2026CNNRuntime(
                **context.runtime_kwargs(), dropout=self.dropout
            )


class _Goel2026CNNRuntime(PipelineLayer):
    """Conv1D(n,32,64,64), ReLU, dropout, flatten and 128-unit regression."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        length, channels = input_sizes[keys_in[0]]
        self.length = length
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * length, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Regress RUL from the paper's three temporal convolution blocks.

        Returns:
            Scalar remaining useful life prediction per input sample.

        Raises:
            ValueError: If input does not match the declared temporal dimension.
        """
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("Goel2026 CNN expects sensor sequences [B,L,D]")
        return self.regression(self.features(x.transpose(1, 2)))
