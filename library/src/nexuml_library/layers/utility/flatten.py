"""Flatten layer for bridging multi-dimensional inputs to 1D."""

from __future__ import annotations
from nexuml.core.discovery import layer


import torch

from nexuml.core.base_layer import PipelineLayer


@layer("Flatten")
class Flatten(PipelineLayer):
    """Flattens arbitrary input dimensions to a 1D vector."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)
