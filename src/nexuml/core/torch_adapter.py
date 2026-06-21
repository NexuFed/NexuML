"""Adapter wrapping plain torch.nn.Modules for TensorDict pipeline compatibility."""

from __future__ import annotations

import torch
import torch.nn as nn

from nexuml.core.base_layer import PipelineLayer


class TorchModuleAdapter(PipelineLayer):
    """Wraps any torch.nn.Module to conform to PipelineLayer interface."""

    def __init__(
        self,
        module: nn.Module,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        self.module = module

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.module(x)
