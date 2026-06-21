"""Regression head for latent space."""

from __future__ import annotations
from nexuml.core.discovery import layer

import math

import torch
import torch.nn as nn

from nexuml.core.base_layer import PipelineLayer


@layer("LatentRegressionHead")
class LatentRegressionHead(PipelineLayer):
    """Regression head operating on latent vectors."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        num_outputs: int = 1,
        hidden_dims: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)

        first_key = keys_in[0]
        input_shape = input_sizes[first_key]
        input_dim = math.prod(input_shape)

        hidden_dims = hidden_dims or []
        all_dims = [input_dim] + hidden_dims + [num_outputs]
        layers: list[nn.Module] = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.ReLU())

        self.head = nn.Sequential(*layers)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.head(x_flat)
