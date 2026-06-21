"""Linear encoder/decoder layer for autoencoder architectures."""

from __future__ import annotations
from nexuml.core.discovery import layer

import math

import torch
import torch.nn as nn

from nexuml.core.base_layer import PipelineLayer


@layer("LinearEncoder")
class LinearEncoder(PipelineLayer):
    """Stacked fully-connected layers with optional batch norm and activation.

    Used for both encoder and decoder in linear autoencoders.
    Input is flattened to 2D before processing.
    """

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        hidden_dims: list[int] | None = None,
        output_dim: int = 8,
        activation: str = "torch.nn.ReLU",
        last_activation: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)

        self.hidden_dims = hidden_dims or []
        self.output_dim = output_dim
        self.last_activation = last_activation

        # Determine input dimension from first key_in
        first_key = keys_in[0]
        if first_key in input_sizes:
            input_shape = input_sizes[first_key]
            self.input_dim = math.prod(input_shape)
        else:
            raise ValueError(
                f"Input key '{first_key}' not found in input_sizes. "
                f"Available: {list(input_sizes.keys())}"
            )

        # Resolve activation class
        activation_cls = _resolve_activation(activation)

        # Build layer stack
        dims = self._create_dim_list()
        self.model = self._build_layers(dims, activation_cls, bias)

    def _create_dim_list(self) -> list[tuple[int, int]]:
        """Create (in_dim, out_dim) pairs for each linear layer.

        Returns:
            list[tuple[int, int]]: Sequential (input, output) dimension pairs.
        """
        all_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        return list(zip(all_dims[:-1], all_dims[1:]))

    def _build_layers(
        self,
        dims: list[tuple[int, int]],
        activation_cls: type,
        bias: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i, (in_d, out_d) in enumerate(dims):
            layers.append(nn.Linear(in_d, out_d, bias=bias))
            is_last = i == len(dims) - 1
            if not is_last or self.last_activation:
                layers.append(nn.BatchNorm1d(out_d))
                layers.append(activation_cls())
        return nn.Sequential(*layers)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        # Flatten to 2D: (batch, *feature_shape) -> (batch, flat_dim)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.model(x_flat)


def _resolve_activation(activation: str) -> type:
    """Resolve activation class from dotted path string.

    Returns:
        type: The resolved activation class, falling back to ``torch.nn.ReLU``.
    """
    parts = activation.rsplit(".", 1)
    if len(parts) == 2:
        import importlib

        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])
    # Fallback: try torch.nn directly
    return getattr(torch.nn, activation, torch.nn.ReLU)
