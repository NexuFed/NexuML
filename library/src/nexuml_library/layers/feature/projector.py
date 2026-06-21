"""Linear projection layers."""

from __future__ import annotations
from nexuml.core.discovery import layer

import importlib
from typing import cast

import torch

from nexuml.core.base_layer import PipelineLayer


def _locate(dotted_path: str | None) -> type | None:
    """Import a class from a dotted path string, e.g. 'torch.nn.GELU'.

    Returns:
        type | None: The resolved class, or None if the path is invalid
            or the import fails.
    """
    if dotted_path is None:
        return None
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        return None
    try:
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])
    except (ImportError, AttributeError):
        return None


@layer("Linear")
class Linear(PipelineLayer):
    """Multi-layer linear projector with optional activation and normalization.

    Args:
        target_dim: Output dimension. If None, uses output_sizes.
        output_dim: Backward-compatible alias for target_dim.
        hidden_dim: Hidden layer size (for n_layers > 1).
        hidden_dims: Backward-compatible explicit hidden-layer sizes. When
            provided, overrides hidden_dim/n_layers for layer construction.
        n_layers: Number of linear layers.
        linear_bias: Whether to use bias.
        bias: Backward-compatible alias for linear_bias.
        activation: Dotted class path, e.g. "torch.nn.GELU".
        normalization: Dotted class path, e.g. "torch.nn.BatchNorm1d".
        skip_last_activation: Skip activation after the last layer.
        flatten_dims: Optional (start, end) dims to flatten before projection.
    """

    def __init__(
        self,
        target_dim: int | None = None,
        output_dim: int | None = None,
        hidden_dim: int = 256,
        hidden_dims: list[int] | None = None,
        n_layers: int = 1,
        linear_bias: bool = True,
        bias: bool | None = None,
        activation: str | None = None,
        normalization: str | None = None,
        skip_last_activation: bool = False,
        flatten_dims: tuple[int, int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(self.keys_in) == 1, "Linear requires exactly one input key."
        assert len(self.keys_out) == 1, "Linear requires exactly one output key."

        input_size = self.input_sizes[cast(list[str], self.keys_in)[0]]
        if target_dim is None and output_dim is not None:
            target_dim = output_dim
        if bias is not None:
            linear_bias = bias

        if target_dim is not None:
            output_size = (target_dim,)
        elif self.output_sizes and self.keys_out[0] in self.output_sizes:
            output_size = self.output_sizes[self.keys_out[0]]
        else:
            raise ValueError("target_dim or output_sizes must be specified.")

        activation_cls = _locate(activation)
        normalization_cls = _locate(normalization)

        layers: list[torch.nn.Module] = []

        if flatten_dims is not None:
            layers.append(torch.nn.Flatten(*flatten_dims))
            s, e = flatten_dims
            input_size = (
                int(torch.prod(torch.tensor(list(input_size)[s - 1 : e if e != -1 else None]))),
            )

        if hidden_dims is not None:
            layer_dims = [input_size[-1], *hidden_dims, output_size[-1]]
        else:
            layer_dims = [input_size[-1]]
            for i in range(max(0, n_layers - 1)):
                layer_dims.append(hidden_dim)
            layer_dims.append(output_size[-1])

        for i, (dim_in, dim_out) in enumerate(zip(layer_dims, layer_dims[1:], strict=False)):
            is_last = i == len(layer_dims) - 2

            layers.append(torch.nn.Linear(dim_in, dim_out, bias=linear_bias))

            if normalization_cls is not None:
                layers.append(normalization_cls(dim_out))
            if activation_cls is not None and not (skip_last_activation and is_last):
                try:
                    layers.append(activation_cls(inplace=True))
                except TypeError:
                    layers.append(activation_cls())

        self.model = torch.nn.Sequential(*layers)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x)


@layer("Dropout")
class Dropout(PipelineLayer):
    """Dropout regularization layer.

    Args:
        p: Dropout probability.
    """

    def __init__(self, p: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert len(self.keys_in) == 1, "Dropout requires exactly one input key."
        assert len(self.keys_out) == 1, "Dropout requires exactly one output key."
        self.dropout = torch.nn.Dropout(p)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.dropout(x)


@layer("Conv1dProjector")
class Conv1dProjector(PipelineLayer):
    """Pointwise Conv1d projector for sequence inputs (B, T, C).

    Equivalent to Linear but uses Conv1d for better memory access patterns.

    Args:
        target_dim: Output channel dimension. If None, uses output_sizes.
        hidden_dim: Hidden size for multi-layer projectors.
        n_layers: Number of projection layers.
        bias: Whether to use bias.
        activation: Dotted class path for activation.
        normalization: Dotted class path for normalization.
    """

    def __init__(
        self,
        target_dim: int | None = None,
        hidden_dim: int = 256,
        n_layers: int = 1,
        bias: bool = True,
        activation: str | None = None,
        normalization: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(self.keys_in) == 1, "Conv1dProjector requires exactly one input key."
        assert len(self.keys_out) == 1, "Conv1dProjector requires exactly one output key."

        input_size = self.input_sizes[cast(list[str], self.keys_in)[0]]
        if target_dim is not None:
            output_size = (target_dim,)
        elif self.output_sizes and self.keys_out[0] in self.output_sizes:
            output_size = self.output_sizes[self.keys_out[0]]
        else:
            raise ValueError("target_dim or output_sizes must be specified.")

        activation_cls = _locate(activation)
        normalization_cls = _locate(normalization)

        dim_in = input_size[-1]
        dim_out = output_size[-1]

        layers: list[torch.nn.Module] = []
        for i in range(n_layers):
            is_last = i == n_layers - 1
            layer_in = dim_in if i == 0 else hidden_dim
            layer_out = dim_out if is_last else hidden_dim
            layers.append(torch.nn.Conv1d(layer_in, layer_out, kernel_size=1, bias=bias))
            if normalization_cls is not None:
                layers.append(normalization_cls(layer_out))
            if activation_cls is not None:
                layers.append(activation_cls())

        self.model = torch.nn.Sequential(*layers)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = self.model(x)
        return x.transpose(-1, -2)


@layer("LowRankProjector")
class LowRankProjector(PipelineLayer):
    """Low-rank factorized projection W ≈ U @ V.

    Reduces parameters from O(C*D) to O((C+D)*r) when r << min(C, D).

    Args:
        target_dim: Output dimension. If None, uses output_sizes.
        rank: Bottleneck rank.
        bias: Whether to use bias.
        activation: Optional activation between U and V.
    """

    def __init__(
        self,
        target_dim: int | None = None,
        rank: int = 64,
        bias: bool = True,
        activation: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(self.keys_in) == 1
        assert len(self.keys_out) == 1

        input_size = self.input_sizes[cast(list[str], self.keys_in)[0]]
        if target_dim is not None:
            output_size = (target_dim,)
        elif self.output_sizes and self.keys_out[0] in self.output_sizes:
            output_size = self.output_sizes[self.keys_out[0]]
        else:
            raise ValueError("target_dim or output_sizes must be specified.")

        dim_in = input_size[-1]
        dim_out = output_size[-1]
        activation_cls = _locate(activation)

        layers: list[torch.nn.Module] = [torch.nn.Linear(dim_in, rank, bias=False)]
        if activation_cls is not None:
            try:
                layers.append(activation_cls(inplace=True))
            except TypeError:
                layers.append(activation_cls())
        layers.append(torch.nn.Linear(rank, dim_out, bias=bias))
        self.model = torch.nn.Sequential(*layers)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x)
