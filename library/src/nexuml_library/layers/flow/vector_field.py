"""Time-conditioned vector-field layer for Flow Matching."""

from __future__ import annotations

import importlib
import math
from typing import cast

import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("TimeConditionedVectorField")
class TimeConditionedVectorField(LayerDefinition):
    """MLP vector field "v_theta(x_t, t)" with sinusoidal time features."""

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 128])
    time_embedding_dim: int = Field(default=16, ge=1)
    activation: str = "torch.nn.SiLU"
    bias: bool = True

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _TimeConditionedVectorFieldRuntime(
            **context.runtime_kwargs(),
            **self.model_dump(),
        )


class _TimeConditionedVectorFieldRuntime(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        hidden_dims: list[int] | None = None,
        time_embedding_dim: int = 16,
        activation: str = "torch.nn.SiLU",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        if len(keys_in) != 2:
            raise ValueError(
                "TimeConditionedVectorField expects two input keys: state and time"
            )
        if len(keys_out) != 1:
            raise ValueError("TimeConditionedVectorField expects exactly one output key")
        if keys_in[0] not in input_sizes:
            raise ValueError(
                f"State key {keys_in[0]!r} not found in input sizes: {list(input_sizes)}"
            )
        hidden_dims = hidden_dims or []
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"hidden_dims must contain positive integers, got {hidden_dims}")

        self.feature_shape = tuple(input_sizes[keys_in[0]])
        self.feature_dim = math.prod(self.feature_shape)
        self.time_embedding_dim = time_embedding_dim
        self.model = self._build_model(
            hidden_dims=hidden_dims,
            activation=activation,
            bias=bias,
        )

    def _build_model(
        self,
        *,
        hidden_dims: list[int],
        activation: str,
        bias: bool,
    ) -> nn.Sequential:
        activation_cls = _resolve_activation(activation)
        dims = [self.feature_dim + self.time_embedding_dim, *hidden_dims, self.feature_dim]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
            if index < len(dims) - 2:
                layers.append(activation_cls())
        return nn.Sequential(*layers)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        batch_size = t.shape[0]
        t_scalar = t.reshape(batch_size, -1)[:, :1]
        if self.time_embedding_dim == 1:
            return t_scalar

        half_dim = self.time_embedding_dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * exponent
        )
        angles = t_scalar * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.time_embedding_dim:
            embedding = torch.cat(
                [embedding, torch.zeros_like(t_scalar)],
                dim=-1,
            )
        return embedding[:, : self.time_embedding_dim]

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned vector field on explicit state/time tensors.

        Returns:
            Predicted velocity with the same shape as the state.

        Raises:
            ValueError: If batch dimensions or flattened feature dimensions do not match.
        """
        if state.shape[0] != time.shape[0]:
            raise ValueError(
                "State and time batch dimensions must match: "
                f"state={tuple(state.shape)}, time={tuple(time.shape)}"
            )
        batch_size = state.shape[0]
        state_flat = state.reshape(batch_size, -1)
        if state_flat.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected flattened state dimension {self.feature_dim}, "
                f"got {state_flat.shape[1]}"
            )
        time_embedding = self._embed_time(time)
        velocity_flat = self.model(torch.cat([state_flat, time_embedding], dim=-1))
        return velocity_flat.reshape_as(state)

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y
        if not isinstance(x, TensorDict):
            raise TypeError("TimeConditionedVectorField requires TensorDict input")

        keys_in = cast(list[str], self.keys_in)
        state = cast(torch.Tensor, x[keys_in[0]])
        time = cast(torch.Tensor, x[keys_in[1]])
        x[self.keys_out[0]] = self.velocity(state, time)
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError(
            "TimeConditionedVectorField consumes state and time through TensorDict"
        )


def _resolve_activation(path: str) -> type[nn.Module]:
    parts = path.rsplit(".", 1)
    if len(parts) == 1:
        activation = getattr(nn, path, None)
    else:
        module = importlib.import_module(parts[0])
        activation = getattr(module, parts[1], None)
    if (
        activation is None
        or not isinstance(activation, type)
        or not issubclass(activation, nn.Module)
    ):
        raise ValueError(f"Activation {path!r} does not resolve to a torch.nn.Module class")
    return activation
