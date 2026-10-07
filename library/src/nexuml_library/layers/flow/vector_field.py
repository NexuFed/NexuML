"""Time-conditioned vector field for Flow Matching."""

from __future__ import annotations

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
    """MLP vector field conditioned by concatenating scalar time."""

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 128])

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
        hidden_dims: list[int],
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        if len(keys_in) != 2 or len(keys_out) != 1:
            raise ValueError("TimeConditionedVectorField requires two inputs and one output")

        feature_dim = math.prod(input_sizes[keys_in[0]])
        dims = [feature_dim + 1, *hidden_dims, feature_dim]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(nn.SiLU())
        self.model = nn.Sequential(*layers)

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Evaluate the vector field.

        Returns:
            Predicted velocity with the same shape as the state.
        """
        batch_size = state.shape[0]
        inputs = torch.cat(
            [state.reshape(batch_size, -1), time.reshape(batch_size, 1)],
            dim=1,
        )
        return self.model(inputs).reshape_as(state)

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
        raise NotImplementedError
