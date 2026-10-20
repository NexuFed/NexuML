"""Time-conditioned vector field for Flow Matching."""

from __future__ import annotations

import math
from typing import cast

import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml_library.layers.feature.projector import Linear


@layer("TimeConditionedVectorField")
class TimeConditionedVectorField(LayerDefinition):
    """Vector field using the reusable linear projector as its backbone."""

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 128])

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        feature_dim = math.prod(context.input_sizes[context.keys_in[0]])
        backbone = Linear(
            target_dim=feature_dim,
            hidden_dims=self.hidden_dims,
            activation="torch.nn.SiLU",
            skip_last_activation=True,
        ).build(
            LayerBuildContext(
                input_sizes={"vector_field_input": (feature_dim + 1,)},
                keys_in=["vector_field_input"],
                keys_out=["vector_field_output"],
            )
        )
        return _TimeConditionedVectorFieldRuntime(
            backbone=backbone,
            **context.runtime_kwargs(),
        )


class _TimeConditionedVectorFieldRuntime(PipelineLayer):
    def __init__(self, backbone: PipelineLayer, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

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
        return self.backbone.forward_tensor(inputs).reshape_as(state)

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
