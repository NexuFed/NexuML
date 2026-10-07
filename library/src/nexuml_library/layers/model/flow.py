"""Continuous flow model layer."""

from __future__ import annotations

from typing import cast

import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml_library.layers.generative.flow.euler import EulerIntegrator
from nexuml_library.layers.generative.flow.vector_field import TimeConditionedVectorField


@layer("Flow")
class Flow(LayerDefinition):
    """Trainable vector field with Euler sampling."""

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 128])
    num_steps: int = Field(default=100, gt=0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        vector_field = TimeConditionedVectorField(hidden_dims=self.hidden_dims).build(context)
        state_key = context.keys_in[0]
        integrator = EulerIntegrator(
            velocity=getattr(vector_field, "velocity"),
            num_steps=self.num_steps,
            input_sizes={state_key: tuple(context.input_sizes[state_key])},
            keys_in=[state_key],
            keys_out=["flow_sample"],
        )
        return _FlowRuntime(
            vector_field=vector_field,
            integrator=integrator,
            **context.runtime_kwargs(),
        )


class _FlowRuntime(PipelineLayer):
    def __init__(
        self,
        vector_field: PipelineLayer,
        integrator: EulerIntegrator,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vector_field = vector_field
        self.integrator = integrator

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        return self.vector_field(x, y)

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned vector field.

        Returns:
            Predicted velocity with the same shape as the state.
        """
        return cast(torch.Tensor, getattr(self.vector_field, "velocity")(state, time))

    def sample(self, x0: torch.Tensor) -> torch.Tensor:
        """Integrate samples from time zero to one.

        Returns:
            Final Euler-integrated state.
        """
        return self.integrator.forward_tensor(x0)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
