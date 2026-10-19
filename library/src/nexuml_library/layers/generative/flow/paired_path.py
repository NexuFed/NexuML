"""Linear Flow Matching path between observed healthy and faulty endpoints."""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml_library.layers.generative.flow.linear_path import _linear_path


@layer("PairedLinearFlowPath")
class PairedLinearFlowPath(LayerDefinition):
    """Interpolate two observed endpoint tensors without Gaussian source noise.

    The source and target may be independently sampled from their respective
    condition distributions. This is unconditional Flow Matching; time is the
    integration coordinate rather than a physical machine age.
    """

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build the runtime path using NexuML TensorDict routing.

        Returns:
            Path runtime with interpolated state, time and target velocity.

        Raises:
            ValueError: If the endpoint shapes or routing contracts disagree.
        """
        if len(context.keys_in) != 2 or len(context.keys_out) != 3:
            raise ValueError("PairedLinearFlowPath requires two inputs and three outputs")
        if context.input_sizes[context.keys_in[0]] != context.input_sizes[context.keys_in[1]]:
            raise ValueError("Healthy and faulty endpoint feature shapes must match")
        return _PairedLinearFlowPathRuntime(**context.runtime_kwargs())


class _PairedLinearFlowPathRuntime(PipelineLayer):
    """Private runtime implementing independent source-target interpolation."""

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        """Sample a time and insert interpolated state, time and velocity.

        Returns:
            Modified TensorDict and unchanged targets.

        Raises:
            TypeError: If the runtime input is not a TensorDict.
            ValueError: If endpoint tensors have incompatible shapes.
        """
        if not self.check_update():
            return x, y
        if not isinstance(x, TensorDict):
            raise TypeError("PairedLinearFlowPath requires TensorDict input")

        keys_in = cast(list[str], self.keys_in)
        healthy = cast(torch.Tensor, x[keys_in[0]])
        faulty = cast(torch.Tensor, x[keys_in[1]])
        if healthy.shape != faulty.shape:
            raise ValueError("Healthy and faulty endpoint tensors must share a shape")
        time = torch.rand((healthy.shape[0], 1), device=healthy.device, dtype=healthy.dtype)
        state, target_velocity = _linear_path(healthy, faulty, time)

        x[self.keys_out[0]] = state
        x[self.keys_out[1]] = time
        x[self.keys_out[2]] = target_velocity
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
