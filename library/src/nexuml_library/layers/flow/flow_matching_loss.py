"""Flow Matching objective layer."""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("FlowMatchingLoss")
class FlowMatchingLoss(LayerDefinition):
    """Per-sample MSE between predicted and target path velocities."""

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _FlowMatchingLossRuntime(**context.runtime_kwargs())


class _FlowMatchingLossRuntime(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        if len(keys_in) != 2:
            raise ValueError("FlowMatchingLoss expects predicted and target velocity input keys")
        if len(keys_out) != 1:
            raise ValueError("FlowMatchingLoss expects exactly one output key")

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y
        if not isinstance(x, TensorDict):
            raise TypeError("FlowMatchingLoss requires TensorDict input")

        keys_in = cast(list[str], self.keys_in)
        predicted = cast(torch.Tensor, x[keys_in[0]])
        target = cast(torch.Tensor, x[keys_in[1]])
        if predicted.shape != target.shape:
            raise ValueError(
                "FlowMatchingLoss requires matching velocity shapes: "
                f"predicted={tuple(predicted.shape)}, target={tuple(target.shape)}"
            )

        batch_size = predicted.shape[0]
        loss = (predicted - target).pow(2).reshape(batch_size, -1).mean(dim=-1)
        x[self.keys_out[0]] = loss
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("FlowMatchingLoss consumes two TensorDict inputs")
