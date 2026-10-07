"""Flow Matching objective."""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("FlowMatchingLoss")
class FlowMatchingLoss(LayerDefinition):
    """Per-sample MSE between predicted and target velocity."""

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _FlowMatchingLossRuntime(**context.runtime_kwargs())


class _FlowMatchingLossRuntime(PipelineLayer):
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
            raise ValueError("Predicted and target velocity shapes must match")

        error = (predicted - target).pow(2).reshape(predicted.shape[0], -1)
        x[self.keys_out[0]] = error.mean(dim=1)
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
