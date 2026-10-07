"""Linear probability path for Flow Matching."""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


def _linear_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = t.reshape(t.shape[0], *([1] * (x1.ndim - 1)))
    return (1.0 - t) * x0 + t * x1, x1 - x0


@layer("LinearFlowPath")
class LinearFlowPath(LayerDefinition):
    """Sample the linear path from standard Gaussian noise to data."""

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _LinearFlowPathRuntime(**context.runtime_kwargs())


class _LinearFlowPathRuntime(PipelineLayer):
    def __init__(self, keys_in: list[str], keys_out: list[str], **kwargs):
        super().__init__(keys_in=keys_in, keys_out=keys_out, **kwargs)
        if len(keys_in) != 1 or len(keys_out) != 3:
            raise ValueError("LinearFlowPath requires one input and three output keys")

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y
        if not isinstance(x, TensorDict):
            raise TypeError("LinearFlowPath requires TensorDict input")

        keys_in = cast(list[str], self.keys_in)
        x1 = cast(torch.Tensor, x[keys_in[0]])
        x0 = torch.randn_like(x1)
        t = torch.rand((x1.shape[0], 1), device=x1.device, dtype=x1.dtype)
        xt, target_velocity = _linear_path(x0, x1, t)

        x[self.keys_out[0]] = xt
        x[self.keys_out[1]] = t
        x[self.keys_out[2]] = target_velocity
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
