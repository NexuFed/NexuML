"""Linear probability-path layer for Flow Matching."""

from __future__ import annotations

from typing import cast

import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


def _broadcast_time(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Reshape per-sample times so they broadcast over "reference".

    Returns:
        Time tensor reshaped for broadcasting over the reference tensor.

    Raises:
        ValueError: If the time and reference batch dimensions differ.
    """
    if t.shape[0] != reference.shape[0]:
        raise ValueError(
            "Time tensor batch dimension must match the reference tensor: "
            f"time={tuple(t.shape)}, reference={tuple(reference.shape)}"
        )
    return t.reshape(t.shape[0], *([1] * (reference.ndim - 1)))


def linear_interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the linear probability path "x_t = (1-t)x_0 + t x_1".

    Returns:
        Interpolated state at time t.

    Raises:
        ValueError: If x0 and x1 do not have identical shapes.
    """
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have identical shapes, got {x0.shape} and {x1.shape}")
    t_broadcast = _broadcast_time(t, x1)
    return (1.0 - t_broadcast) * x0 + t_broadcast * x1


def linear_target_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Return the constant velocity of the linear path, "u_t = x_1 - x_0".

    Returns:
        Constant target velocity along the linear path.

    Raises:
        ValueError: If x0 and x1 do not have identical shapes.
    """
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have identical shapes, got {x0.shape} and {x1.shape}")
    return x1 - x0


@layer("LinearFlowPath")
class LinearFlowPath(LayerDefinition):
    """Sample a linear Flow Matching path from Gaussian source to data.

    The layer reads one data tensor "x_1" and writes three tensors in this
    order: the interpolated state "x_t", the sampled time "t", and the
    target velocity "u_t = x_1 - x_0".
    """

    noise_std: float = Field(default=1.0, gt=0.0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _LinearFlowPathRuntime(**context.runtime_kwargs(), **self.model_dump())


class _LinearFlowPathRuntime(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        if len(keys_in) != 1:
            raise ValueError(f"LinearFlowPath expects exactly one input key, got {keys_in}")
        if len(keys_out) != 3:
            raise ValueError(
                "LinearFlowPath expects exactly three output keys "
                "for state, time, and target velocity"
            )
        self.noise_std = noise_std

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
        if not x1.is_floating_point():
            raise TypeError("LinearFlowPath requires floating-point feature tensors")

        x0 = torch.randn_like(x1) * self.noise_std
        t = torch.rand((x1.shape[0], 1), device=x1.device, dtype=x1.dtype)
        xt = linear_interpolate(x0, x1, t)
        target_velocity = linear_target_velocity(x0, x1)

        x[self.keys_out[0]] = xt
        x[self.keys_out[1]] = t
        x[self.keys_out[2]] = target_velocity
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("LinearFlowPath routes multiple outputs through TensorDict")
