"""ODE solvers used by continuous-flow models."""

from __future__ import annotations

from collections.abc import Callable

import torch

VectorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def euler_integrate(
    vector_field: VectorField,
    x0: torch.Tensor,
    *,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 100,
) -> torch.Tensor:
    """Integrate "dx/dt = v(x,t)" with fixed-step explicit Euler.

    Returns:
        State after integration from t_start to t_end.

    Raises:
        ValueError: If num_steps is non-positive or the vector field changes state shape.
        TypeError: If x0 is not a floating-point tensor.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not x0.is_floating_point():
        raise TypeError("Euler integration requires a floating-point initial state")

    state = x0
    dt = (float(t_end) - float(t_start)) / num_steps
    for step in range(num_steps):
        time_value = float(t_start) + step * dt
        time = torch.full(
            (state.shape[0], 1),
            time_value,
            device=state.device,
            dtype=state.dtype,
        )
        velocity = vector_field(state, time)
        if velocity.shape != state.shape:
            raise ValueError(
                "Vector field output must match state shape during Euler integration: "
                f"state={tuple(state.shape)}, velocity={tuple(velocity.shape)}"
            )
        state = state + dt * velocity
    return state
