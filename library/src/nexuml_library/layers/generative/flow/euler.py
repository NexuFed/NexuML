"""Euler integration layer for continuous flows."""

from __future__ import annotations

from collections.abc import Callable

import torch

from nexuml.core.base_layer import PipelineLayer


class EulerIntegrator(PipelineLayer):
    """Fixed-step Euler integrator over a learned vector field."""

    def __init__(
        self,
        velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_steps: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self.velocity = velocity
        self.num_steps = num_steps
        self.last_trajectory: torch.Tensor | None = None

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("Euler integration requires a floating-point state")

        state = x
        trajectory = [state]
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            time = torch.full(
                (state.shape[0], 1),
                step * dt,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self.velocity(state, time)
            if velocity.shape != state.shape:
                raise ValueError("Vector field output must match the state shape")
            state = state + dt * velocity
            trajectory.append(state)
        self.last_trajectory = torch.stack(trajectory, dim=1).detach()
        return state
