"""Matched direct-future and conditional-flow models for per-machine degradation.

Flow interpolation s in [0,1] is an artificial transport coordinate. The
requested horizon h is measured in *physical C-MAPSS engine cycles*. The
conditional vector field is trained against observed same-engine transitions
only, never random pairings of different machines.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DynamicMethod = Literal["direct_future", "flow"]


class MachineFutureDynamics(nn.Module):
    """Parameter-matched direct and conditional-flow latent forecasters.

    In both variants the neural network has exactly 2D+2 inputs and D outputs.
    The flow conditions on initial state and actual physical horizon; the direct
    model predicts a residual from the same initial state and horizon.
    """

    def __init__(
        self,
        *,
        method: DynamicMethod,
        dimension: int = 128,
        hidden_dim: int = 128,
        horizon_scale: float = 130.0,
    ) -> None:
        super().__init__()
        if method not in ("direct_future", "flow"):
            raise ValueError("Unknown within-machine dynamics method")
        if dimension < 1 or hidden_dim < 1 or horizon_scale <= 0:
            raise ValueError("Dynamics dimensions and horizon scale must be positive")
        self.method = method
        self.dimension = dimension
        self.horizon_scale = horizon_scale
        self.velocity = nn.Sequential(
            nn.Linear(2 * dimension + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dimension),
        )

    def _field(
        self, interpolated: Tensor, initial: Tensor, s: Tensor, horizon: Tensor
    ) -> Tensor:
        """Return conditional physical-horizon displacement.

        Returns:
            Neural displacement vector of shape [batch, dimension].

        Raises:
            ValueError: If a batch is incorrectly shaped.
        """
        if (
            initial.ndim != 2
            or initial.shape[1] != self.dimension
            or interpolated.shape != initial.shape
            or s.shape != (len(initial), 1)
            or horizon.shape != (len(initial), 1)
        ):
            raise ValueError("Forecast inputs must share [B,D] and [B,1] shapes")
        return self.velocity(
            torch.cat(
                (interpolated, initial, s, horizon / self.horizon_scale), dim=-1
            )
        )

    def training_loss(
        self,
        start: Tensor,
        end: Tensor,
        horizon: Tensor,
    ) -> Tensor:
        """Supervise a paired, same-engine transition.

        Flow uses the conditional linear probability path with an independently
        sampled interpolation coordinate; direct prediction uses a fixed
        interpolation coordinate 0 and the identical endpoint displacement.

        Returns:
            Mean squared endpoint displacement or flow velocity error.

        Raises:
            ValueError: If paired endpoints or cycle horizons are inconsistent.
        """
        if end.shape != start.shape or horizon.shape != (len(start), 1):
            raise ValueError("Paired dynamics endpoints or horizons are misaligned")
        if not bool(torch.isfinite(horizon).all()) or not bool((horizon > 0).all()):
            raise ValueError("Paired physical cycle horizons must be finite and positive")
        if self.method == "flow":
            s = torch.rand_like(horizon)
            state = torch.lerp(start, end, s)
        else:
            s = torch.zeros_like(horizon)
            state = start
        velocity = self._field(state, start, s, horizon)
        return F.mse_loss(velocity, end - start)

    def forecast(
        self, start: Tensor, horizon: Tensor, *, integration_steps: int = 8
    ) -> Tensor:
        """Predict the latent state at an actual future cycle horizon.

        Flow integrates its *conditional interpolation* ODE from s=0 to s=1
        for a fixed physical horizon. It does not interpret s as engine age.

        Returns:
            Predicted [B,D] future latent representation.

        Raises:
            ValueError: If an invalid integration count or horizon is supplied.
        """
        if integration_steps < 1:
            raise ValueError("Flow integration requires at least one Euler step")
        if horizon.shape != (len(start), 1) or not bool((horizon >= 0).all()):
            raise ValueError("Requested forecast horizons must be nonnegative [B,1]")
        if not bool(torch.isfinite(horizon).all()):
            raise ValueError("Forecast horizon must be finite")
        if self.method == "direct_future":
            result = start + self._field(
                start, start, torch.zeros_like(horizon), horizon
            )
        else:
            result = start
            for step in range(integration_steps):
                interpolation = torch.full_like(horizon, step / integration_steps)
                result = result + self._field(
                    result, start, interpolation, horizon
                ) / integration_steps
        # The learned residual at an exactly zero horizon is not guaranteed
        # to vanish. Enforce the identity boundary algebraically.
        return torch.where(horizon == 0, start, result)

    def parameter_count(self) -> int:
        """Report a matched neural parameter budget.

        Returns:
            Number of learned neural parameters.
        """
        return sum(parameter.numel() for parameter in self.parameters())
