"""Post-training conversion of one-step battery capacity forecasts to EOL RUL.

A learned capacity forecaster is iterated without access to future measurements.
A non-crossing forecast is *censored*, represented by -1, never assigned a
fabricated remaining life. This is an inference utility, not a trained head.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def rollout_capacity_to_eol(
    predict_next: Callable[[Tensor], Tensor],
    history_normalized: Tensor,
    *,
    capacity_offset_ah: float,
    capacity_scale_ah: float,
    end_of_life_ah: float,
    max_cycles: int,
) -> Tensor:
    """Forecast future capacities until the documented battery EOL threshold.

    Args:
        predict_next: Trained next-cycle model, returning raw capacity in Ah.
        history_normalized: [batch,lookback,1] using training-only min/max.
        capacity_offset_ah: Training-only minimum capacity in Ah.
        capacity_scale_ah: Training-only max-minus-min capacity in Ah.
        end_of_life_ah: Actual chemistry-specific failure threshold.
        max_cycles: Conservative finite extrapolation horizon.

    Returns:
        Predicted cycles to first EOL crossing, or -1 if not crossed in horizon.

    Raises:
        ValueError: If dimensions or protocol parameters are invalid.
    """
    if (
        history_normalized.ndim != 3
        or history_normalized.shape[-1] != 1
        or history_normalized.shape[1] == 0
        or not bool(torch.isfinite(history_normalized).all())
        or capacity_scale_ah <= 0
        or end_of_life_ah <= 0
        or max_cycles <= 0
    ):
        raise ValueError("Invalid battery capacity history or EOL rollout settings")
    history = history_normalized.detach().clone()
    result = torch.full(
        (len(history),), -1, dtype=torch.long, device=history.device
    )
    last_capacity = history[:, -1, 0] * capacity_scale_ah + capacity_offset_ah
    result[last_capacity <= end_of_life_ah] = 0
    with torch.no_grad():
        for step in range(1, max_cycles + 1):
            capacity = predict_next(history)
            if capacity.shape != (len(history), 1) or not torch.isfinite(capacity).all():
                raise ValueError("Battery model must emit finite [batch,1] raw Ah")
            newly_crossed = (capacity[:, 0] <= end_of_life_ah) & (result < 0)
            result[newly_crossed] = step
            next_normalized = (capacity - capacity_offset_ah) / capacity_scale_ah
            history = torch.cat((history[:, 1:], next_normalized.unsqueeze(1)), dim=1)
            if bool((result >= 0).all()):
                break
    return result
