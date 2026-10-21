"""TensorDict interaction semantics without a simulator-framework dependency."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Any

import numpy as np
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.types import InteractionContract


@dataclass(frozen=True, slots=True)
class InteractionStep:
    """One transition; termination and time-limit truncation stay distinct."""

    observation: TensorDict
    reward: Tensor | None
    terminated: Tensor
    truncated: Tensor
    info: Mapping[str, Any] | None = None
    groups: TensorDict | None = None


class EnvironmentRuntime(Protocol):
    """Execution-local environment; implementations own deterministic cleanup."""

    @property
    def contract(self) -> InteractionContract:
        """Portable description matching this runtime's vectorization."""
        ...

    def reset(self, *, seed: int | None = None, mask: Tensor | None = None) -> TensorDict:
        """Start new trajectories and return tensor observations."""
        ...

    def step(self, action: TensorDict) -> InteractionStep:
        """Apply a tensor action and return explicit transition signals."""
        ...

    def close(self) -> None:
        """Release all runtime resources, including after failure."""
        ...


class RGBEnvironmentRuntime(Protocol):
    """Optional evaluation rendering, independent of policy observation fields."""

    @property
    def render_fps(self) -> float | None:
        """Playback rate supplied by the renderer, when available."""
        ...

    def render_frame(self) -> np.ndarray:
        """Return one CPU uint8 HWC RGB frame."""
        ...
