"""TensorDict interaction semantics without a simulator-framework dependency."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Any, runtime_checkable

import numpy as np
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.types import InteractionContract


@dataclass(frozen=True, slots=True)
class InteractionTrace:
    """Fixed-step action receipt; native frames are not wall or sensor time."""

    source: str
    requested_action: TensorDict
    native_action: Tensor
    issued_monotonic_ns: int
    native_clock: str
    native_frame_before: int
    native_frame_after: int
    control_period_s: float


@dataclass(frozen=True, slots=True)
class InteractionStep:
    """One transition; termination and time-limit truncation stay distinct."""

    observation: TensorDict
    reward: Tensor | None
    terminated: Tensor
    truncated: Tensor
    info: Mapping[str, Any] | None = None
    groups: TensorDict | None = None
    trace: InteractionTrace | None = None


class EnvironmentRuntime(Protocol):
    """Execution-local environment; implementations own deterministic cleanup."""

    @property
    def contract(self) -> InteractionContract:
        """Portable description matching this runtime's vectorization."""
        ...

    def reset(self, *, seed: int | None = None, mask: Tensor | None = None) -> TensorDict:
        """Start new trajectories and return tensor observations.

        During play, package-owned inputs are raw; runtime-owned inputs remain processed.
        """
        ...

    def step(self, action: TensorDict) -> InteractionStep:
        """Apply a synchronous tensor action and return pre-reset ending observations."""
        ...

    def close(self) -> None:
        """Release all runtime resources, including after failure."""
        ...


@runtime_checkable
class TraceableEnvironmentRuntime(EnvironmentRuntime, Protocol):
    """Optional fixed-step action recording; not an asynchronous command API."""

    def step_recorded(self, action: TensorDict, *, source: str) -> InteractionStep:
        """Step through the same validator and record the provider submission."""
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
