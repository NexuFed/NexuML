"""High-level joint observations/commands, never an actuator servo loop."""

from typing import Protocol


class Ros2Transport(Protocol):
    """Injectable transport shared by production middleware and deterministic tests."""

    def receive_joint_state(self, timeout_s: float) -> tuple[list[float], list[float]]:
        """Receive a fresh, ordered position/velocity sample or raise TimeoutError."""
        ...

    def send_joint_command(self, positions: list[float], control_period_s: float) -> None:
        """Publish bounded joint-position targets with a trajectory duration."""
        ...

    def reset(self, timeout_s: float) -> None:
        """Invoke the configured simulation reset hook or raise TimeoutError."""
        ...

    def close(self) -> None:
        """Release resources owned by this transport only."""
        ...
