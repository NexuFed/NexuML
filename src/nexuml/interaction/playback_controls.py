"""Bounded input from a qualified simulated playback viewer."""

from queue import Empty, Full, Queue
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal


PlaybackCommand = Literal["pause", "resume", "step", "reset", "speed", "task_reset"]
PlaybackValue = int | float | tuple[str, str] | None
PlaybackRequest = tuple[PlaybackCommand, PlaybackValue]
PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 2.0)


class PlaybackControls:
    """Thread-safe inbox; requests never call a simulator or touch a policy."""

    def __init__(
        self,
        num_envs: int,
        *,
        selected_reset: bool,
        task_options: Mapping[str, tuple[str, ...]] | None = None,
    ):
        self.num_envs = num_envs
        self.selected_reset = selected_reset
        options = dict(task_options or {})
        if any(
            type(name) is not str
            or not name.isidentifier()
            or type(choices) is not tuple
            or not choices
            or any(type(choice) is not str or not choice.isidentifier() for choice in choices)
            or len(set(choices)) != len(choices)
            for name, choices in options.items()
        ):
            raise ValueError("Playback task reset options require named, unique finite choices")
        self.task_options = MappingProxyType(options)
        self._requests: Queue[PlaybackRequest] = Queue(maxsize=16)

    def accepts_task_option(self, name: str, choice: str) -> bool:
        """Check a task reset option against the provider's declaration.

        Returns:
            Whether the provider declared this exact name and choice.
        """
        return (
            type(name) is str and type(choice) is str and choice in self.task_options.get(name, ())
        )

    def submit(self, command: PlaybackCommand, value: PlaybackValue = None) -> bool:
        """Accept only known, bounded requests from the viewer.

        Returns:
            True if queued; false for a rejected request or a full inbox.
        """
        if type(command) is not str:
            return False
        if command == "reset":
            if value is not None and (
                not self.selected_reset or type(value) is not int or not 0 <= value < self.num_envs
            ):
                return False
        elif command == "speed":
            if type(value) not in (int, float) or value not in PLAYBACK_SPEEDS:
                return False
        elif command == "task_reset":
            if type(value) is not tuple or len(value) != 2 or not self.accepts_task_option(*value):
                return False
        elif command not in ("pause", "resume", "step") or value is not None:
            return False
        try:
            self._requests.put_nowait((command, value))
        except Full:
            return False
        return True

    def receive(self, *, timeout: float = 0.0) -> PlaybackRequest | None:
        """Read one admitted request on the owner thread.

        Returns:
            One request or None after a bounded wait.
        """
        try:
            return self._requests.get(timeout=timeout)
        except Empty:
            return None
