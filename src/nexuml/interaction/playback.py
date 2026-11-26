"""Inference-only playback over the ordinary simulation interaction boundary."""

from dataclasses import dataclass, field
from collections.abc import Callable
from threading import Event
from time import monotonic, monotonic_ns
import math
from typing import Literal, TYPE_CHECKING, cast

import torch

from nexuml.core.components import (
    BenchmarkEnvironmentDefinition,
    EnvironmentBuildContext,
    EnvironmentDefinition,
    PlaybackViewer,
)
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import RLVideoSpec
from nexuml.evaluation.policy import _evaluation_state, _validate_evaluation_contract
from nexuml.interaction.media import capture_rgb
from nexuml.interaction.playback_controls import PlaybackControls
from nexuml.interaction.runtime import RGBEnvironmentRuntime

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    """Actual counts at a native control boundary, not an evaluation score."""

    steps: int
    transitions: int
    completed_episodes: int
    reason: Literal["step_limit", "episode_limit", "stop_requested"]
    interventions: dict[str, int] = field(default_factory=dict)


def play_policy(
    policy: CompiledPolicy,
    definition: EnvironmentDefinition,
    *,
    num_envs: int = 1,
    steps: int | None = None,
    episodes: int | None = None,
    seed: int | None = None,
    viewer: PlaybackViewer = "headless",
    deterministic: bool = True,
    real_time: bool = False,
    stop: Event | None = None,
    video: RLVideoSpec | None = None,
    media_sink: Callable[[list["np.ndarray"], dict], None] | None = None,
) -> PlaybackResult:
    """Play in the caller's ``definition.job_scope()``, like standalone evaluation.

    Process package-owned inputs with policy state local to this session.
    An episode limit can be exceeded by simultaneous ends on the final tick.

    Returns:
        Completed native steps, transitions, and episodes, without formal metrics.

    Raises:
        ValueError: If the definition, settings, contract or native batch is unsupported.
    """
    if isinstance(definition, BenchmarkEnvironmentDefinition):
        raise ValueError("Playback requires a playable environment, not a benchmark suite")
    definition.validate_playback(viewer)
    if video is not None and video.enabled and (media_sink is None or num_envs != 1):
        raise ValueError("Playback recording requires a single environment and an artifact sink")
    for name, count in (("num_envs", num_envs), ("steps", steps), ("episodes", episodes)):
        if count is None and name != "num_envs":
            continue
        if type(count) is not int or count < 1:
            raise ValueError(f"Playback {name} must be a positive integer")
    declared = definition.describe()
    _validate_evaluation_contract(policy.contract, declared)
    period = declared.control_period_s if real_time else None
    if real_time and (period is None or not math.isfinite(period) or period <= 0):
        raise ValueError("Real-time playback requires a finite positive native control period")
    anchor = next(policy.parameters(), None)
    if anchor is None:
        anchor = next(policy.buffers(), torch.empty(0))
    context = EnvironmentBuildContext(
        purpose="play",
        seed=seed,
        device=str(anchor.device),
        num_envs=num_envs,
        viewer=viewer,
        render_mode="rgb_array" if video is not None and video.enabled else None,
    )
    expected = (num_envs,) if num_envs > 1 or declared.groups else ()
    stop = stop or Event()
    if stop.is_set():
        return PlaybackResult(0, 0, 0, "stop_requested")
    ticks = completed = 0
    interventions: dict[str, int] = {}
    frames = []
    times = []
    source_size = None
    fps = None
    recording_finished = False

    def publish_recording(reason: str) -> None:
        nonlocal recording_finished, frames
        if recording_finished or not frames:
            return
        recording_finished = True
        assert media_sink is not None
        captured, frames = frames, []
        media_sink(
            captured,
            {
                "fps": fps,
                "source_size": source_size,
                "native_ticks": list(range(1, len(captured) + 1)),
                "monotonic_capture_ns": times,
                "end": reason,
                "seed": seed,
                "deterministic": deterministic,
                "viewer": viewer,
                "interventions": dict(interventions),
            },
        )

    reason: Literal["step_limit", "episode_limit", "stop_requested"]
    with _evaluation_state(policy, seed):
        runtime = definition.build(context)
        try:
            _validate_evaluation_contract(policy.contract, runtime.contract, num_envs=num_envs)
            if runtime.contract.physical != declared.physical:
                raise ValueError("Playback runtime provenance differs from its definition")
            if video is not None and video.enabled:
                if not callable(getattr(runtime, "render_frame", None)):
                    raise ValueError("Playback environment does not provide RGB recording")
                fps = video.fps or getattr(runtime, "render_fps", None)
                if fps is None or not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Recording requires renderer FPS metadata or video.fps")
            controls = getattr(runtime, "playback_controls", None)
            if controls is not None:
                if not isinstance(controls, PlaybackControls) or controls.num_envs != num_envs:
                    raise ValueError("Playback control inbox does not match the native batch")
                period = declared.control_period_s
                if period is None or not math.isfinite(period) or period <= 0:
                    raise ValueError("Viewer speed controls require a finite native control period")
            paused, speed = False, 1.0
            observation = runtime.reset(seed=seed)
            state = None
            stateful = policy.processor is not None or (
                bool(policy.contract.groups)
                and any(actor.processor is not None for actor in policy.actors.values())
            )
            while True:
                if stop.is_set():
                    reason = "stop_requested"
                    break
                single_step = False
                if controls is not None:
                    request = controls.receive(timeout=0.05 if paused else 0.0)
                    if request is not None:
                        command, value = request
                        intervention: str = command
                        if command == "pause":
                            paused = True
                        elif command == "resume":
                            paused = False
                        elif command == "step":
                            if not paused:
                                continue
                            single_step = True
                        elif command == "speed":
                            assert isinstance(value, (int, float))
                            speed = float(value)
                        elif command == "reset":
                            mask = None
                            if value is not None:
                                assert type(value) is int
                                mask = torch.arange(num_envs, device=anchor.device) == value
                            observation = runtime.reset(mask=mask)
                            if mask is None:
                                state = None
                            elif state is not None:
                                state = policy.reset_state(state, mask.reshape(expected))
                        elif command == "task_reset":
                            reset_with_option = getattr(runtime, "reset_with_task_option", None)
                            if (
                                not isinstance(value, tuple)
                                or not controls.accepts_task_option(*value)
                                or not callable(reset_with_option)
                            ):
                                raise ValueError(
                                    "Playback runtime did not authorize that task reset"
                                )
                            observation = reset_with_option(*value)
                            state = None
                            intervention = f"task_reset:{value[0]}:{value[1]}"
                        interventions[intervention] = interventions.get(intervention, 0) + 1
                    if stop.is_set():
                        reason = "stop_requested"
                        break
                    if paused and not single_step:
                        continue
                if tuple(observation.batch_size) != expected:
                    raise ValueError("Playback observation does not match the native batch")
                started = monotonic() if period is not None else 0.0
                if stateful:
                    if state is None:
                        state = policy.initialize_state(expected, device=anchor.device)
                    action, state = policy.step(observation, state, deterministic=deterministic)
                else:
                    action = policy(observation, deterministic=deterministic)
                transition = runtime.step(action)
                flags = (transition.terminated, transition.truncated)
                if (
                    tuple(transition.observation.batch_size) != expected
                    or any(
                        not isinstance(flag, torch.Tensor)
                        or flag.dtype != torch.bool
                        or flag.shape != (*expected, 1)
                        for flag in flags
                    )
                    or flags[0].device != flags[1].device
                ):
                    raise ValueError("Playback transition does not match the native batch")
                done = (flags[0] | flags[1]).reshape(num_envs)
                ends = int(done.count_nonzero().item())
                completed += ends
                ticks += 1
                if video is not None and video.enabled and not recording_finished:
                    frame, source_size = capture_rgb(
                        cast(RGBEnvironmentRuntime, runtime),
                        video.max_size,
                        expected_size=source_size,
                    )
                    frames.append(frame)
                    times.append(monotonic_ns())
                    if len(frames) == video.max_frames:
                        publish_recording("capture_limit")
                if steps is not None and ticks == steps:
                    reason = "step_limit"
                    break
                if episodes is not None and completed >= episodes:
                    reason = "episode_limit"
                    break
                if stop.is_set():
                    reason = "stop_requested"
                    break
                if ends:
                    observation = runtime.reset(mask=done)
                    if state is not None:
                        state = policy.reset_state(state, done.reshape(expected).to(anchor.device))
                else:
                    observation = transition.observation
                if period is not None:
                    stop.wait(max(0.0, started + period / speed - monotonic()))
            publish_recording(reason)
        except (KeyboardInterrupt, SystemExit) as error:
            try:
                publish_recording("cancelled")
            except Exception as recording_error:
                error.add_note(f"Playback recording cleanup also failed: {recording_error}")
            raise
        finally:
            runtime.close()
    return PlaybackResult(ticks, ticks * num_envs, completed, reason, interventions)
