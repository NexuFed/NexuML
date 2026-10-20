"""Fresh environment-based policy evaluation, independent of any training engine."""

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import random
import sys

import numpy as np
import torch

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.policy import CompiledPolicy
from nexuml.core.config import ResolvedConfig
from nexuml.core.serialization import lower_component, lower_model
from nexuml.core.types import (
    InteractionContract,
    PolicyEvaluationSpec,
    ReinforcementLearningSpec,
    ScenarioSpec,
)


@contextmanager
def _evaluation_state(policy, seed):
    modes = [(module, module.training) for module in policy.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = sorted(
        {
            tensor.device.index or 0
            for tensor in (*policy.parameters(), *policy.buffers())
            if tensor.is_cuda
        }
    )
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % (2**32))
                # Do not initialize or reseed unrelated visible accelerators.
                torch.random.default_generator.manual_seed(seed)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
            policy.eval()
            with torch.inference_mode():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training


def _validate_evaluation_contract(policy_contract, actual):
    # Inference archives own isolated model classes and may predate optional metadata.
    policy_contract = InteractionContract.model_validate(lower_model(policy_contract))
    actual = InteractionContract.model_validate(lower_model(actual))
    if actual.num_envs != 1:
        raise ValueError("Evaluation requires a single unbatched environment")
    if actual.action_space != policy_contract.action_space:
        raise ValueError("Evaluation action-space semantics do not match the policy")
    if actual.control_period_s != policy_contract.control_period_s:
        raise ValueError("Evaluation control period does not match the policy")
    if policy_contract.physical is not None and actual.physical != policy_contract.physical:
        raise ValueError("Evaluation physical system identity does not match the policy")
    for role in ("observations", "actions"):
        expected_fields = getattr(policy_contract, role)
        actual_fields = getattr(actual, role)
        if role == "actions" and actual_fields.keys() != expected_fields.keys():
            raise ValueError("Evaluation action keys do not match the policy")
        for key, expected in expected_fields.items():
            found = actual_fields.get(key)
            fields = ("shape", "dtype", "modality", "num_values")
            if role == "actions":
                fields += ("low", "high", "rate_hz", "description")
            if expected.physical is not None:
                fields += ("physical",)
            if expected.axes is not None:
                fields += ("axes",)
            if found is None or any(
                getattr(found, name) != getattr(expected, name) for name in fields
            ):
                raise ValueError(f"Evaluation {role} field {key!r} is incompatible with the policy")


def resolve_policy_evaluation(
    scenario: ScenarioSpec,
) -> tuple[EnvironmentDefinition, PolicyEvaluationSpec, int | None]:
    """Select episode options without changing the scenario's original training mode.

    Returns:
        Environment definition, validated episode/media options and effective seed.

    Raises:
        ValueError: If policy evaluation inputs are absent.
    """
    if scenario.interaction is None or scenario.policy is None:
        raise ValueError("Policy evaluation requires scenario.interaction and scenario.policy")
    spec = getattr(scenario, "policy_evaluation", None)
    if spec is not None:
        spec = PolicyEvaluationSpec.model_validate(lower_model(spec))
        seed = spec.seed
    elif isinstance(scenario.training, ReinforcementLearningSpec):
        spec = PolicyEvaluationSpec.model_validate(lower_model(scenario.training.evaluation))
        seed = spec.seed if spec.seed is not None else scenario.training.seed
    else:
        raise ValueError("Offline policy evaluation requires scenario.policy_evaluation")
    definition = scenario.interaction.evaluation_environment or scenario.interaction.environment
    return definition, spec, seed


def evaluate_policy(
    policy: CompiledPolicy,
    scenario: ScenarioSpec,
    *,
    media_sink: Callable | None = None,
    frame_step: int = 0,
    provenance: dict | None = None,
) -> dict[str, float]:
    """Evaluate the complete deployable mapping using a fresh environment.

    Returns:
        Episode returns/lengths and termination/truncation statistics.
    """
    definition, spec, seed = resolve_policy_evaluation(scenario)
    if spec.episodes == 0:
        return {}
    evaluation_context = {}
    if spec.video.enabled:
        evaluation_context = {
            "environment": lower_component(definition),
            "contract_hash": hashlib.sha256(
                json.dumps(lower_model(policy.contract), sort_keys=True).encode()
            ).hexdigest(),
            "config_hash": hashlib.sha256(
                ResolvedConfig.from_scenario(scenario).to_yaml().encode()
            ).hexdigest()[:16],
        }
    return _evaluate_episodes(
        policy,
        definition,
        spec,
        seed,
        media_sink=media_sink,
        frame_step=frame_step,
        provenance=provenance,
        evaluation_context=evaluation_context,
        require_reward=isinstance(scenario.training, ReinforcementLearningSpec),
    )


def _evaluate_episodes(
    policy,
    definition,
    spec,
    seed,
    *,
    media_sink,
    frame_step,
    provenance,
    evaluation_context,
    require_reward,
) -> dict[str, float]:
    """Execute scalar episodes and bounded media independently of training.

    Returns:
        Episode returns/lengths and termination/truncation statistics.

    Raises:
        ValueError: If evaluation I/O, rewards, or requested rendering are unsupported.
    """
    device = next(policy.parameters(), torch.tensor(0)).device
    if spec.video.enabled and media_sink is None:
        raise ValueError("Evaluation recording requires an artifact output sink")
    returns, lengths = [], []
    successes, final_errors = [], []
    task_errors, violations = {}, {}
    outcome_signature = None
    reward_signature = None
    terminated = truncated = 0
    frames = []
    source_size = None
    capped = False
    fps = None
    runtime = None
    with _evaluation_state(policy, seed):
        try:
            _validate_evaluation_contract(policy.contract, definition.describe())
            runtime = definition.build(
                EnvironmentBuildContext(
                    purpose="eval",
                    seed=seed,
                    device=str(device),
                    num_envs=1,
                    render_mode="rgb_array" if spec.video.enabled else None,
                )
            )
            _validate_evaluation_contract(policy.contract, runtime.contract)
            if spec.video.enabled:
                if not callable(getattr(runtime, "render_frame", None)):
                    raise ValueError("Evaluation environment does not support RGB recording")
                fps = spec.video.fps or getattr(runtime, "render_fps", None)
                if fps is None or not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Recording requires renderer FPS metadata or video.fps")

            def capture():
                nonlocal source_size, capped
                if len(frames) >= spec.video.max_frames:
                    capped = True
                    return
                frame = getattr(runtime, "render_frame")()
                if (
                    not isinstance(frame, np.ndarray)
                    or frame.dtype != np.uint8
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                    or not frame.size
                ):
                    raise ValueError("Recording frames must be nonempty CPU uint8 HWC RGB arrays")
                from PIL import Image

                height, width = frame.shape[:2]
                if source_size is None:
                    source_size = [width, height]
                elif source_size != [width, height]:
                    raise ValueError("Recording frame dimensions changed within the episode")
                ratio = min(1.0, spec.video.max_size / max(width, height))
                size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
                if size != (width, height):
                    frame = np.asarray(
                        Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
                    )
                frames.append(frame.copy())

            for episode in range(spec.episodes):
                observation = runtime.reset(seed=None if seed is None else seed + episode)
                if observation.batch_size:
                    raise ValueError("Evaluation requires unbatched episode observations")
                if spec.video.enabled and episode == 0:
                    capture()
                total, steps = 0.0, 0
                episode_success, final_error = False, None
                episode_errors = {}
                while True:
                    transition = runtime.step(policy(observation, deterministic=spec.deterministic))
                    reward = transition.reward
                    if reward is None and require_reward:
                        raise ValueError("RL evaluation requires a task reward")
                    if reward is not None and (
                        not isinstance(reward, torch.Tensor)
                        or reward.numel() != 1
                        or reward.is_complex()
                        or not torch.isfinite(reward).all()
                    ):
                        raise ValueError("RL evaluation requires a finite scalar task reward")
                    if reward_signature is not None and reward_signature != (reward is not None):
                        raise ValueError("Task rewards must be supplied consistently")
                    reward_signature = reward is not None
                    if (
                        any(
                            not isinstance(flag, torch.Tensor)
                            or flag.numel() != 1
                            or flag.dtype != torch.bool
                            for flag in (transition.terminated, transition.truncated)
                        )
                        or transition.terminated.device != transition.truncated.device
                        or (reward is not None and reward.device != transition.terminated.device)
                        or transition.observation.batch_size
                    ):
                        raise ValueError("Evaluation requires single-environment transitions")
                    info = transition.info or {}
                    errors = info.get("task_errors", {})
                    if (
                        not isinstance(errors, Mapping)
                        or any(
                            not isinstance(name, str) or not name.isidentifier() for name in errors
                        )
                        or set(errors).intersection(
                            {"joint_error_rad", "constraint_violations", "action_violations"}
                        )
                    ):
                        raise ValueError("Task outcomes require named scalar task_errors")
                    counters = {
                        name: info[name]
                        for name in ("constraint_violations", "action_violations")
                        if name in info
                    }
                    signature = (
                        "task_success" in info,
                        "joint_error_rad" in info,
                        tuple(sorted(errors)),
                        tuple(sorted(counters)),
                    )
                    if reward is None and not signature[0]:
                        raise ValueError("Rewardless evaluation requires validated task outcomes")
                    if outcome_signature is not None and signature != outcome_signature:
                        raise ValueError(
                            "Task outcomes must be supplied consistently across steps/episodes"
                        )
                    outcome_signature = signature
                    if signature[0] or signature[1]:
                        success, error = info.get("task_success"), info.get("joint_error_rad")
                        if (
                            not isinstance(success, torch.Tensor)
                            or success.numel() != 1
                            or success.dtype != torch.bool
                            or success.device != transition.terminated.device
                        ):
                            raise ValueError(
                                "Task outcomes require one boolean task_success on the task device"
                            )
                        episode_success |= bool(success.item())
                        if signature[1]:
                            if (
                                not isinstance(error, torch.Tensor)
                                or error.numel() != 1
                                or error.device != transition.terminated.device
                                or not error.is_floating_point()
                                or not torch.isfinite(error).all()
                                or (error < 0).any()
                            ):
                                raise ValueError(
                                    "Task outcomes require finite nonnegative joint_error_rad "
                                    "on the task device when supplied"
                                )
                            final_error = float(error.item())
                    for name, value in (*errors.items(), *counters.items()):
                        if (
                            not isinstance(value, torch.Tensor)
                            or value.numel() != 1
                            or value.device != transition.terminated.device
                            or value.is_complex()
                            or not torch.isfinite(value).all()
                            or (value < 0).any()
                            or (name in errors and not value.is_floating_point())
                            or (name in counters and value.item() % 1 != 0)
                        ):
                            raise ValueError(
                                "Task outcomes require finite nonnegative scalar metrics"
                            )
                        if name in counters:
                            violations[name] = violations.get(name, 0.0) + float(value.item())
                        else:
                            episode_errors[name] = float(value.item())
                    if reward is not None:
                        total += float(reward.item())
                    steps += 1
                    term = bool(transition.terminated.item())
                    trunc = bool(transition.truncated.item())
                    if spec.video.enabled and episode == 0:
                        capture()
                    limit = (
                        spec.max_steps_per_episode is not None
                        and steps >= spec.max_steps_per_episode
                    )
                    if term or trunc or limit:
                        terminated += int(term)
                        truncated += int(trunc or (limit and not term))
                        break
                    observation = transition.observation
                if reward_signature:
                    returns.append(total)
                lengths.append(steps)
                if outcome_signature[0]:
                    successes.append(episode_success)
                if outcome_signature[1]:
                    assert final_error is not None
                    final_errors.append(final_error)
                for name, value in episode_errors.items():
                    task_errors.setdefault(name, []).append(value)
        finally:
            primary = sys.exception()
            if runtime is not None:
                try:
                    runtime.close()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(f"Evaluation cleanup also failed: {error}")
    metrics = {
        "eval/rl/length_mean": sum(lengths) / len(lengths),
        "eval/rl/terminated": float(terminated),
        "eval/rl/truncated": float(truncated),
    }
    if returns:
        values = torch.tensor(returns, dtype=torch.float64)
        metrics["eval/rl/return_mean"] = float(values.mean())
        metrics["eval/rl/return_std"] = float(values.std(unbiased=False))
    if successes:
        metrics["eval/rl/task_success_rate"] = sum(successes) / len(successes)
    if final_errors:
        metrics["eval/rl/final_joint_error_rad_mean"] = sum(final_errors) / len(final_errors)
    for name, values in task_errors.items():
        metrics[f"eval/rl/final_{name}_mean"] = sum(values) / len(values)
    metrics.update({f"eval/rl/{name}": value for name, value in violations.items()})
    # Existing dashboards/packages retain their keys; new consumers use the policy namespace.
    metrics.update(
        {key.replace("eval/rl/", "eval/policy/", 1): value for key, value in metrics.items()}
    )
    if spec.video.enabled:
        metadata = {
            **evaluation_context,
            "frame_step": frame_step,
            "metrics": metrics,
            "evaluation": spec.model_dump(mode="json"),
            "seeds": [None if seed is None else seed + i for i in range(spec.episodes)],
            "fps": fps,
            "captured_frames": len(frames),
            "capture_capped": capped,
            "source_size": source_size,
            "captured_size": [frames[0].shape[1], frames[0].shape[0]],
            "provenance": provenance or {},
        }
        backend = getattr(runtime, "render_backend", None)
        if backend is not None:
            metadata["renderer"] = backend
            metadata["source_fps"] = getattr(runtime, "render_fps", None)
        # Encoding and logger dispatch are also isolated from learner random streams/modes.
        with _evaluation_state(policy, seed):
            assert media_sink is not None
            media_sink(frames, metadata)
    return metrics
