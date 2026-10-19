"""Lightning-owned manual optimization and fresh policy evaluation."""

from collections.abc import Callable
from contextlib import contextmanager
import hashlib
import json
import math
import random
import sys
from typing import Any

import lightning as L
import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from nexuml.core.compiler import compile
from nexuml.core.components import (
    EnvironmentBuildContext,
    EnvironmentDefinition,
    RLAlgorithmBuildContext,
)
from nexuml.core.policy import CompiledPolicy
from nexuml.core.config import ResolvedConfig
from nexuml.core.serialization import lower_component, lower_model, restore_model_data
from nexuml.core.types import (
    InteractionContract,
    RLEvaluationSpec,
    ReinforcementLearningSpec,
    ScenarioSpec,
)
from nexuml.reinforcement.collector import (
    RolloutDataset,
    build_collector,
    synchronize_policy,
)


class NexuRLLightningModule(L.LightningModule):
    """Finite rollout batches with checkpoint-visible algorithm-owned auxiliaries."""

    automatic_optimization = False
    frames: torch.Tensor
    batches: torch.Tensor

    def __init__(self, scenario, policy=None, runtime_metadata=None):
        super().__init__()
        if not isinstance(scenario, ScenarioSpec):
            scenario = ScenarioSpec.model_validate(restore_model_data(scenario, ScenarioSpec))
        if not isinstance(scenario.training, ReinforcementLearningSpec):
            raise ValueError("NexuRLLightningModule requires reinforcement training")
        assert scenario.interaction is not None and scenario.policy is not None
        evaluation = (
            scenario.policy_evaluation
            if scenario.policy_evaluation is not None
            else scenario.training.evaluation
        )
        scenario.interaction.environment.validate_training(
            scenario.training.model_copy(update={"evaluation": evaluation})
        )
        resolve_policy_evaluation(scenario)
        self.scenario = scenario
        self.training_spec = scenario.training
        if policy is None:
            if scenario.training.seed is not None:
                L.seed_everything(scenario.training.seed, workers=True)
            contract = scenario.interaction.environment.describe()
            policy = CompiledPolicy(
                compile(scenario), scenario.policy.action_adapter.build(contract), contract
            )
        self.policy: CompiledPolicy = policy
        self.algorithm: Any = scenario.training.algorithm.build(
            RLAlgorithmBuildContext(policy, policy.contract)
        )
        if not isinstance(self.algorithm, torch.nn.Module):
            raise TypeError("RL algorithms must build a checkpointable nn.Module")
        self._collector = None
        self._evaluation_cache = None
        self._next_evaluation_frame = None
        self.register_buffer("frames", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("batches", torch.tensor(0, dtype=torch.int64))
        self.save_hyperparameters(
            {"scenario": lower_model(scenario), "runtime_metadata": runtime_metadata or {}}
        )

    @property
    def pipeline(self):
        return self.policy.pipeline

    @property
    def evaluation_spec(self):
        spec = self.scenario.policy_evaluation
        return self.training_spec.evaluation if spec is None else spec

    def forward(self, observation: TensorDict, *, deterministic=True):
        return self.policy(observation, deterministic=deterministic)

    def configure_optimizers(self):
        return self.algorithm.configure_optimizers()

    def train_dataloader(self):
        remaining = max(0, self.training_spec.total_frames - int(self.frames))
        if remaining and self._collector is None:
            assert self.scenario.interaction is not None
            # Placement and checkpoint restoration precede this hook; resume respects the budget.
            self._collector = build_collector(
                self.scenario.interaction.environment,
                self.training_spec.model_copy(
                    update={"total_frames": remaining, "evaluation": self.evaluation_spec}
                ),
                self.algorithm.rollout_policy,
                self.device,
                self.global_rank,
            )
        return DataLoader(
            RolloutDataset(
                self._collector if self._collector is not None else (),
                batches=-(-remaining // self.training_spec.frames_per_batch),
            ),
            batch_size=None,
            num_workers=0,
            collate_fn=lambda rollout: rollout,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def training_step(self, rollout, batch_idx):
        self._evaluation_cache = None
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        metrics = self.algorithm.update(rollout, lightning_module=self, optimizers=optimizers)
        self.frames.add_(rollout.numel())
        self.batches.add_(1)
        for name, value in metrics.items():
            self.log(
                f"train/rl/{name}", value, on_step=True, on_epoch=False, batch_size=rollout.numel()
            )
        self.log("train/rl/frames", self.frames.float(), on_step=True, on_epoch=False, batch_size=1)
        self.log(
            "train/rl/reward",
            rollout["next", "reward"].float().mean(),
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )
        if int(self.batches) % self.training_spec.collector.policy_sync_interval_batches == 0:
            synchronize_policy(self._collector, self.algorithm.rollout_policy)
        interval = self.evaluation_spec.every_n_frames
        if interval is not None and self._next_evaluation_frame is not None:
            if int(self.frames) >= self._next_evaluation_frame:
                self.evaluate_current_policy()
                self._next_evaluation_frame = (int(self.frames) // interval + 1) * interval

    def on_fit_start(self):
        assert self.scenario.interaction is not None
        self.scenario.interaction.environment.validate_training(
            self.training_spec.model_copy(update={"evaluation": self.evaluation_spec}),
            device=str(self.device),
        )
        from nexuml.tracking.logger import preflight_evaluation_media

        spec = self.evaluation_spec
        preflight_evaluation_media(self.trainer.loggers, spec.video)
        if spec.every_n_frames is not None:
            self._next_evaluation_frame = (
                int(self.frames) // spec.every_n_frames + 1
            ) * spec.every_n_frames
        if spec.at_start and int(self.frames) == 0:
            self.evaluate_current_policy()

    def on_fit_end(self):
        spec = self.evaluation_spec
        # Fit-end also runs when a resumed checkpoint has no remaining training batches.
        if spec.episodes and (spec.at_start or spec.every_n_frames or spec.video.enabled):
            self.evaluate_current_policy()

    def evaluate_current_policy(self) -> dict[str, float]:
        """Report one evaluation per unchanged policy/frame/configuration state.

        Returns:
            The existing scalar episode metrics, reusing only matching local results.
        """
        from nexuml.tracking.logger import log_evaluation_media

        definition, spec, seed = resolve_policy_evaluation(self.scenario)
        key = (
            id(self.policy),
            int(self.frames),
            spec.model_dump_json(),
            seed,
            repr(definition),
            tuple(tensor._version for tensor in self.policy.parameters()),
            tuple(tensor._version for tensor in self.policy.buffers()),
        )
        if self._evaluation_cache is not None and self._evaluation_cache[0] == key:
            return dict(self._evaluation_cache[1])
        step = int(self.frames)

        def publish(frames, metadata):
            log_evaluation_media(self.trainer.loggers, frames, metadata, spec.video, step=step)

        metrics = evaluate_policy(self.policy, self.scenario, media_sink=publish, frame_step=step)
        if metrics:
            for logger in self.trainer.loggers:
                logger.log_metrics(metrics, step=step)
        self._evaluation_cache = (key, dict(metrics))
        return metrics

    def teardown(self, stage):
        self.close_collection()

    def close_collection(self):
        if self._collector is not None:
            collector, self._collector = self._collector, None
            collector.shutdown()


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
) -> tuple[EnvironmentDefinition, RLEvaluationSpec, int | None]:
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
        spec = RLEvaluationSpec.model_validate(lower_model(spec))
        seed = spec.seed
    elif isinstance(scenario.training, ReinforcementLearningSpec):
        spec = RLEvaluationSpec.model_validate(lower_model(scenario.training.evaluation))
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
    )


def _evaluate_episodes(
    policy, definition, spec, seed, *, media_sink, frame_step, provenance, evaluation_context
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
    outcome_signature = None
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
                while True:
                    transition = runtime.step(policy(observation, deterministic=spec.deterministic))
                    if transition.reward is None or transition.reward.numel() != 1:
                        raise ValueError("RL evaluation requires a task reward")
                    if not torch.isfinite(transition.reward).all():
                        raise ValueError("RL evaluation requires a finite task reward")
                    if (
                        transition.terminated.numel() != 1
                        or transition.truncated.numel() != 1
                        or transition.observation.batch_size
                    ):
                        raise ValueError("Evaluation requires single-environment transitions")
                    info = transition.info or {}
                    signature = ("task_success" in info, "joint_error_rad" in info)
                    if outcome_signature is not None and signature != outcome_signature:
                        raise ValueError(
                            "Task outcomes must be supplied consistently across steps/episodes"
                        )
                    outcome_signature = signature
                    if any(signature):
                        success, error = info.get("task_success"), info.get("joint_error_rad")
                        if (
                            not isinstance(success, torch.Tensor)
                            or success.numel() != 1
                            or success.dtype != torch.bool
                            or success.device != transition.reward.device
                        ):
                            raise ValueError(
                                "Task outcomes require one boolean task_success on the task device"
                            )
                        episode_success |= bool(success.item())
                        if signature[1]:
                            if (
                                not isinstance(error, torch.Tensor)
                                or error.numel() != 1
                                or error.device != transition.reward.device
                                or not error.is_floating_point()
                                or not torch.isfinite(error).all()
                                or (error < 0).any()
                            ):
                                raise ValueError(
                                    "Task outcomes require finite nonnegative joint_error_rad "
                                    "on the task device when supplied"
                                )
                            final_error = float(error.item())
                    total += float(transition.reward.item())
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
                returns.append(total)
                lengths.append(steps)
                if outcome_signature[0]:
                    successes.append(episode_success)
                if outcome_signature[1]:
                    assert final_error is not None
                    final_errors.append(final_error)
        finally:
            primary = sys.exception()
            if runtime is not None:
                try:
                    runtime.close()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(f"Evaluation cleanup also failed: {error}")
    values = torch.tensor(returns, dtype=torch.float64)
    metrics = {
        "eval/rl/return_mean": float(values.mean()),
        "eval/rl/return_std": float(values.std(unbiased=False)),
        "eval/rl/length_mean": sum(lengths) / len(lengths),
        "eval/rl/terminated": float(terminated),
        "eval/rl/truncated": float(truncated),
    }
    if successes:
        metrics["eval/rl/task_success_rate"] = sum(successes) / len(successes)
    if final_errors:
        metrics["eval/rl/final_joint_error_rad_mean"] = sum(final_errors) / len(final_errors)
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
