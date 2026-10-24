"""Isolated episodic evaluation, frame cadence, and opt-in real recording proofs."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import ClassVar

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict
from torch import nn

from nexuml.cli.overrides import apply_overrides
from nexuml.core.components import EnvironmentDefinition
from nexuml.core.config import ResolvedConfig
from nexuml.core.discovery import environment
from nexuml.core.types import (
    InteractionContract,
    LoggingSpec,
    MLflowSpec,
    RLEvaluationSpec,
    RLVideoSpec,
    ReinforcementLearningSpec,
    TensorBoardSpec,
    TensorFieldContract,
)
from nexuml.interaction.runtime import InteractionStep
from nexuml.training.reinforcement import NexuRLLightningModule, evaluate_policy
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo, pendulum_ppo


def contract():
    return InteractionContract(
        observations={"observation": TensorFieldContract(shape=(4,), dtype="float32")},
        actions={"action": TensorFieldContract(shape=(), dtype="int64", num_values=2)},
        action_space="discrete",
    )


class FakeRuntime:
    render_fps: float | None = 10

    def __init__(self):
        self.contract = contract()
        self.closed = False
        self.steps = 0
        self.resets = []
        self.renders = 0
        self.frame = np.zeros((9, 17, 3), dtype=np.uint8)
        self.failure = None
        self.close_failure = False
        self.terminal = True

    def reset(self, *, seed=None):
        self.steps = 0
        self.resets.append(seed)
        random.random(), np.random.random(), torch.rand(1)
        return TensorDict({"observation": torch.zeros(4)}, batch_size=[])

    def step(self, action):
        if self.failure:
            raise self.failure
        self.steps += 1
        return InteractionStep(
            observation=TensorDict({"observation": torch.ones(4)}, batch_size=[]),
            reward=torch.tensor(1.0),
            terminated=torch.tensor(self.terminal and self.steps == 3),
            truncated=torch.tensor(not self.terminal and self.steps == 3),
        )

    def render_frame(self):
        self.renders += 1
        self.frame.fill(self.steps * 50)
        return self.frame

    def close(self):
        self.closed = True
        if self.close_failure:
            raise RuntimeError("close failure")


@environment("test.evaluation.environment")
class FakeEnvironment(EnvironmentDefinition):
    factory: ClassVar = FakeRuntime
    instances: ClassVar[list] = []
    contexts: ClassVar[list] = []

    def describe(self):
        return contract()

    def build(self, context):
        runtime = type(self).factory()
        self.instances.append(runtime)
        self.contexts.append(context)
        return runtime


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 2), nn.Dropout())
        self.contract = contract()
        self.actions = []

    def forward(self, observation, *, deterministic=True):
        logits = self.net(observation["observation"])
        action = (
            logits.argmax(-1)
            if deterministic
            else torch.distributions.Categorical(logits=logits).sample()
        )
        self.actions.append(int(action))
        return TensorDict({"action": action}, batch_size=[])


@pytest.fixture
def fake(monkeypatch):
    FakeEnvironment.instances.clear()
    FakeEnvironment.contexts.clear()
    monkeypatch.setattr(FakeEnvironment, "factory", FakeRuntime)
    scenario = cartpole_ppo(total_frames=32, frames_per_batch=16)
    scenario.interaction.environment = FakeEnvironment()
    scenario.training.evaluation = RLEvaluationSpec(episodes=2, seed=7, max_steps_per_episode=5)
    return scenario, FakePolicy()


@pytest.mark.parametrize(
    "values",
    [
        {"every_n_frames": 0},
        {"every_n_frames": -1},
        {"every_n_frames": True},
        {"video": {"format": "avi"}},
        {"video": {"fps": float("nan")}},
        {"video": {"fps": float("inf")}},
        {"video": {"fps": 0}},
        {"video": {"max_frames": 0}},
        {"video": {"max_size": 0}},
        {"episodes": 0, "video": {"enabled": True}},
    ],
)
def test_invalid_evaluation_configuration(values):
    with pytest.raises(ValidationError):
        RLEvaluationSpec.model_validate(values)


@pytest.mark.parametrize("with_error", [False, True])
@pytest.mark.parametrize("record", [False, True])
def test_task_outcomes_are_separate_from_returns_and_reset_per_episode(
    fake, monkeypatch, with_error, record
):
    scenario, policy = fake
    scenario.training.evaluation.video = RLVideoSpec(enabled=record)

    def factory():
        runtime = FakeRuntime()
        original = runtime.step

        def step(action):
            transition = original(action)
            first_episode = len(runtime.resets) == 1
            info = {"task_success": torch.tensor(first_episode and runtime.steps == 2)}
            if with_error:
                info["joint_error_rad"] = torch.tensor(
                    (0.1 if first_episode else 0.2) * runtime.steps
                )
            return replace(transition, info=info)

        monkeypatch.setattr(runtime, "step", step)
        return runtime

    monkeypatch.setattr(FakeEnvironment, "factory", factory)
    recordings = []
    metrics = evaluate_policy(policy, scenario, media_sink=lambda f, m: recordings.append((f, m)))
    assert metrics["eval/rl/return_mean"] == 3
    assert metrics["eval/rl/task_success_rate"] == 0.5
    assert metrics["eval/rl/task_successes"] == 1
    assert metrics["eval/rl/episodes"] == 2
    if with_error:
        assert metrics["eval/rl/final_joint_error_rad_mean"] == pytest.approx(0.45)
    else:
        assert "eval/rl/final_joint_error_rad_mean" not in metrics
    assert FakeEnvironment.instances[-1].closed
    if record:
        assert len(recordings) == 1 and len(recordings[0][0]) == 4
        assert recordings[0][1]["metrics"] == metrics
    assert evaluate_policy(policy, scenario, media_sink=lambda f, m: None) == metrics


@pytest.mark.parametrize(
    "info",
    [
        {"task_success": None},
        {"task_success": True},
        {"task_success": torch.tensor(1.0)},
        {"task_success": torch.ones(2, dtype=torch.bool)},
        {"task_success": torch.tensor(True, device="meta")},
        {"joint_error_rad": torch.tensor(0.1)},
        {"task_success": torch.ones(2, dtype=torch.bool), "joint_error_rad": torch.tensor(0.1)},
        {"task_success": torch.tensor(1.0), "joint_error_rad": torch.tensor(0.1)},
        {"task_success": True, "joint_error_rad": torch.tensor(0.1)},
        *[
            {"task_success": torch.tensor(False), "joint_error_rad": value}
            for value in (
                torch.tensor(float("nan")),
                torch.tensor(float("inf")),
                torch.tensor(-0.1),
                torch.ones(2),
                torch.tensor(1),
                torch.tensor(0.1, device="meta"),
                None,
                0.1,
            )
        ],
    ],
)
def test_invalid_task_outcomes_fail_and_close_runtime(fake, monkeypatch, info):
    scenario, policy = fake

    def factory():
        runtime = FakeRuntime()
        original = runtime.step
        monkeypatch.setattr(runtime, "step", lambda action: replace(original(action), info=info))
        return runtime

    monkeypatch.setattr(FakeEnvironment, "factory", factory)
    with pytest.raises(ValueError, match="Task outcomes"):
        evaluate_policy(policy, scenario)
    assert FakeEnvironment.instances[-1].closed


@pytest.mark.parametrize("disappear", ["step", "episode"])
@pytest.mark.parametrize(
    "before,after",
    [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
)
def test_task_outcome_presence_cannot_change_or_reuse_stale_values(
    fake, monkeypatch, disappear, before, after
):
    scenario, policy = fake

    def factory():
        runtime = FakeRuntime()
        original = runtime.step

        def step(action):
            transition = original(action)
            present = runtime.steps == 1 if disappear == "step" else len(runtime.resets) == 1
            signature = before if present else after
            info = {}
            if signature >= 1:
                info["task_success"] = torch.tensor(False)
            if signature == 2:
                info["joint_error_rad"] = torch.tensor(0.1)
            return replace(transition, info=info)

        monkeypatch.setattr(runtime, "step", step)
        return runtime

    monkeypatch.setattr(FakeEnvironment, "factory", factory)
    with pytest.raises(ValueError, match="supplied consistently"):
        evaluate_policy(policy, scenario)
    assert FakeEnvironment.instances[-1].closed


@pytest.mark.parametrize("change", ["period", "rate", "mapping", "joint-order", "control-origin"])
def test_replay_rejects_same_shape_but_changed_control_meaning(change):
    from nexuml.training.reinforcement import _validate_evaluation_contract
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment

    expected = IsaacLabEnvironment(task="Isaac-Reach-Franka").describe()
    if change == "period":
        actual = expected.model_copy(update={"control_period_s": 1 / 60})
    else:
        action = expected.actions["action"]
        assert action.description is not None
        changes = {
            "rate": {"rate_hz": 60.0},
            "mapping": {"description": action.description.replace("0.5 * action", "1.0 * action")},
            "joint-order": {
                "description": action.description.replace(
                    "panda_joint1, panda_joint2", "panda_joint2, panda_joint1"
                )
            },
            "control-origin": {
                "description": action.description.replace("default joint pose", "zero joint pose")
            },
        }
        actual = expected.model_copy(
            update={"actions": {"action": action.model_copy(update=changes[change])}}
        )
    assert actual.actions["action"].shape == expected.actions["action"].shape
    with pytest.raises(ValueError, match="control period|incompatible"):
        _validate_evaluation_contract(expected, actual)


@pytest.mark.parametrize("format", ["gif", "mp4", "both"])
def test_configuration_round_trip_and_ordinary_overrides(format):
    scenario = cartpole_ppo(total_frames=32, frames_per_batch=16)
    apply_overrides(
        scenario,
        [
            "training.evaluation.every_n_frames=20",
            "training.evaluation.at_start=true",
            "training.evaluation.video.enabled=true",
            f"training.evaluation.video.format={format}",
        ],
    )
    restored = ResolvedConfig.from_yaml(
        ResolvedConfig.from_scenario(scenario).to_yaml()
    ).to_scenario()
    assert isinstance(restored.training, ReinforcementLearningSpec)
    assert restored.training.evaluation.every_n_frames == 20
    assert restored.training.evaluation.at_start
    assert restored.training.evaluation.video.format == format
    assert restored.training.evaluation.video.enabled
    assert RLEvaluationSpec().video == RLVideoSpec()
    assert RLEvaluationSpec().every_n_frames is None and not RLEvaluationSpec().at_start


@pytest.mark.parametrize("deterministic", [True, False])
def test_evaluation_preserves_rng_parameters_gradients_and_mixed_modes(fake, deterministic):
    scenario, policy = fake
    scenario.training.evaluation.deterministic = deterministic
    policy.net[1].eval()
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = copy.deepcopy(policy.state_dict())
    flags = [module.training for module in policy.modules()]
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    expected = random.random(), np.random.random(), torch.rand(3)
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    first = evaluate_policy(policy, scenario)
    actions = policy.actions.copy()
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    policy.actions.clear()
    assert evaluate_policy(policy, scenario) == first
    assert policy.actions == actions
    assert first["eval/rl/return_mean"] == 3 and first["eval/rl/length_mean"] == 3
    assert first["eval/rl/terminated"] == 2 and first["eval/rl/truncated"] == 0
    assert "eval/rl/task_success_rate" not in first
    assert "eval/rl/final_joint_error_rad_mean" not in first
    assert [module.training for module in policy.modules()] == flags
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in before.items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in policy.parameters())
    assert all(runtime.resets == [7, 8] and runtime.closed for runtime in FakeEnvironment.instances)
    assert all(
        context.purpose == "eval" and context.num_envs == 1 and context.render_mode is None
        for context in FakeEnvironment.contexts
    )


@pytest.mark.parametrize("failure", [RuntimeError("step failure"), KeyboardInterrupt()])
def test_failure_cleanup_preserves_primary_error_and_modes(fake, monkeypatch, failure):
    scenario, policy = fake
    runtime = FakeRuntime()
    runtime.failure, runtime.close_failure = failure, True
    monkeypatch.setattr(FakeEnvironment, "factory", lambda: runtime)
    policy.net[1].eval()
    flags = [m.training for m in policy.modules()]
    with pytest.raises(type(failure)) as caught:
        evaluate_policy(policy, scenario)
    assert caught.value is failure
    assert "close failure" in " ".join(caught.value.__notes__)
    assert runtime.closed and [m.training for m in policy.modules()] == flags


def test_close_failure_without_primary_error_is_visible(fake, monkeypatch):
    scenario, policy = fake
    runtime = FakeRuntime()
    runtime.close_failure = True
    monkeypatch.setattr(FakeEnvironment, "factory", lambda: runtime)
    with pytest.raises(RuntimeError, match="close failure"):
        evaluate_policy(policy, scenario)
    assert runtime.closed and policy.training


def test_build_failure_restores_policy_modes_and_rng(fake, monkeypatch):
    scenario, policy = fake
    policy.net[1].eval()
    flags = [module.training for module in policy.modules()]
    state = torch.get_rng_state().clone()

    def fail():
        torch.rand(3)
        raise RuntimeError("build failure")

    monkeypatch.setattr(FakeEnvironment, "factory", fail)
    with pytest.raises(RuntimeError, match="build failure"):
        evaluate_policy(policy, scenario)
    assert torch.equal(torch.get_rng_state(), state)
    assert [module.training for module in policy.modules()] == flags


@pytest.mark.parametrize("change", ["shape", "dtype", "categories"])
def test_required_contract_fields_rejected_before_build(fake, monkeypatch, change):
    scenario, policy = fake
    actual = contract()
    if change == "categories":
        actual.actions["action"] = TensorFieldContract(shape=(), dtype="int64", num_values=3)
    else:
        actual.observations["observation"] = TensorFieldContract(
            shape=(5,) if change == "shape" else (4,),
            dtype="float64" if change == "dtype" else "float32",
        )
    monkeypatch.setattr(FakeEnvironment, "describe", lambda self: actual)
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_policy(policy, scenario)
    assert not FakeEnvironment.instances


def test_override_contract_rejection_and_disabled_evaluation(fake, monkeypatch):
    scenario, policy = fake
    scenario.training.evaluation.episodes = 0
    assert evaluate_policy(policy, scenario) == {} and not FakeEnvironment.instances
    scenario.training.evaluation.episodes = 1
    wrong = contract().model_copy(update={"action_space": "continuous"})
    monkeypatch.setattr(FakeEnvironment, "describe", lambda self: wrong)
    with pytest.raises(ValueError, match="action-space"):
        evaluate_policy(policy, scenario)
    assert not FakeEnvironment.instances


def test_runtime_batch_and_reward_validation_closes_owned_runtime(fake, monkeypatch):
    scenario, policy = fake
    runtime = FakeRuntime()
    runtime.contract = contract().model_copy(update={"num_envs": 2})
    monkeypatch.setattr(FakeEnvironment, "factory", lambda: runtime)
    with pytest.raises(ValueError, match="single unbatched"):
        evaluate_policy(policy, scenario)
    assert runtime.closed


@pytest.mark.parametrize("reward", [None, torch.tensor(float("nan")), torch.ones(2)])
def test_invalid_episode_reward_is_not_reported_as_success(fake, monkeypatch, reward):
    scenario, policy = fake
    runtime = FakeRuntime()
    step = runtime.step
    monkeypatch.setattr(runtime, "step", lambda action: replace(step(action), reward=reward))
    monkeypatch.setattr(FakeEnvironment, "factory", lambda: runtime)
    with pytest.raises(ValueError, match="task reward"):
        evaluate_policy(policy, scenario)
    assert runtime.closed


def test_capture_copies_resizes_bounds_and_continues_all_episodes(fake):
    scenario, policy = fake
    scenario.training.evaluation.video = RLVideoSpec(enabled=True, max_frames=2, max_size=8)
    recordings = []
    metrics = evaluate_policy(
        policy,
        scenario,
        frame_step=32,
        media_sink=lambda frames, meta: recordings.append((frames, meta)),
    )
    frames, metadata = recordings[0]
    assert metrics["eval/rl/length_mean"] == 3 and metadata["seeds"] == [7, 8]
    assert len(recordings) == 1 and len(frames) == 2
    assert all(frame.shape == (4, 8, 3) and frame.dtype == np.uint8 for frame in frames)
    assert frames[0].max() == 0 and frames[1].min() == 50
    assert metadata["capture_capped"] and metadata["frame_step"] == 32
    assert metadata["source_size"] == [17, 9] and metadata["fps"] == 10
    assert len(metadata["config_hash"]) == 16 and len(metadata["contract_hash"]) == 64
    assert FakeEnvironment.instances[0].renders == 2
    assert FakeEnvironment.instances[0].closed
    assert FakeEnvironment.contexts[0].render_mode == "rgb_array"


def test_terminal_frame_capture_and_horizon_statistics(fake, monkeypatch):
    scenario, policy = fake
    scenario.training.evaluation.video = RLVideoSpec(enabled=True)
    recordings = []
    evaluate_policy(policy, scenario, media_sink=lambda f, m: recordings.append((f, m)))
    assert len(recordings[0][0]) == 4 and recordings[0][0][-1].min() == 150
    assert not recordings[0][1]["capture_capped"]
    scenario.training.evaluation.max_steps_per_episode = 2
    metrics = evaluate_policy(policy, scenario, media_sink=lambda f, m: None)
    assert metrics["eval/rl/length_mean"] == 2 and metrics["eval/rl/truncated"] == 2
    assert metrics["eval/rl/native_truncated"] == 0
    assert metrics["eval/rl/evaluation_horizon_limits"] == 2


def test_native_truncation_is_not_an_evaluator_horizon_or_measured_timeout(fake, monkeypatch):
    scenario, policy = fake
    original = FakeRuntime.__init__

    def initialize(runtime):
        original(runtime)
        runtime.terminal = False

    monkeypatch.setattr(FakeRuntime, "__init__", initialize)
    metrics = evaluate_policy(policy, scenario)
    assert metrics["eval/rl/episodes"] == 2
    assert metrics["eval/rl/native_truncated"] == 2
    assert metrics["eval/rl/evaluation_horizon_limits"] == 0
    assert "eval/rl/timeouts" not in metrics


@pytest.mark.parametrize("error", ["fps", "empty", "dtype", "render"])
def test_invalid_rendering_prerequisites(fake, monkeypatch, error):
    scenario, policy = fake
    scenario.training.evaluation.video = RLVideoSpec(enabled=True)
    runtime = FakeRuntime()
    if error == "fps":
        runtime.render_fps = None
    elif error == "empty":
        runtime.frame = np.zeros((0, 1, 3), dtype=np.uint8)
    elif error == "dtype":
        runtime.frame = runtime.frame.astype(np.float32)
    else:
        monkeypatch.setattr(runtime, "render_frame", None)
    monkeypatch.setattr(FakeEnvironment, "factory", lambda: runtime)
    with pytest.raises(ValueError, match="FPS|RGB|uint8"):
        evaluate_policy(policy, scenario, media_sink=lambda f, m: None)
    assert runtime.closed


def test_explicit_fps_and_training_seed_fallback(fake, monkeypatch):
    scenario, policy = fake
    scenario.training.seed = 9
    scenario.training.evaluation.seed = None
    scenario.training.evaluation.video = RLVideoSpec(enabled=True, fps=15)
    runtime = FakeRuntime()
    runtime.render_fps = None
    monkeypatch.setattr(FakeEnvironment, "factory", lambda: runtime)
    recordings = []
    evaluate_policy(
        policy, scenario, media_sink=lambda frames, metadata: recordings.append(metadata)
    )
    assert recordings[0]["fps"] == 15 and recordings[0]["seeds"] == [9, 10]


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_scheduled_evaluation_follows_sync_before_next_collection(tmp_path, monkeypatch):
    from nexuml.training.lightning import NexuSession
    from nexuml.reinforcement.collector import RolloutDataset
    import nexuml.training.reinforcement as module

    order = []
    scenario = cartpole_ppo(total_frames=32, frames_per_batch=16)
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    scenario.training.evaluation = RLEvaluationSpec(
        episodes=1, max_steps_per_episode=2, every_n_frames=16
    )
    session = NexuSession(
        scenario, log_dir=tmp_path, enable_loggers=False, enable_progress_bar=False
    )
    sync = module.synchronize_policy
    evaluate = module.evaluate_policy

    def synchronize(collector, policy):
        sync(collector, policy)
        order.append(("sync", int(session.lightning_module.get_buffer("frames"))))

    def evaluate_after_sync(policy, scenario, **kwargs):
        order.append(("eval", kwargs["frame_step"]))
        return evaluate(policy, scenario, **kwargs)

    def collect(dataset):
        for batch in dataset.collector:
            order.append(("collect", int(session.lightning_module.get_buffer("frames"))))
            yield batch

    monkeypatch.setattr(module, "synchronize_policy", synchronize)
    monkeypatch.setattr(module, "evaluate_policy", evaluate_after_sync)
    monkeypatch.setattr(RolloutDataset, "__iter__", collect)
    session.run()
    assert order == [
        ("collect", 0),
        ("sync", 16),
        ("eval", 16),
        ("collect", 16),
        ("sync", 32),
        ("eval", 32),
    ]


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("interval,expected", [(16, [0, 16, 32]), (20, [0, 32]), (5, [0, 16, 32])])
def test_cadence_actual_frames_final_dedup_and_default_fit(
    tmp_path, monkeypatch, interval, expected
):
    import nexuml.training.reinforcement as module
    from nexuml.training.lightning import NexuSession

    calls = []
    original = module.evaluate_policy

    def evaluate(policy, scenario, **kwargs):
        calls.append(kwargs["frame_step"])
        return original(policy, scenario, **kwargs)

    monkeypatch.setattr(module, "evaluate_policy", evaluate)
    scenario = cartpole_ppo(total_frames=32, frames_per_batch=16)
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    scenario.training.evaluation = RLEvaluationSpec(
        episodes=1, max_steps_per_episode=3, at_start=True, every_n_frames=interval
    )
    result = NexuSession(
        scenario, log_dir=tmp_path, enable_loggers=False, enable_progress_bar=False
    ).run()
    assert calls == expected and result.interaction_results
    assert isinstance(result.lightning_module, NexuRLLightningModule)
    result.lightning_module.evaluate_current_policy()
    assert calls == expected
    scenario.training.evaluation.seed = 6
    result.lightning_module.evaluate_current_policy()
    assert calls[-1] == 32 and len(calls) == len(expected) + 1
    scenario.training.evaluation.seed = None
    result.lightning_module.evaluate_current_policy()
    scenario.training.seed = 9
    result.lightning_module.evaluate_current_policy()
    assert len(calls) == len(expected) + 3
    calls.clear()
    scenario.training.evaluation = RLEvaluationSpec(episodes=1, max_steps_per_episode=3)
    session = NexuSession(
        scenario, log_dir=tmp_path / "default", enable_loggers=False, enable_progress_bar=False
    )
    session.fit()
    assert not calls


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("factory", [cartpole_ppo, pendulum_ppo])
def test_evaluation_does_not_change_training_continuation(factory, tmp_path):
    from nexuml.training.lightning import NexuSession

    states = []
    for scheduled in (False, True):
        scenario = factory(total_frames=32, frames_per_batch=16)
        scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
        scenario.training.evaluation = RLEvaluationSpec(
            episodes=2,
            max_steps_per_episode=3,
            deterministic=False,
            at_start=scheduled,
            every_n_frames=16 if scheduled else None,
        )
        session = NexuSession(
            scenario,
            log_dir=tmp_path / str(scheduled),
            enable_loggers=False,
            enable_progress_bar=False,
        )
        session.fit()
        states.append(copy.deepcopy(session.lightning_module.state_dict()))
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(value, states[1][key]) for key, value in states[0].items())


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("factory", [cartpole_ppo, pendulum_ppo])
def test_scheduled_partial_and_full_budget_resume(factory, tmp_path, monkeypatch):
    from lightning.pytorch.callbacks import Callback
    from nexuml.training.lightning import NexuSession
    import nexuml.training.reinforcement as module

    calls = []
    original = module.evaluate_policy

    def evaluate(policy, scenario, **kwargs):
        calls.append(kwargs["frame_step"])
        return original(policy, scenario, **kwargs)

    monkeypatch.setattr(module, "evaluate_policy", evaluate)
    partial = tmp_path / "partial.ckpt"

    class Interrupt(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            trainer.save_checkpoint(partial)
            raise RuntimeError("checkpoint interruption")

    scenario = factory(total_frames=32, frames_per_batch=16)
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    scenario.training.evaluation = RLEvaluationSpec(
        episodes=1, max_steps_per_episode=3, at_start=True, every_n_frames=20
    )
    session = NexuSession(
        scenario, log_dir=tmp_path / "first", enable_loggers=False, enable_progress_bar=False
    )
    session.trainer.callbacks.append(Interrupt())  # ty: ignore[unresolved-attribute]
    with pytest.raises(RuntimeError, match="checkpoint interruption"):
        session.fit()
    assert calls == [0]
    calls.clear()
    resumed = NexuSession.from_trainer_checkpoint(
        partial, log_dir=tmp_path / "resumed", enable_loggers=False, enable_progress_bar=False
    )
    result = resumed.run()
    assert calls == [32] and result.interaction_results
    complete = tmp_path / "complete.ckpt"
    resumed.trainer.save_checkpoint(complete)
    calls.clear()
    completed = NexuSession.from_trainer_checkpoint(
        complete, log_dir=tmp_path / "complete", enable_loggers=False, enable_progress_bar=False
    )
    assert completed.run().interaction_results
    assert calls == [32] and completed.lightning_module._collector is None


@pytest.mark.requires_simulator
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("pygame")
@pytest.mark.requires_optional("imageio_ffmpeg")
@pytest.mark.parametrize("factory", [cartpole_ppo, pendulum_ppo])
@pytest.mark.parametrize("format", ["gif", "mp4", "both"])
@pytest.mark.parametrize("backend", ["tensorboard", "mlflow"])
def test_real_headless_recordings(factory, format, backend, tmp_path, monkeypatch):
    from nexuml.training.lightning import NexuSession
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if backend == "mlflow":
        pytest.importorskip("mlflow")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    scenario = factory(total_frames=32, frames_per_batch=16)
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    scenario.training.evaluation = RLEvaluationSpec(
        episodes=2,
        max_steps_per_episode=3,
        at_start=True,
        every_n_frames=16,
        video=RLVideoSpec(enabled=True, format=format, max_frames=4, max_size=65),
    )
    scenario.logging = LoggingSpec(
        tensorboard=TensorBoardSpec(log_dir=str(tmp_path / "tensorboard"))
        if backend == "tensorboard"
        else None,
        mlflow=MLflowSpec(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", auto_nested_runs=False
        )
        if backend == "mlflow"
        else None,
        temp_artifact_dir=str(tmp_path / "staging"),
        diagram=None,
    )
    session = NexuSession(scenario, log_dir=tmp_path / "training", enable_progress_bar=False)
    result = session.run()
    loggers = session.trainer_loggers
    assert isinstance(loggers, list)
    logger = loggers[0]
    if backend == "tensorboard":
        root = Path(logger.log_dir) / "artifacts/evaluation"
        events = EventAccumulator(logger.log_dir).Reload().Images("eval/rl/episode")
        assert [event.step for event in events] == [0, 16, 32]
    else:
        assert len(loggers) == 1 and type(logger).__name__ == "MLFlowLogger"
        root = Path(
            logger.experiment.download_artifacts(logger.run_id, "evaluation", str(tmp_path))
        )
    formats = {"gif", "mp4"} if format == "both" else {format}
    assert {file.suffix for file in root.iterdir()} == {".json", *[f".{f}" for f in formats]}
    assert len(list(root.glob("*.json"))) == 3
    data = json.loads((root / "frame_000000000032.json").read_text())
    assert data["captured_frames"] == 4 and data["frame_step"] == 32
    assert result.interaction_results["eval/rl/length_mean"] <= 3
    assert session.lightning_module._collector is None
    for file in root.iterdir():
        if file.suffix == ".gif":
            from PIL import Image

            with Image.open(file) as image:
                assert max(image.size) <= 65 and getattr(image, "n_frames") > 1
        elif file.suffix == ".mp4":
            import imageio.v2 as imageio

            with imageio.get_reader(file) as reader:
                assert getattr(reader, "count_frames")() == 4
                assert getattr(reader, "get_meta_data")()["codec"] == "h264"


@pytest.mark.requires_simulator
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("pygame")
@pytest.mark.requires_optional("imageio_ffmpeg")
@pytest.mark.parametrize("factory", [cartpole_ppo, pendulum_ppo])
def test_recorded_interrupt_resume_export_reload_and_cli_replay(factory, tmp_path, monkeypatch):
    from lightning.pytorch.callbacks import Callback
    from typer.testing import CliRunner
    from nexuml.cli.main import app
    from nexuml.core.export import export_package, load_inference_package
    from nexuml.training.lightning import NexuSession

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    scenario = factory(total_frames=32, frames_per_batch=16)
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    scenario.training.evaluation = RLEvaluationSpec(
        episodes=1,
        max_steps_per_episode=3,
        every_n_frames=16,
        at_start=True,
        video=RLVideoSpec(enabled=True, format="both", max_frames=4, max_size=64),
    )
    scenario.logging = LoggingSpec(
        tensorboard=TensorBoardSpec(log_dir=str(tmp_path / "tensorboard")),
        temp_artifact_dir=str(tmp_path / "staging"),
        diagram=None,
    )
    partial = tmp_path / "partial.ckpt"

    class Interrupt(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            trainer.save_checkpoint(partial)
            raise RuntimeError("recorded interruption")

    first = NexuSession(scenario, log_dir=tmp_path / "first", enable_progress_bar=False)
    first.trainer.callbacks.append(Interrupt())  # ty: ignore[unresolved-attribute]
    with pytest.raises(RuntimeError, match="recorded interruption"):
        first.fit()
    assert first.lightning_module._collector is None
    resumed = NexuSession.from_trainer_checkpoint(
        partial, log_dir=tmp_path / "resumed", enable_progress_bar=False
    )
    result = resumed.run()
    assert int(result.lightning_module.get_buffer("frames")) == 32
    loggers = resumed.trainer_loggers
    assert isinstance(loggers, list) and result.policy is not None
    resumed_media = Path(loggers[0].log_dir) / "artifacts/evaluation"
    assert [f.name for f in resumed_media.glob("*.json")] == ["frame_000000000032.json"]
    package = export_package(result.policy, tmp_path / "policy", trainer=result.trainer)
    policy, _, _ = load_inference_package(package)
    observation = TensorDict(
        {
            key: torch.zeros(field.shape, dtype=getattr(torch, field.dtype))
            for key, field in result.policy.contract.observations.items()
        },
        batch_size=[],
    )
    assert torch.equal(policy(observation)["action"], result.policy(observation)["action"])
    before = set((tmp_path / "tensorboard").glob("**/artifacts/evaluation/*.json"))
    replay = CliRunner().invoke(
        app, ["evaluate", str(package), "-O", "training.evaluation.video.format=mp4"]
    )
    assert replay.exit_code == 0, replay.output + repr(replay.exception)
    created = set((tmp_path / "tensorboard").glob("**/artifacts/evaluation/*.json")) - before
    assert len(created) == 1
    replay_json = created.pop()
    assert replay_json.stem == "frame_000000000000"
    assert {file.suffix for file in replay_json.parent.iterdir()} == {".json", ".mp4"}
    assert resumed.lightning_module._collector is None
