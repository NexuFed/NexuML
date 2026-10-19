"""Simulator-free contracts and explicitly opted-in execution smoke checks."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import (
    InteractionContract,
    RLEvaluationSpec,
    ReinforcementLearningSpec,
    TensorFieldContract,
)
from nexuml.training.lightning import NexuSession
from nexuml.training.reinforcement import evaluate_policy
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo
from nexuml_library.scenarios.reinforcement.simulation import (
    visual_cartpole_ppo,
    mujoco_humanoid_ppo,
    mujoco_reacher_ppo,
)


@pytest.mark.parametrize("factory", [visual_cartpole_ppo, mujoco_humanoid_ppo, mujoco_reacher_ppo])
def test_optional_scenario_compiles_without_simulator_materialization(factory, monkeypatch):
    def no_simulator(*args, **kwargs):
        raise AssertionError("description/compilation must not boot a simulator")

    monkeypatch.setattr(GymnasiumEnvironment, "_make", no_simulator)
    spec = factory(total_frames=16, frames_per_batch=16)
    assert spec.interaction is not None and spec.policy is not None
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(spec).to_yaml()).to_scenario()
    assert restored.interaction is not None and restored.policy is not None
    contract = restored.interaction.environment.describe()
    policy = CompiledPolicy(
        compile(restored), restored.policy.action_adapter.build(contract), contract
    )
    observation = TensorDict(
        {
            key: torch.zeros(field.shape, dtype=getattr(torch, field.dtype))
            for key, field in contract.observations.items()
        },
        batch_size=[],
    )
    assert policy(observation)["action"].shape == contract.actions["action"].shape
    if spec.name == "visual-cartpole-ppo":
        assert contract.observations["image"].modality == "image"
        assert contract.observations["image"].dtype == "uint8"


@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("factory", [mujoco_humanoid_ppo, mujoco_reacher_ppo])
def test_missing_mujoco_reports_the_simulator_extra(factory, monkeypatch):
    import gymnasium

    def missing_simulator(*args, **kwargs):
        raise gymnasium.error.DependencyNotInstalled("MuJoCo intentionally unavailable")

    monkeypatch.setattr(gymnasium, "make", missing_simulator)
    spec = factory()
    assert spec.interaction is not None
    with pytest.raises(ImportError, match=r"nexuml-library\[mujoco\]"):
        spec.interaction.environment.build(EnvironmentBuildContext(purpose="train"))


@pytest.mark.requires_simulator
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("pygame")
def test_visual_policy_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    _fit_smoke(visual_cartpole_ppo, tmp_path)


@pytest.mark.requires_simulator
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("mujoco")
@pytest.mark.parametrize("factory", [mujoco_humanoid_ppo, mujoco_reacher_ppo])
def test_mujoco_policy_smoke(factory, tmp_path):
    _fit_smoke(factory, tmp_path)


def _fit_smoke(factory, tmp_path):
    spec = factory(total_frames=16, frames_per_batch=16)
    assert isinstance(spec.training, ReinforcementLearningSpec)
    spec.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    spec.training.evaluation = RLEvaluationSpec(episodes=1, max_steps_per_episode=2)
    result = NexuSession(
        spec, log_dir=tmp_path, enable_loggers=False, enable_progress_bar=False
    ).run()
    assert result.policy is not None and result.interaction_results
    assert result.trainer.global_step > 0


@pytest.fixture
def outcome_runtime(monkeypatch):
    class Native:
        def __init__(self, info):
            self.info = info
            self.closed = False

        def reset(self, *, seed=None):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.ones(1, dtype=np.float32), 2.0, False, True, self.info

        def close(self):
            self.closed = True

    def build(infos, device="cpu"):
        natives = [Native(info) for info in infos]
        pending = iter(natives)
        monkeypatch.setattr(GymnasiumEnvironment, "_make", lambda self: next(pending))
        definition = GymnasiumEnvironment(
            contract=InteractionContract(
                observations={"observation": TensorFieldContract(shape=(1,), dtype="float32")},
                actions={"action": TensorFieldContract(shape=(), dtype="int64", num_values=2)},
                action_space="discrete",
            )
        )
        runtime = definition.build(
            EnvironmentBuildContext(purpose="eval", num_envs=len(infos), device=device)
        )
        observation = runtime.reset(seed=7)
        actions = TensorDict(
            {"action": torch.zeros(observation.batch_size, dtype=torch.int64, device=device)},
            batch_size=observation.batch_size,
        )
        return runtime, natives, actions

    return build


@pytest.mark.parametrize(
    "value,expected",
    [
        (False, False),
        (True, True),
        (0, False),
        (1, True),
        (0.0, False),
        (1.0, True),
        (np.bool_(True), True),
        (np.int64(0), False),
        (np.float32(1), True),
        (np.array(1.0), True),
    ],
)
@pytest.mark.parametrize("num_envs", [1, 3])
def test_native_success_is_validated_and_keeps_batch_device_and_reward(
    outcome_runtime, value, expected, num_envs
):
    infos = [{"success": value, "private": object()} for _ in range(num_envs)]
    if num_envs > 1:
        infos[-1]["success"] = not expected
    runtime, natives, actions = outcome_runtime(infos)
    try:
        transition = runtime.step(actions)
        expected_values = [expected] * num_envs
        if num_envs > 1:
            expected_values[-1] = not expected
        success = transition.info["task_success"]
        assert success.shape == transition.reward.shape
        assert success.dtype == torch.bool and success.device == transition.reward.device
        assert success.flatten().tolist() == expected_values
        assert torch.all(transition.reward == 2.0) and torch.all(transition.truncated)
        assert not transition.terminated.any()
        assert set(transition.info) == {"task_success"}
        assert set(transition.observation.keys()) == {"observation"}
        for native in natives:
            native.info["success"] = not expected
        assert success.flatten().tolist() == expected_values
    finally:
        runtime.close()
    assert all(native.closed for native in natives)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "1",
        [],
        [1],
        np.array([1]),
        np.ones(2),
        np.array([[True]]),
        float("nan"),
        float("inf"),
        -1,
        0.5,
        2,
        complex(1, 0),
        object(),
    ],
)
def test_native_success_rejects_ambiguous_flags(outcome_runtime, value):
    runtime, _, actions = outcome_runtime([{"success": value}])
    try:
        with pytest.raises(ValueError, match="success"):
            runtime.step(actions)
    finally:
        runtime.close()


@pytest.mark.parametrize("infos", [[{}, {"success": True}], [{"success": False}, {}]])
def test_native_success_requires_consistent_rows(outcome_runtime, infos):
    runtime, _, actions = outcome_runtime(infos)
    try:
        with pytest.raises(ValueError, match="success.*consisten"):
            runtime.step(actions)
    finally:
        runtime.close()


@pytest.mark.parametrize("initial", [False, True])
def test_native_success_cannot_appear_or_disappear_after_reset(outcome_runtime, initial):
    runtime, natives, actions = outcome_runtime([{"success": False} if initial else {}])
    try:
        transition = runtime.step(actions)
        assert (transition.info is not None) == initial
        runtime.reset(seed=8)
        natives[0].info = {} if initial else {"success": True}
        with pytest.raises(ValueError, match="success.*consisten"):
            runtime.step(actions)
    finally:
        runtime.close()


@pytest.mark.requires_gpu
def test_native_success_uses_requested_cuda_device(outcome_runtime):
    runtime, _, actions = outcome_runtime([{"success": 0.0}, {"success": 1.0}], "cuda:0")
    try:
        transition = runtime.step(actions)
        assert transition.info["task_success"].device == torch.device("cuda:0")
        assert transition.info["task_success"].flatten().tolist() == [False, True]
    finally:
        runtime.close()


@pytest.mark.parametrize("success", [1.0, 0.5])
def test_native_outcome_reaches_evaluation_and_failures_close_runtime(
    outcome_runtime, monkeypatch, success
):
    runtime, natives, _ = outcome_runtime([{"success": success}])
    scenario = cartpole_ppo(total_frames=16, frames_per_batch=16)
    scenario.interaction.environment = GymnasiumEnvironment(contract=runtime.contract)
    scenario.training.evaluation = RLEvaluationSpec(episodes=2, max_steps_per_episode=3)
    monkeypatch.setattr(GymnasiumEnvironment, "build", lambda self, context: runtime)
    policy = CompiledPolicy(
        compile(scenario), scenario.policy.action_adapter.build(runtime.contract), runtime.contract
    )
    if success == 1.0:
        metrics = evaluate_policy(policy, scenario)
        assert metrics["eval/rl/task_success_rate"] == 1.0
        assert metrics["eval/rl/return_mean"] == 2.0
        assert metrics["eval/rl/length_mean"] == 1.0
        assert metrics["eval/rl/truncated"] == 2
        assert "eval/rl/final_joint_error_rad_mean" not in metrics
    else:
        with pytest.raises(ValueError, match="Native task success"):
            evaluate_policy(policy, scenario)
    assert all(native.closed for native in natives)
