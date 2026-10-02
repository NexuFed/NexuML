"""Simulator-free contracts and explicitly opted-in execution smoke checks."""

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import RLEvaluationSpec, ReinforcementLearningSpec
from nexuml.training.lightning import NexuSession
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
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
