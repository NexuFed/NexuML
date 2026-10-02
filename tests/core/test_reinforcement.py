"""Finite-frame CPU integration, portability, synchronization and cleanup."""

import copy
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from tensordict import TensorDict

from nexuml.core.config import ResolvedConfig
from nexuml.core.export import (
    export_package,
    load_package,
    load_inference_package,
    load_package_for_training,
)
from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.types import (
    CollectorSpec,
    LayerSpec,
    PipelineSpec,
    PolicySpec,
    RLEvaluationSpec,
    ReinforcementLearningSpec,
    ScenarioSpec,
)
from nexuml.reinforcement.collector import RolloutDataset, synchronize_policy
from nexuml.training.lightning import NexuSession
from nexuml.training.reinforcement import NexuRLLightningModule
from nexuml.training.reinforcement import evaluate_policy
from nexuml_library.action_adapters.adapters import CategoricalActionAdapter
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.reinforcement.policies import PolicyHead
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo, pendulum_ppo


def smoke_scenario(factory):
    scenario = factory(total_frames=32, frames_per_batch=16)
    assert isinstance(scenario.training, ReinforcementLearningSpec)
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    scenario.training.evaluation = RLEvaluationSpec(episodes=2, max_steps_per_episode=10, seed=3)
    return scenario


def test_rl_configuration_requires_interaction_and_nested_components_round_trip():
    with pytest.raises(ValidationError, match="interaction and policy"):
        ScenarioSpec(name="invalid", training=ReinforcementLearningSpec(algorithm=PPO()))
    with pytest.raises(ValidationError):
        CollectorSpec(policy_sync_interval_batches=0)
    with pytest.raises(ValidationError):
        ReinforcementLearningSpec(algorithm=PPO(), total_frames=0)
    scenario = smoke_scenario(cartpole_ppo)
    assert isinstance(scenario.training, ReinforcementLearningSpec)
    scenario.training.algorithm = PPO(
        critic=PipelineSpec(
            stages={
                "value": [
                    LayerSpec(
                        component=PolicyHead(),
                        keys_in=["observation"],
                        keys_out=["state_value"],
                    )
                ]
            }
        )
    )
    text = ResolvedConfig.from_scenario(scenario).to_yaml()
    restored = ResolvedConfig.from_yaml(text).to_scenario()
    assert isinstance(restored.training, ReinforcementLearningSpec)
    assert isinstance(restored.training.algorithm, PPO)
    assert restored.training.algorithm.critic is not None
    assert isinstance(restored.training.algorithm.critic.stages["value"][0].component, PolicyHead)
    assert isinstance(restored.policy, PolicySpec)
    assert isinstance(restored.policy.action_adapter, CategoricalActionAdapter)
    assert "mode: reinforcement" in text


def test_copied_policy_synchronization_changes_rollout_weights():
    learner = nn.Linear(2, 1)

    class FakeCollector:
        def __init__(self):
            self.policy = copy.deepcopy(learner)

        def update_policy_weights_(self, weights):
            weights.to_module(self.policy)

    collector = FakeCollector()
    old_weight = collector.policy.weight.detach().clone()
    with torch.no_grad():
        learner.weight.add_(1)
    assert torch.equal(collector.policy.weight, old_weight)
    synchronize_policy(collector, learner)
    assert torch.equal(collector.policy.weight, learner.weight)
    assert not torch.equal(collector.policy.weight, old_weight)


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("factory", [pendulum_ppo, cartpole_ppo])
def test_cpu_fit_evaluate_checkpoint_and_worker_local_materialization(
    factory, tmp_path, monkeypatch
):
    builds = []
    original = GymnasiumEnvironment.build

    def traced_build(self, context):
        runtime = original(self, context)
        builds.append((context, runtime))
        return runtime

    monkeypatch.setattr(GymnasiumEnvironment, "build", traced_build)
    session = NexuSession(
        smoke_scenario(factory),
        log_dir=tmp_path,
        enable_loggers=False,
        enable_progress_bar=False,
    )
    runtime = session.runtime
    assert runtime.data_module is None
    assert isinstance(runtime.lightning_module, NexuRLLightningModule)
    assert runtime.lightning_module._collector is None and not builds
    before = {key: value.clone() for key, value in session.pipeline.state_dict().items()}
    result = session.run()
    assert isinstance(result.lightning_module, NexuRLLightningModule)
    assert result.policy is not None
    assert result.validation_results == [] and result.test_results == []
    assert int(result.lightning_module.frames) == 32
    assert int(result.lightning_module.algorithm.updates) >= 1
    assert result.trainer.optimizers and result.trainer.global_step > 0
    assert {
        "train/rl/loss_objective",
        "train/rl/loss_critic",
        "train/rl/loss_entropy",
        "train/rl/clip_fraction",
        "train/rl/reward",
        "train/rl/frames",
    } <= set(result.trainer.logged_metrics)
    assert any(
        not torch.equal(value, session.pipeline.state_dict()[key]) for key, value in before.items()
    )
    assert result.interaction_results["eval/rl/length_mean"] <= 10
    assert [context.purpose for context, _ in builds] == ["train", "eval"]
    assert all(not environment.envs for _, environment in builds)
    checkpoint = tmp_path / "resume.ckpt"
    result.trainer.save_checkpoint(checkpoint)
    restored = NexuSession.from_trainer_checkpoint(checkpoint, enable_loggers=False)
    assert isinstance(restored.lightning_module, NexuRLLightningModule)
    assert int(restored.lightning_module.frames) == 32
    assert restored.lightning_module._collector is None
    for key, value in result.lightning_module.algorithm.critic.state_dict().items():
        assert torch.equal(value, restored.lightning_module.algorithm.critic.state_dict()[key])
    payload = torch.load(checkpoint, weights_only=False)
    assert len(payload["optimizer_states"]) == 2
    assert any(key.startswith("algorithm.critic.") for key in payload["state_dict"])
    export_dir = export_package(result.policy, tmp_path / "policy", trainer=result.trainer)
    assert not (export_dir / "training_state.pt").exists()
    assert not (export_dir / "lightning.ckpt").exists()
    state = torch.load(export_dir / "state_dict.pt", weights_only=True)
    assert set(state) == set(result.pipeline.state_dict())
    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert metadata["training_mode"] == "reinforcement"
    assert metadata["algorithm"]["type"] == "PPO"
    assert metadata["environment"]["type"] == "GymnasiumEnvironment"
    assert len(metadata["interaction_contract_hash"]) == 64
    assert metadata["training_state_available"] is False
    observation = TensorDict(
        {
            key: torch.zeros((2, *field.shape), dtype=getattr(torch, field.dtype))
            for key, field in result.policy.contract.observations.items()
        },
        batch_size=[2],
    )
    expected = result.policy(observation)["action"].detach()
    torch.save(observation.to_dict(), tmp_path / "observation.pt")
    torch.save(expected, tmp_path / "expected.pt")
    for loader in (load_package, load_inference_package):
        policy, _, _ = loader(export_dir)
        assert hasattr(policy, "action_adapter")
        assert torch.equal(policy(observation)["action"], expected)
    training_reload = load_package_for_training(export_dir)
    assert isinstance(training_reload.lightning_module, NexuRLLightningModule)
    assert training_reload.lightning_module._collector is None
    assert training_reload.report.matched
    with pytest.raises(ValueError, match="complete CompiledPolicy"):
        export_package(result.pipeline, tmp_path / "incomplete")
    code = """
import importlib.abc, sys, torch
from pathlib import Path
from tensordict import TensorDict
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torchrl", "gymnasium", "rclpy", "mujoco", "carla"}:
            raise ImportError("optional runtime blocked: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from nexuml.core.export import load_package, load_inference_package, infer
root = Path(sys.argv[1])
observation = TensorDict(torch.load(root / "observation.pt", weights_only=True), batch_size=[2])
expected = torch.load(root / "expected.pt", weights_only=True)
for loader in (load_package, load_inference_package):
    policy, config, metadata = loader(root / "policy")
    assert torch.equal(infer(policy, observation)["action"], expected)
    assert metadata["training_state_available"] is False
    assert not any(name.split(".")[0] in {"torchrl", "gymnasium", "rclpy"} for name in sys.modules)
"""
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / "library/src"))),
    }
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("total_frames, frames_per_batch, num_envs", [(32, 16, 1), (9, 4, 3)])
def test_lightning_collects_only_after_the_previous_learner_update(
    tmp_path,
    monkeypatch,
    total_frames,
    frames_per_batch,
    num_envs,
):
    collected_after_batches = []
    spec = smoke_scenario(cartpole_ppo)
    assert isinstance(spec.training, ReinforcementLearningSpec)
    spec.training.total_frames = total_frames
    spec.training.frames_per_batch = frames_per_batch
    spec.training.collector = CollectorSpec(num_envs_per_collector=num_envs)
    session = NexuSession(
        spec,
        log_dir=tmp_path,
        enable_loggers=False,
        enable_progress_bar=False,
    )

    def traced_iteration(self):
        for rollout in self.collector:
            collected_after_batches.append(int(session.lightning_module.get_buffer("batches")))
            yield rollout

    monkeypatch.setattr(RolloutDataset, "__iter__", traced_iteration)
    session.fit()
    assert collected_after_batches == [0, 1]


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_environment_is_closed_when_learning_raises(tmp_path, monkeypatch):
    environments = []
    original = GymnasiumEnvironment.build

    def build(self, context):
        runtime = original(self, context)
        environments.append(runtime)
        return runtime

    monkeypatch.setattr(GymnasiumEnvironment, "build", build)
    session = NexuSession(smoke_scenario(cartpole_ppo), log_dir=tmp_path, enable_loggers=False)

    def fail(*args, **kwargs):
        raise RuntimeError("test update failure")

    monkeypatch.setattr(session.lightning_module.algorithm, "update", fail)
    with pytest.raises(RuntimeError, match="test update failure"):
        session.fit()
    assert environments and all(not environment.envs for environment in environments)
    assert session.lightning_module._collector is None


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("name", ["Pendulum-v1", "CartPole-v1"])
def test_vectorized_tensor_shapes_and_reset_masks(name):
    from nexuml.reinforcement.torchrl_env import _TorchRLEnvironmentAdapter
    from torchrl.envs.utils import check_env_specs

    runtime = GymnasiumEnvironment(env_name=name).build(
        EnvironmentBuildContext(purpose="train", num_envs=2)
    )
    env = _TorchRLEnvironmentAdapter(runtime)
    try:
        check_env_specs(env)
        assert env.rollout(3).batch_size == torch.Size((2, 3))
        observation = runtime.reset(seed=5)
        partial = runtime.reset(seed=7, mask=torch.tensor([True, False]))
        assert torch.equal(partial["observation"][1], observation["observation"][1])
    finally:
        env.close()


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_evaluation_override_is_fresh_deterministic_and_always_closed(tmp_path, monkeypatch):
    scenario = smoke_scenario(pendulum_ppo)
    assert scenario.interaction is not None
    override = GymnasiumEnvironment(env_name="Pendulum-v1", kwargs={"max_episode_steps": 2})
    scenario.interaction.evaluation_environment = override
    session = NexuSession(scenario, log_dir=tmp_path, enable_loggers=False)
    policy = session.runtime.policy
    assert policy is not None
    environments = []
    original = GymnasiumEnvironment.build

    def build(self, context):
        assert self == override and context.purpose == "eval"
        runtime = original(self, context)
        environments.append(runtime)
        return runtime

    monkeypatch.setattr(GymnasiumEnvironment, "build", build)
    metrics = evaluate_policy(policy, scenario)
    assert metrics == evaluate_policy(policy, scenario)
    assert metrics["eval/rl/length_mean"] == 2
    assert metrics["eval/rl/truncated"] == 2
    assert metrics["eval/rl/terminated"] == 0
    assert all(not runtime.envs for runtime in environments)
    assert policy.training

    def fail(*args, **kwargs):
        raise ValueError("test policy failure")

    monkeypatch.setattr(policy, "forward", fail)
    with pytest.raises(ValueError, match="test policy failure"):
        evaluate_policy(policy, scenario)
    assert all(not runtime.envs for runtime in environments)
    assert policy.training


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_partial_checkpoint_resumes_remaining_frames_and_optimizer_state(tmp_path):
    from lightning.pytorch.callbacks import Callback

    checkpoint = tmp_path / "partial.ckpt"

    class InterruptAfterCheckpoint(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            trainer.save_checkpoint(checkpoint)
            raise RuntimeError("interrupted after checkpoint")

    session = NexuSession(
        smoke_scenario(cartpole_ppo),
        log_dir=tmp_path / "first",
        enable_loggers=False,
        enable_progress_bar=False,
    )
    session.trainer.callbacks.append(InterruptAfterCheckpoint())  # ty: ignore[unresolved-attribute]
    with pytest.raises(RuntimeError, match="interrupted after checkpoint"):
        session.fit()
    assert isinstance(session.lightning_module, NexuRLLightningModule)
    assert int(session.lightning_module.frames) == 16
    assert session.lightning_module._collector is None
    checkpoint_data = torch.load(checkpoint, weights_only=False)
    optimizer_step = max(
        float(state["step"]) for state in checkpoint_data["optimizer_states"][0]["state"].values()
    )
    resumed = NexuSession.from_trainer_checkpoint(
        checkpoint,
        log_dir=tmp_path / "resumed",
        enable_loggers=False,
        enable_progress_bar=False,
    )
    assert isinstance(resumed.lightning_module, NexuRLLightningModule)
    assert int(resumed.lightning_module.frames) == 16
    result = resumed.run()
    assert int(resumed.lightning_module.frames) == 32
    assert int(resumed.lightning_module.algorithm.updates) == 4
    assert (
        max(float(state["step"]) for state in result.trainer.optimizers[0].state.values())
        > optimizer_step
    )


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_process_collection_updates_and_closes_workers(tmp_path):
    existing_workers = {worker.pid for worker in multiprocessing.active_children()}
    scenario = smoke_scenario(cartpole_ppo)
    assert isinstance(scenario.training, ReinforcementLearningSpec)
    scenario.training.collector = CollectorSpec(backend="process", num_collectors=2)
    session = NexuSession(
        scenario,
        log_dir=tmp_path,
        enable_loggers=False,
        enable_progress_bar=False,
    )
    assert isinstance(session.lightning_module, NexuRLLightningModule)
    assert session.lightning_module._collector is None
    assert {worker.pid for worker in multiprocessing.active_children()} <= existing_workers
    result = session.run()
    assert int(session.lightning_module.frames) == 32
    assert int(session.lightning_module.algorithm.updates) == 4
    assert result.interaction_results
    assert session.lightning_module._collector is None
    assert {worker.pid for worker in multiprocessing.active_children()} <= existing_workers


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("name", ["pendulum-ppo", "cartpole-ppo"])
def test_cli_discovery_config_train_and_export_checkpoint(name, tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from nexuml.cli.main import app
    from nexuml.core.scenario_registry import get_scenario_registry
    from nexuml.core.types import ExportSpec

    scenario = smoke_scenario(get_scenario_registry().get(name))
    export_dir = tmp_path / "cli-policy"
    scenario.exports = [ExportSpec(output=str(export_dir))]
    config = tmp_path / "scenario.yaml"
    ResolvedConfig.from_scenario(scenario).save(config)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["train", "--config", str(config)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "RL evaluation:" in result.output
    policy, _, _ = load_inference_package(export_dir)
    assert hasattr(policy, "action_adapter")
    checkpoint = next((tmp_path / ".experiments/checkpoints").glob("*.ckpt"))
    exported = runner.invoke(
        app,
        [
            "export",
            name,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "checkpoint-policy"),
        ],
    )
    assert exported.exit_code == 0, exported.output + repr(exported.exception)
    restored, _, _ = load_package(tmp_path / "checkpoint-policy")
    assert hasattr(restored, "action_adapter")
    for key, value in policy.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
