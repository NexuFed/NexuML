"""Offline supervised policy assembly, budget and artifact regression proofs."""

import json
import subprocess
import sys

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict
from torch import nn

from nexuml import nn_module
from nexuml.core.export import export_package, load_package_for_training
from nexuml.core.types import (
    DataSpec,
    InteractionContract,
    InteractionSpec,
    LayerSpec,
    PhysicalTensorContract,
    PipelineSpec,
    PolicySpec,
    ScenarioSpec,
    TensorFieldContract,
    TrainingSpec,
)
from nexuml.training.lightning import NexuLightningModule, NexuSession, create_runtime_artifacts
from nexuml_library.action_adapters.adapters import DirectActionAdapter
from nexuml_library.data.synthetic import SyntheticDataset
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.layers.loss.regression_loss import RegressionLoss


def _scenario(*, steps=8, offline=True):
    observation = TensorFieldContract(
        shape=(3,),
        dtype="float32",
        physical=PhysicalTensorContract(units=("rad",), frame="joint", joints=("a", "b", "c")),
    )
    contract = InteractionContract(
        observations={"observation": observation},
        actions={"action": TensorFieldContract(shape=(3,), dtype="float32")},
        action_space="continuous",
    )
    return ScenarioSpec(
        name="offline-policy-training",
        data=DataSpec(
            source=SyntheticDataset(feature_key="observation", feature_shape=(3,), num_samples=32),
            input_contracts={"observation": observation},
            train_split=0.75,
            val_split=0.125,
            test_split=0.125,
        ),
        pipeline=PipelineSpec(
            stages={
                "actor": [
                    LayerSpec(
                        component=nn_module(nn.Linear, 3, 3),
                        keys_in=["observation"],
                        keys_out=["command"],
                    ),
                    LayerSpec(
                        component=RegressionLoss(),
                        keys_in=["command"],
                        keys_out=["imitation_loss"],
                        label_key="reconstruction_target",
                    ),
                ]
            }
        ),
        training=TrainingSpec(
            max_epochs=-1,
            max_steps=steps,
            batch_size=8,
            loss_keys={"imitation_loss": 1.0},
            lr=0.05,
            seed=17,
            deterministic=True,
            accelerator="cpu",
            devices=1,
        ),
        interaction=InteractionSpec(environment=GymnasiumEnvironment(contract=contract))
        if offline
        else None,
        policy=PolicySpec(action_adapter=DirectActionAdapter(source_key="command"))
        if offline
        else None,
    )


def _session(scenario, path):
    return NexuSession(
        scenario,
        log_dir=path,
        enable_progress_bar=False,
        enable_loggers=False,
    )


@pytest.fixture(autouse=True)
def _no_interaction_or_rl(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline training/export must not build a simulator or RL learner")

    monkeypatch.setattr(GymnasiumEnvironment, "build", forbidden)
    monkeypatch.setattr(GymnasiumEnvironment, "_make", forbidden)
    monkeypatch.setattr("nexuml.training.lightning.NexuRLLightningModule.__init__", forbidden)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_supervised_policy_optimizes_same_pipeline_with_seeded_step_budget(tmp_path, monkeypatch):
    gradients = []

    def record_gradients(module):
        gradients.extend(parameter.grad.clone() for parameter in module.pipeline.parameters())

    monkeypatch.setattr(NexuLightningModule, "on_after_backward", record_gradients)
    states = []
    for attempt in range(2):
        session = _session(_scenario(), tmp_path / str(attempt))
        runtime = session.runtime
        assert runtime.policy is not None
        assert isinstance(runtime.lightning_module, NexuLightningModule)
        assert runtime.policy.pipeline is runtime.pipeline is runtime.lightning_module.pipeline
        before = {key: value.clone() for key, value in runtime.pipeline.state_dict().items()}
        x, y = runtime.data_module.dataset[0]
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        initial_loss = runtime.pipeline(x.clone(), y)[0]["imitation_loss"].item()

        result = session.run()

        assert result.policy is runtime.policy and result.pipeline is runtime.pipeline
        assert result.trainer.global_step == result.trainer.max_steps == 8
        assert result.trainer.max_epochs == -1
        assert torch.are_deterministic_algorithms_enabled()
        assert any(
            not torch.equal(value, result.pipeline.state_dict()[key])
            for key, value in before.items()
        )
        assert result.pipeline(x.clone(), y)[0]["imitation_loss"].item() < initial_loss
        assert all(torch.isfinite(value) for value in result.trainer.logged_metrics.values())
        assert set(result.trainer.optimizers[0].state) == set(result.pipeline.parameters())
        torch.testing.assert_close(
            result.policy(x)["action"], result.pipeline(x.clone())[0]["command"]
        )
        states.append({key: value.clone() for key, value in result.pipeline.state_dict().items()})
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key])
    assert len(gradients) == 2 * 8 * 2
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.count_nonzero() for gradient in gradients)


def test_ordinary_supervised_training_retains_defaults_and_no_policy(tmp_path):
    scenario = _scenario(offline=False)
    scenario.training = TrainingSpec(
        max_epochs=1, batch_size=8, loss_keys={"imitation_loss": 1}, accelerator="cpu", devices=1
    )
    result = _session(scenario, tmp_path).run()
    assert result.policy is None
    assert result.trainer.max_steps == -1 and result.trainer.global_step == 3
    assert scenario.training.seed is None and scenario.training.deterministic is False


def test_offline_policy_checkpoint_export_reload_and_resume(tmp_path):
    session = _session(_scenario(steps=3), tmp_path / "run")
    result = session.run()
    observation = TensorDict({"observation": torch.tensor([[0.1, 0.2, 0.3]])}, batch_size=[1])
    expected = result.policy(observation)["action"].detach()
    checkpoint_path = tmp_path / "resume.ckpt"
    result.trainer.save_checkpoint(checkpoint_path)
    restored = NexuSession.from_trainer_checkpoint(
        checkpoint_path,
        log_dir=tmp_path / "restore",
        enable_progress_bar=False,
        enable_loggers=False,
    )
    assert (
        restored.runtime.policy.pipeline is restored.pipeline is restored.lightning_module.pipeline
    )
    assert torch.equal(restored.runtime.policy(observation)["action"], expected)
    state = torch.load(checkpoint_path, weights_only=False)
    assert all(key.startswith("pipeline.") for key in state["state_dict"])

    export_dir = export_package(result.policy, tmp_path / "policy", trainer=result.trainer)
    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert metadata["artifact_kind"] == "policy" and not metadata["training_state_available"]
    assert not (export_dir / "training_state.pt").exists()
    assert not (export_dir / "lightning.ckpt").exists()
    assert "algorithm" not in metadata and "training_mode" not in metadata
    reloaded = load_package_for_training(export_dir)
    assert isinstance(reloaded.lightning_module, NexuLightningModule)
    assert reloaded.policy.pipeline is reloaded.pipeline is reloaded.lightning_module.pipeline
    assert torch.equal(reloaded.policy(observation)["action"], expected)
    assert reloaded.pipeline.loss_keys == {"imitation_loss": 1.0}
    assert reloaded.report.matched and not reloaded.report.missing

    torch.save({"observation": observation.to_dict(), "action": expected}, tmp_path / "expected.pt")
    code = """
import importlib.abc, sys, torch
from pathlib import Path
from tensordict import TensorDict
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('nexuml.training') or fullname.split('.')[0] in {
            'lightning', 'torchrl', 'gymnasium', 'isaaclab', 'isaacsim', 'mujoco', 'rclpy'
        }:
            raise AssertionError('training/simulator import: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
from nexuml.core.export import load_package, load_inference_package
root = Path(sys.argv[1])
saved = torch.load(root / 'expected.pt', weights_only=True)
observation = TensorDict(saved['observation'], batch_size=[1])
for source in (root / 'policy', root / 'policy/pipeline.package'):
    for loader in (load_package, load_inference_package):
        policy, config, metadata = loader(source)
        assert torch.equal(policy(observation)['action'], saved['action'])
        assert config.training.max_steps == 3 and config.training.seed == 17
        assert config.training.loss_keys == {'imitation_loss': 1.0}
        assert metadata['training_state_available'] is False
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    state["hyper_parameters"]["scenario"]["training"]["max_steps"] = 5
    torch.save(state, checkpoint_path)
    resumed = NexuSession.from_trainer_checkpoint(
        checkpoint_path,
        log_dir=tmp_path / "resume",
        enable_progress_bar=False,
        enable_loggers=False,
    )
    resumed.fit()
    assert resumed.trainer.global_step == 5
    assert not torch.equal(resumed.runtime.policy(observation)["action"], expected)
    assert all(int(value["step"]) == 5 for value in resumed.trainer.optimizers[0].state.values())


@pytest.mark.parametrize("offline", [True, False])
def test_legacy_checkpoint_missing_budget_seed_and_deterministic_restores_defaults(
    tmp_path, offline
):
    session = _session(_scenario(steps=1, offline=offline), tmp_path / "run")
    session.fit()
    checkpoint = tmp_path / "legacy.ckpt"
    session.trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    training = saved["hyper_parameters"]["scenario"]["training"]
    for key in ("max_steps", "seed", "deterministic"):
        training.pop(key)
    training["max_epochs"] = 1
    torch.save(saved, checkpoint)
    restored = NexuSession.from_trainer_checkpoint(
        checkpoint, log_dir=tmp_path / "restored", enable_progress_bar=False, enable_loggers=False
    )
    assert restored.scenario.training.max_steps is None
    assert restored.scenario.training.seed is None and not restored.scenario.training.deterministic
    assert restored.trainer.max_steps == -1
    assert (restored.runtime.policy is not None) == offline
    for key, value in session.pipeline.state_dict().items():
        assert torch.equal(value, restored.pipeline.state_dict()[key])


@pytest.mark.parametrize("steps", [0, -1])
def test_nonpositive_step_budget_is_rejected(steps):
    with pytest.raises(ValidationError, match="max_steps"):
        TrainingSpec(max_steps=steps)


@pytest.mark.parametrize("mismatch", ["shape", "sample_shape", "missing", "dtype", "physical"])
def test_incompatible_dataset_inputs_fail_before_fit(mismatch):
    scenario = _scenario()
    if mismatch == "shape":
        scenario.data.input_contracts = {}
        scenario.data.input_shapes = {"observation": [4]}
    elif mismatch == "sample_shape":
        scenario.data.source = SyntheticDataset(feature_key="observation", feature_shape=(4,))
    elif mismatch == "missing":
        scenario.data.source = SyntheticDataset(feature_key="other", feature_shape=(3,))
    elif mismatch == "dtype":
        field = scenario.interaction.environment.contract.observations["observation"]
        scenario.interaction.environment.contract.observations["observation"] = field.model_copy(
            update={"dtype": "float64"}
        )
    else:
        field = scenario.data.input_contracts["observation"]
        scenario.data.input_contracts = {
            "observation": field.model_copy(
                update={"physical": field.physical.model_copy(update={"units": ("degree",)})}
            )
        }
    with pytest.raises(ValueError, match="Dataset input.*observation"):
        create_runtime_artifacts(scenario)


def test_undeclared_shapes_hydrate_from_actual_offline_inputs():
    scenario = _scenario()
    scenario.data.input_contracts = {}
    runtime = create_runtime_artifacts(scenario)
    assert runtime.policy.pipeline is runtime.pipeline
    assert runtime.lightning_module.scenario.data.input_shapes == {"observation": [3]}


@pytest.mark.parametrize("state_kind", ["parameter", "buffer", "nonpersistent_buffer"])
def test_stateful_offline_adapters_are_rejected_before_fit(monkeypatch, state_kind):
    adapter = nn.Module()
    if state_kind == "parameter":
        adapter.register_parameter("weight", nn.Parameter(torch.ones(1)))
    else:
        adapter.register_buffer("offset", torch.ones(1), persistent=state_kind == "buffer")
    monkeypatch.setattr(DirectActionAdapter, "build", lambda self, contract: adapter)
    with pytest.raises(ValueError, match="stateless action adapter.*checkpoint owner"):
        create_runtime_artifacts(_scenario())


def test_training_reload_rejects_incompatible_dataset_contract(tmp_path):
    runtime = create_runtime_artifacts(_scenario())
    export_dir = export_package(runtime.policy, tmp_path / "policy")
    scenario = _scenario()
    scenario.data.input_contracts = {}
    scenario.data.input_shapes = {"observation": [4]}
    with pytest.raises(ValueError, match="Dataset input.*observation"):
        load_package_for_training(export_dir, scenario=scenario)
