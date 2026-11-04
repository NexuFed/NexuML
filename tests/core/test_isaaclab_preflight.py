"""Training entry points reject unsupported Isaac settings before native allocation."""

import builtins

import pytest
import torch

from nexuml.core.components import EnvironmentDefinition
from nexuml.core.types import RLEvaluationSpec, RLVideoSpec, ReinforcementLearningSpec
from nexuml.reinforcement.collector import build_collector
from nexuml.training.lightning import NexuSession
from nexuml.training.reinforcement import NexuRLLightningModule
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo
from nexuml_library.scenarios.reinforcement.isaaclab import (
    isaac_cartpole_ppo,
    isaac_franka_reach_ppo,
    isaac_so101_joint_reach_ppo,
)


def forbidden(*args, **kwargs):
    raise AssertionError("preflight must precede runtime or learner allocation")


@pytest.fixture(autouse=True)
def no_native_runtime(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root.startswith(("isaaclab", "isaacsim")) or root in {"omni", "pxr"}:
            raise AssertionError(f"preflight must not import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(IsaacLabEnvironment, "build", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


@pytest.mark.parametrize("entry", ["module", "session", "collector"])
@pytest.mark.parametrize(
    "section,changes,match",
    [
        ("collector", {"backend": "process"}, "synchronous direct collector"),
        ("collector", {"backend": "ray"}, "synchronous direct collector"),
        ("collector", {"sync": False}, "synchronous direct collector"),
        ("collector", {"num_collectors": 2}, "num_collectors=1"),
        ("collector", {"env_device": "cpu"}, "CUDA device for collector.env_device"),
        ("collector", {"storing_device": "cpu"}, "CUDA device for collector.storing_device"),
        ("collector", {"policy_device": "cuda:1"}, "same CUDA device index"),
        (None, {"devices": 2}, "training.devices=1|supports one learner"),
        (None, {"strategy": "ddp"}, "training.strategy='auto'"),
        (None, {"frames_per_batch": 1_025}, "frames_per_batch must be divisible"),
        (None, {"evaluation": RLEvaluationSpec()}, "fresh-process replay"),
        ("evaluation", {"at_start": True}, "fresh-process replay"),
        ("evaluation", {"every_n_frames": 64}, "fresh-process replay"),
        ("evaluation", {"video": RLVideoSpec(enabled=True)}, "fresh-process replay"),
    ],
)
def test_invalid_training_fails_before_allocation(
    entry, section, changes, match, tmp_path, monkeypatch
):
    scenario = isaac_cartpole_ppo()
    training = scenario.training
    if section is None:
        scenario.training = training.model_copy(update=changes)
    else:
        setattr(training, section, getattr(training, section).model_copy(update=changes))
    monkeypatch.setattr(PPO, "build", forbidden)
    if entry == "module":
        monkeypatch.setattr("nexuml.training.reinforcement.compile", forbidden)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torchrl.collectors":
            raise AssertionError("preflight must precede TorchRL collector imports")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match=match):
        if entry == "module":
            NexuRLLightningModule(scenario)
        elif entry == "session":
            session = NexuSession(scenario, log_dir=tmp_path, enable_loggers=False)
            session.build_runtime()
        else:
            build_collector(scenario.interaction.environment, scenario.training, None, "cuda:0")


@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize(
    "factory", [isaac_cartpole_ppo, isaac_franka_reach_ppo, isaac_so101_joint_reach_ppo]
)
def test_valid_defaults_construct_cpu_module_but_reject_cpu_fit(factory, monkeypatch):
    module = NexuRLLightningModule(factory())
    assert module.device == torch.device("cpu")
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())
    assert module._collector is None
    monkeypatch.setattr("nexuml.tracking.logger.preflight_evaluation_media", forbidden)
    monkeypatch.setattr(module, "evaluate_current_policy", forbidden)
    with pytest.raises(ValueError, match="CUDA device for learner device"):
        module.on_fit_start()


@pytest.mark.requires_optional("torchrl")
def test_cpu_session_fit_rejects_resolved_device_before_media(tmp_path, monkeypatch):
    session = NexuSession(
        isaac_cartpole_ppo(),
        accelerator="cpu",
        devices=1,
        log_dir=tmp_path,
        enable_loggers=False,
        enable_progress_bar=False,
    )
    monkeypatch.setattr("nexuml.tracking.logger.preflight_evaluation_media", forbidden)
    with pytest.raises(ValueError, match="CUDA device for learner device"):
        session.fit()
    assert session.lightning_module._collector is None


@pytest.mark.requires_optional("torchrl")
def test_fit_start_revalidates_mutated_training(monkeypatch):
    module = NexuRLLightningModule(isaac_cartpole_ppo())
    module.training_spec.collector.sync = False
    monkeypatch.setattr("nexuml.tracking.logger.preflight_evaluation_media", forbidden)
    with pytest.raises(ValueError, match="synchronous direct collector"):
        module.on_fit_start()
    assert module._collector is None


@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize(
    "field,value,match",
    [
        ("frames_per_batch", 1_025, "frames_per_batch must be divisible"),
        ("evaluation", RLEvaluationSpec(), "fresh-process replay"),
    ],
)
def test_collector_revalidates_postconstructor_mutations(field, value, match, monkeypatch):
    module = NexuRLLightningModule(isaac_cartpole_ppo())
    setattr(module.training_spec, field, value)
    assert module.scenario.interaction is not None
    monkeypatch.setattr("nexuml.reinforcement.collector.EnvironmentFactory.__call__", forbidden)
    with pytest.raises(ValueError, match=match):
        build_collector(
            module.scenario.interaction.environment,
            module.training_spec,
            module.algorithm.rollout_policy,
            torch.device("cuda:0"),
        )
    assert module._collector is None


@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_gymnasium_keeps_optional_noop_validation_and_evaluation_defaults(tmp_path):
    assert GymnasiumEnvironment.validate_training is EnvironmentDefinition.validate_training
    session = NexuSession(
        cartpole_ppo(total_frames=32, frames_per_batch=16),
        accelerator="cpu",
        devices=1,
        log_dir=tmp_path,
        enable_loggers=False,
        enable_progress_bar=False,
    )
    module = session.lightning_module
    assert isinstance(module, NexuRLLightningModule)
    assert isinstance(module.training_spec, ReinforcementLearningSpec)
    module.trainer = session.trainer
    module.on_fit_start()
    assert module.training_spec.evaluation == RLEvaluationSpec()
    assert module._collector is None
