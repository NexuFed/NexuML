"""Isaac static contracts and preflight require neither a simulator nor CUDA."""

import json
import os
import subprocess
import sys
import textwrap
from inspect import signature
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.cli.overrides import apply_overrides
from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentDefinition
from nexuml.core.config import ResolvedConfig
from nexuml.core.policy import CompiledPolicy
from nexuml.core.scenario_registry import get_scenario_registry
from nexuml.core.serialization import (
    lower_component,
    lower_model,
    restore_component,
    restore_model_data,
)
from nexuml.core.types import (
    CollectorSpec,
    RLEvaluationSpec,
    ReinforcementLearningSpec,
    RLVideoSpec,
    StrategySpec,
)
from nexuml_library.action_adapters.adapters import GaussianActionAdapter
from nexuml_library.environments.isaaclab import (
    IsaacLabEnvironment,
    IsaacBindings,
    isaac_task_config,
)
from nexuml_library.environments._isaaclab_tasks import configure_state_reference
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.isaaclab import (
    isaac_cartpole_ppo,
    isaac_franka_reach_ppo,
    isaac_so101_joint_reach_ppo,
)


_REVISION = "70ab24fec3eabf51222817db0cdb5a4e7133bd7e"
_REFERENCES = [
    (isaac_cartpole_ppo, "isaac-cartpole-ppo", "Isaac-Cartpole", 4, 1, 1 / 60),
    (isaac_franka_reach_ppo, "isaac-franka-reach-ppo", "Isaac-Reach-Franka", 32, 7, 1 / 30),
    (
        isaac_so101_joint_reach_ppo,
        "isaac-so101-joint-reach-ppo",
        "NexuML-SO101-Joint-Reach",
        15,
        5,
        1 / 30,
    ),
]


@pytest.mark.parametrize("factory,name,task,obs_size,action_size,period", _REFERENCES)
def test_static_contract_identity_configuration_and_inference(
    factory, name, task, obs_size, action_size, period, monkeypatch
):
    def forbidden_build(*args, **kwargs):
        raise AssertionError("static paths must not materialize Isaac")

    monkeypatch.setattr(IsaacLabEnvironment, "build", forbidden_build)
    assert get_scenario_registry().get(name) is factory
    assert list(signature(factory).parameters) == ["total_frames", "frames_per_batch", "num_envs"]
    spec = factory()
    definition = spec.interaction.environment
    assert definition == IsaacLabEnvironment(
        task=task, task_config=isaac_task_config(configure_state_reference)
    )
    assert definition.model_dump() == {
        "task": task,
        "revision": _REVISION,
        "physics": "isaacsim_physx",
        "presets": (),
        "task_config": isaac_task_config(configure_state_reference).model_dump(),
        "contract": None,
        "bindings": None,
        "viewer": None,
    }
    with pytest.raises(ValidationError, match="frozen"):
        definition.task = "Isaac-Cartpole"
    contract = definition.describe()
    assert contract.num_envs == 1
    assert contract.control_period_s == period
    assert contract.action_space == "continuous"
    assert set(contract.observations) == {"observation"}
    assert set(contract.actions) == {"action"}
    observation, action = contract.observations["observation"], contract.actions["action"]
    assert observation.shape == (obs_size,)
    assert observation.dtype == "float32" and observation.modality == "vector"
    assert action.shape == (action_size,) and action.dtype == "float32"
    assert action.low == -1 and action.high == 1
    assert action.rate_hz == 1 / period
    for identity in (task, "isaacsim_physx"):
        assert identity in action.description
    lowered = lower_component(definition)
    assert lowered == {
        "type": "IsaacLabEnvironment",
        "version": "2",
        "params": definition.model_dump(mode="json"),
    }
    assert restore_component(kind="environment", value=lowered) == definition
    assert isinstance(spec.policy.action_adapter, GaussianActionAdapter)
    training = spec.training
    assert isinstance(training, ReinforcementLearningSpec)
    assert isinstance(training.algorithm, PPO)
    assert training.seed == 0 and training.accelerator == "gpu" and training.devices == 1
    assert training.strategy == "auto"
    assert training.collector == CollectorSpec(
        num_envs_per_collector=32, env_device="cuda:0", storing_device="cuda:0"
    )
    assert training.evaluation == RLEvaluationSpec(episodes=0)
    assert spec.logging.tensorboard is not None
    definition.validate_training(training)

    spec = factory(total_frames=256, frames_per_batch=64, num_envs=4)
    apply_overrides(
        spec,
        [
            "training.total_frames=512",
            "training.frames_per_batch=128",
            "training.collector.num_envs_per_collector=8",
        ],
    )
    resolved = ResolvedConfig.from_scenario(spec)
    yaml = resolved.to_yaml()
    restored = ResolvedConfig.from_yaml(yaml).to_scenario()
    json_config = json.loads(json.dumps(lower_model(resolved), allow_nan=False))
    from_json = ResolvedConfig.model_validate(restore_model_data(json_config, ResolvedConfig))
    assert lower_model(from_json) == lower_model(resolved)
    assert lower_model(restored) == lower_model(spec)
    assert restored.interaction is not None and restored.policy is not None
    assert isinstance(restored.training, ReinforcementLearningSpec)
    assert restored.interaction.environment.describe() == contract
    assert restored.training.total_frames == 512
    assert restored.training.frames_per_batch == 128
    assert restored.training.collector.num_envs_per_collector == 8
    assert _REVISION in yaml and "IsaacLabEnvironment" in yaml
    pipeline = compile(restored)
    assert restored.policy.action_adapter is not None
    policy = CompiledPolicy(pipeline, restored.policy.action_adapter.build(contract), contract)
    for batch_shape in ((), (3,), (2, 3)):
        inputs = TensorDict(
            {"observation": torch.zeros(*batch_shape, obs_size)}, batch_size=batch_shape
        )
        expected = policy(inputs)["action"]
        assert set(inputs.keys()) == {"observation"}
        assert torch.equal(expected, policy(inputs)["action"])
        for actions in (expected, policy(inputs, deterministic=False)["action"]):
            assert actions.shape == (*batch_shape, action_size)
            assert actions.dtype == torch.float32 and torch.isfinite(actions).all()
            assert (actions >= -1).all() and (actions <= 1).all()


def test_control_descriptions_preserve_physical_mapping_and_joint_order():
    cartpole = IsaacLabEnvironment().describe().actions["action"].description
    assert cartpole is not None
    assert "slider_to_cart" in cartpole and "continuous joint effort = 100 * action N" in cartpole
    franka = IsaacLabEnvironment(task="Isaac-Reach-Franka").describe().actions["action"].description
    assert franka is not None
    assert ", ".join(f"panda_joint{index}" for index in range(1, 8)) in franka
    assert "default joint pose + 0.5 * action rad" in franka
    so101 = IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach").describe()
    assert so101.actions["action"].description is not None
    assert so101.observations["observation"].description is not None
    joints = "shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll"
    assert joints in so101.actions["action"].description
    assert joints in so101.observations["observation"].description
    assert "positions (rad), velocities (rad/s), and target positions (rad)" in (
        so101.observations["observation"].description
    )
    assert "gripper is fixed at 0.2555162024919699 rad" in so101.actions["action"].description
    assert (
        "absolute joint-position targets = 0.5 * action rad" in so101.actions["action"].description
    )


def test_custom_task_keeps_explicit_contract_and_literal_bindings_without_resolution():
    contract = IsaacLabEnvironment().describe()
    definition = IsaacLabEnvironment(
        task="installed_tasks:Custom-Cartpole",
        physics="custom_backend",
        presets=("custom_preset",),
        contract=contract,
        bindings=IsaacBindings(observations={"observation": ("custom", "state")}),
    )
    assert definition.describe() == contract
    assert restore_component(kind="environment", value=lower_component(definition)) == definition


@pytest.mark.parametrize("physics", ["newton_mjwarp", "isaacsim_physx", "ovphysx"])
def test_direct_cartpole_has_explicit_portable_control_and_backend_identity(physics):
    definition = IsaacLabEnvironment(task="Isaac-Cartpole-Direct", physics=physics)
    contract = definition.describe()
    assert contract.observations["observation"].shape == (4,)
    assert contract.actions["action"].shape == (1,)
    assert contract.control_period_s == 1 / 60
    assert contract.actions["action"].description is not None
    assert physics in contract.actions["action"].description
    assert restore_component(kind="environment", value=lower_component(definition)) == definition
    with pytest.raises(ValidationError, match="validated PhysX"):
        IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach", physics="newton_mjwarp")


@pytest.mark.parametrize(
    "factory", [isaac_cartpole_ppo, isaac_franka_reach_ppo, isaac_so101_joint_reach_ppo]
)
def test_complete_policy_package_inference_with_simulation_and_learning_blocked(factory, tmp_path):
    from nexuml.core.export import export_package

    scenario = factory()
    assert scenario.interaction is not None and scenario.policy is not None
    contract = scenario.interaction.environment.describe()
    policy = CompiledPolicy(
        compile(scenario), scenario.policy.action_adapter.build(contract), contract
    )
    observation = TensorDict(
        {"observation": torch.zeros(2, *contract.observations["observation"].shape)}, batch_size=[2]
    )
    expected = policy(observation)["action"].detach()
    export_package(policy, tmp_path / "policy")
    torch.save(observation.to_dict(), tmp_path / "observation.pt")
    torch.save(expected, tmp_path / "expected.pt")
    torch.save(policy.state_dict(), tmp_path / "expected-state.pt")
    (tmp_path / "parent-context.json").write_text(
        json.dumps(
            {
                "torch": str(torch.__version__),
                "precision": torch.get_float32_matmul_precision(),
                "threads": torch.get_num_threads(),
                "training": policy.training,
                "device": str(next(policy.parameters()).device),
            }
        )
    )
    code = """
import importlib.abc, json, sys, torch
from pathlib import Path
from tensordict import TensorDict
blocked = {"torchrl", "gymnasium", "PIL", "imageio", "imageio_ffmpeg", "moviepy", "omni", "pxr"}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in blocked or root.startswith(("isaaclab", "isaacsim")):
            raise ImportError("optional runtime blocked: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from nexuml.core.export import load_inference_package, load_package, infer
root = Path(sys.argv[1])
observation = TensorDict(torch.load(root / "observation.pt", weights_only=True), batch_size=[2])
expected = torch.load(root / "expected.pt", weights_only=True)
expected_state = torch.load(root / "expected-state.pt", weights_only=True)
parent_context = json.loads((root / "parent-context.json").read_text())
print("PARENT_CONTEXT", json.dumps(parent_context))
print("CHILD_INITIAL_PRECISION", torch.get_float32_matmul_precision())
# Exact parity compares the same numerical execution policy, not two precision modes.
torch.set_float32_matmul_precision(parent_context["precision"])
print("CHILD_CONTEXT", json.dumps({"torch": str(torch.__version__),
    "precision": torch.get_float32_matmul_precision(), "threads": torch.get_num_threads()}))
for loader in (load_package, load_inference_package):
    policy, config, metadata = loader(root / "policy")
    torch.testing.assert_close(policy.state_dict(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(infer(policy, observation)["action"], expected, rtol=0, atol=0)
    assert metadata["training_state_available"] is False
    assert metadata["environment"]["type"] == "IsaacLabEnvironment"
    assert not any(
        key.startswith(("algorithm.", "critic.", "optimizer.")) for key in policy.state_dict()
    )
assert not any(name.split(".")[0] in blocked for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "params",
    [
        {"task": "Isaac-Cartpole-Direct-v0"},
        {"task": "IsaacContrib-Stack-Cube-SO101-v0"},
        {"revision": "v3.0.0-EA"},
        {"physics": "physics=physx"},
        {"presets": ("",)},
        {"contract": {}},
        {"image_shape": [3, 64, 64]},
        {"action_scale": 1},
        {"kwargs": {}},
    ],
)
def test_provider_rejects_unapproved_profiles_and_configuration(params):
    with pytest.raises(ValidationError):
        IsaacLabEnvironment.model_validate(params)


@pytest.fixture
def training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not probe CUDA or allocate a runtime")

    for name in ("is_available", "init", "_lazy_init", "current_device", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(IsaacLabEnvironment, "build", forbidden)
    return isaac_cartpole_ppo().training


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"backend": "process"}, "synchronous direct collector"),
        ({"backend": "ray"}, "synchronous direct collector"),
        ({"sync": False}, "synchronous direct collector"),
        ({"num_collectors": 2}, "num_collectors=1"),
        ({"env_device": "cpu"}, "CUDA device for collector.env_device"),
        ({"env_device": "mps"}, "CUDA device for collector.env_device"),
        ({"env_device": "auto"}, "CUDA device for collector.env_device"),
        ({"env_device": "cuda:-1"}, "CUDA device for collector.env_device"),
        ({"storing_device": "cpu"}, "CUDA device for collector.storing_device"),
        ({"policy_device": "cpu"}, "CUDA device for collector.policy_device"),
        ({"env_device": "cuda:1"}, "same CUDA device index"),
        ({"storing_device": "cuda:1"}, "same CUDA device index"),
        ({"policy_device": "cuda:1"}, "same CUDA device index"),
    ],
)
def test_preflight_rejects_collector_topology_and_devices(training, changes, match):
    training.collector = training.collector.model_copy(update=changes)
    with pytest.raises(ValueError, match=match):
        IsaacLabEnvironment().validate_training(training)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"devices": "auto"}, "training.devices=1"),
        ({"devices": "1"}, "training.devices=1"),
        ({"devices": 2}, "training.devices=1"),
        ({"strategy": "ddp"}, "training.strategy='auto'"),
        (
            {"strategy": StrategySpec(factory="lightning.pytorch.strategies:SingleDeviceStrategy")},
            "training.strategy='auto'",
        ),
        ({"frames_per_batch": 1_025}, "frames_per_batch must be divisible"),
    ],
)
def test_preflight_rejects_learner_topology_and_frame_geometry(training, changes, match):
    training = training.model_copy(update=changes)
    with pytest.raises(ValueError, match=match):
        IsaacLabEnvironment().validate_training(training)


@pytest.mark.parametrize("device", ["cpu", "mps", "auto", "cuda:1"])
def test_preflight_rejects_non_cuda_or_mismatched_resolved_learner(training, device):
    with pytest.raises(ValueError, match="CUDA"):
        IsaacLabEnvironment().validate_training(training, device=device)


@pytest.mark.parametrize(
    "changes",
    [
        {"episodes": 1},
        {"at_start": True},
        {"every_n_frames": 64},
        {"video": RLVideoSpec(enabled=True)},
    ],
)
def test_preflight_rejects_every_automatic_evaluation_setting(training, changes):
    # A copied spec also exercises video=True independently of the episodes guard.
    training.evaluation = training.evaluation.model_copy(update=changes)
    with pytest.raises(ValueError, match="automatic fit evaluation") as error:
        IsaacLabEnvironment().validate_training(training)
    assert "fresh-process" in str(error.value)
    assert "nexuml evaluate <policy-package> --device cuda:0" in str(error.value)
    assert "--override training.evaluation.episodes=3" in str(error.value)


def test_explicit_policy_evaluation_cannot_bypass_isaac_training_preflight(monkeypatch):
    from nexuml.training.reinforcement import NexuRLLightningModule
    from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo

    scenario = isaac_cartpole_ppo()
    scenario.policy_evaluation = RLEvaluationSpec(episodes=1, at_start=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Isaac preflight must reject automatic evaluation before compile/native build")

    monkeypatch.setattr("nexuml.training.reinforcement.compile", forbidden)
    with pytest.raises(ValueError, match="automatic fit evaluation"):
        NexuRLLightningModule(scenario)


@pytest.mark.parametrize(
    "env,storage,policy,learner",
    [
        ("cuda:0", "cuda:0", "auto", None),
        ("cuda", "cuda:0", "auto", "cuda:0"),
        ("cuda:0", "cuda", "cuda", "cuda"),
        ("cuda:2", "cuda:2", "cuda:2", "cuda:2"),
    ],
)
def test_preflight_accepts_same_cuda_device_without_probing(
    training, env, storage, policy, learner
):
    training.collector = CollectorSpec(
        num_envs_per_collector=32,
        env_device=env,
        storing_device=storage,
        policy_device=policy,
    )
    assert IsaacLabEnvironment().validate_training(training, device=learner) is None


def test_base_provider_validation_remains_optional_no_op():
    class PortableEnvironment(EnvironmentDefinition):
        def describe(self):
            return IsaacLabEnvironment().describe()

        def build(self, context):
            raise AssertionError("validation must not build")

    training = ReinforcementLearningSpec(algorithm=PPO())
    assert PortableEnvironment().validate_training(training) is None
    assert PortableEnvironment().validate_training(training, device="cpu") is None


def test_fresh_discovery_yaml_compile_and_inference_forbid_native_imports():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        attempts = []
        class BlockNative(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split('.')[0]
                if root.startswith(('isaaclab', 'isaacsim')) or root in {'omni', 'pxr'}:
                    attempts.append(fullname)
                    raise AssertionError('unexpected native import: ' + fullname)
        sys.meta_path.insert(0, BlockNative())

        import torch
        from tensordict import TensorDict
        from nexuml.core.compiler import compile
        from nexuml.core.config import ResolvedConfig
        from nexuml.core.policy import CompiledPolicy
        from nexuml.core.scenario_registry import get_scenario_registry
        from nexuml_library.environments.isaaclab import IsaacLabEnvironment

        def forbidden_build(*args, **kwargs):
            raise AssertionError('static paths must not build a simulator')
        IsaacLabEnvironment.build = forbidden_build
        registry = get_scenario_registry()
        for name in (
            'isaac-cartpole-ppo', 'isaac-franka-reach-ppo',
            'isaac-so101-joint-reach-ppo', 'isaac-custom-cartpole-ppo',
        ):
            scenario = registry.get(name)()
            restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())
            contract = restored.interaction.environment.describe()
            restored.interaction.environment.validate_training(restored.training, device='cuda')
            policy = CompiledPolicy(
                compile(restored), restored.policy.action_adapter.build(contract), contract
            )
            observations = TensorDict(
                {'observation': torch.zeros(2, *contract.observations['observation'].shape)},
                batch_size=[2],
            )
            actions = policy(observations)['action']
            assert actions.shape == (2, *contract.actions['action'].shape)
            assert torch.isfinite(actions).all() and actions.abs().max() <= 1
        assert not attempts, attempts
        """
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "library/src")]),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
