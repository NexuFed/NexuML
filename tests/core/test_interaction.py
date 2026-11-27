"""Portable interaction definitions and existing-registry round trips."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError
from torch import nn

from nexuml.core.components import (
    ActionAdapterDefinition,
    EnvironmentBuildContext,
    EnvironmentDefinition,
    RLAlgorithmDefinition,
)
from nexuml.core.config import ResolvedConfig
from nexuml.core.discovery import Scanner, action_adapter, environment, rl_algorithm
from nexuml.core.registry import ComponentRegistry
from nexuml.core.serialization import lower_component, restore_component
from nexuml.core.types import (
    DataSpec,
    InteractionContract,
    InteractionSpec,
    PolicySpec,
    ScenarioSpec,
    TensorFieldContract,
    TrainingSpec,
)
from nexuml_library.data.synthetic import SyntheticDataset


def robot_contract():
    return InteractionContract(
        observations={
            "camera": TensorFieldContract(shape=(3, 16, 16), dtype="uint8", modality="image"),
            "joints": TensorFieldContract(shape=(2,), dtype="float32", modality="proprioception"),
        },
        actions={
            "command": TensorFieldContract(
                shape=(2,),
                dtype="float32",
                modality="joint_command",
                low=[-1, -2],
                high=[1, 2],
                rate_hz=20,
            ),
        },
        action_space="continuous",
        num_envs=8,
        control_period_s=0.05,
    )


@environment("test.interaction.environment", version="2")
class TestEnvironment(EnvironmentDefinition):
    __test__ = False
    seed: int = 7

    def describe(self):
        return robot_contract()

    def build(self, context):
        raise AssertionError("portable resolution must not build a live environment")


@action_adapter("test.interaction.adapter", version="3")
class TestAdapter(ActionAdapterDefinition):
    __test__ = False

    def build(self, contract):
        return nn.Identity()


@rl_algorithm("test.interaction.algorithm", version="4")
class TestAlgorithm(RLAlgorithmDefinition):
    __test__ = False

    def build(self, context):
        return nn.Linear(2, 1)


def test_interaction_round_trip_and_description_never_build_runtime():
    scenario = ScenarioSpec(
        name="hybrid",
        data=DataSpec(source=SyntheticDataset(feature_shape=(2,), num_samples=8)),
        interaction=InteractionSpec(environment=TestEnvironment()),
        policy=PolicySpec(action_adapter=TestAdapter()),
    )
    text = ResolvedConfig.from_scenario(scenario).to_yaml()
    restored = ResolvedConfig.from_yaml(text).to_scenario()
    assert restored.interaction is not None and restored.policy is not None
    assert restored.interaction.environment == TestEnvironment()
    assert isinstance(restored.policy.action_adapter, TestAdapter)
    assert isinstance(restored.data.source, SyntheticDataset)
    assert isinstance(restored.training, TrainingSpec)
    assert restored.interaction.environment.describe() == robot_contract()
    contract = InteractionContract.model_validate_json(robot_contract().model_dump_json())
    assert contract == robot_contract()
    assert contract.observations["camera"].shape == (3, 16, 16)  # not (8, 3, 16, 16)
    assert "test.interaction.environment" in text
    assert "version: '2'" in text
    assert "TestEnvironment" not in text


def test_all_new_roles_use_common_registry_and_stable_identity(monkeypatch):
    registry = ComponentRegistry()
    module = ModuleType("interaction_test_library")
    setattr(module, "Environment", TestEnvironment)
    setattr(module, "Adapter", TestAdapter)
    setattr(module, "Algorithm", TestAlgorithm)
    scanner = Scanner()
    scanner.scan_module(module)
    for item in scanner.items:
        registry.register(item.key, item.obj)
    monkeypatch.setattr("nexuml.core.serialization.get_component_registry", lambda: registry)
    for definition in (TestEnvironment(), TestAdapter(), TestAlgorithm()):
        document = lower_component(definition)
        assert document["type"] == definition.component_name
        assert document["version"] == definition.component_version
        assert restore_component(kind=definition.kind, value=document) == definition
        json.dumps(document, allow_nan=False)
    assert {entry.kind for entry in registry.entries()} >= {
        "environment",
        "action_adapter",
        "rl_algorithm",
    }
    with pytest.raises(ValidationError):
        TestEnvironment.model_validate({"seed": 7, "runtime": object()})
    with pytest.raises(ValidationError):
        setattr(TestEnvironment(), "seed", 3)
    with pytest.raises(TypeError, match="EnvironmentDefinition"):
        environment("wrong.role")(TestAdapter)


@pytest.mark.parametrize(
    "update",
    [
        {"shape": (0,)},
        {"shape": (-1,)},
        {"shape": (True,)},
        {"dtype": ""},
        {"rate_hz": 0},
        {"low": [0]},
        {"low": 2, "high": 1},
        {"high": float("inf")},
    ],
)
def test_tensor_contract_rejects_invalid_metadata(update):
    with pytest.raises(ValidationError):
        TensorFieldContract.model_validate({"shape": (2,), "dtype": "float32", **update})


def test_build_context_and_scalar_discrete_action():
    assert EnvironmentBuildContext(purpose="eval", num_envs=8).num_envs == 8
    with pytest.raises(ValueError):
        EnvironmentBuildContext(purpose="train", num_envs=0)
    scalar = TensorFieldContract(shape=(), dtype="int64", num_values=2)
    assert scalar.model_dump(mode="json")["shape"] == []


def test_core_supervised_imports_block_all_optional_interaction_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torchrl', 'gymnasium', 'mujoco', 'rclpy', 'carla'}:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import nexuml
from nexuml.core.compiler import compile
from nexuml.core.types import ScenarioSpec, TrainingSpec
from nexuml.interaction.runtime import EnvironmentRuntime
scenario = ScenarioSpec(name='base')
assert isinstance(scenario.training, TrainingSpec)
compile(scenario)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(Path(__file__).resolve().parents[2] / "src"),
                    str(Path(__file__).resolve().parents[2] / "library/src"),
                ]
            ),
        },
    )
    assert result.returncode == 0, result.stderr
