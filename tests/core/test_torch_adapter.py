"""Tests for portable direct PyTorch modules."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict

from nexuml import NnModuleLayer, nn_module
from nexuml.core.compiler import compile
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.registry import get_component_registry
from nexuml.core.torch_adapter import TorchModuleAdapter
from nexuml.core.types import DataSpec, LayerSpec, PipelineSpec, ScenarioSpec


def _context(
    *,
    keys_in: list[str] | None = None,
    keys_out: list[str] | None = None,
    label_key: str | None = None,
    label_in_x: bool = False,
) -> LayerBuildContext:
    return LayerBuildContext(
        input_sizes={"x": (4,)},
        keys_in=["x"] if keys_in is None else keys_in,
        keys_out=["y"] if keys_out is None else keys_out,
        label_key=label_key,
        label_in_x=label_in_x,
    )


def _scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="direct_module",
        pipeline=PipelineSpec(
            stages={
                "model": [
                    LayerSpec(
                        component=nn_module(torch.nn.Linear, 4, 2, bias=False),
                        keys_in=["x"],
                        keys_out=["y"],
                    )
                ]
            }
        ),
        data=DataSpec(input_shapes={"x": [4]}),
    )


def test_nn_module_captures_factory_and_normalizes_values() -> None:
    definition = nn_module(torch.nn.Linear, 4, 2, bias=False)

    assert definition.factory == "torch.nn.modules.linear:Linear"
    assert definition.args == [4, 2]
    assert definition.kwargs == {"bias": False}
    assert definition.component_name == "NnModule"
    assert NnModuleLayer.model_json_schema()["properties"].keys() >= {"factory", "args", "kwargs"}

    nested = NnModuleLayer(
        factory="torch.nn.modules.linear:Identity",
        args=cast(Any, ({"z": (1, None), "a": [True, 2.5]},)),
    )
    assert nested.args == [{"a": [True, 2.5], "z": [1, None]}]


def test_direct_modules_use_only_the_universal_registry_identity() -> None:
    names = {entry.name for entry in get_component_registry().entries(kind="layer")}

    assert "NnModule" in names
    assert names.isdisjoint({"IdentityLayer", "Dropout", "Flatten"})


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor(1),
        torch.nn.Identity(),
        torch.float32,
        torch.device("cpu"),
        len,
        {1},
        {1: "value"},
        float("nan"),
        float("inf"),
    ],
)
def test_nn_module_rejects_nonportable_constructor_values(value: object) -> None:
    with pytest.raises(ValueError, match="unsupported|finite|string keys"):
        nn_module(cast(Any, torch.nn.Identity), value)


def test_nn_module_rejects_process_local_factories() -> None:
    def local_factory() -> torch.nn.Module:
        return torch.nn.Identity()

    class FactoryOwner:
        def factory(self) -> torch.nn.Module:
            return torch.nn.Identity()

    with pytest.raises(TypeError, match="live torch.nn.Module"):
        nn_module(cast(Any, torch.nn.Identity()))
    for factory in (lambda: torch.nn.Identity(), local_factory, FactoryOwner().factory):
        with pytest.raises(ValueError, match="top-level|bound methods"):
            nn_module(cast(Any, factory))


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (_context(keys_in=[]), "exactly one input key"),
        (_context(keys_out=["y", "z"]), "exactly one input key"),
        (_context(label_key="target"), "cannot consume labels"),
        (_context(label_in_x=True), "cannot consume labels"),
    ],
)
def test_nn_module_enforces_routing_contract(context: LayerBuildContext, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        nn_module(torch.nn.Identity).build(context)


def test_nn_module_validates_factory_and_output_types() -> None:
    with pytest.raises(ValueError, match="Could not import"):
        NnModuleLayer(factory="missing_package:factory").build(_context())
    with pytest.raises(TypeError, match="must return torch.nn.Module"):
        NnModuleLayer(factory="builtins:str").build(_context())

    layer = nn_module(torch.nn.LSTM, 4, 2).build(_context())
    with pytest.raises(TypeError, match="must return one torch.Tensor"):
        layer.forward_tensor(torch.randn(2, 4))


def test_direct_module_runtime_preserves_pytorch_semantics() -> None:
    layer = nn_module(torch.nn.Linear, 4, 2).build(_context())
    assert isinstance(layer, TorchModuleAdapter)
    assert list(layer.state_dict()) == ["module.weight", "module.bias"]

    layer.eval()
    assert not layer.module.training
    layer.train()
    assert layer.module.training
    layer.to(dtype=torch.float64)
    assert all(parameter.dtype == torch.float64 for parameter in layer.parameters())

    x = TensorDict({"x": torch.randn(3, 4, dtype=torch.float64)}, batch_size=[3])
    output, labels = layer(x, None)
    assert labels is None
    assert output["y"].shape == (3, 2)


def test_direct_module_round_trip_compiles_deterministically() -> None:
    scenario = _scenario()
    yaml_text = ResolvedConfig.from_scenario(scenario).to_yaml()
    restored = ResolvedConfig.from_yaml(yaml_text)
    component = restored.pipeline.stages["model"][0].component

    assert isinstance(component, NnModuleLayer)
    assert component == scenario.pipeline.stages["model"][0].component
    assert ResolvedConfig.from_scenario(restored.to_scenario()).to_yaml() == yaml_text

    original_pipeline = compile(scenario)
    restored_pipeline = compile(restored.to_scenario())
    assert original_pipeline.input_sizes["y"] == (2,)
    assert list(original_pipeline.state_dict()) == list(restored_pipeline.state_dict())


def test_nn_module_restores_in_fresh_core_only_process() -> None:
    workspace = Path(__file__).resolve().parents[2]
    site_packages = next(path for path in sys.path if path.endswith("site-packages"))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(workspace / "src"), site_packages])
    script = """
from nexuml.core.serialization import restore_component
from nexuml.core.torch_adapter import NnModuleLayer

value = {
    "type": "NnModule",
    "version": "1",
    "params": {"factory": "torch.nn.modules.dropout:Dropout", "args": [], "kwargs": {"p": 0.5}},
}
assert isinstance(restore_component(kind="layer", value=value), NnModuleLayer)
"""

    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
