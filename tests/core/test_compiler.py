"""Tests for nexuml.core.compiler."""

from __future__ import annotations

import pytest
from typing import ClassVar
import torch
from tensordict import TensorDict

from nexuml.core.compiler import PipelineCompileContext, compile, compile_pipeline
from nexuml.core.components import LayerDefinition
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import (
    DataSpec,
    LayerSpec,
    PipelineSpec,
    ScenarioSpec,
    TensorFieldContract,
    TrainingSpec,
)
from nexuml.core.torch_adapter import TorchModuleAdapter, nn_module
from nexuml_library.layers.model.linear_encoder import LinearEncoder
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data


def _simple_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="test_simple",
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        component=LinearEncoder(output_dim=4),
                        keys_in=["features"],
                        keys_out=["latent"],
                    ),
                ],
            }
        ),
        training=TrainingSpec(max_epochs=1, batch_size=4),
        data=synthetic_vector_data(feature_shape=(16,), num_samples=32),
    )


def test_compiler_success():
    scenario = _simple_scenario()
    pipeline = compile(scenario)
    assert pipeline.stages
    assert "latent" in pipeline.input_sizes


def test_compiler_forward_on_compiled_pipeline():
    scenario = _simple_scenario()
    pipeline = compile(scenario)
    x = TensorDict({"features": torch.randn(2, 16)}, batch_size=[2])
    x_out, _ = pipeline(x, None)
    assert "latent" in x_out.keys()


def test_compiler_key_contract_failure():
    scenario = ScenarioSpec(
        name="test_missing_input",
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        component=LinearEncoder(output_dim=4),
                        keys_in=["not_features"],
                        keys_out=["latent"],
                    ),
                ],
            }
        ),
        training=TrainingSpec(max_epochs=1, batch_size=4),
        data=DataSpec(input_shapes={"features": [16]}),
    )
    with pytest.raises(Exception):  # KeyError or ValueError during shape propagation
        compile(scenario)


@pytest.mark.parametrize("typed", [False, True])
def test_inferred_dtypes_preserve_existing_floating_dummy_rng_draws(typed):
    data = (
        DataSpec(input_contracts={"features": TensorFieldContract(shape=(4,), dtype="float32")})
        if typed
        else DataSpec(input_shapes={"features": [4]})
    )
    scenario = ScenarioSpec(
        name="dummy-rng",
        data=data,
        training=TrainingSpec(),
        pipeline=PipelineSpec(
            stages={
                "copy": [
                    LayerSpec(
                        component=nn_module(torch.nn.Identity),
                        keys_in=["features"],
                        keys_out=["copy"],
                    ),
                    LayerSpec(
                        component=nn_module(torch.nn.Identity),
                        keys_in=["copy"],
                        keys_out=["result"],
                    ),
                ]
            }
        ),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        for _ in range(1 if typed else 3):
            torch.randn(2, 4)
        expected = torch.random.get_rng_state()
        torch.manual_seed(17)
        compile(scenario)
        assert torch.equal(torch.random.get_rng_state(), expected)


@pytest.mark.parametrize("kind", ["quantized", "float8"])
def test_legacy_nonstandard_intermediate_dtypes_keep_their_shape_probe_fallback(kind):
    if kind == "float8" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch version has no float8 dtype")

    class Cast(torch.nn.Module):
        def forward(self, value):
            if kind == "quantized":
                return torch.quantize_per_tensor(value, 0.25, 0, torch.quint8)
            return value.to(torch.float8_e4m3fn)

    class CastDefinition(LayerDefinition):
        component_name: ClassVar[str] = "CastFixture"

        def build(self, context):
            return TorchModuleAdapter(module=Cast(), **context.runtime_kwargs())

    spec = PipelineSpec(
        stages={
            "model": [
                LayerSpec(component=CastDefinition(), keys_in=["features"], keys_out=["cast"]),
                LayerSpec(
                    component=nn_module(torch.nn.Identity), keys_in=["cast"], keys_out=["output"]
                ),
            ]
        }
    )
    pipeline = compile_pipeline(
        spec,
        context=PipelineCompileContext(input_sizes={"features": (4,)}),
        resolved_config=ResolvedConfig.from_scenario(_simple_scenario()),
    )
    value = torch.ones(2, 4)
    result, _ = pipeline(TensorDict({"features": value}, [2]), None)
    expected = Cast()(value)
    assert result["output"].shape == expected.shape and result["output"].dtype == expected.dtype
    if kind == "quantized":
        torch.testing.assert_close(result["output"].dequantize(), expected.dequantize())
    else:
        torch.testing.assert_close(result["output"].float(), expected.float())
