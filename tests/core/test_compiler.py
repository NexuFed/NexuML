"""Tests for nexuml.core.compiler."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.registry import get_registry
from nexuml.core.types import DataSpec, LayerSpec, PipelineSpec, ScenarioSpec, TrainingSpec
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data


def _simple_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="test_simple",
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        type_key="LinearEncoder",
                        keys_in=["features"],
                        keys_out=["latent"],
                        params={"output_dim": 4},
                    ),
                ],
            }
        ),
        training=TrainingSpec(max_epochs=1, batch_size=4),
        data=synthetic_vector_data(feature_shape=(16,), num_samples=32),
    )


def test_compiler_success():
    scenario = _simple_scenario()
    pipeline = compile(scenario, get_registry())
    assert pipeline.stages
    assert "latent" in pipeline.input_sizes


def test_compiler_forward_on_compiled_pipeline():
    scenario = _simple_scenario()
    pipeline = compile(scenario, get_registry())
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
                        type_key="LinearEncoder",
                        keys_in=["not_features"],
                        keys_out=["latent"],
                        params={"output_dim": 4},
                    ),
                ],
            }
        ),
        training=TrainingSpec(max_epochs=1, batch_size=4),
        data=DataSpec(params={"feature_shape": [16]}),
    )
    with pytest.raises(Exception):  # KeyError or ValueError during shape propagation
        compile(scenario, get_registry())
