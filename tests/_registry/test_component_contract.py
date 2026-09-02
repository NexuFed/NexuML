"""Focused contracts for typed component definitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nexuml.core.components import ComponentDefinition
from nexuml.core.config import ResolvedConfig
from nexuml.core.registry import ComponentEntry, get_component_registry
from nexuml.core.types import (
    DataSpec,
    EvalAlgorithmSpec,
    EvaluationSpec,
    LayerSpec,
    LoaderSpec,
    PipelineSpec,
    ScenarioSpec,
)
from nexuml.data.loaders.definitions import TorchLoader
from nexuml_library.data.synthetic import SyntheticDataset
from nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator import AnomalyEvaluator
from nexuml_library.layers.model.linear_encoder import LinearEncoder


_ENTRIES = get_component_registry().entries()


@pytest.mark.parametrize(
    "entry",
    _ENTRIES,
    ids=lambda entry: f"{entry.kind}:{entry.name}@{entry.version}",
)
def test_discovered_definition_contract(entry: ComponentEntry) -> None:
    definition_type = entry.definition_type

    assert issubclass(definition_type, ComponentDefinition)
    assert definition_type.kind == entry.kind
    assert definition_type.component_name == entry.name
    assert definition_type.component_version == entry.version
    assert definition_type.model_json_schema()["type"] == "object"


def test_representative_scenario_round_trip_restores_all_component_roles() -> None:
    scenario = ScenarioSpec(
        name="typed-round-trip",
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        component=LinearEncoder(output_dim=4),
                        keys_in=["features"],
                        keys_out=["latent"],
                    )
                ]
            }
        ),
        data=DataSpec(
            source=SyntheticDataset(feature_shape=(16,), num_samples=8),
            input_shapes={"features": [16]},
            loader=LoaderSpec(backend=TorchLoader(), batch_size=4),
        ),
        evaluation=EvaluationSpec(algorithms=[EvalAlgorithmSpec(algorithm=AnomalyEvaluator())]),
    )

    yaml_text = ResolvedConfig.from_scenario(scenario).to_yaml()
    restored = ResolvedConfig.from_yaml(yaml_text)

    assert isinstance(restored.pipeline.stages["encode"][0].component, LinearEncoder)
    assert isinstance(restored.data.source, SyntheticDataset)
    assert isinstance(restored.data.loader.backend, TorchLoader)
    assert isinstance(restored.evaluation.algorithms[0].algorithm, AnomalyEvaluator)
    assert "nexuml_library." not in yaml_text


def test_definition_values_are_strict_frozen_and_finite() -> None:
    definition = LinearEncoder(output_dim=4)

    with pytest.raises(ValidationError):
        definition.output_dim = 8  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        LinearEncoder.model_validate({"output_dim": 4, "unknown": True})
    with pytest.raises(ValidationError):
        AnomalyEvaluator(max_fpr=float("inf"))

    schema = LinearEncoder.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["output_dim"]["default"] == 8


def test_legacy_layer_selector_syntax_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LayerSpec.model_validate(
            {
                "type_key": "LinearEncoder",
                "keys_in": ["features"],
                "keys_out": ["latent"],
                "params": {"output_dim": 4},
            }
        )
