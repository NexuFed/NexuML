"""Tests for nexuml.core.diagram."""

from __future__ import annotations

from nexuml.core.compiler import compile
from nexuml.core.diagram import build_pipeline_mermaid_diagram, export_mermaid_diagram
from nexuml.core.registry import get_registry
from nexuml.core.types import LayerSpec, PipelineSpec, ScenarioSpec, TrainingSpec
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data


def _make_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="test_diagram",
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


def test_build_pipeline_mermaid_diagram():
    scenario = _make_scenario()
    pipeline = compile(scenario, get_registry())
    diagram = build_pipeline_mermaid_diagram(pipeline)
    assert "flowchart" in diagram
    assert "encode" in diagram


def test_export_mermaid_diagram(tmp_path):
    scenario = _make_scenario()
    pipeline = compile(scenario, get_registry())
    path = tmp_path / "diagram.md"
    export_mermaid_diagram(pipeline, path)
    assert path.exists()
    assert "flowchart" in path.read_text()
