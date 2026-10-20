"""Native catalog facts stay separate from ordinary schemas and runtime instances."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import Field

from nexuml.api.contracts import component_io
from nexuml.core.components import (
    ContextInput,
    DataSourceDefinition,
    EvalAlgorithmDefinition,
    EvalBuildContext,
    EvalContextInputs,
    FeatureKey,
    LabelKey,
)


class Report(EvalAlgorithmDefinition):
    context_inputs = EvalContextInputs(feature_key=ContextInput(required=True))
    prediction: FeatureKey
    mask: FeatureKey | None = None
    labels: list[LabelKey] | None = None
    output_key: str = "not-an-input"

    def build(self, context):
        raise AssertionError("Catalog inspection must not build")


def test_native_annotations_do_not_change_values_or_schema():
    value = Report(prediction="loss", labels=["class"])
    assert value.model_dump() == {
        "prediction": "loss",
        "mask": None,
        "labels": ["class"],
        "output_key": "not-an-input",
    }
    schema = Report.model_json_schema()
    assert schema["properties"]["prediction"]["type"] == "string"
    assert not any(key.startswith("x-nexuml") for key in schema)
    io = component_io(Report)
    assert io["inputs_known"] and io["labels"] is None
    assert [
        (
            binding["source"],
            binding["field"],
            binding["domain"],
            binding["collection"],
            binding["required"],
        )
        for binding in io["inputs"]
    ] == [
        ("context", "feature_key", "x", False, True),
        ("parameter", "prediction", "x", False, True),
        ("parameter", "mask", "x", False, False),
        ("parameter", "labels", "y", True, False),
    ]


def test_constraints_preserve_scalar_nullable_and_collection_roles():
    class Constrained(Report):
        prediction: FeatureKey = Field(min_length=1)
        labels: list[LabelKey] | None = Field(default=None, min_length=1)
        scores: list[FeatureKey] = Field(min_length=1)

    bindings = {item["field"]: item for item in component_io(Constrained)["inputs"]}
    assert bindings["prediction"]["domain"] == "x"
    assert not bindings["prediction"]["collection"]
    assert bindings["labels"]["domain"] == "y"
    assert bindings["labels"]["collection"]
    assert bindings["scores"]["domain"] == "x"
    assert bindings["scores"]["collection"] and bindings["scores"]["required"]


def test_unknown_and_known_empty_are_distinct():
    class Unknown(DataSourceDefinition):
        def build(self):
            raise AssertionError("No dataset scan or download")

    class Empty(Unknown):
        fixed_label_keys = ()

    class UnknownReport(Report):
        context_inputs = None

    assert component_io(Unknown)["labels"] is None
    assert component_io(Empty)["labels"] == []
    assert component_io(UnknownReport)["inputs_known"] is False
    assert len(component_io(UnknownReport)["inputs"]) == 3

    from nexuml_library.data.exported import ExportedDataset

    exported = ExportedDataset(root="unused", label_keys=["configured"])
    assert exported.model_dump()["label_keys"] == ["configured"]
    assert component_io(ExportedDataset)["labels"] is None


@pytest.mark.parametrize("value", [None, "", "override"])
def test_context_resolver_preserves_empty_and_default_semantics(value):
    inputs = EvalContextInputs(feature_key=ContextInput(default="latent"))
    assert inputs.resolve(EvalBuildContext(feature_key=value)).feature_key == (value or "latent")
    assert inputs.resolve(EvalBuildContext(label_key="unused")).label_key is None
    required = EvalContextInputs(feature_key=ContextInput(required=True))
    if value is None:
        with pytest.raises(ValueError, match="explicit feature_key"):
            required.resolve(EvalBuildContext())
    else:
        assert required.resolve(EvalBuildContext(feature_key=value)).feature_key == value
    with pytest.raises(FrozenInstanceError):
        setattr(inputs, "feature_key", None)


def test_catalog_uses_external_native_contracts_without_instances_or_runtime_io(
    tmp_path,
    monkeypatch,
):
    from nexuml.api.worker import catalog
    from nexuml.core.registry import ComponentRegistry

    package = tmp_path / "native_contract_library"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "components.py").write_text("""
from pathlib import Path
from nexuml.core.components import (
    ContextInput, DataSourceDefinition, EvalAlgorithmDefinition,
    EvalContextInputs, FeatureKey, LabelKey,
)
from nexuml.core.discovery import data_source, eval_algorithm

@data_source("external-images")
class Images(DataSourceDefinition):
    root: str
    fixed_label_keys = ("external-label",)
    def model_post_init(self, context):
        raise AssertionError("Discovery instantiated a required-parameter source")
    def build(self):
        list(Path(self.root).iterdir())
        raise AssertionError("Discovery built or scanned a source")

@eval_algorithm("external-report")
class Report(EvalAlgorithmDefinition):
    context_inputs = EvalContextInputs(label_key=ContextInput(default="external-label"))
    score: FeatureKey
    labels: list[LabelKey] | None = None
    def build(self, context):
        raise AssertionError("Discovery built an evaluator")
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = ComponentRegistry()
    registry.scan(["native_contract_library"])
    monkeypatch.setattr("nexuml.core.registry.get_component_registry", lambda: registry)
    # Library/module discovery remains trusted import; dataset I/O must never occur.
    no_scan = Mock(side_effect=AssertionError("Dataset path scanned"))
    monkeypatch.setattr(Path, "iterdir", no_scan)
    result = catalog()
    rows = {entry["name"]: entry for entry in result["components"]}
    assert rows["external-images"]["schema"]["required"] == ["root"]
    assert rows["external-images"]["io"]["labels"] == ["external-label"]
    assert rows["external-report"]["io"] == {
        "labels": None,
        "inputs_known": True,
        "inputs": [
            {
                "source": "context",
                "field": "label_key",
                "domain": "y",
                "collection": False,
                "required": False,
                "default": "external-label",
            },
            {
                "source": "parameter",
                "field": "score",
                "domain": "x",
                "collection": False,
                "required": True,
                "default": None,
            },
            {
                "source": "parameter",
                "field": "labels",
                "domain": "y",
                "collection": True,
                "required": False,
                "default": None,
            },
        ],
    }
    no_scan.assert_not_called()


def test_all_builtin_catalog_schemas_have_no_extensions_or_authored_extras():
    import json
    from nexuml.api.worker import catalog

    rows = catalog()["components"]
    assert rows
    for row in rows:
        assert "x-nexuml-" not in json.dumps(row["schema"]), row["name"]
        assert set(row["io"]) == {"inputs", "inputs_known", "labels"}
