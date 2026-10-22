"""Native evaluator declarations agree with validation and runtime construction."""

from __future__ import annotations

import json
from typing import get_type_hints

import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.core.components import (
    ContextInput,
    EvalAlgorithmDefinition,
    EvalBuildContext,
    EvalContextInputs,
    FeatureKey,
    LabelKey,
)
from nexuml.core.serialization import (
    lower_component,
    lower_model,
    restore_component,
    restore_model_data,
)
from nexuml.core.types import AxisKeySpec, EvalAlgorithmSpec
from nexuml.evaluation.algorithm import ContractError, EvalAlgorithm, get_declared_axis
from nexuml.evaluation.registry import create_algorithm
from nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator import (
    AnomalyEvaluator,
    AnomalyVisualizer,
)
from nexuml_library.evaluation.visualizers.class_histogram import ClassHistogramVisualizer
from nexuml_library.evaluation.visualizers.latent import LatentVisualizer
from nexuml_library.evaluation.visualizers.reconstruction import ReconstructionVisualizer


_DEFINITIONS = [
    ReconstructionVisualizer(reconstructed_key="reconstructed"),
    LatentVisualizer(),
    ClassHistogramVisualizer(),
    AnomalyEvaluator(),
    AnomalyVisualizer(),
]
_INPUTS = [
    {"feature_key": None},
    {"feature_key": "latent", "label_key": "y_true"},
    {"label_key": "y_true"},
    {"label_key": "y_true"},
    {"feature_key": "latent", "label_key": "y_true"},
]


@pytest.fixture(params=[False, True], ids=["direct", "factory"])
def build(request):
    def materialize(definition, **context):
        if request.param:
            spec = EvalAlgorithmSpec(algorithm=definition, **context)
            before = spec.model_dump()
            runtime = create_algorithm(spec)
            assert spec.model_dump() == before
            return runtime
        return definition.build(EvalBuildContext(**context))

    return materialize


@pytest.mark.parametrize(
    ("definition", "inputs"),
    list(zip(_DEFINITIONS, _INPUTS, strict=True)),
    ids=[definition.component_name for definition in _DEFINITIONS],
)
@pytest.mark.parametrize(
    "context",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"feature_key": None, "label_key": None}, id="null"),
        pytest.param({"feature_key": "", "label_key": ""}, id="empty"),
        pytest.param({"feature_key": "embedding", "label_key": "target"}, id="override"),
        pytest.param({"feature_key": "embedding"}, id="feature-only"),
        pytest.param({"label_key": "target"}, id="label-only"),
        pytest.param({"feature_key": "", "label_key": "target"}, id="mixed"),
    ],
)
def test_context_declaration_matches_direct_and_factory_builds(build, definition, inputs, context):
    assert definition.context_inputs == EvalContextInputs(
        **{
            name: ContextInput(required=default is None, default=default)
            for name, default in inputs.items()
        }
    )
    if isinstance(definition, ReconstructionVisualizer) and context.get("feature_key") is None:
        with pytest.raises(ValueError, match="feature_key"):
            build(definition, **context)
        return

    runtime = build(definition, **context)

    for name, default in inputs.items():
        value = context.get(name)
        expected = (value or default) if default is not None else value
        assert getattr(runtime, name) == expected
    for name in {"feature_key", "label_key"} - inputs.keys():
        assert not hasattr(runtime, name)


def test_reconstruction_consumes_scalar_optional_and_collection_keys(build):
    runtime = build(
        ReconstructionVisualizer(
            reconstructed_key="prediction", mask_key="visible", label_keys=["class", "domain"]
        ),
        feature_key="original",
        label_key="unused",
    )
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x = TensorDict(
        {"original": original, "prediction": original + 1, "visible": mask}, batch_size=[2]
    )
    y = TensorDict({"class": torch.tensor([0, 1]), "domain": torch.tensor([2, 3])}, batch_size=[2])

    runtime.eval_batch(x, y)

    assert runtime.results() == {"reconstruction_mse_sample": 1.0}
    stored = runtime._storage.get()
    assert torch.equal(stored["mask"], mask)
    assert torch.equal(stored["label__class"], y["class"].float())
    assert torch.equal(stored["label__domain"], y["domain"].float())


def test_latent_consumes_context_feature_and_label_keys(build):
    runtime = build(LatentVisualizer(), feature_key="embedding", label_key="target")
    x = TensorDict({"embedding": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, batch_size=[2])
    y = TensorDict({"target": torch.tensor([0, 1])}, batch_size=[2])

    runtime.fit_batch(x, y)
    runtime.eval_batch(x, y)

    for storage in (runtime._train_storage, runtime._test_storage):
        stored = storage.get()
        assert torch.equal(stored["feature"], x["embedding"])
        assert torch.equal(stored["label"], y["target"].float())


def test_histogram_consumes_context_label_key(build):
    runtime = build(ClassHistogramVisualizer(), label_key="target")
    x = TensorDict({}, batch_size=[2])
    y = TensorDict({"target": torch.tensor([0, 1])}, batch_size=[2])

    runtime.fit_batch(x, y)
    runtime.eval_batch(x, y)

    assert runtime.results() == {
        "n_classes": 2.0,
        "train_samples": 2.0,
        "test_samples": 2.0,
        "imbalance_ratio": 1.0,
    }


def test_anomaly_evaluator_consumes_score_decision_and_mixed_domain_groups(build):
    runtime = build(
        AnomalyEvaluator(score_key="score", decision_key="decision", group_keys=["group", "other"]),
        label_key="target",
    )
    x = TensorDict(
        {
            "score": torch.tensor([0.1, 0.9]),
            "decision": torch.tensor([0, 1]),
            "group": torch.tensor([9, 9]),
            "other": torch.tensor([2, 3]),
        },
        batch_size=[2],
    )
    y = TensorDict({"target": torch.tensor([0, 1]), "group": torch.tensor([4, 5])}, batch_size=[2])

    runtime.eval_batch(x, y)
    runtime.eval_end()

    assert runtime.results()["score_mean"] == pytest.approx(0.5)
    assert runtime.results()["f1_decision"] == 1.0
    assert torch.equal(runtime._labels[0], y["target"].float())
    assert runtime._group_vals == {"group": [4, 5], "other": [2, 3]}


def test_anomaly_visualizer_consumes_score_and_context_keys(build):
    runtime = build(
        AnomalyVisualizer(score_key="score"), feature_key="embedding", label_key="target"
    )
    x = TensorDict(
        {"score": torch.tensor([0.1, 0.9]), "embedding": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        batch_size=[2],
    )
    y = TensorDict({"target": torch.tensor([0, 1])}, batch_size=[2])

    runtime.eval_batch(x, y)

    assert torch.equal(runtime._scores[0], x["score"])
    assert torch.equal(runtime._features[0], x["embedding"])
    assert torch.equal(runtime._labels[0], y["target"].float())


def test_key_annotations_remain_native_in_scalar_optional_and_list_fields():
    reconstruction = get_type_hints(ReconstructionVisualizer, include_extras=True)
    assert reconstruction["reconstructed_key"] == FeatureKey
    assert reconstruction["mask_key"] == FeatureKey | None
    assert reconstruction["label_keys"] == list[LabelKey] | None
    evaluator = get_type_hints(AnomalyEvaluator, include_extras=True)
    assert evaluator["score_key"] == FeatureKey
    assert evaluator["decision_key"] == FeatureKey | None
    assert evaluator["group_keys"] == list[str] | None
    visualizer = get_type_hints(AnomalyVisualizer, include_extras=True)
    assert visualizer["score_key"] == FeatureKey
    assert visualizer["group_keys"] == list[str] | None


@pytest.mark.parametrize("definition", _DEFINITIONS, ids=lambda value: value.component_name)
def test_native_definitions_preserve_plain_schema_serialization_and_validation(definition):
    definition_type = type(definition)
    serialized = lower_component(definition)
    assert serialized["params"] == json.loads(definition.model_dump_json())
    assert restore_component(kind="eval_algorithm", value=serialized) == definition
    assert "context_inputs" not in serialized["params"]
    schema = definition_type.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "x-nexuml-" not in json.dumps(schema)
    assert definition_type.model_config.get("json_schema_extra") is None
    with pytest.raises(ValidationError) as unknown:
        definition_type.model_validate({**serialized["params"], "unknown": True})
    assert unknown.value.errors()[0]["type"] == "extra_forbidden"
    field = next(iter(definition_type.model_fields))
    with pytest.raises(ValidationError) as frozen:
        setattr(definition, field, getattr(definition, field))
    assert frozen.value.errors()[0]["type"] == "frozen_instance"


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (
            {"reconstructed_key": ""},
            {"reconstructed_key": "", "mask_key": None, "label_keys": None},
        ),
        (
            {"reconstructed_key": "prediction", "mask_key": "", "label_keys": []},
            {"reconstructed_key": "prediction", "mask_key": "", "label_keys": []},
        ),
        (
            {"reconstructed_key": "prediction", "mask_key": None, "label_keys": ["", "class"]},
            {"reconstructed_key": "prediction", "mask_key": None, "label_keys": ["", "class"]},
        ),
    ],
)
def test_key_values_keep_plain_strings_nulls_and_lists(params, expected):
    definition = ReconstructionVisualizer.model_validate(params)
    assert definition.model_dump(mode="json", include=set(expected)) == expected
    schema = definition.model_json_schema()["properties"]
    assert schema["reconstructed_key"]["type"] == "string"
    assert schema["mask_key"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert schema["label_keys"]["anyOf"] == [
        {"items": {"type": "string"}, "type": "array"},
        {"type": "null"},
    ]


@pytest.mark.parametrize("decision_key", [None, "", "decision"])
def test_anomaly_key_values_keep_plain_empty_strings_and_nulls(decision_key):
    definition = AnomalyEvaluator(score_key="", decision_key=decision_key)
    serialized = lower_component(definition)
    assert serialized["params"]["score_key"] == ""
    assert serialized["params"]["decision_key"] == decision_key
    assert restore_component(kind="eval_algorithm", value=serialized) == definition
    assert AnomalyVisualizer(score_key="").model_dump()["score_key"] == ""


@pytest.mark.parametrize(
    ("definition_type", "params", "location"),
    [
        (ReconstructionVisualizer, {}, ("reconstructed_key",)),
        (ReconstructionVisualizer, {"reconstructed_key": None}, ("reconstructed_key",)),
        (ReconstructionVisualizer, {"reconstructed_key": 1}, ("reconstructed_key",)),
        (ReconstructionVisualizer, {"reconstructed_key": "r", "mask_key": 1}, ("mask_key",)),
        (
            ReconstructionVisualizer,
            {"reconstructed_key": "r", "label_keys": "class"},
            ("label_keys",),
        ),
        (
            ReconstructionVisualizer,
            {"reconstructed_key": "r", "label_keys": [1]},
            ("label_keys", 0),
        ),
        (AnomalyEvaluator, {"score_key": None}, ("score_key",)),
        (AnomalyEvaluator, {"decision_key": 1}, ("decision_key",)),
        (AnomalyVisualizer, {"score_key": 1}, ("score_key",)),
    ],
)
def test_native_key_validation_reports_the_field(definition_type, params, location):
    with pytest.raises(ValidationError) as error:
        definition_type.model_validate(params)
    assert error.value.errors()[0]["loc"] == location


@pytest.mark.parametrize("method", ["tsne", "umap"])
def test_native_method_choices_round_trip_and_build(build, method):
    definition = LatentVisualizer(method=method)
    serialized = lower_component(definition)
    assert serialized["params"]["method"] == method
    assert restore_component(kind="eval_algorithm", value=serialized) == definition
    assert build(definition).method == method
    schema = definition.model_json_schema()["properties"]["method"]
    assert schema["enum"] == ["tsne", "umap"]
    assert schema["type"] == "string"
    assert schema["default"] == "tsne"


@pytest.mark.parametrize("method", ["pca", "TSNE", "", None, 1])
@pytest.mark.parametrize("via_config", [False, True], ids=["direct", "config"])
def test_invalid_method_fails_before_runtime_construction(monkeypatch, method, via_config):
    def unexpected_runtime(*args, **kwargs):
        pytest.fail("invalid method reached runtime construction")

    monkeypatch.setattr(
        "nexuml_library.evaluation.visualizers.latent._LatentVisualizerRuntime", unexpected_runtime
    )
    with pytest.raises(ValidationError) as error:
        definition = (
            restore_component(
                kind="eval_algorithm",
                value={"type": "latent_visualizer", "version": "1", "params": {"method": method}},
            )
            if via_config
            else LatentVisualizer.model_validate({"method": method})
        )
        create_algorithm(EvalAlgorithmSpec(algorithm=definition))
    assert error.value.errors()[0]["loc"] == ("method",)
    assert error.value.errors()[0]["type"] == "literal_error"


def test_unknown_external_context_is_not_resolved_or_guessed(build):
    class Runtime(EvalAlgorithm):
        def __init__(self, context):
            self.context = context

        def results(self):
            return {}

    class External(EvalAlgorithmDefinition):
        something_key: str = "ordinary"

        def build(self, context):
            return Runtime(context)

    class NoContext(External):
        context_inputs = EvalContextInputs()

        def build(self, context):
            return Runtime(self.context_inputs.resolve(context))

    assert External.context_inputs is None
    assert NoContext.context_inputs == EvalContextInputs()
    definition = External()
    assert definition.model_dump() == {"something_key": "ordinary"}
    assert build(definition, feature_key="", label_key="external").context == EvalBuildContext(
        feature_key="", label_key="external"
    )
    assert (
        build(NoContext(), feature_key="unused", label_key="unused").context == EvalBuildContext()
    )


def test_materialization_preserves_surrounding_axis_provenance_and_config():
    spec = EvalAlgorithmSpec(
        algorithm=ReconstructionVisualizer(
            reconstructed_key="prediction", mask_key="mask", label_keys=["target"]
        ),
        feature_key="original",
        label_key="target",
        axis_keys=[
            "target",
            AxisKeySpec(key="prediction", source="x"),
            AxisKeySpec(key="target", source="y"),
            AxisKeySpec(key="basename", source="metadata"),
        ],
    )
    serialized = lower_model(spec)
    restored = EvalAlgorithmSpec.model_validate(restore_model_data(serialized, EvalAlgorithmSpec))
    summary = restored.contract_summary()

    runtime = create_algorithm(restored)

    assert runtime.feature_key == "original"
    assert runtime.reconstructed_key == "prediction"
    assert lower_model(restored) == serialized
    assert restored.contract_summary() == summary
    assert summary["axis_keys"] == [
        "target",
        {"key": "prediction", "source": "x"},
        {"key": "target", "source": "y"},
        {"key": "basename", "source": "metadata"},
    ]


@pytest.mark.parametrize("source", ["x", "y", "metadata"])
def test_axis_access_keeps_exact_provenance_without_fallback(source):
    x = TensorDict({"group": torch.tensor([1, 2])}, batch_size=[2])
    y = TensorDict({"group": torch.tensor([3, 4])}, batch_size=[2])
    metadata = pd.DataFrame({"group": [5, 6]})
    axis = AxisKeySpec(key="group", source=source)
    selected = {"x": x["group"], "y": y["group"], "metadata": metadata["group"]}

    assert get_declared_axis(x, y, axis, metadata=metadata).tolist() == selected[source].tolist()
    assert get_declared_axis(x, y, "group") is y["group"]
    with pytest.raises(ContractError, match=f"source={source}"):
        get_declared_axis(
            TensorDict({}, batch_size=[2]) if source == "x" else x,
            None if source == "y" else y,
            axis,
            metadata=None if source == "metadata" else metadata,
        )
