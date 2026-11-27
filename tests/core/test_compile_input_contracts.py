"""Typed compile inputs preserve feature axes and numeric dtypes across ordinary graphs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict
from torch import nn

from nexuml import NnModuleLayer, nn_module
from nexuml.core.compiler import (
    compile,
    compile_context_from_data,
    compile_context_from_interaction,
)
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import _infer_io_keys, export_onnx, export_package, load_inference_package
from nexuml.core.serialization import lower_model, restore_model_data
from nexuml.core.types import (
    DataSpec,
    InteractionContract,
    LayerSpec,
    PhysicalTensorContract,
    PipelineSpec,
    ScenarioSpec,
    TensorFieldContract,
    TrainingSpec,
)
from nexuml.training.lightning import _hydrate_scenario_from_dataset
from nexuml.training.reinforcement import _validate_evaluation_contract
from nexuml_library.data.synthetic import SyntheticDataset
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data


def _contracts() -> dict[str, TensorFieldContract]:
    return {
        "tokens": TensorFieldContract(
            shape=(2, 3), dtype="int64", axes=("agent", "time"), modality="language"
        ),
        "mask": TensorFieldContract(shape=(2, 3), dtype="bool", axes=("agent", "time")),
        "image_history": TensorFieldContract(
            shape=(2, 3, 3, 8, 8),
            dtype="uint8",
            axes=("agent", "time", "channel", "height", "width"),
            modality="image",
        ),
        "state": TensorFieldContract(
            shape=(2, 3, 5),
            dtype="float32",
            axes=("agent", "time", "joint"),
            physical=PhysicalTensorContract(
                units=("rad",), frame="joint", sensor_id="fixture:joint-state"
            ),
        ),
    }


def _identity(key: str, output: str) -> LayerSpec:
    return LayerSpec(component=nn_module(nn.Identity), keys_in=[key], keys_out=[output])


def _scenario(data: DataSpec | None = None, layers: list[LayerSpec] | None = None) -> ScenarioSpec:
    return ScenarioSpec(
        name="typed_compile_inputs",
        data=DataSpec(input_contracts=_contracts()) if data is None else data,
        pipeline=PipelineSpec(
            stages={
                "model": layers
                if layers is not None
                else [
                    _identity("state", "copied_state"),
                    _identity("tokens", "copied_tokens"),
                    LayerSpec(
                        component=nn_module(nn.Embedding, 16, 4),
                        keys_in=["copied_tokens"],
                        keys_out=["embedding"],
                    ),
                    _identity("embedding", "copied_embedding"),
                    _identity("mask", "copied_mask"),
                    _identity("image_history", "copied_image_history"),
                ]
            }
        ),
        training=TrainingSpec(max_epochs=1, batch_size=4),
    )


def _interaction(fields: dict[str, TensorFieldContract]) -> InteractionContract:
    return InteractionContract(
        observations=fields,
        actions={"action": TensorFieldContract(shape=(1,), dtype="float32")},
        action_space="continuous",
    )


def _batch(batch_size: int) -> TensorDict:
    return TensorDict(
        {
            "tokens": torch.arange(batch_size * 6, dtype=torch.int64).reshape(batch_size, 2, 3)
            % 16,
            "mask": torch.ones(batch_size, 2, 3, dtype=torch.bool),
            "image_history": torch.full((batch_size, 2, 3, 3, 8, 8), 127, dtype=torch.uint8),
            "state": torch.arange(batch_size * 30, dtype=torch.float32).reshape(
                batch_size, 2, 3, 5
            ),
        },
        batch_size=[batch_size],
    )


@pytest.mark.parametrize(
    "axes",
    [
        ("agent",),
        ("agent", "time", "channel"),
        ("agent", "agent"),
        ("", "time"),
        (" ", "time"),
        ("batch", "time"),
        ("agent", "environment"),
    ],
)
def test_feature_axes_reject_wrong_rank_duplicates_blanks_and_outer_batch_names(axes):
    with pytest.raises(ValidationError):
        TensorFieldContract(shape=(2, 3), dtype="float32", axes=axes)


@pytest.mark.parametrize("key", ["", " ", "\t"])
def test_input_contract_keys_cannot_be_blank(key):
    with pytest.raises(ValidationError, match="keys"):
        DataSpec(input_contracts={key: TensorFieldContract(shape=(3,), dtype="float32")})


@pytest.mark.parametrize("legacy_shape", [[3, 2], [6], [2, 3, 1]])
def test_overlapping_shape_hints_must_match_every_feature_axis(legacy_shape):
    with pytest.raises(ValidationError, match="disagree"):
        DataSpec(
            input_shapes={"tokens": legacy_shape},
            input_contracts={"tokens": _contracts()["tokens"]},
        )


def test_legacy_shape_and_typed_contract_union_has_no_invented_features():
    contracts = _contracts()
    data = DataSpec(
        input_shapes={"tokens": [2, 3], "legacy": [7, 11]},
        input_contracts=contracts,
        num_classes=9,
        skip_pipeline_stages=["prepared"],
    )
    context = compile_context_from_data(data)

    assert context.input_sizes == {"legacy": (7, 11), **{k: v.shape for k, v in contracts.items()}}
    assert context.input_metadata == {key: lower_model(leaf) for key, leaf in contracts.items()}
    assert context.input_metadata["state"]["physical"]["sensor_id"] == "fixture:joint-state"
    assert context.num_classes == 9 and context.skip_stages == ("prepared",)
    assert "features" not in context.input_sizes
    assert "legacy" not in context.input_metadata
    assert _infer_io_keys(ResolvedConfig.from_scenario(_scenario(data)))[0] == list(
        context.input_sizes
    )


@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    ],
)
def test_data_and_interaction_share_numeric_dtypes_in_context_dummy_and_runtime(dtype, monkeypatch):
    leaf = TensorFieldContract(shape=(2, 3), dtype=dtype, axes=("agent", "time"))
    data = DataSpec(input_contracts={"input": leaf})
    data_context = compile_context_from_data(data)
    interaction_context = compile_context_from_interaction(_interaction({"input": leaf}))
    assert data_context.input_sizes == interaction_context.input_sizes == {"input": (2, 3)}
    assert data_context.input_metadata == interaction_context.input_metadata

    seen = []

    def capture(_module, tensor):
        seen.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    monkeypatch.setattr(nn.Identity, "forward", capture)
    pipeline = compile(_scenario(data, [_identity("input", "output")]))
    assert seen == [((2, 2, 3), getattr(torch, dtype))]
    x = TensorDict({"input": torch.zeros(4, 2, 3, dtype=getattr(torch, dtype))}, batch_size=[4])
    output, labels = pipeline(x, None)
    assert labels is None and output.batch_size == torch.Size([4])
    assert output["output"].shape == (4, 2, 3)
    assert output["output"].dtype == getattr(torch, dtype)


@pytest.mark.parametrize("dtype", ["float128", "complex64", "str", "torch.int64"])
def test_unsupported_input_dtype_fails_in_both_contexts_before_layer_build(dtype, monkeypatch):
    leaf = TensorFieldContract(shape=(2, 3), dtype=dtype)
    data = DataSpec(input_contracts={"tokens": leaf})

    def no_build(*_args, **_kwargs):
        pytest.fail("Unsupported input dtype reached layer construction")

    monkeypatch.setattr(NnModuleLayer, "build", no_build)
    for build_context in (
        lambda: compile_context_from_data(data),
        lambda: compile_context_from_interaction(_interaction({"tokens": leaf})),
        lambda: compile(_scenario(data, [_identity("tokens", "output")])),
    ):
        with pytest.raises(ValueError, match="tokens.*unsupported dtype"):
            build_context()


def test_multimodal_graph_preserves_dtype_and_feature_axes_during_compile_and_runtime(monkeypatch):
    builds = {}
    seen = {}
    original_build = NnModuleLayer.build

    def capture_build(definition, context: LayerBuildContext):
        output_key = context.keys_out[0]
        builds[output_key] = copy.deepcopy(dict(context.metadata))
        layer = original_build(definition, context)

        def capture_input(_module, args):
            td = args[0]
            tensor = td[context.keys_in[0]]
            seen[output_key] = (tuple(tensor.shape), tensor.dtype, tuple(td.batch_size))

        layer.register_forward_pre_hook(capture_input)
        return layer

    monkeypatch.setattr(NnModuleLayer, "build", capture_build)
    scenario = _scenario()
    pipeline = compile(scenario)
    expected_inputs = {
        "copied_tokens": ((2, 3), torch.int64),
        "embedding": ((2, 3), torch.int64),
        "copied_embedding": ((2, 3, 4), torch.float32),
        "copied_mask": ((2, 3), torch.bool),
        "copied_image_history": ((2, 3, 3, 8, 8), torch.uint8),
        "copied_state": ((2, 3, 5), torch.float32),
    }
    assert seen == {
        key: ((2, *shape), dtype, (2,)) for key, (shape, dtype) in expected_inputs.items()
    }
    assert pipeline.input_sizes["embedding"] == (2, 3, 4)
    assert pipeline.input_sizes["image_history"] == (2, 3, 3, 8, 8)
    for build_metadata in builds.values():
        for key, leaf in scenario.data.input_contracts.items():
            assert build_metadata["input_metadata"][key] == lower_model(leaf)
    for key, intermediate in (
        ("copied_tokens", "copied_state"),
        ("embedding", "copied_tokens"),
        ("copied_embedding", "embedding"),
    ):
        metadata = builds[key]["input_metadata"][intermediate]
        assert metadata["dtype"] == ("int64" if intermediate == "copied_tokens" else "float32")
        assert tuple(metadata["shape"]) == pipeline.input_sizes[intermediate]
        assert metadata.get("axes") is None and metadata.get("physical") is None

    seen.clear()
    x = _batch(4)
    output, labels = pipeline(x.clone(), None)
    assert labels is None and output.batch_size == torch.Size([4])
    assert seen == {
        key: ((4, *shape), dtype, (4,)) for key, (shape, dtype) in expected_inputs.items()
    }
    for key in ("tokens", "mask", "image_history", "state"):
        torch.testing.assert_close(output[f"copied_{key}"], x[key], rtol=0, atol=0)
    assert output["embedding"].shape == (4, 2, 3, 4)
    assert output["embedding"].dtype == torch.float32
    torch.testing.assert_close(output["copied_embedding"], output["embedding"])


def test_input_contracts_roundtrip_through_json_lowering_and_resolved_yaml():
    data = DataSpec(input_contracts=_contracts())
    assert DataSpec.model_validate_json(data.model_dump_json(exclude_unset=True)) == data
    lowered = json.loads(json.dumps(lower_model(data)))
    assert lowered["input_contracts"]["image_history"]["axes"] == [
        "agent",
        "time",
        "channel",
        "height",
        "width",
    ]
    assert DataSpec.model_validate(restore_model_data(lowered, DataSpec)) == data
    config = ResolvedConfig.from_scenario(_scenario(data))
    restored = ResolvedConfig.from_yaml(config.to_yaml())
    assert restored.data == data
    assert restored.to_scenario().data.input_contracts == data.input_contracts


@pytest.mark.parametrize("source_location", ["source", "datasets"])
def test_legacy_source_shape_fallback_and_explicit_hints_are_unchanged(source_location):
    source = SyntheticDataset(feature_shape=(5, 7), num_samples=4)
    data = DataSpec.model_validate(
        {
            "feature_key": "legacy",
            source_location: source if source_location == "source" else [{"source": source}],
        }
    )
    fallback = compile_context_from_data(data)
    assert fallback.input_sizes == {"legacy": (5, 7)} and fallback.input_metadata == {}
    explicit = compile_context_from_data(data.model_copy(update={"input_shapes": {"x": [2, 9]}}))
    assert explicit.input_sizes == {"x": (2, 9)} and explicit.input_metadata == {}
    default = compile_context_from_data(DataSpec())
    assert default.input_sizes == {"features": (128,)} and default.input_metadata == {}


def test_fixed_legacy_json_without_new_fields_keeps_default_contracts_and_axes():
    data = DataSpec.model_validate_json('{"input_shapes": {"features": [7]}}')
    leaf = TensorFieldContract.model_validate_json(
        '{"shape": [2, 3], "dtype": "float32", "modality": "generic"}'
    )
    assert data.input_contracts == {} and leaf.axes is None
    context = compile_context_from_data(data)
    assert context.input_sizes == {"features": (7,)} and context.input_metadata == {}


def test_legacy_data_object_without_input_contracts_keeps_shape_fallback():
    legacy_data = SimpleNamespace(
        input_shapes={},
        source=SyntheticDataset(feature_shape=(5, 7), num_samples=4),
        datasets=[],
        feature_key="features",
        num_classes=None,
        skip_pipeline_stages=[],
    )
    assert not hasattr(legacy_data, "input_contracts")
    context = compile_context_from_data(cast(Any, legacy_data))
    assert context.input_sizes == {"features": (5, 7)} and context.input_metadata == {}


def test_ordinary_supervised_source_still_compiles_and_runs_without_typed_inputs():
    data = synthetic_vector_data(feature_shape=(7,), num_samples=8)
    pipeline = compile(_scenario(data, [_identity("features", "output")]))
    x = TensorDict({"features": torch.randn(4, 7)}, batch_size=[4])
    output, _ = pipeline(x.clone(), None)
    assert data.input_contracts == {} and pipeline.input_sizes["output"] == (7,)
    assert output.batch_size == torch.Size([4])
    torch.testing.assert_close(output["output"], x["features"])


def test_contracts_only_portable_package_preserves_config_io_keys_shapes_and_dtypes(tmp_path: Path):
    scenario = _scenario()
    pipeline = compile(scenario)
    x = _batch(4)
    expected, _ = pipeline(x.clone(), None)
    export_dir = export_package(pipeline, tmp_path / "typed_inputs")
    loaded, config, metadata = load_inference_package(export_dir)
    actual, _ = loaded(x.clone(), None)

    assert config.data.input_shapes == {}
    assert config.data.input_contracts == scenario.data.input_contracts
    assert config.to_yaml() == pipeline.resolved_config.to_yaml()
    assert metadata["x_keys"] == list(scenario.data.input_contracts)
    assert _infer_io_keys(config)[0] == metadata["x_keys"]
    assert "features" not in metadata["x_keys"]
    assert loaded.input_sizes == pipeline.input_sizes
    assert actual.batch_size == expected.batch_size == torch.Size([4])
    assert set(actual.keys()) == set(expected.keys())
    for key in expected.keys():
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype
        torch.testing.assert_close(actual[key], expected[key])


def test_onnx_export_uses_integer_contract_dummy_without_real_onnx_export(monkeypatch, tmp_path):
    pytest.importorskip("onnxscript")
    data = DataSpec(input_contracts={"tokens": _contracts()["tokens"]})
    pipeline = compile(_scenario(data, [_identity("tokens", "output")]))
    captured = {}

    def capture(wrapper, args, _path, **kwargs):
        captured["dummy"] = args[0]
        captured["output"] = wrapper(*args)
        captured.update(kwargs)

    monkeypatch.setattr(torch.onnx, "export", capture)
    export_onnx(pipeline, tmp_path / "tokens.onnx", input_key="tokens", output_key="output")
    assert captured["dummy"].shape == (1, 2, 3)
    assert captured["dummy"].dtype == captured["output"].dtype == torch.int64
    assert captured["output"].shape == (1, 2, 3)
    assert captured["input_names"] == ["tokens"]
    assert captured["dynamic_axes"] == {"tokens": {0: "batch"}, "output": {0: "batch"}}


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "missing"])
def test_dataset_hydration_checks_actual_first_sample_before_training(mismatch):
    scenario = _scenario()
    sample = _batch(1)[0]
    assert isinstance(sample, TensorDict)
    if mismatch == "shape":
        sample["tokens"] = torch.zeros(3, 2, dtype=torch.int64)
    elif mismatch == "dtype":
        sample["tokens"] = sample["tokens"].float()
    else:
        del sample["tokens"]
    data_module = SimpleNamespace(dataset=[(sample, None)])

    with pytest.raises(ValueError, match=f"Dataset input.*tokens.*{mismatch}|missing.*tokens"):
        _hydrate_scenario_from_dataset(scenario, cast(Any, data_module))


def test_dataset_hydration_preserves_valid_contracts_and_all_feature_dimensions():
    scenario = _scenario()
    data_module = SimpleNamespace(dataset=[(_batch(1)[0], None)])
    hydrated = _hydrate_scenario_from_dataset(scenario, cast(Any, data_module))
    assert hydrated.data.input_contracts == scenario.data.input_contracts
    assert hydrated.data.input_shapes == {
        key: list(leaf.shape) for key, leaf in _contracts().items()
    }
    assert compile_context_from_data(hydrated.data).input_metadata == {
        key: lower_model(leaf) for key, leaf in _contracts().items()
    }


@pytest.mark.parametrize("actual_axes", [("time", "agent"), None])
def test_evaluator_rejects_same_shape_with_changed_or_missing_feature_axes(actual_axes):
    expected = _interaction({"tokens": _contracts()["tokens"]})
    data = expected.model_dump()
    data["observations"]["tokens"]["axes"] = actual_axes
    actual = InteractionContract.model_validate(data)
    with pytest.raises(ValueError, match="Evaluation observations field 'tokens'"):
        _validate_evaluation_contract(expected, actual)
    _validate_evaluation_contract(expected, expected)
    legacy = expected.model_dump()
    legacy["observations"]["tokens"].pop("axes")
    _validate_evaluation_contract(InteractionContract.model_validate(legacy), expected)
