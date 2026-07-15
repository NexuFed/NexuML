"""Tests for nexuml.core.export and data/export/runner."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.export import (
    export_onnx,
    export_package,
    export_safetensors,
    infer,
    load_package,
)
from nexuml.core.registry import get_registry
from nexuml.core.types import LayerSpec, PipelineSpec, ScenarioSpec, TrainingSpec
from nexuml.data.export.runner import export_data_module
from nexuml.training.lightning import create_data_module_from_spec
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data


def _make_simple_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="test_export",
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        type_key="LinearEncoder",
                        keys_in=["features"],
                        keys_out=["latent"],
                        params={"hidden_dims": [8], "output_dim": 4},
                    ),
                ],
                "decode": [
                    LayerSpec(
                        type_key="LinearEncoder",
                        keys_in=["latent"],
                        keys_out=["reconstructed"],
                        params={"hidden_dims": [8], "output_dim": 16},
                    ),
                ],
            }
        ),
        training=TrainingSpec(max_epochs=1, batch_size=4, loss_keys={"reconstruction_loss": 1.0}),
        data=synthetic_vector_data(feature_shape=(16,), num_samples=32),
    )


@pytest.fixture
def compiled_pipeline(tmp_path):
    scenario = _make_simple_scenario()
    return compile(scenario, get_registry())


def test_export_package_and_reload(compiled_pipeline, tmp_path):
    export_dir = tmp_path / "exported"
    export_package(compiled_pipeline, export_dir)
    assert (export_dir / "pipeline.package").exists()

    loaded, config, metadata = load_package(export_dir, get_registry())
    assert loaded is not None
    assert metadata.get("schema_version") == 2


def test_export_safetensors(compiled_pipeline, tmp_path):
    path = tmp_path / "model.safetensors"
    export_safetensors(compiled_pipeline, path)
    assert path.exists()


@pytest.mark.requires_optional("onnxscript")
def test_export_onnx(compiled_pipeline, tmp_path):
    path = tmp_path / "model.onnx"
    export_onnx(
        compiled_pipeline,
        path,
        input_key="features",
        output_key="reconstructed",
    )
    assert path.exists()


def test_infer(compiled_pipeline):
    x = TensorDict({"features": torch.randn(1, 16)}, batch_size=[1])
    x_out = infer(compiled_pipeline, x)
    assert "reconstructed" in x_out.keys()


def test_export_data_module_numpy(tmp_path):
    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="numpy",
        splits=["train"],
    )
    assert (export_dir / "config.yaml").exists()


def _expected_first_train_sample(data_module):
    data_module.setup()
    return data_module._train_ds[0]


def _assert_exported_labels_match(
    sample: dict[str, torch.Tensor],
    expected_y: TensorDict | None,
    label_prefix: str = "label__",
) -> None:
    """Assert every expected label is present in the exported sample under the prefixed key.

    Derives the exported label key from ``expected_y`` rather than hardcoding a
    specific target name, so the roundtrip assertion stays valid if the synthetic
    dataset's label set changes.
    """
    if expected_y is None:
        return
    for label_key in expected_y.keys():
        exported_key = f"{label_prefix}{label_key}"
        assert exported_key in sample, f"missing exported label key {exported_key!r}"
        expected = expected_y[label_key]
        assert isinstance(expected, torch.Tensor)
        assert torch.equal(sample[exported_key], expected)


def test_export_data_module_numpy_mmap_roundtrip(tmp_path):
    from nexuml.data.export.numpy_mmap import NumpyMmapBackend

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="numpy_mmap",
        splits=["train"],
    )

    sample = NumpyMmapBackend.load_sample(export_dir, 0)
    assert torch.equal(sample["features"], expected_x["features"])
    _assert_exported_labels_match(sample, expected_y)


def test_export_data_module_torch_roundtrip(tmp_path):
    from nexuml.data.export.torch_files import TorchBackend

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="torch",
        splits=["train"],
    )

    sample = TorchBackend.load_sample(export_dir, 0)
    assert torch.equal(sample["features"], expected_x["features"])
    _assert_exported_labels_match(sample, expected_y)


def test_export_data_module_tensordict_memmap_roundtrip(tmp_path):
    from nexuml.data.export.tensordict_memmap import TensorDictMemmapBackend

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="tensordict_memmap",
        splits=["train"],
    )

    sample = TensorDictMemmapBackend.load_sample(export_dir, 0)
    assert torch.equal(sample["features"], expected_x["features"])
    _assert_exported_labels_match(sample, expected_y)


def test_export_data_module_webdataset_roundtrip(tmp_path):
    from nexuml.data.export.webdataset import WebDatasetBackend

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="webdataset",
        splits=["train"],
    )

    sample = WebDatasetBackend.load_sample(export_dir, 0)
    # The label key is stored losslessly (npy encoding); the "features" key goes
    # through lossy audio (wav) encoding since the dataset's default modality is
    # "audio", so only shape/finiteness is asserted for it.
    assert sample["features"].numel() == expected_x["features"].numel()
    assert torch.isfinite(sample["features"]).all()
    _assert_exported_labels_match(sample, expected_y)


def test_export_data_module_tensor_shards_roundtrip(
    tmp_path,
):
    from nexuml.data.export.tensor_shards import (
        TensorShardsBackend,
    )

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="tensor_shards",
        splits=["train"],
        samples_per_shard=8,
    )

    sample = TensorShardsBackend.load_sample(
        export_dir,
        0,
    )

    assert torch.equal(
        sample["features"],
        expected_x["features"],
    )
    _assert_exported_labels_match(
        sample,
        expected_y,
    )


def test_tensor_shards_zero_pad_partial_shard(
    tmp_path,
):
    from nexuml.data.export.tensor_shards import (
        TensorShardsBackend,
    )

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="tensor_shards",
        splits=["train"],
        samples_per_shard=8,
    )

    manifest = TensorShardsBackend.load_manifest(export_dir)
    last_entry = manifest["shards"][-1]
    shard = TensorShardsBackend.load_shard(
        export_dir,
        last_entry,
    )

    valid_count = int(shard["num_samples"])
    capacity = int(shard["capacity"])

    assert capacity == 8
    assert valid_count <= capacity
    assert int(shard["num_padding_samples"]) == (capacity - valid_count)

    assert shard["valid_mask"][:valid_count].all()
    assert not shard["valid_mask"][valid_count:].any()

    assert torch.equal(
        shard["indices"][valid_count:],
        torch.full(
            (capacity - valid_count,),
            -1,
            dtype=torch.long,
        ),
    )

    for tensor in shard["features"].values():
        assert tensor.shape[0] == capacity
        assert torch.count_nonzero(tensor[valid_count:]) == 0


def test_tensor_shards_do_not_mix_splits(
    tmp_path,
):
    from nexuml.data.export.tensor_shards import (
        TensorShardsBackend,
    )

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="tensor_shards",
        splits=["train", "val", "test"],
        samples_per_shard=8,
    )

    manifest = TensorShardsBackend.load_manifest(export_dir)

    seen_indices: set[int] = set()

    for entry in manifest["shards"]:
        split = entry["split"]

        assert f"data/shards/{split}/" in entry["path"]

        shard = TensorShardsBackend.load_shard(
            export_dir,
            entry,
        )

        assert shard["split"] == split

        valid_indices = shard["indices"][shard["valid_mask"]].tolist()

        for index in valid_indices:
            assert index not in seen_indices
            seen_indices.add(index)

    assert len(seen_indices) == manifest["num_samples"]
