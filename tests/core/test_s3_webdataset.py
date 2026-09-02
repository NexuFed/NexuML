from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from nexuml.core.types import LoaderSpec, ScenarioSpec
from nexuml.data.loaders.definitions import DaliLoader
from nexuml.data.export.webdataset import WebDatasetBackend
from nexuml.data.exported import ExportedDataset


class FakeS3:
    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})
        self.uploads: list[tuple[str, str, bytes]] = []

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def upload_file(self, source, bucket, key):
        data = Path(source).read_bytes()
        self.objects[key] = data
        self.uploads.append((bucket, key, data))

    def download_file(self, bucket, key, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])

    def put_object(self, *, Bucket, Key, Body):
        self.objects[Key] = bytes(Body)


def _fake_wds2idx(monkeypatch):
    import nexuml.data.export.webdataset as webdataset

    monkeypatch.setattr(webdataset.shutil, "which", lambda name: "/usr/bin/wds2idx")

    def run(args, **kwargs):
        del kwargs
        Path(args[2]).write_text("idx")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(webdataset.subprocess, "run", run)


def test_webdataset_uses_lossless_numpy_and_writes_index(tmp_path, monkeypatch):
    _fake_wds2idx(monkeypatch)
    backend = WebDatasetBackend(samples_per_shard=2)
    backend.initialize(tmp_path, 1, {"features": (3,)})
    backend.start_split("train")
    expected = torch.tensor([1.25, 2.5, 3.75], dtype=torch.float32)
    backend.save_sample(0, {"features": expected})
    backend.end_split("train")
    metadata = backend.finalize()

    tar_path = tmp_path / metadata["shards"][0]
    idx_path = tmp_path / metadata["index_paths"][0]
    assert tar_path.exists()
    assert idx_path.exists()

    with tarfile.open(tar_path, "r") as archive:
        member = archive.extractfile("00000000.features.npy")
        assert member is not None
        restored = np.load(io.BytesIO(member.read()), allow_pickle=False)
    assert np.array_equal(restored, expected.numpy())
    assert metadata["key_specs"]["features"]["encoding"] == "npy"
    assert metadata["sample_index_file"] == "data/webdataset_index.json"


def test_s3_webdataset_uploads_closed_shard_and_index_without_sample_index(tmp_path, monkeypatch):
    _fake_wds2idx(monkeypatch)
    s3 = FakeS3()
    backend = WebDatasetBackend(
        samples_per_shard=1,
        s3_uri="s3://bucket/dataset",
        s3_client=s3,
    )
    backend.initialize(tmp_path, 2, {"features": (1,)})
    backend.start_split("train")
    backend.save_sample(0, {"features": torch.tensor([1.0])})
    backend.save_sample(1, {"features": torch.tensor([2.0])})
    backend.end_split("train")
    metadata = backend.finalize()

    uploaded_keys = {key for _bucket, key, _data in s3.uploads}
    assert "dataset/data/shards/train/shard-000000.tar" in uploaded_keys
    assert "dataset/data/shards/train/shard-000000.idx" in uploaded_keys
    assert "dataset/data/shards/train/shard-000001.tar" in uploaded_keys
    assert "dataset/data/shards/train/shard-000001.idx" in uploaded_keys
    assert "dataset/data/webdataset_index.json" not in uploaded_keys
    assert "sample_index_file" not in metadata
    assert not list(tmp_path.rglob("*.tar"))
    assert not list(tmp_path.rglob("*.idx"))
    assert metadata["storage_uri"] == "s3://bucket/dataset"


def _remote_dataset_objects() -> dict[str, bytes]:
    config = {
        "format_version": 2,
        "backend": "webdataset",
        "writer": "webdataset",
        "num_samples": 2,
        "label_names": [],
        "num_classes": {},
        "modality": "generic",
        "x_keys": ["features"],
        "y_keys": [],
        "label_prefix": "label__",
        "feature_shapes": {"features": [3]},
        "key_specs": {
            "features": {
                "key": "features",
                "role": "x",
                "shape": [3],
                "dtype": "float32",
                "encoding": "npy",
                "storage": {"type": "webdataset", "member_ext": "features.npy"},
            }
        },
        "extra": {
            "metadata_file": "metadata.csv",
            "metadata_format": "csv",
            "shards": [
                "data/shards/train/shard-000000.tar",
                "data/shards/val/shard-000000.tar",
            ],
            "index_paths": [
                "data/shards/train/shard-000000.idx",
                "data/shards/val/shard-000000.idx",
            ],
        },
    }
    metadata = pd.DataFrame(
        {
            "sample_id": ["train-0", "val-0"],
            "split": ["train", "val"],
            "export_index": [0, 1],
        }
    ).to_csv(index=False)
    return {
        "dataset/config.yaml": yaml.safe_dump(config).encode(),
        "dataset/metadata.csv": metadata.encode(),
        "dataset/data/shards/train/shard-000000.tar": b"train tar",
        "dataset/data/shards/train/shard-000000.idx": b"train idx",
        "dataset/data/shards/val/shard-000000.tar": b"val tar",
        "dataset/data/shards/val/shard-000000.idx": b"val idx",
    }


def test_remote_exported_dataset_keeps_s3_paths_and_rejects_python_tensor_reads():
    dataset = ExportedDataset(
        "s3://bucket/dataset",
        s3_client=FakeS3(_remote_dataset_objects()),
    )

    assert str(dataset.root / dataset.extra["shards"][0]) == (
        "s3://bucket/dataset/data/shards/train/shard-000000.tar"
    )
    with pytest.raises(RuntimeError, match="DALI WebDataset loader"):
        dataset[0]


def test_scenario_hydration_uses_exported_feature_shapes_without_sample_reads():
    from nexuml.training.lightning import _hydrate_scenario_from_dataset

    dataset = ExportedDataset(
        "s3://bucket/dataset",
        s3_client=FakeS3(_remote_dataset_objects()),
    )

    hydrated = _hydrate_scenario_from_dataset(
        ScenarioSpec(name="remote"),
        cast(Any, SimpleNamespace(dataset=dataset)),
    )

    assert hydrated.data.input_shapes == {"features": [3]}


def test_remote_split_only_exposes_its_webdataset_shards():
    dataset = ExportedDataset(
        "s3://bucket/dataset",
        s3_client=FakeS3(_remote_dataset_objects()),
    )
    train = dataset.get_split("train")

    assert train.extra["shards"] == ["data/shards/train/shard-000000.tar"]
    assert train.extra["index_paths"] == ["data/shards/train/shard-000000.idx"]


def test_dali_materializes_s3_paths_and_preserves_rank_sharding(monkeypatch):
    dataset = ExportedDataset(
        "s3://bucket/dataset",
        s3_client=FakeS3(_remote_dataset_objects()),
    ).get_split("train")

    import nexuml.data.loaders.dali_backend as dali_backend
    import nexuml.data.loaders.dali_multimodal as dali_multimodal

    monkeypatch.setattr(dali_backend, "_check_dali_available", lambda: None)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    captured = {}

    def build_webdataset_loader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metadata=None)

    monkeypatch.setattr(dali_multimodal, "build_webdataset_loader", build_webdataset_loader)

    module = SimpleNamespace(
        loader_spec=LoaderSpec(backend=DaliLoader(), batch_size=8, num_workers=1)
    )
    dali_backend.DaliLoaderBackend().create_loader(module, dataset, split="train")

    assert Path(captured["shard_paths"][0]).read_bytes() == b"train tar"
    assert Path(captured["index_paths"][0]).read_bytes() == b"train idx"
    assert captured["global_rank"] == 2
    assert captured["world_size"] == 4
