"""LeRobot frame ingestion: real local v3 fixtures, no Hub or simulator access."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from nexuml.core.serialization import lower_component, restore_component
from nexuml.core.types import DataSpec, DatasetSpec, LoaderSpec, TargetSpec
from nexuml.data.creator import NexuDataCreator
from nexuml.data.super_dataset import SuperDataset
from nexuml_library.data.lerobot import LeRobotDataSource


def _source(path, **overrides):
    return LeRobotDataSource.model_validate(
        {
            "root": str(path),
            "repo_id": "local/tiny",
            "split_episodes": {"train": (2,), "val": (0,), "test": (1,)},
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"split_episodes": {}},
        {"split_episodes": {"train": ()}},
        {"split_episodes": {"train": (-1,)}},
        {"split_episodes": {"train": (True,)}},
        {"split_episodes": {"train": (1.0,)}},
        {"split_episodes": {"train": ("1",)}},
        {"split_episodes": {"fit": (0,)}},
        {"split_episodes": {"train": (0, 0)}},
        {"split_episodes": {"train": (0,), "val": (0,)}},
        {"input_keys": ()},
        {"input_keys": ("",)},
        {"input_keys": (" ",)},
        {"input_keys": ("observation.state", "observation.state")},
        {"input_keys": ("action",)},
        {"root": " "},
        {"repo_id": ""},
        {"action_key": ""},
    ],
)
def test_invalid_definition(overrides):
    with pytest.raises(ValidationError):
        _source("unused", **overrides)


def test_definition_roundtrip_and_frozen_fields():
    source = _source("unused")
    serialized = lower_component(source)
    assert serialized["type"] == "LeRobotDataset"
    assert serialized["params"]["split_episodes"]["train"] == [2]
    assert restore_component(kind="data_source", value=serialized) == source
    assert LeRobotDataSource.model_validate_json(source.model_dump_json()) == source
    with pytest.raises(ValidationError, match="frozen"):
        source.root = "changed"


def test_discovery_and_definition_do_not_import_optional_backends():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoRobotics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'lerobot', 'isaaclab', 'isaacsim'}:
            raise AssertionError(f'optional runtime imported: {fullname}')

sys.meta_path.insert(0, NoRobotics())
import nexuml_library.data.lerobot as module
from nexuml.core.discovery import Scanner
scanner = Scanner()
scanner.scan_module(module)
items = scanner.by_kind('data_source')
assert any(item.key == 'LeRobotDataset' for item in items)
source = module.LeRobotDataSource(root='unused', repo_id='local/tiny',
                                 split_episodes={'train': (0,)})
assert source.model_dump()['input_keys'] == ('observation.state',)
assert not any(name.split('.')[0] in {'lerobot', 'isaaclab', 'isaacsim'}
               for name in sys.modules)
""",
        ],
        env={**os.environ, "CPU_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _write_dataset(root: Path, *, video: bool = False, image: bool = False) -> Path:
    dataset_class = pytest.importorskip("lerobot.datasets.lerobot_dataset").LeRobotDataset
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["joint", "gripper"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["joint", "gripper"]},
        "policy_revision": {"dtype": "string", "shape": (1,), "names": None},
    }
    kwargs = {}
    if video:
        pytest.importorskip("av")
        config = pytest.importorskip("lerobot.configs").RGBEncoderConfig
        try:
            kwargs["rgb_encoder"] = config(vcodec="h264", crf=0, preset="ultrafast")
        except ValueError as error:
            pytest.skip(f"public CPU H.264 encoder unavailable: {error}")
    if video or image:
        features["observation.images.wrist"] = {
            "dtype": "video" if video else "image",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        }
    dataset = dataset_class.create(
        repo_id="local/tiny",
        root=root,
        fps=10,
        features=features,
        robot_type="test-only",
        use_videos=video,
        video_backend="pyav",
        encoder_threads=1,
        **kwargs,
    )
    try:
        for episode in range(3):
            for frame in range(3):
                values = {
                    "observation.state": np.array([episode, frame], dtype=np.float32),
                    "action": np.array([episode + 10, frame + 20], dtype=np.float32),
                    "task": f"test-task-{episode}",
                    "policy_revision": f"test-revision-{episode}",
                }
                if video or image:
                    values["observation.images.wrist"] = np.full(
                        (32, 32, 3), episode * 60 + frame * 10, dtype=np.uint8
                    )
                dataset.add_frame(values)
            dataset.save_episode()
    finally:
        dataset.finalize()
    return root


@pytest.fixture
def local_dataset(tmp_path):
    return _write_dataset(tmp_path / "state")


@pytest.fixture
def local_video_dataset(tmp_path):
    return _write_dataset(tmp_path / "video", video=True)


@pytest.fixture
def forbid_network(monkeypatch):
    metadata = pytest.importorskip("lerobot.datasets.dataset_metadata")
    dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    hub = pytest.importorskip("huggingface_hub")
    datasets = pytest.importorskip("datasets")
    requests = pytest.importorskip("requests")

    def forbidden(*args, **kwargs):
        pytest.fail("network, Hub entrypoint or download-capable dataset constructor called")

    for module, names in (
        (metadata, ("snapshot_download", "sync_bucket", "get_safe_version")),
        (hub, ("snapshot_download", "hf_hub_download")),
        (datasets, ("load_dataset",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    for name in ("repo_info", "list_repo_refs"):
        monkeypatch.setattr(hub.HfApi, name, forbidden)
    monkeypatch.setattr(dataset.LeRobotDataset, "__init__", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.mark.requires_optional("lerobot")
@pytest.mark.usefixtures("forbid_network")
class TestLocalLeRobot:
    def test_state_parity_splits_and_nested_views(self, local_dataset):
        dataset = _source(local_dataset).build()
        assert len(dataset) == 9
        assert dataset.data is None
        assert dataset.label_names == []
        assert dataset.num_classes == {}
        assert dataset.info["robot_type"] == "test-only"
        assert "origin" not in dataset.provenance
        assert dataset.provenance["split_episodes"] == {"train": (2,), "val": (0,), "test": (1,)}
        assert dataset.meta["index"].tolist() == list(range(9))
        assert dataset.meta["timestamp"].tolist() == pytest.approx([0, 0.1, 0.2] * 3)
        episode_sets = []
        for split, episode in (("train", 2), ("val", 0), ("test", 1)):
            view = dataset.get_split(split)
            episode_sets.append(set(view.meta["episode_index"]))
            assert len(view) == 3
            assert set(view.meta["task"]) == {f"test-task-{episode}"}
            assert set(view.meta["policy_revision"]) == {f"test-revision-{episode}"}
            for frame in range(3):
                x, y = view[frame]
                assert set(x.keys()) == {"observation.state"}
                assert set(y.keys()) == {"action"}
                torch.testing.assert_close(
                    x["observation.state"], torch.tensor([episode, frame]).float()
                )
                torch.testing.assert_close(
                    y["action"], torch.tensor([episode + 10, frame + 20]).float()
                )
            nested = view.take([2, 0]).take([1])
            torch.testing.assert_close(
                nested[0][0]["observation.state"], view[0][0]["observation.state"]
            )
            assert nested.meta.iloc[0]["source_index"] == episode * 3
        assert all(
            not a & b for index, a in enumerate(episode_sets) for b in episode_sets[index + 1 :]
        )

    def test_filtered_reader_index_is_not_global_index(self, local_dataset):
        dataset = _source(local_dataset, split_episodes={"train": (2,)}).build()
        assert dataset.meta["source_index"].tolist() == [0, 1, 2]
        assert dataset.meta["index"].tolist() == [6, 7, 8]
        torch.testing.assert_close(
            dataset.take([2, 0]).get_split("train")[0][0]["observation.state"],
            torch.tensor([2.0, 2.0]),
        )

    def test_samples_clone_maintained_reader_tensors(self, local_dataset, monkeypatch):
        dataset = _source(local_dataset).build()
        get_item = dataset.reader.get_item
        captured = {}

        def capture(index):
            item = get_item(index)
            captured.update(item)
            return item

        monkeypatch.setattr(dataset.reader, "get_item", capture)
        x, y = dataset[0]
        for key, tensor in (("observation.state", x["observation.state"]), ("action", y["action"])):
            assert tensor.data_ptr() != captured[key].data_ptr()
            torch.testing.assert_close(tensor, captured[key])
            tensor.fill_(-100)
            assert not torch.any(captured[key] == -100)

    def test_declared_provenance_is_retained_not_inferred_as_labels(self, local_dataset):
        path = local_dataset / "meta/info.json"
        info = json.loads(path.read_text())
        info["provenance"] = {"origin": "synthetic-test", "policy_revision": "test-revision"}
        path.write_text(json.dumps(info))
        dataset = _source(local_dataset).build()
        assert dataset.info["provenance"] == info["provenance"]
        assert dataset.meta.iloc[0]["provenance"] == info["provenance"]
        assert dataset.meta.iloc[0]["episode_metadata"]["tasks"] == ["test-task-0"]
        assert dataset.label_names == []

    def test_standard_creator_and_loader(self, local_dataset):
        spec = DataSpec(
            datasets=[
                DatasetSpec(source=_source(local_dataset), split_type="keep", modality="robot")
            ],
            input_shapes={"observation.state": [2]},
            feature_key="observation.state",
            targets=[TargetSpec(type="regression", key="action", num_outputs=2)],
            loader=LoaderSpec(batch_size=2, num_workers=0, shuffle_train=False),
        )
        module = NexuDataCreator().build(spec)
        module.setup()
        x, y = next(iter(module.train_dataloader()))
        torch.testing.assert_close(x["observation.state"], torch.tensor([[2.0, 0.0], [2.0, 1.0]]))
        torch.testing.assert_close(y["action"], torch.tensor([[12.0, 20.0], [12.0, 21.0]]))
        assert module.dataset.meta is not None
        assert set(module.dataset.meta["split"]) == {"train", "val", "test"}

    @pytest.mark.parametrize("fixture_name", ["local_dataset", "local_video_dataset"])
    def test_multisource_keep_preserves_pairs_and_filtered_nested_views(
        self, request, fixture_name, monkeypatch
    ):
        root = request.getfixturevalue(fixture_name)
        source = _source(root, split_episodes={"train": (2,), "val": (1,)})
        spec = DataSpec(
            datasets=[DatasetSpec(source=source, split_type="keep", modality="robot")] * 2,
            loader=LoaderSpec(batch_size=2, num_workers=0, shuffle_train=False),
        )
        module = NexuDataCreator().build(spec)
        module.setup()
        dataset = module.dataset
        assert isinstance(dataset, SuperDataset)
        assert len(dataset) == 12
        assert dataset.label_names == []
        assert dataset.meta is not None
        assert set(dataset.meta["index"]) == set(range(3, 9))
        calls = []
        for name, child in dataset.meta_data_list.items():
            reader = getattr(child, "reader")
            get_item = reader.get_item

            def count_item(index, get_item=get_item, name=name):
                calls.append((name, index))
                return get_item(index)

            monkeypatch.setattr(reader, "get_item", count_item)

        x, y = next(iter(module.train_dataloader()))
        torch.testing.assert_close(x["observation.state"], torch.tensor([[2.0, 0.0], [2.0, 1.0]]))
        torch.testing.assert_close(y["action"], torch.tensor([[12.0, 20.0], [12.0, 21.0]]))
        assert len(calls) == 2  # One maintained reader call per sample, including video decoding.
        calls.clear()
        rows = dataset.get_split("train").take([5, 0, 3]).take([1, 0])
        assert rows.meta is not None
        assert rows.meta["index"].tolist() == [6, 8]
        assert rows.meta["source_index"].tolist() == [3, 5]
        assert rows.meta["source_dataset"].nunique() == 2
        for index, frame in enumerate((0, 2)):
            x, y = rows[index]
            assert y is not None
            torch.testing.assert_close(x["observation.state"], torch.tensor([2.0, float(frame)]))
            torch.testing.assert_close(y["action"], torch.tensor([12.0, float(frame + 20)]))
            sample_index = x["sample_index"]
            assert isinstance(sample_index, torch.Tensor)
            assert sample_index.item() == index
        assert calls == list(
            zip(rows.meta["source_dataset"], rows.meta["source_index"], strict=True)
        )

    @pytest.mark.parametrize("bad_source", [0, 1])
    @pytest.mark.parametrize("split_type", ["fit", "all"])
    def test_multisource_rejects_unsafe_split_before_merge(
        self, local_dataset, bad_source, split_type, monkeypatch
    ):
        datasets = [DatasetSpec(source=_source(local_dataset), split_type="keep") for _ in range(2)]
        datasets[bad_source] = DatasetSpec(source=_source(local_dataset), split_type=split_type)

        def forbidden(*args, **kwargs):
            pytest.fail("unsafe LeRobot split reached multisource merging")

        monkeypatch.setattr("nexuml.data.creator.SuperDataset", forbidden)
        with pytest.raises(ValueError, match="split_type='keep'"):
            NexuDataCreator().build_dataset(DataSpec(datasets=datasets))

    @pytest.mark.parametrize("split_type", ["fit", "all", "train"])
    def test_creator_rejects_overwritten_episode_splits(self, local_dataset, split_type):
        spec = DataSpec(
            datasets=[DatasetSpec(source=_source(local_dataset), split_type=split_type)]
        )
        with pytest.raises(ValueError, match="split_type='keep'"):
            NexuDataCreator().build_dataset(spec)

    def test_creator_source_shortcut_is_unsafe(self, local_dataset):
        with pytest.raises(ValueError, match="split_type='keep'"):
            NexuDataCreator().build_dataset(DataSpec(source=_source(local_dataset)))

    @pytest.mark.parametrize("split_type", ["fit", "all"])
    def test_direct_split_meta_rejects_overwritten_episode_splits(self, local_dataset, split_type):
        dataset = _source(local_dataset).build()
        assert dataset.meta is not None
        dataset.meta["split"] = split_type
        with pytest.raises(ValueError, match="split_type='keep'"):
            dataset.split_meta([0.7, 0.15, 0.15])

    @pytest.mark.parametrize("version", ["v2.1", "v3.1", None])
    def test_unsupported_schema(self, local_dataset, version):
        path = local_dataset / "meta/info.json"
        info = json.loads(path.read_text())
        info["codebase_version"] = version
        path.write_text(json.dumps(info))
        with pytest.raises(ValueError, match="only v3.0"):
            _source(local_dataset).build()

    def test_missing_selected_episode(self, local_dataset):
        with pytest.raises(ValueError, match="episodes.*absent.*99"):
            _source(local_dataset, split_episodes={"train": (99,)}).build()

    @pytest.mark.parametrize(
        "keys", [{"input_keys": ("missing.state",)}, {"action_key": "missing.action"}]
    )
    def test_missing_requested_key(self, local_dataset, keys):
        with pytest.raises(ValueError, match="Requested LeRobot key.*absent"):
            _source(local_dataset, **keys).build()

    def test_missing_shape_information(self, local_dataset):
        path = local_dataset / "meta/info.json"
        info = json.loads(path.read_text())
        del info["features"]["observation.state"]["shape"]
        path.write_text(json.dumps(info))
        with pytest.raises(ValueError, match="shape information"):
            _source(local_dataset).build()

    @pytest.mark.parametrize("dtype", ["string", "language"])
    def test_text_annotations_require_explicit_processing(self, local_dataset, dtype):
        path = local_dataset / "meta/info.json"
        info = json.loads(path.read_text())
        info["features"]["policy_revision"]["dtype"] = dtype
        path.write_text(json.dumps(info))
        with pytest.raises(ValueError, match="must be a tensor feature"):
            _source(local_dataset, input_keys=("policy_revision",)).build()

    @pytest.mark.parametrize(
        "pattern",
        [
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/episodes/*/*.parquet",
            "data/*/*.parquet",
        ],
    )
    def test_missing_local_files_never_download(self, local_dataset, pattern):
        for path in local_dataset.glob(pattern):
            path.unlink()
        with pytest.raises(FileNotFoundError, match="local LeRobot"):
            _source(local_dataset).build()

    def test_missing_root_never_downloads(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="never downloads"):
            _source(tmp_path / "missing").build()

    def test_selected_episode_missing_from_existing_data_shard(self, local_dataset):
        parquet = pytest.importorskip("pyarrow.parquet")
        compute = pytest.importorskip("pyarrow.compute")
        path = next(local_dataset.glob("data/*/*.parquet"))
        table = parquet.read_table(path)
        parquet.write_table(table.filter(compute.not_equal(table["episode_index"], 2)), path)
        with pytest.raises(FileNotFoundError, match="incomplete for episodes"):
            _source(local_dataset).build()

    def test_video_build_does_not_decode_frames(self, local_video_dataset, monkeypatch):
        reader_module = pytest.importorskip("lerobot.datasets.dataset_reader")

        def forbidden(*args, **kwargs):
            pytest.fail("build decoded a video or fetched a frame")

        monkeypatch.setattr(reader_module, "decode_video_frames", forbidden)
        monkeypatch.setattr(reader_module.DatasetReader, "get_item", forbidden)
        dataset = _source(
            local_video_dataset, input_keys=("observation.state", "observation.images.wrist")
        ).build()
        assert len(dataset.meta) == 9
        assert "observation.images.wrist" not in dataset.meta

    def test_actual_video_frame_action_timestamp_parity(self, local_video_dataset):
        dataset = _source(
            local_video_dataset, input_keys=("observation.state", "observation.images.wrist")
        ).build()
        for split, episode in (("train", 2), ("val", 0), ("test", 1)):
            view = dataset.get_split(split).take([2, 0])
            assert set(view.meta["episode_index"]) == {episode}
            for index, frame in enumerate((2, 0)):
                row = view.meta.iloc[index]
                direct = dataset.reader.get_item(int(row["source_index"]))
                x, y = view[index]
                torch.testing.assert_close(
                    x["observation.images.wrist"], direct["observation.images.wrist"]
                )
                torch.testing.assert_close(x["observation.state"], direct["observation.state"])
                torch.testing.assert_close(y["action"], direct["action"])
                assert x["observation.images.wrist"].shape == (3, 32, 32)
                assert x["observation.images.wrist"].mean().item() == pytest.approx(
                    (episode * 60 + frame * 10) / 255, abs=2 / 255
                )
                assert row["timestamp"] == pytest.approx(direct["timestamp"].item())
                assert row["timestamp"] == pytest.approx(frame / 10)

    def test_missing_video_never_downloads(self, local_video_dataset):
        for path in local_video_dataset.glob("videos/*/*/*.mp4"):
            path.unlink()
        with pytest.raises(FileNotFoundError, match="never downloads"):
            _source(local_video_dataset).build()

    @pytest.mark.parametrize("select_image", [False, True])
    def test_remote_image_cells_rejected_before_reader_construction(
        self, tmp_path, select_image, monkeypatch
    ):
        root = _write_dataset(tmp_path / "image", image=True)
        arrow = pytest.importorskip("pyarrow")
        parquet = pytest.importorskip("pyarrow.parquet")
        path = next(root.glob("data/*/*.parquet"))
        table = parquet.read_table(path)
        column = table.schema.get_field_index("observation.images.wrist")
        field = table.schema.field(column)
        remote_cells = arrow.array(
            [{"bytes": None, "path": "https://invalid.example/remote.png"}] * len(table),
            type=field.type,
        )
        parquet.write_table(table.set_column(column, field, remote_cells), path)

        def forbidden(*args, **kwargs):
            pytest.fail("image-backed dataset reached metadata/reader construction")

        metadata = pytest.importorskip("lerobot.datasets.dataset_metadata")
        reader = pytest.importorskip("lerobot.datasets.dataset_reader")
        monkeypatch.setattr(metadata.LeRobotDatasetMetadata, "__init__", forbidden)
        monkeypatch.setattr(reader.DatasetReader, "__init__", forbidden)
        keys = (
            ("observation.state", "observation.images.wrist")
            if select_image
            else ("observation.state",)
        )
        with pytest.raises(ValueError, match="Image-backed.*unsupported for offline ingestion"):
            _source(root, input_keys=keys).build()
