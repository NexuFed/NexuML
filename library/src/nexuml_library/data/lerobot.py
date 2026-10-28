"""Offline, frame-level LeRobot v3 ingestion through the maintained reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, field_validator, model_validator
from tensordict import TensorDict

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


_SPLIT_ERROR = (
    "LeRobot episode splits must be preserved. Use "
    "DataSpec(datasets=[DatasetSpec(source=..., split_type='keep')]), "
    "not DataSpec.source or frame-level fit/all splits."
)


@data_source("LeRobotDataset")
class LeRobotDataSource(DataSourceDefinition):
    """Read explicit local episodes without downloads or NexuML preprocessing.

    Use ``DatasetSpec(source=..., split_type="keep")`` to retain episode splits.
    LeRobot and its dataset dependencies are imported only by ``build()``.
    """

    root: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    split_episodes: dict[
        Literal["train", "val", "test"], tuple[Annotated[int, Field(strict=True, ge=0)], ...]
    ]
    input_keys: tuple[str, ...] = ("observation.state",)
    action_key: str = "action"

    @field_validator("root", "repo_id", "action_key")
    @classmethod
    def _nonempty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("root, repo_id and action_key must be nonempty")
        return value

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        episodes = [episode for split in self.split_episodes.values() for episode in split]
        if not episodes:
            raise ValueError("split_episodes must select at least one episode")
        if len(episodes) != len(set(episodes)):
            raise ValueError("episode IDs must be unique within and across splits")
        if not self.input_keys or any(not key.strip() for key in self.input_keys):
            raise ValueError("input_keys must contain nonempty keys")
        if len(set(self.input_keys)) != len(self.input_keys):
            raise ValueError("input_keys must be unique")
        if self.action_key in self.input_keys:
            raise ValueError("input_keys must not include action_key")
        return self

    def build(self) -> NexuDataset:
        """Build a local reader, rejecting missing files before upstream Hub fallback.

        Returns:
            A frame dataset with explicit episode splits and source metadata.
        """
        return _LeRobotDatasetRuntime(self)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing local LeRobot file: {path}. Restore a complete finalized v3.0 "
            "dataset at root; this data source never downloads files."
        )


def _local_info(root: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    _require_file(root / "meta/info.json")
    with (root / "meta/info.json").open() as file:
        info = json.load(file)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Unsupported LeRobot schema {info.get('codebase_version')!r}; "
            "only v3.0 is supported. Convert explicitly with upstream tools."
        )
    image_keys = [
        key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "image"
    ]
    if image_keys:
        raise ValueError(
            f"Image-backed LeRobot features are unsupported for offline ingestion: {image_keys}. "
            "Use local state/video data; convert image-backed datasets explicitly with "
            "upstream tools before ingestion. Image cells may contain remote references, "
            "even when those features are not selected."
        )
    for key in keys:
        feature = info.get("features", {}).get(key)
        if feature is None:
            raise ValueError(f"Requested LeRobot key {key!r} is absent from meta/info.json")
        shape = feature.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
        ):
            raise ValueError(f"Requested LeRobot key {key!r} requires positive shape information")
        if feature.get("dtype") in (None, "string", "language"):
            raise ValueError(f"Requested LeRobot key {key!r} must be a tensor feature")

    # Metadata's constructor downloads on missing tasks/episodes; check before calling it.
    _require_file(root / "meta/stats.json")
    _require_file(root / "meta/tasks.parquet")
    if not any(path.is_file() for path in (root / "meta/episodes").glob("*/*.parquet")):
        raise FileNotFoundError(
            f"Missing local LeRobot episode metadata under {root / 'meta/episodes'}. "
            "Finalize or restore the dataset locally; downloads are disabled."
        )
    return info


class _LeRobotDatasetRuntime(NexuDataset):
    def __init__(self, definition: LeRobotDataSource):
        root = Path(definition.root).expanduser().resolve()
        keys = (*definition.input_keys, definition.action_key)
        self.info = _local_info(root, keys)

        from datasets import Value
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.dataset_reader import DatasetReader

        metadata = LeRobotDatasetMetadata(repo_id=definition.repo_id, root=root, token=False)
        episode_records = metadata.episodes
        tasks = metadata.tasks
        if episode_records is None or tasks is None:
            raise ValueError("LeRobot metadata must contain recorded episodes and tasks")
        self._episode_splits = {
            episode: split
            for split, episodes in definition.split_episodes.items()
            for episode in episodes
        }
        episodes = sorted(self._episode_splits)
        available = metadata.filter_episodes(lambda episode: True, candidates=episodes)
        if missing := sorted(set(episodes) - set(available)):
            raise ValueError(f"Selected LeRobot episodes are absent from local metadata: {missing}")
        self.episode_metadata = {}
        for episode in episodes:
            if episode >= len(episode_records):
                raise ValueError(f"LeRobot episode metadata is incomplete for episode {episode}")
            episode_meta = episode_records[episode]
            if episode_meta["episode_index"] != episode:
                raise ValueError(f"LeRobot episode metadata is misindexed for episode {episode}")
            self.episode_metadata[episode] = episode_meta
            _require_file(root / metadata.get_data_file_path(episode))
            for key in metadata.video_keys:
                _require_file(root / metadata.get_video_file_path(episode, key))

        self.reader = DatasetReader(
            meta=metadata,
            root=root,
            episodes=episodes,
            tolerance_s=1e-4,
            video_backend="pyav",
            delta_timestamps=None,
            image_transforms=None,
        )
        if not self.reader.try_load():
            raise FileNotFoundError(
                f"Local LeRobot data/video under {root} is incomplete for episodes {episodes}. "
                "Restore the selected episode files locally; downloads are disabled."
            )

        frames = self.reader.hf_dataset
        assert frames is not None  # try_load() succeeded.
        required_columns = {"episode_index", "frame_index", "timestamp", "index", "task_index"}
        required_columns.update(key for key in keys if key not in metadata.video_keys)
        if missing := sorted(required_columns - set(frames.column_names)):
            raise ValueError(f"Required LeRobot frame columns are missing: {missing}")
        scalar_columns = [
            key for key, feature in frames.features.items() if isinstance(feature, Value)
        ]
        # Read only scalar columns, without transforms or decoding visual features.
        meta = frames.select_columns(scalar_columns).with_format(None).to_pandas()
        meta["source_index"] = range(len(meta))
        meta["source_dataset"] = definition.repo_id
        meta["split"] = meta["episode_index"].map(self._episode_splits)
        meta["task"] = meta["task_index"].map(
            {int(row.task_index): task for task, row in tasks.iterrows()}
        )
        meta["episode_metadata"] = meta["episode_index"].map(self.episode_metadata)
        if "provenance" in self.info:
            meta["provenance"] = [self.info["provenance"] for _ in range(len(meta))]
        self.provenance = {
            "repo_id": definition.repo_id,
            "root": str(root),
            "codebase_version": self.info["codebase_version"],
            "split_episodes": definition.model_dump()["split_episodes"],
            "input_keys": definition.input_keys,
            "action_key": definition.action_key,
        }
        self.input_keys = definition.input_keys
        self.action_key = definition.action_key
        # Keep the reader separate from self.data: base take() must subset only metadata.
        super().__init__(meta=meta, label_names=[], modality="robot")

    def validate_split_type(self, split_type: str) -> None:
        if split_type != "keep":
            raise ValueError(_SPLIT_ERROR)

    def split_meta(self, split_ratio: list[float]) -> None:
        if self.meta is None or not self.meta["split"].equals(
            self.meta["episode_index"].map(self._episode_splits)
        ):
            raise ValueError(_SPLIT_ERROR)

    def __getitem__(self, index: int) -> tuple[TensorDict, TensorDict]:
        if self.meta is None:
            raise ValueError("LeRobot frame metadata is unavailable")
        source_index = int(self.meta.iloc[index]["source_index"])
        try:
            item = self.reader.get_item(source_index)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Missing local LeRobot frame/video at reader index {source_index}. "
                "Restore the dataset locally; downloads are disabled."
            ) from error
        x = TensorDict({key: item[key].clone() for key in self.input_keys}, batch_size=[])
        y = TensorDict({self.action_key: item[self.action_key].clone()}, batch_size=[])
        return x, y


__all__ = ["LeRobotDataSource"]
