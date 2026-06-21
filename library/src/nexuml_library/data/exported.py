"""Dataset source for loading NexuML exported datasets."""

from __future__ import annotations
from nexuml.core.discovery import data_source

import copy
import io
import json
import tarfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from PIL import Image
from tensordict import TensorDict

from nexuml.data.dataset import _KEEP_DATA, NexuDataset


@data_source("ExportedDataset")
class ExportedDataset(NexuDataset):
    """Load a dataset previously written by ``export_data_module``."""

    def __init__(
        self,
        root: str | Path,
        split: str | list[str] | None = None,
        feature_keys: list[str] | None = None,
        label_keys: list[str] | None = None,
        label_prefix: str = "label__",
    ):
        self.root = Path(root)
        config = yaml.safe_load((self.root / "config.yaml").read_text()) or {}
        self.config = cast(dict[str, Any], config)
        self.backend = str(config.get("writer") or config.get("backend", "numpy"))
        self.modality = str(config.get("modality", "generic"))
        self.label_prefix = str(config.get("label_prefix", label_prefix))
        self.supports_dali_loader = self.backend in {
            "numpy",
            "numpy_mmap",
            "webdataset",
            "tensordict_memmap",
            "torch",
        }
        self.feature_shapes = {
            key: tuple(shape) for key, shape in (config.get("feature_shapes", {}) or {}).items()
        }
        self.key_specs = cast(dict[str, dict[str, Any]], config.get("key_specs", {}) or {})
        self.extra = cast(dict[str, Any], config.get("extra", {}) or {})

        metadata = self._load_metadata(config)
        if split is not None and "split" in metadata.columns:
            requested = [split] if isinstance(split, str) else list(split)
            metadata = metadata[metadata["split"].isin(requested)].reset_index(drop=True)

        if "export_index" not in metadata.columns:
            metadata = metadata.reset_index(drop=True)
            metadata["export_index"] = np.arange(len(metadata))

        stored_x, stored_y = self._resolve_stored_keys(config)
        metadata_label_keys = [
            key for key in list(config.get("label_names", []) or []) if key in metadata.columns
        ]

        requested_x = list(
            feature_keys or cast(list[str], config.get("x_keys", []) or []) or stored_x.keys()
        )
        requested_y = list(
            label_keys
            or cast(list[str], config.get("y_keys", []) or [])
            or list(stored_y.keys()) + [key for key in metadata_label_keys if key not in stored_y]
        )

        missing_x = [key for key in requested_x if key not in stored_x]
        if missing_x:
            raise KeyError(f"Requested feature keys are not present in export: {missing_x}")
        missing_y = [
            key for key in requested_y if key not in stored_y and key not in metadata.columns
        ]
        if missing_y:
            raise KeyError(f"Requested label keys are not present in export: {missing_y}")

        self.x_keys = requested_x
        self.file_label_keys = [key for key in requested_y if key in stored_y]
        self.meta_label_keys = [
            key
            for key in requested_y
            if key not in self.file_label_keys and key in metadata.columns
        ]
        self.y_keys = self.file_label_keys + self.meta_label_keys
        self._stored_x_keys = {logical: stored_x[logical] for logical in self.x_keys}
        self._stored_y_keys = {logical: stored_y[logical] for logical in self.file_label_keys}

        self._mmap_arrays: dict[str, np.ndarray] = {}
        self._tensordict_memmap: TensorDict | None = None
        self._webdataset_index: dict[str, dict[str, Any]] | None = None
        self._cached_export_idx: int | None = None
        self._cached_payload: dict[str, torch.Tensor] | None = None

        super().__init__(
            meta=metadata,
            label_names=[],
            do_split=False,
            modality=self.modality,
        )
        self.label_names = list(self.y_keys)
        if len(self.x_keys) == 1:
            self.feature_key = self.x_keys[0]
        config_num_classes = cast(dict[str, int], config.get("num_classes", {}) or {})
        self.num_classes = {
            key: int(config_num_classes[key])
            for key in self.label_names
            if key in config_num_classes
        }

    def _load_metadata(self, config: dict[str, Any]) -> pd.DataFrame:
        extra = cast(dict[str, Any], config.get("extra", {}) or {})
        metadata_file = extra.get("metadata_file")
        metadata_format = extra.get("metadata_format")

        if metadata_file is not None:
            metadata_path = self.root / str(metadata_file)
            if metadata_format == "csv" or metadata_path.suffix == ".csv":
                return pd.read_csv(metadata_path)
            return pd.read_parquet(metadata_path)

        parquet_path = self.root / "metadata.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        return pd.read_csv(self.root / "metadata.csv")

    def _resolve_stored_keys(self, config: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
        stored_x: dict[str, str] = {}
        stored_y: dict[str, str] = {}
        extra = cast(dict[str, Any], config.get("extra", {}) or {})

        stored_keys = list(self.key_specs) or [
            str(key) for key in cast(list[Any], extra.get("stored_keys", []) or [])
        ]
        if not stored_keys:
            if self.backend == "numpy":
                stored_keys = sorted(
                    path.name for path in (self.root / "data").iterdir() if path.is_dir()
                )
            elif self.backend == "numpy_mmap":
                stored_keys = sorted(path.stem for path in (self.root / "data").glob("*.npy"))
            else:
                stored_keys = sorted(self.feature_shapes.keys())

        for stored_key in stored_keys:
            spec = self.key_specs.get(stored_key, {})
            logical_key = str(spec.get("key") or _strip_label_prefix(stored_key, self.label_prefix))
            role = str(
                spec.get("role") or ("y" if stored_key.startswith(self.label_prefix) else "x")
            )
            if role == "y":
                stored_y[logical_key] = stored_key
            else:
                stored_x[logical_key] = stored_key

        return stored_x, stored_y

    def _mmap_array(self, stored_key: str) -> np.ndarray:
        if stored_key not in self._mmap_arrays:
            self._mmap_arrays[stored_key] = np.load(
                self.root / "data" / f"{stored_key}.npy", mmap_mode="r"
            )
        return self._mmap_arrays[stored_key]

    def _torch_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        payload = torch.load(
            self.root / "data" / f"{export_idx:08d}.pt", map_location="cpu", weights_only=False
        )
        return {str(key): _to_tensor(value) for key, value in payload.items()}

    def _numpy_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        return {
            stored_key: torch.from_numpy(
                np.load(
                    self.root / "data" / stored_key / f"{export_idx:08d}.npy", allow_pickle=False
                )
            )
            for stored_key in {*self._stored_x_keys.values(), *self._stored_y_keys.values()}
        }

    def _numpy_mmap_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        return {
            stored_key: _to_tensor(self._mmap_array(stored_key)[export_idx])
            for stored_key in {*self._stored_x_keys.values(), *self._stored_y_keys.values()}
        }

    def _tensordict_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        if self._tensordict_memmap is None:
            self._tensordict_memmap = TensorDict.load_memmap(self.root / "data")
        sample = cast(TensorDict, self._tensordict_memmap[export_idx])
        return {str(key): _to_tensor(value) for key, value in sample.items()}

    def _webdataset_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        if self._webdataset_index is None:
            index_path = self.root / "data" / "webdataset_index.json"
            self._webdataset_index = json.loads(index_path.read_text())

        sample_id = f"{export_idx:08d}"
        if sample_id not in self._webdataset_index:
            raise IndexError(f"Sample index {export_idx} is not present in the WebDataset export")
        sample_entry = self._webdataset_index[sample_id]
        shard_path = self.root / "data" / "shards" / sample_entry["shard"]

        payload: dict[str, torch.Tensor] = {}
        with tarfile.open(shard_path, "r") as handle:
            for stored_key, entry in sample_entry["components"].items():
                member = handle.getmember(entry["member"])
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"Could not read WebDataset member {entry['member']}")
                spec = self.key_specs.get(stored_key, {})
                payload[stored_key] = _decode_webdataset_component(
                    extracted.read(),
                    encoding=str(entry["encoding"]),
                    modality=self.modality
                    if stored_key in self._stored_x_keys.values()
                    else "generic",
                    layout=cast(str | None, spec.get("layout")),
                    shape=cast(list[int] | None, spec.get("shape")),
                    dtype=cast(str | None, spec.get("dtype")),
                )
        return payload

    def _payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        if self._cached_export_idx == export_idx and self._cached_payload is not None:
            return self._cached_payload

        if self.backend == "numpy":
            payload = self._numpy_payload(export_idx)
        elif self.backend == "numpy_mmap":
            payload = self._numpy_mmap_payload(export_idx)
        elif self.backend == "tensordict_memmap":
            payload = self._tensordict_payload(export_idx)
        elif self.backend == "torch":
            payload = self._torch_payload(export_idx)
        elif self.backend == "webdataset":
            payload = self._webdataset_payload(export_idx)
        else:
            raise ValueError(f"Unsupported exported dataset backend: {self.backend}")

        self._cached_export_idx = export_idx
        self._cached_payload = payload
        return payload

    def clone_with_meta(
        self,
        meta: pd.DataFrame,
        data=_KEEP_DATA,
    ) -> "ExportedDataset":
        clone = copy.copy(self)
        clone.meta = meta.reset_index(drop=True)
        if data is not _KEEP_DATA:
            clone.data = cast(Any, data)
        clone.label_names = list(self.label_names)
        clone.num_classes = dict(self.num_classes)
        return clone

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        export_idx = int(row.get("export_index", idx))
        payload = self._payload(export_idx)
        return TensorDict(
            {
                logical_key: payload[stored_key]
                for logical_key, stored_key in self._stored_x_keys.items()
            },
            batch_size=[],
        )

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict | None:
        if not self.y_keys:
            return None

        export_idx = int(row.get("export_index", idx))
        payload = self._payload(export_idx) if self.file_label_keys else {}
        return TensorDict(
            {
                **{
                    logical_key: payload[stored_key]
                    for logical_key, stored_key in self._stored_y_keys.items()
                },
                **{key: _to_tensor(row[key]) for key in self.meta_label_keys},
            },
            batch_size=[],
        )


def _strip_label_prefix(key: str, label_prefix: str) -> str:
    return key[len(label_prefix) :] if key.startswith(label_prefix) else key


def _decode_webdataset_component(
    payload: bytes,
    *,
    encoding: str,
    modality: str,
    layout: str | None,
    shape: list[int] | None,
    dtype: str | None,
) -> torch.Tensor:
    if encoding == "npy":
        return torch.from_numpy(np.load(io.BytesIO(payload), allow_pickle=False).copy())
    if encoding == "pt":
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        return _to_tensor(value)
    if encoding == "txt":
        return torch.tensor(list(payload), dtype=torch.uint8)
    if encoding == "bin":
        if dtype is None or shape is None:
            raise ValueError("Binary WebDataset payloads require dtype and shape metadata")
        array = np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape)
        return torch.from_numpy(array.copy())
    if encoding in {"png", "jpg", "jpeg"}:
        array = np.asarray(Image.open(io.BytesIO(payload)))
        return torch.from_numpy(_layout_from_payload(array, layout, modality).copy())
    if encoding == "wav":
        audio, _sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        return torch.from_numpy(_layout_from_payload(np.asarray(audio), layout, modality).copy())
    if encoding == "mp4":
        raise NotImplementedError(
            "Torch-side decoding of WebDataset MP4 payloads is not implemented."
        )
    raise ValueError(f"Unsupported WebDataset encoding: {encoding}")


def _layout_from_payload(array: np.ndarray, layout: str | None, modality: str) -> np.ndarray:
    if modality == "image" and layout == "CHW":
        return np.moveaxis(array, -1, 0)
    if modality == "video" and layout == "TCHW":
        return np.moveaxis(array, -1, 1)
    if modality == "audio" and layout == "CT":
        return np.moveaxis(array, -1, 0)
    return array


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return torch.as_tensor(value.item())
    return torch.as_tensor(value.item() if hasattr(value, "item") else value)
