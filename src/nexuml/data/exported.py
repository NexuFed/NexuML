"""Dataset source for loading NexuML exported datasets."""

from __future__ import annotations

import copy
import io
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tensordict import TensorDict

from nexuml.data.dataset import _KEEP_DATA, NexuDataset
from nexuml.data.export import get_export_backend
from nexuml.storage.s3 import S3Client, S3Path, is_s3_uri


class ExportedDataset(NexuDataset):
    """Load a dataset previously written by ``export_data_module``."""

    def __init__(
        self,
        root: str | Path,
        split: str | list[str] | None = None,
        feature_keys: list[str] | None = None,
        label_keys: list[str] | None = None,
        label_prefix: str = "label__",
        s3_endpoint_url: str | None = None,
        s3_region: str | None = None,
        s3_profile: str | None = None,
        s3_client: Any | None = None,
    ):
        root_text = str(root)
        self._s3 = (
            S3Client(
                endpoint_url=s3_endpoint_url,
                region=s3_region,
                profile=s3_profile,
                client=s3_client,
            )
            if is_s3_uri(root_text)
            else None
        )
        self.root: Path | S3Path = S3Path.parse(root_text) if self._s3 is not None else Path(root)

        config = yaml.safe_load(self._read_text("config.yaml")) or {}
        self.config = cast(dict[str, Any], config)
        self.backend = str(config.get("writer") or config.get("backend", "numpy"))
        if self._s3 is not None and self.backend != "webdataset":
            raise ValueError("Remote ExportedDataset currently supports WebDataset exports only")
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
        self._webdataset_tempdir = (
            tempfile.TemporaryDirectory(prefix="nexuml-webdataset-")
            if self._s3 is not None
            else None
        )
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
        self._filter_webdataset_paths_to_meta()

    @property
    def is_remote(self) -> bool:
        """Whether tensors are stored in S3."""
        return self._s3 is not None

    def _read_bytes(self, relative: str) -> bytes:
        if self._s3 is not None:
            assert isinstance(self.root, S3Path)
            return self._s3.read_bytes(self.root / relative)
        assert isinstance(self.root, Path)
        return (self.root / relative).read_bytes()

    def _read_text(self, relative: str) -> str:
        return self._read_bytes(relative).decode("utf-8")

    def _filter_webdataset_paths_to_meta(self) -> None:
        if self.backend != "webdataset" or self.meta is None or "split" not in self.meta.columns:
            return
        splits = {str(value) for value in self.meta["split"].dropna().unique()}
        if len(splits) != 1:
            return
        split = next(iter(splits))
        prefix = f"data/shards/{split}/"
        self.extra = copy.deepcopy(self.extra)
        for key in ("shards", "index_paths"):
            paths = self.extra.get(key)
            if isinstance(paths, list):
                self.extra[key] = [str(path) for path in paths if str(path).startswith(prefix)]

    def _materialize_webdataset_paths(self, relative_paths: list[str]) -> list[str]:
        if self._s3 is None:
            return [str(self.root / relative_path) for relative_path in relative_paths]

        assert isinstance(self.root, S3Path)
        assert self._webdataset_tempdir is not None
        local_root = Path(self._webdataset_tempdir.name)
        materialized = []
        for relative_path in relative_paths:
            destination = local_root / relative_path
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(self.root / relative_path, destination)
            materialized.append(str(destination))
        return materialized

    def _load_metadata(self, config: dict[str, Any]) -> pd.DataFrame:
        extra = cast(dict[str, Any], config.get("extra", {}) or {})
        metadata_file = str(extra.get("metadata_file") or "metadata.parquet")
        metadata_format = str(extra.get("metadata_format") or "parquet")

        if self._s3 is not None:
            data = io.BytesIO(self._read_bytes(metadata_file))
            if metadata_format == "csv" or metadata_file.endswith(".csv"):
                return pd.read_csv(data)
            return pd.read_parquet(data)

        assert isinstance(self.root, Path)
        metadata_path = self.root / metadata_file
        if metadata_path.exists():
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
            if self._s3 is not None:
                stored_keys = sorted(self.feature_shapes.keys())
            elif self.backend == "numpy":
                assert isinstance(self.root, Path)
                stored_keys = sorted(
                    path.name for path in (self.root / "data").iterdir() if path.is_dir()
                )
            elif self.backend == "numpy_mmap":
                assert isinstance(self.root, Path)
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

    def _local_root(self) -> Path:
        if not isinstance(self.root, Path):
            raise RuntimeError(
                "S3 ExportedDataset tensors are consumed through the DALI WebDataset loader; "
                "per-sample Python reads are intentionally unsupported"
            )
        return self.root

    def _mmap_array(self, stored_key: str) -> np.ndarray:
        root = self._local_root()
        if stored_key not in self._mmap_arrays:
            self._mmap_arrays[stored_key] = np.load(
                root / "data" / f"{stored_key}.npy", mmap_mode="r"
            )
        return self._mmap_arrays[stored_key]

    def _torch_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        root = self._local_root()
        payload = torch.load(
            root / "data" / f"{export_idx:08d}.pt", map_location="cpu", weights_only=False
        )
        return {str(key): _to_tensor(value) for key, value in payload.items()}

    def _numpy_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        root = self._local_root()
        return {
            stored_key: torch.from_numpy(
                np.load(root / "data" / stored_key / f"{export_idx:08d}.npy", allow_pickle=False)
            )
            for stored_key in {*self._stored_x_keys.values(), *self._stored_y_keys.values()}
        }

    def _numpy_mmap_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        return {
            stored_key: _to_tensor(self._mmap_array(stored_key)[export_idx])
            for stored_key in {*self._stored_x_keys.values(), *self._stored_y_keys.values()}
        }

    def _tensordict_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        root = self._local_root()
        if self._tensordict_memmap is None:
            self._tensordict_memmap = TensorDict.load_memmap(root / "data")
        sample: TensorDict = self._tensordict_memmap[export_idx]  # ty: ignore[invalid-assignment]
        return {str(key): _to_tensor(value) for key, value in sample.items()}

    def _webdataset_payload(self, export_idx: int) -> dict[str, torch.Tensor]:
        root = self._local_root()
        if self._webdataset_index is None:
            index_name = str(self.extra.get("sample_index_file", "data/webdataset_index.json"))
            index_path = root / index_name
            self._webdataset_index = json.loads(index_path.read_text())

        sample_id = f"{export_idx:08d}"
        if sample_id not in self._webdataset_index:
            raise IndexError(f"Sample index {export_idx} is not present in the WebDataset export")
        sample_entry = self._webdataset_index[sample_id]
        shard_path = root / sample_entry["shard"]

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
        if self._s3 is not None:
            self._local_root()
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
            try:
                backend_cls = get_export_backend(self.backend)
            except KeyError as exc:
                raise ValueError(f"Unsupported exported dataset backend: {self.backend}") from exc
            payload = backend_cls.load_sample(self._local_root(), export_idx)

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
        clone.extra = copy.deepcopy(self.extra)
        clone._filter_webdataset_paths_to_meta()
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
        import soundfile as sf  # ty: ignore[unresolved-import]

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
