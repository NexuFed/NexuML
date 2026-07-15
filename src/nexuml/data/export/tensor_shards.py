"""Non-prebatched tensor-shard export backend."""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any

import torch

from nexuml.data.export.backend import ExportBackend, register_export_backend

_MANIFEST = "tensor_shards_index.json"


def _manifest_path(root: Path) -> Path:
    return root / "data" / _MANIFEST


def _shard_path(root: Path, split: str, shard_id: int) -> Path:
    return root / "data" / "shards" / split / f"shard-{shard_id:06d}.pt"


def _normalise_split(split: str) -> str:
    split = str(split).strip().replace("/", "_").replace("\\", "_")
    if split in {"", ".", ".."}:
        raise ValueError(f"Invalid split name: {split!r}")
    return split


def _torch_dtype(dtype: object | None) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype

    name = str(dtype).removeprefix("torch.")
    name = {
        "float": "float32",
        "double": "float64",
        "half": "float16",
    }.get(name, name)

    result = getattr(torch, name, None)
    if not isinstance(result, torch.dtype):
        raise TypeError(f"Unsupported dtype: {dtype!r}")
    return result


@register_export_backend("tensor_shards")
class TensorShardsBackend(ExportBackend):
    """Write arbitrary fixed-shape tensor keys into sample shards.

    A shard is a storage unit with ``N`` samples along dimension zero. It is
    NOT a training batch; the runtime loader may create arbitrary batch sizes
    from the shard contents.
    """

    def __init__(
        self,
        *,
        samples_per_shard: int = 4096,
        **_kwargs: Any,
    ) -> None:
        if samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be > 0")

        self.samples_per_shard = int(samples_per_shard)

        self._root: Path | None = None
        self._num_samples = 0
        self._feature_shapes: dict[str, tuple[int, ...]] = {}
        self._storage_dtypes: dict[str, torch.dtype] = {}
        self._float_dtype: torch.dtype | None = None

        self._next_index = 0
        self._saved = 0

        self._split: str | None = None
        self._next_shard_id: dict[str, int] = {}
        self._split_sample_counts: dict[str, int] = {}

        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_count = 0
        self._buffer_start = 0

        self._shards: list[dict[str, Any]] = []

    def initialize(
        self,
        export_dir: Path,
        num_samples: int,
        feature_shapes: dict[str, tuple[int, ...]],
        dtype: object | None = None,
    ) -> None:
        if num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if not feature_shapes:
            raise ValueError("feature_shapes must not be empty")

        self._root = Path(export_dir)
        self._num_samples = int(num_samples)
        self._feature_shapes = {
            str(key): tuple(int(dim) for dim in shape) for key, shape in feature_shapes.items()
        }
        self._float_dtype = _torch_dtype(dtype)

        if (
            self._float_dtype is not None
            and not torch.empty((), dtype=self._float_dtype).is_floating_point()
        ):
            raise ValueError("dtype must be a floating-point dtype")

        (self._root / "data" / "shards").mkdir(parents=True, exist_ok=True)

        self._storage_dtypes.clear()
        self._next_index = 0
        self._saved = 0
        self._split = None
        self._next_shard_id.clear()
        self._split_sample_counts.clear()
        self._buffer.clear()
        self._buffer_count = 0
        self._buffer_start = 0
        self._shards.clear()

    # Optional hooks. Add the small runner patch shown below to use them.
    def start_split(self, split: str) -> None:
        split = _normalise_split(split)
        if split == self._split:
            return
        if self._buffer_count:
            self._flush()
        self._split = split
        self._next_shard_id.setdefault(split, 0)
        self._split_sample_counts.setdefault(split, 0)

    def end_split(self, split: str) -> None:
        split = _normalise_split(split)
        if split != self._split:
            raise RuntimeError(f"Cannot end split {split!r}; active split is {self._split!r}")
        if self._buffer_count:
            self._flush()
        self._split = None

    def save_sample(
        self,
        index: int,
        features: dict[str, torch.Tensor],
    ) -> None:
        self.save_batch(
            index,
            {key: tensor.unsqueeze(0) for key, tensor in features.items()},
        )

    def save_batch(
        self,
        start_index: int,
        features: dict[str, torch.Tensor],
    ) -> None:
        self._require_initialized()

        if start_index != self._next_index:
            raise ValueError(
                "Tensor-shard export requires contiguous ordered writes: "
                f"expected {self._next_index}, got {start_index}"
            )

        batch, batch_size = self._prepare_batch(features)
        if start_index + batch_size > self._num_samples:
            raise IndexError(
                f"Batch ending at {start_index + batch_size} exceeds "
                f"declared size {self._num_samples}"
            )
        if batch_size == 0:
            return

        if self._split is None:
            self.start_split("all")

        offset = 0
        while offset < batch_size:
            if self._buffer_count == 0:
                self._allocate_buffer(self._next_index)

            take = min(
                self.samples_per_shard - self._buffer_count,
                batch_size - offset,
            )
            dst = slice(self._buffer_count, self._buffer_count + take)
            src = slice(offset, offset + take)

            for key in self._feature_shapes:
                self._buffer[key][dst].copy_(batch[key][src])

            self._buffer_count += take
            self._next_index += take
            self._saved += take
            assert self._split is not None
            self._split_sample_counts[self._split] += take
            offset += take

            if self._buffer_count == self.samples_per_shard:
                self._flush()

    def finalize(self) -> dict[str, Any]:
        root = self._require_initialized()

        if self._buffer_count:
            self._flush()

        if self._saved != self._num_samples:
            raise RuntimeError(f"Saved {self._saved} samples, expected {self._num_samples}")

        manifest = {
            "format_version": 1,
            "format": "tensor_shards",
            "num_samples": self._num_samples,
            "samples_per_shard": self.samples_per_shard,
            "num_shards": len(self._shards),
            "feature_shapes": {key: list(shape) for key, shape in self._feature_shapes.items()},
            "storage_dtypes": {
                key: str(dtype).removeprefix("torch.")
                for key, dtype in self._storage_dtypes.items()
            },
            "split_sample_counts": self._split_sample_counts,
            "shards": self._shards,
        }
        self._write_json_atomic(_manifest_path(root), manifest)

        shards_by_split: dict[str, list[str]] = {}
        for shard in self._shards:
            shards_by_split.setdefault(shard["split"], []).append(shard["path"])

        return {
            "format": "tensor_shards",
            "dtype": (
                None if self._float_dtype is None else str(self._float_dtype).removeprefix("torch.")
            ),
            "samples_saved": self._saved,
            "samples_per_shard": self.samples_per_shard,
            "num_shards": len(self._shards),
            "shards": [entry["path"] for entry in self._shards],
            "shards_by_split": shards_by_split,
            "shard_index_file": str(_manifest_path(root).relative_to(root)),
            "key_specs": {
                key: {
                    "encoding": "pt",
                    "storage": {
                        "type": "tensor_shard",
                        "path": "data/shards",
                        "index_file": f"data/{_MANIFEST}",
                        "samples_per_shard": self.samples_per_shard,
                    },
                }
                for key in self._feature_shapes
            },
        }

    @staticmethod
    def load_manifest(export_dir: Path) -> dict[str, Any]:
        path = _manifest_path(Path(export_dir))
        manifest = json.loads(path.read_text())
        if manifest.get("format") != "tensor_shards":
            raise ValueError(f"Invalid tensor-shard manifest: {path}")
        return manifest

    @staticmethod
    def load_shard(
        export_dir: Path,
        shard: int | str | dict[str, Any],
    ) -> dict[str, Any]:
        """Load a complete shard for the future windowed loader."""
        root = Path(export_dir)

        if isinstance(shard, int):
            manifest = TensorShardsBackend.load_manifest(root)
            try:
                relative_path = manifest["shards"][shard]["path"]
            except IndexError as exc:
                raise IndexError(f"Shard index out of range: {shard}") from exc
        elif isinstance(shard, str):
            relative_path = shard
        else:
            relative_path = shard["path"]

        path = root / relative_path
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
            raise ValueError(f"Invalid tensor shard: {path}")

        payload["features"] = {
            str(key): tensor.detach().cpu() for key, tensor in payload["features"].items()
        }
        return payload

    @staticmethod
    def load_sample(
        export_dir: Path,
        index: int,
    ) -> dict[str, torch.Tensor]:
        """Map-style compatibility path; the window loader should use load_shard."""
        manifest = TensorShardsBackend.load_manifest(export_dir)
        num_samples = int(manifest["num_samples"])
        if not 0 <= index < num_samples:
            raise IndexError(f"Sample index {index} out of bounds for {num_samples}")

        shards = manifest["shards"]
        starts = [int(entry["start_index"]) for entry in shards]
        shard_position = bisect_right(starts, index) - 1
        if shard_position < 0:
            raise IndexError(f"No shard contains sample {index}")

        entry = shards[shard_position]
        local_index = index - int(entry["start_index"])
        if local_index >= int(entry["num_samples"]):
            raise IndexError(f"No shard contains sample {index}")

        payload = TensorShardsBackend.load_shard(export_dir, entry)
        return {key: tensor[local_index].clone() for key, tensor in payload["features"].items()}

    def _prepare_batch(
        self,
        features: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], int]:
        expected = set(self._feature_shapes)
        actual = set(features)

        if missing := expected - actual:
            raise KeyError(f"Missing feature keys: {sorted(missing)}")
        if unexpected := actual - expected:
            raise KeyError(f"Unexpected feature keys: {sorted(unexpected)}")

        result: dict[str, torch.Tensor] = {}
        batch_size: int | None = None

        for key, sample_shape in self._feature_shapes.items():
            tensor = features[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Feature {key!r} must be a tensor, got {type(tensor).__name__}")
            if tensor.ndim == 0:
                raise ValueError(f"Feature {key!r} has no leading batch dimension")

            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError(
                    f"Batch-size mismatch for {key!r}: {tensor.shape[0]} != {batch_size}"
                )

            actual_shape = tuple(int(dim) for dim in tensor.shape[1:])
            if actual_shape != sample_shape:
                raise ValueError(
                    f"Feature {key!r} has shape {actual_shape}, expected {sample_shape}"
                )

            tensor = tensor.detach()
            if self._float_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=self._float_dtype)
            tensor = tensor.to(device="cpu")

            old_dtype = self._storage_dtypes.get(key)
            if old_dtype is None:
                self._storage_dtypes[key] = tensor.dtype
            elif old_dtype != tensor.dtype:
                raise TypeError(f"Feature {key!r} changed dtype: {old_dtype} -> {tensor.dtype}")

            result[key] = tensor

        assert batch_size is not None
        return result, batch_size

    def _allocate_buffer(self, start_index: int) -> None:
        self._buffer = {
            key: torch.empty(
                (self.samples_per_shard, *shape),
                dtype=self._storage_dtypes[key],
            )
            for key, shape in self._feature_shapes.items()
        }
        self._buffer_count = 0
        self._buffer_start = start_index

    def _flush(self) -> None:
        root = self._require_initialized()
        if self._buffer_count == 0:
            return
        if self._split is None:
            raise RuntimeError("Cannot flush without an active split")

        split = self._split
        shard_id = self._next_shard_id[split]
        path = _shard_path(root, split, shard_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        count = self._buffer_count
        features = (
            self._buffer
            if count == self.samples_per_shard
            else {key: tensor[:count].clone() for key, tensor in self._buffer.items()}
        )

        payload = {
            "format_version": 1,
            "format": "tensor_shard",
            "split": split,
            "shard_id": shard_id,
            "start_index": self._buffer_start,
            "num_samples": count,
            "indices": torch.arange(
                self._buffer_start,
                self._buffer_start + count,
                dtype=torch.int64,
            ),
            "features": features,
        }
        self._torch_save_atomic(path, payload)

        self._shards.append(
            {
                "split": split,
                "shard_id": shard_id,
                "path": str(path.relative_to(root)),
                "start_index": self._buffer_start,
                "end_index": self._buffer_start + count,
                "num_samples": count,
            }
        )
        self._next_shard_id[split] += 1
        self._buffer = {}
        self._buffer_count = 0

    def _require_initialized(self) -> Path:
        if self._root is None:
            raise RuntimeError("Backend has not been initialized")
        return self._root

    @staticmethod
    def _torch_save_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
