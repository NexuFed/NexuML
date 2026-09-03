"""WebDataset tar-shard export backend."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nexuml.data.export.backend import ExportBackend, register_export_backend
from nexuml.storage.s3 import S3Client, S3Path


def _write_tar_bytes(handle: tarfile.TarFile, member_name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=member_name)
    info.size = len(data)
    handle.addfile(info, io.BytesIO(data))


def _safe_split(value: str) -> str:
    split = str(value).strip()
    if not split or split in {".", ".."} or "/" in split or "\\" in split:
        raise ValueError(f"Invalid WebDataset split name: {value!r}")
    return split


@register_export_backend("webdataset")
class WebDatasetBackend(ExportBackend):
    """Write lossless NumPy components into indexed WebDataset tar shards."""

    def __init__(
        self,
        *,
        samples_per_shard: int = 256,
        create_index: bool = True,
        s3_uri: str | None = None,
        s3_endpoint_url: str | None = None,
        s3_region: str | None = None,
        s3_profile: str | None = None,
        s3_client: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        if samples_per_shard < 1:
            raise ValueError("samples_per_shard must be positive")
        self.samples_per_shard = int(samples_per_shard)
        self.create_index = bool(create_index)
        self._s3_root = S3Path.parse(s3_uri) if s3_uri is not None else None
        self._s3 = (
            S3Client(
                endpoint_url=s3_endpoint_url,
                region=s3_region,
                profile=s3_profile,
                client=s3_client,
            )
            if self._s3_root is not None
            else None
        )

        self._export_dir: Path | None = None
        self._dtype: np.dtype[Any] | None = None
        self._key_specs: dict[str, dict[str, Any]] = {}
        self._sample_index: dict[str, dict[str, Any]] = {}
        self._saved = 0

        self._split: str | None = None
        self._next_shard_id: dict[str, int] = {}
        self._current_tar: tarfile.TarFile | None = None
        self._current_tar_path: Path | None = None
        self._current_shard_relative: str | None = None
        self._current_shard_samples = 0

        self._shard_paths: list[str] = []
        self._index_paths: list[str] = []

    @property
    def remote_uri(self) -> str | None:
        """S3 dataset root for remote exports."""
        return None if self._s3_root is None else str(self._s3_root)

    def initialize(
        self,
        export_dir: Path,
        num_samples: int,
        feature_shapes: dict[str, tuple[int, ...]],
        dtype: np.dtype[Any] | str | None = None,
    ) -> None:
        """Prepare local shard staging.

        Args:
            export_dir: Local export/staging directory.
            num_samples: Declared number of exported samples.
            feature_shapes: Tensor shapes supplied by the export runner.
            dtype: Optional floating-point storage dtype.
        """
        del num_samples, feature_shapes
        self._export_dir = Path(export_dir)
        self._dtype = None if dtype is None else np.dtype(dtype)
        (self._export_dir / "data" / "shards").mkdir(parents=True, exist_ok=True)

    def start_split(self, split: str) -> None:
        """Start one physically isolated dataset split."""
        if self._current_tar is not None:
            self._close_current_shard()
        self._split = _safe_split(split)
        self._next_shard_id.setdefault(self._split, 0)

    def end_split(self, split: str) -> None:
        """Finish one dataset split and close its current shard.

        Raises:
            RuntimeError: If the requested split is not the active split.
        """
        selected = _safe_split(split)
        if self._split != selected:
            raise RuntimeError(f"Cannot end split {selected!r}; active split is {self._split!r}")
        self._close_current_shard()
        self._split = None

    def _new_shard(self) -> None:
        if self._export_dir is None:
            raise RuntimeError("WebDataset backend has not been initialized")
        split = self._split or "all"
        shard_id = self._next_shard_id.setdefault(split, 0)
        self._next_shard_id[split] = shard_id + 1
        relative = f"data/shards/{split}/shard-{shard_id:06d}.tar"
        path = self._export_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self._current_tar_path = path
        self._current_shard_relative = relative
        self._current_shard_samples = 0
        self._current_tar = tarfile.open(path, "w")

    def _ensure_shard(self) -> tarfile.TarFile:
        if self._current_tar is None:
            self._new_shard()
        elif self._current_shard_samples >= self.samples_per_shard:
            self._close_current_shard()
            self._new_shard()
        assert self._current_tar is not None
        return self._current_tar

    def _generate_index(self, tar_path: Path) -> Path:
        index_path = tar_path.with_suffix(".idx")
        if not self.create_index:
            return index_path
        tool = shutil.which("wds2idx")
        if tool is None:
            raise RuntimeError(
                "WebDataset index generation requires DALI's wds2idx utility; "
                "install the nexuml[dali] extra or set create_index=False"
            )
        subprocess.run(
            [tool, str(tar_path), str(index_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return index_path

    def _upload(self, local_path: Path, relative: str) -> None:
        if self._s3 is None or self._s3_root is None:
            return
        self._s3.upload_file(local_path, self._s3_root / relative)

    def _close_current_shard(self) -> None:
        if self._current_tar is None:
            return
        assert self._current_tar_path is not None
        assert self._current_shard_relative is not None

        self._current_tar.close()
        tar_path = self._current_tar_path
        tar_relative = self._current_shard_relative
        idx_path = self._generate_index(tar_path)
        idx_relative = str(Path(tar_relative).with_suffix(".idx")).replace("\\", "/")

        self._shard_paths.append(tar_relative)
        if self.create_index:
            self._index_paths.append(idx_relative)

        if self._s3 is not None:
            self._upload(tar_path, tar_relative)
            if self.create_index:
                self._upload(idx_path, idx_relative)
            tar_path.unlink(missing_ok=True)
            if self.create_index:
                idx_path.unlink(missing_ok=True)

        self._current_tar = None
        self._current_tar_path = None
        self._current_shard_relative = None
        self._current_shard_samples = 0

    def _encode(self, value: torch.Tensor) -> tuple[bytes, list[int], str]:
        tensor = value.detach().cpu()
        if self._dtype is not None and tensor.is_floating_point():
            torch_dtype = torch.from_numpy(np.empty((), dtype=self._dtype)).dtype
            tensor = tensor.to(torch_dtype)
        array = tensor.numpy()
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue(), list(array.shape), str(array.dtype)

    def save_sample(self, index: int, features: dict[str, torch.Tensor]) -> None:
        """Append one tensor sample to the active tar shard.

        Raises:
            ValueError: If a component key contains a path separator.
        """
        tar_handle = self._ensure_shard()
        sample_id = f"{index:08d}"
        components: dict[str, dict[str, str]] = {}

        for key, value in features.items():
            if "/" in key or "\\" in key:
                raise ValueError(f"WebDataset keys must not contain path separators: {key!r}")
            payload, shape, dtype = self._encode(value)
            member_ext = f"{key}.npy"
            member_name = f"{sample_id}.{member_ext}"
            _write_tar_bytes(tar_handle, member_name, payload)
            self._key_specs.setdefault(
                key,
                {
                    "encoding": "npy",
                    "dtype": dtype,
                    "layout": None,
                    "shape": shape,
                    "storage": {
                        "type": "webdataset",
                        "path": "data/shards",
                        "member_ext": member_ext,
                    },
                },
            )
            components[key] = {"member": member_name, "encoding": "npy"}

        index_buffer = io.BytesIO()
        np.save(index_buffer, np.asarray(index, dtype=np.int64), allow_pickle=False)
        _write_tar_bytes(tar_handle, f"{sample_id}.__index.npy", index_buffer.getvalue())

        assert self._current_shard_relative is not None
        if self._s3 is None:
            self._sample_index[sample_id] = {
                "shard": self._current_shard_relative,
                "components": components,
            }
        self._current_shard_samples += 1
        self._saved += 1

    def finalize(self) -> dict[str, Any]:
        """Close the final shard and return metadata for ``config.yaml``.

        Returns:
            Backend-specific export metadata.

        Raises:
            RuntimeError: If the backend was never initialized.
        """
        self._close_current_shard()
        if self._export_dir is None:
            raise RuntimeError("WebDataset backend has not been initialized")

        metadata: dict[str, Any] = {
            "format": "webdataset",
            "dtype": None if self._dtype is None else self._dtype.name,
            "samples_saved": self._saved,
            "key_specs": self._key_specs,
            "shards": list(self._shard_paths),
            "index_paths": list(self._index_paths),
            "storage_uri": self.remote_uri,
        }
        if self._s3 is None:
            index_file = self._export_dir / "data" / "webdataset_index.json"
            index_file.parent.mkdir(parents=True, exist_ok=True)
            index_file.write_text(json.dumps(self._sample_index, indent=2, sort_keys=True))
            metadata["sample_index_file"] = "data/webdataset_index.json"
        return metadata

    def publish_export_metadata(self, config_path: Path, metadata_path: Path) -> None:
        """Upload the final lightweight export metadata for an S3 export."""
        if self._s3 is None:
            return
        self._upload(config_path, "config.yaml")
        self._upload(metadata_path, metadata_path.name)

    @staticmethod
    def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]:
        """Load one sample from a local WebDataset export.

        Returns:
            Mapping of stored keys to tensors.

        Raises:
            IndexError: If the sample is absent from the export index.
            FileNotFoundError: If a referenced tar member cannot be read.
        """
        index_data = json.loads((export_dir / "data" / "webdataset_index.json").read_text())
        sample_id = f"{index:08d}"
        if sample_id not in index_data:
            raise IndexError(f"Sample index {index} is not present in the WebDataset export")

        sample_entry = index_data[sample_id]
        with tarfile.open(export_dir / sample_entry["shard"], "r") as handle:
            result: dict[str, torch.Tensor] = {}
            for key, entry in sample_entry["components"].items():
                member = handle.getmember(entry["member"])
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"Could not read tar member {member.name}")
                result[key] = torch.from_numpy(
                    np.load(io.BytesIO(extracted.read()), allow_pickle=False).copy()
                )
            return result
