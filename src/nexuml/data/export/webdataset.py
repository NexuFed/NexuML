"""WebDataset tar-shard export backend."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from nexuml.data.export.backend import ExportBackend, register_export_backend


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _infer_layout(array: np.ndarray, modality: str) -> str | None:
    if modality == "image" and array.ndim == 3:
        if array.shape[0] in {1, 3, 4}:
            return "CHW"
        if array.shape[-1] in {1, 3, 4}:
            return "HWC"
    if modality == "video" and array.ndim == 4:
        if array.shape[1] in {1, 3, 4}:
            return "TCHW"
        if array.shape[-1] in {1, 3, 4}:
            return "THWC"
    if modality == "audio" and array.ndim == 2:
        if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            return "CT"
        return "TC"
    if modality == "audio" and array.ndim == 1:
        return "T"
    return None


def _layout_for_payload(array: np.ndarray, layout: str | None, modality: str) -> np.ndarray:
    if modality == "image" and layout == "CHW":
        return np.moveaxis(array, 0, -1)
    if modality == "video" and layout == "TCHW":
        return np.moveaxis(array, 1, -1)
    if modality == "audio" and layout == "CT":
        return np.moveaxis(array, 0, -1)
    return array


def _layout_from_payload(array: np.ndarray, layout: str | None, modality: str) -> np.ndarray:
    if modality == "image" and layout == "CHW":
        return np.moveaxis(array, -1, 0)
    if modality == "video" and layout == "TCHW":
        return np.moveaxis(array, -1, 1)
    if modality == "audio" and layout == "CT":
        return np.moveaxis(array, -1, 0)
    return array


def _write_tar_bytes(handle: tarfile.TarFile, member_name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=member_name)
    info.size = len(data)
    handle.addfile(info, io.BytesIO(data))


@register_export_backend("webdataset")
class WebDatasetBackend(ExportBackend):
    """Write samples into WebDataset tar shards."""

    def __init__(
        self,
        *,
        modality: str = "generic",
        x_keys: list[str] | None = None,
        y_keys: list[str] | None = None,
        transform_applied: bool = False,
        generic_payload: str = "npy",
        samples_per_shard: int = 256,
        audio_sample_rate: int = 16000,
        video_fps: float = 30.0,
        **_kwargs,
    ) -> None:
        self.modality = str(modality).lower()
        self.x_keys = set(x_keys or [])
        self.y_keys = set(y_keys or [])
        self.transform_applied = bool(transform_applied)
        self.generic_payload = generic_payload
        self.samples_per_shard = int(samples_per_shard)
        self.audio_sample_rate = int(audio_sample_rate)
        self.video_fps = float(video_fps)
        self._export_dir: Path | None = None
        self._feature_shapes: dict[str, tuple[int, ...]] = {}
        self._dtype: np.dtype[Any] | None = None
        self._key_specs: dict[str, dict[str, Any]] = {}
        self._sample_index: dict[str, dict[str, Any]] = {}
        self._current_shard_id: int | None = None
        self._current_shard_name: str | None = None
        self._current_tar: tarfile.TarFile | None = None
        self._saved: int = 0

    def initialize(
        self,
        export_dir: Path,
        num_samples: int,
        feature_shapes: dict[str, tuple[int, ...]],
        dtype: np.dtype[Any] | str | None = None,
    ) -> None:
        self._export_dir = export_dir
        self._feature_shapes = dict(feature_shapes)
        self._dtype = None if dtype is None else np.dtype(dtype)
        (export_dir / "data" / "shards").mkdir(parents=True, exist_ok=True)

    def _ensure_shard(self, index: int) -> tuple[tarfile.TarFile, str]:
        if self._export_dir is None:
            raise RuntimeError("Backend has not been initialized")
        shard_id = index // self.samples_per_shard
        shard_name = f"shard-{shard_id:06d}.tar"
        if self._current_shard_id != shard_id:
            if self._current_tar is not None:
                self._current_tar.close()
            self._current_shard_id = shard_id
            self._current_shard_name = shard_name
            self._current_tar = tarfile.open(self._export_dir / "data" / "shards" / shard_name, "w")
        tar = self._current_tar
        assert tar is not None
        return tar, shard_name

    def _choose_encoding(self, key: str, tensor: torch.Tensor | str | bytes) -> str:
        if key in self.y_keys:
            return self.generic_payload
        if self.transform_applied:
            return self.generic_payload
        if key not in self.x_keys:
            return self.generic_payload
        if self.modality == "image":
            return "png"
        if self.modality == "audio":
            return "wav"
        if self.modality == "video":
            return "mp4"
        if self.modality == "text":
            if isinstance(tensor, (str, bytes)):
                return "txt"
            if isinstance(tensor, torch.Tensor) and tensor.dtype in {torch.uint8, torch.int8}:
                return "txt"
        return self.generic_payload

    def _encode_component(
        self, key: str, value: torch.Tensor | str | bytes
    ) -> tuple[str, bytes, dict[str, Any]]:
        encoding = self._choose_encoding(key, value)
        layout: str | None = None
        dtype_name: str | None = None
        shape: list[int] | None = None

        if isinstance(value, bytes):
            payload = value
            encoding = "txt" if encoding == "txt" else encoding
            dtype_name = "bytes"
            shape = [len(payload)]
        elif isinstance(value, str):
            payload = value.encode("utf-8")
            encoding = "txt"
            dtype_name = "utf8"
            shape = [len(payload)]
        else:
            tensor = value.detach().cpu()
            if self._dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(torch.from_numpy(np.empty((), dtype=self._dtype)).dtype)
            array = tensor.numpy()
            shape = list(array.shape)
            dtype_name = str(array.dtype)
            layout = _infer_layout(array, self.modality if key in self.x_keys else "generic")

            if encoding == "png":
                image_array = _layout_for_payload(array, layout, "image")
                image = Image.fromarray(_normalize_to_uint8(image_array))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                payload = buffer.getvalue()
            elif encoding == "wav":
                audio_array = _layout_for_payload(array, layout, "audio")
                buffer = io.BytesIO()
                sf.write(buffer, audio_array, self.audio_sample_rate, format="WAV")
                payload = buffer.getvalue()
            elif encoding == "mp4":
                payload = _encode_video(array, layout, fps=self.video_fps)
            elif encoding == "pt":
                buffer = io.BytesIO()
                torch.save(tensor, buffer)
                payload = buffer.getvalue()
            elif encoding == "bin":
                payload = array.tobytes(order="C")
            elif encoding == "txt":
                payload = bytes(array.tolist())
            else:
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                payload = buffer.getvalue()

        spec: dict[str, Any] = {
            "encoding": encoding,
            "dtype": dtype_name,
            "layout": layout,
            "shape": shape,
            "storage": {
                "type": "webdataset",
                "path": "data/shards",
            },
        }
        if encoding == "wav":
            spec["sample_rate"] = self.audio_sample_rate
        if encoding == "mp4":
            spec["fps"] = self.video_fps
        return encoding, payload, spec

    def save_sample(self, index: int, features: dict[str, torch.Tensor]) -> None:
        tar_handle, shard_name = self._ensure_shard(index)
        sample_id = f"{index:08d}"
        sample_components: dict[str, dict[str, Any]] = {}

        for key, value in features.items():
            encoding, payload, spec = self._encode_component(key, value)
            ext = f"{key}.{encoding}"
            member_name = f"{sample_id}.{ext}"
            _write_tar_bytes(tar_handle, member_name, payload)
            spec["storage"] = {
                **spec["storage"],
                "member_ext": ext,
            }
            self._key_specs.setdefault(key, spec)
            sample_components[key] = {
                "member": member_name,
                "encoding": encoding,
            }

        index_buffer = io.BytesIO()
        np.save(index_buffer, np.asarray(index, dtype=np.int64), allow_pickle=False)
        index_member = f"{sample_id}.__index.npy"
        _write_tar_bytes(tar_handle, index_member, index_buffer.getvalue())

        self._sample_index[sample_id] = {
            "shard": shard_name,
            "components": sample_components,
        }
        self._saved += 1

    def finalize(self) -> dict[str, Any]:
        if self._current_tar is not None:
            self._current_tar.close()
            self._current_tar = None
        if self._export_dir is None:
            raise RuntimeError("Backend has not been initialized")

        shard_paths = sorted(
            str(path.relative_to(self._export_dir))
            for path in (self._export_dir / "data" / "shards").glob("*.tar")
        )
        index_path = self._export_dir / "data" / "webdataset_index.json"
        index_path.write_text(json.dumps(self._sample_index, indent=2, sort_keys=True))
        return {
            "format": "webdataset",
            "dtype": None if self._dtype is None else self._dtype.name,
            "samples_saved": self._saved,
            "key_specs": self._key_specs,
            "shards": shard_paths,
            "sample_index_file": str(index_path.relative_to(self._export_dir)),
        }

    @staticmethod
    def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]:
        index_data = json.loads((export_dir / "data" / "webdataset_index.json").read_text())
        sample_id = f"{index:08d}"
        if sample_id not in index_data:
            raise IndexError(f"Sample index {index} is not present in WebDataset export")

        components = index_data[sample_id]["components"]
        shard_path = export_dir / "data" / "shards" / index_data[sample_id]["shard"]
        with tarfile.open(shard_path, "r") as handle:
            members = {member.name: member for member in handle.getmembers()}
            result: dict[str, torch.Tensor] = {}
            for key, entry in components.items():
                member = members[entry["member"]]
                payload = handle.extractfile(member)
                if payload is None:
                    raise FileNotFoundError(f"Could not read tar member {member.name}")
                result[key] = _decode_component(payload.read(), encoding=entry["encoding"], key=key)
        return result


def _decode_component(data: bytes, *, encoding: str, key: str) -> torch.Tensor:
    if encoding == "npy":
        return torch.from_numpy(np.load(io.BytesIO(data), allow_pickle=False).copy())
    if encoding == "pt":
        value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if encoding == "txt":
        return torch.tensor(list(data), dtype=torch.uint8)
    if encoding == "bin":
        raise ValueError(
            f"Raw binary component '{key}' requires manifest-based reinterpretation and cannot be "
            "decoded without ExportedDataset metadata."
        )
    if encoding in {"png", "jpg", "jpeg"}:
        image = np.asarray(Image.open(io.BytesIO(data)))
        return torch.from_numpy(image.copy())
    if encoding == "wav":
        audio, _sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return torch.from_numpy(audio.copy())
    if encoding == "mp4":
        raise NotImplementedError(
            "Torch-side WebDataset MP4 decoding is not available without a video decoder stack."
        )
    raise ValueError(f"Unsupported WebDataset component encoding: {encoding}")


def _encode_video(array: np.ndarray, layout: str | None, *, fps: float) -> bytes:
    frames = _layout_for_payload(array, layout, "video")
    frames = _normalize_to_uint8(frames)
    if frames.ndim != 4:
        raise ValueError(f"Video export expects a 4D tensor, got shape {frames.shape}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("WebDataset video export requires ffmpeg on PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_dir / f"frame_{idx:06d}.png")

        output_path = frame_dir / "sample.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame_%06d.png"),
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return output_path.read_bytes()
