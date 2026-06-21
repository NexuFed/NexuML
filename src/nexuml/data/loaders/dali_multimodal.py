"""General DALI pipeline helpers for keyed TensorDict outputs."""

from __future__ import annotations

import io
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from nvidia.dali import types as _dali_types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from tensordict import TensorDict

import nvidia.dali.fn as fn


def _dali_type(name: str) -> Any:
    """Return a DALI type constant by name.

    NVIDIA DALI stubs are incomplete; this single accessor centralises the
    ``ty: ignore`` and avoids scattering per-constant ignores across the file.
    """
    return getattr(_dali_types, name)


INDEX_OUTPUT = "__index"


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value.item())
    return np.asarray(value)


def _decode_text_bytes(sample: np.ndarray) -> np.ndarray:
    raw = np.asarray(sample, dtype=np.uint8).tobytes()
    return np.frombuffer(raw.decode("utf-8").encode("utf-8"), dtype=np.uint8)


def _load_pt_tensor(sample: np.ndarray) -> np.ndarray:
    payload = io.BytesIO(np.asarray(sample, dtype=np.uint8).tobytes())
    value = torch.load(payload, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


_DALI_DTYPE_MAP: dict[str, str] = {
    "float16": "FLOAT16",
    "float32": "FLOAT",
    "float64": "FLOAT64",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "uint8": "UINT8",
    "uint16": "UINT16",
    "uint32": "UINT32",
    "uint64": "UINT64",
    "bool": "BOOL",
}


def _dali_dtype(dtype_name: str | None) -> Any:
    if dtype_name is None:
        return None
    dali_name = _DALI_DTYPE_MAP.get(dtype_name)
    if dali_name is None:
        raise KeyError(f"Unsupported DALI dtype mapping for '{dtype_name}'")
    return _dali_type(dali_name)


def _layout_transpose(node, *, modality: str, layout: str | None):
    if modality == "image" and layout == "CHW":
        return fn.transpose(node, perm=[2, 0, 1])
    if modality == "video" and layout == "TCHW":
        return fn.transpose(node, perm=[0, 3, 1, 2])
    if modality == "audio" and layout == "CT":
        return fn.transpose(node, perm=[1, 0])
    return node


class KeyedDaliIterator(DALIGenericIterator):
    """DALI iterator that returns `(x, y)` TensorDict pairs keyed semantically."""

    metadata: pd.DataFrame | None = None  # set externally after construction (dali_backend.py)

    def __init__(
        self,
        *,
        x_keys: Sequence[str],
        y_keys: Sequence[str],
        metadata_labels: TensorDict | None,
        output_map: Sequence[str],
        size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(output_map=list(output_map), size=-1 if size is None else size, **kwargs)
        self.x_keys = list(x_keys)
        self.y_keys = list(y_keys)
        self.metadata_labels = metadata_labels

    def __next__(self) -> tuple[TensorDict, TensorDict | None]:  # ty: ignore[invalid-method-override] — see class comment
        out = super().__next__()[0]
        x = TensorDict(
            {key: out[key] for key in self.x_keys},
            batch_size=[out[self.x_keys[0]].shape[0]]
            if self.x_keys
            else [out[INDEX_OUTPUT].shape[0]],
        )
        x["sample_index"] = out[INDEX_OUTPUT].reshape(-1).long()

        y_dict: dict[str, torch.Tensor] = {key: out[key] for key in self.y_keys if key in out}
        if self.metadata_labels is not None:
            indices = out[INDEX_OUTPUT].reshape(-1).long().cpu()
            label_batch = cast(TensorDict, self.metadata_labels[indices])
            for key, value in label_batch.items():
                y_dict[str(key)] = cast(torch.Tensor, value)

        y = TensorDict(y_dict, batch_size=x.batch_size) if y_dict else None  # ty: ignore[invalid-argument-type]
        return x, y


class ExternalSampleIterator:
    """External source iterator backed by a dataset's `__getitem__`."""

    def __init__(
        self,
        dataset: Any,
        *,
        x_keys: Sequence[str],
        y_keys: Sequence[str],
        num_shards: int,
        shard_id: int,
        shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.x_keys = list(x_keys)
        self.y_keys = list(y_keys)
        self.shuffle = shuffle
        indices = list(range(len(dataset)))
        self.indices = indices[
            len(indices) * shard_id // num_shards : len(indices) * (shard_id + 1) // num_shards
        ]
        self.n = len(self.indices)

    def __iter__(self):
        self.i = 0
        self._order = list(self.indices)
        if self.shuffle:
            random.Random().shuffle(self._order)
        return self

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
        idx = self._order[self.i]
        self.i += 1
        x, y = self.dataset[idx]
        outputs: list[np.ndarray] = []
        for key in self.x_keys:
            outputs.append(_to_numpy(x[key]))
        for key in self.y_keys:
            if y is None or key not in y:
                raise KeyError(f"Missing DALI external-source label key '{key}'")
            outputs.append(_to_numpy(y[key]))
        outputs.append(np.asarray(idx, dtype=np.int64))
        return tuple(outputs)


@pipeline_def
def external_source_pipeline(external_source: Iterable, *, num_outputs: int):
    """Build a DALI pipeline backed by an external Python source.

    Returns:
        A single DALI output node when ``num_outputs == 1``,
        otherwise a tuple of output nodes.
    """
    outputs = fn.external_source(
        source=external_source,
        num_outputs=num_outputs,
        device="cpu",
        batch=False,
        parallel=False,
        cycle="quiet",
    )
    if num_outputs == 1:
        return outputs
    return tuple(outputs)


@pipeline_def
def audio_file_pipeline(
    *,
    files: list[str],
    target_sr: int,
    mono: bool,
    num_samples: int | None,
    shuffle: bool,
    shard_id: int,
    num_shards: int,
    layout: str | None,
):
    """Build a DALI pipeline that decodes audio files from paths.

    Returns:
        Tuple of decoded audio tensor and integer file-index label.
    """
    encoded, label = fn.readers.file(
        files=files,
        labels=list(range(len(files))),
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        seed=42,
        name="Reader",
    )
    audio, _sample_rate = fn.decoders.audio(
        encoded,
        dtype=_dali_type("FLOAT"),
        downmix=mono,
        sample_rate=target_sr,
        device="cpu",
    )
    if num_samples is not None:
        audio = fn.slice(
            audio,
            start=0,
            end=int(num_samples),
            axes=[0],
            out_of_bounds_policy="pad",
        )
        if mono and layout == "T":
            audio = fn.reshape(audio, shape=[int(num_samples)])
    if layout == "CT":
        audio = fn.transpose(audio, perm=[1, 0])
    return audio, label


@pipeline_def
def image_file_pipeline(
    *,
    files: list[str],
    shuffle: bool,
    shard_id: int,
    num_shards: int,
    layout: str | None,
):
    """Build a DALI pipeline that decodes image files from paths.

    Returns:
        Tuple of decoded image tensor and integer file-index label.
    """
    encoded, label = fn.readers.file(
        files=files,
        labels=list(range(len(files))),
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        seed=42,
        name="Reader",
    )
    image = fn.decoders.image(encoded, device="cpu")
    image = _layout_transpose(image, modality="image", layout=layout)
    return image, label


@pipeline_def
def text_file_pipeline(
    *,
    files: list[str],
    shuffle: bool,
    shard_id: int,
    num_shards: int,
):
    """Build a DALI pipeline that reads text files as byte strings.

    Returns:
        Tuple of text byte tensor and integer file-index label.
    """
    encoded, label = fn.readers.file(
        files=files,
        labels=list(range(len(files))),
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        seed=42,
        name="Reader",
    )
    text = fn.python_function(encoded, function=_decode_text_bytes, num_outputs=1)
    return text, label


@pipeline_def
def video_file_pipeline(
    *,
    files: list[str],
    shuffle: bool,
    shard_id: int,
    num_shards: int,
    sequence_length: int,
    layout: str | None,
    reader_device: str,
):
    """Build a DALI pipeline that decodes video files from paths.

    Returns:
        Tuple of decoded video tensor and integer file-index label.
    """
    video, label = cast(
        Sequence[Any],
        fn.readers.video(
            device=reader_device,
            filenames=files,
            labels=list(range(len(files))),
            random_shuffle=shuffle,
            num_shards=num_shards,
            shard_id=shard_id,
            seed=42,
            sequence_length=sequence_length,
            pad_sequences=True,
            name="Reader",
        ),
    )
    video = _layout_transpose(video, modality="video", layout=layout)
    return video, label


@pipeline_def
def numpy_file_pipeline(
    *,
    files: list[str],
    shuffle: bool,
    shard_id: int,
    num_shards: int,
):
    """Build a DALI pipeline that loads numpy array files from paths.

    Returns:
        Tuple of loaded numpy data tensor and integer file-index label.
    """
    data = fn.readers.numpy(
        files=files,
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        name="Reader",
    )
    label = fn.get_property(data, key="source_info", name="SourceInfo")
    label = fn.python_function(label, function=_path_to_index, num_outputs=1)
    return data, label


def _path_to_index(path_tensor: np.ndarray) -> np.ndarray:
    text = bytes(path_tensor.tolist()).decode("utf-8")
    return np.asarray(int(Path(text).stem), dtype=np.int64)


@dataclass
class WebDatasetComponentSpec:
    """Component specification for a WebDataset tar shard."""

    key: str
    member_ext: str
    encoding: str
    layout: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    modality: str = "generic"


@pipeline_def
def webdataset_pipeline(
    *,
    paths: list[str],
    index_paths: list[str] | None,
    shuffle: bool,
    shard_id: int,
    num_shards: int,
    components: list[WebDatasetComponentSpec],
):
    """Build a DALI pipeline that reads a WebDataset tar archive.

    Returns:
        Tuple of decoded tensors, one per component.

    Raises:
        ValueError: If a component encoding is unsupported.
    """
    ext = [component.member_ext for component in components]
    dtypes = [
        _dali_dtype(component.dtype) if component.encoding == "bin" else None
        for component in components
    ]
    reader_kwargs: dict[str, Any] = {
        "paths": paths,
        "ext": ext,
        "random_shuffle": shuffle,
        "num_shards": num_shards,
        "shard_id": shard_id,
        "name": "Reader",
    }
    if index_paths:
        reader_kwargs["index_paths"] = index_paths
    if any(dtype is not None for dtype in dtypes):
        reader_kwargs["dtypes"] = [dtype or _dali_type("UINT8") for dtype in dtypes]

    raw_outputs = fn.readers.webdataset(**reader_kwargs)
    if not isinstance(raw_outputs, (tuple, list)):
        raw_outputs = (raw_outputs,)

    processed = []
    for raw, component in zip(raw_outputs, components, strict=True):
        raw_node = cast(Any, raw)  # DALI stubs under-type dynamic WebDataset outputs.
        if component.encoding == "npy":
            node = fn.decoders.numpy(raw_node)
        elif component.encoding == "pt":
            node = fn.python_function(raw_node, function=_load_pt_tensor, num_outputs=1)
        elif component.encoding == "txt":
            node = fn.python_function(raw_node, function=_decode_text_bytes, num_outputs=1)
        elif component.encoding == "bin":
            shape = component.shape or []
            node = fn.reshape(raw_node, shape=shape)
        elif component.encoding in {"png", "jpg", "jpeg"}:
            node = fn.decoders.image(raw_node, device="cpu")
            node = _layout_transpose(node, modality="image", layout=component.layout)
        elif component.encoding == "wav":
            node, _sample_rate = fn.decoders.audio(
                raw_node, dtype=_dali_type("FLOAT"), device="cpu"
            )
            node = _layout_transpose(node, modality="audio", layout=component.layout)
            # Keep exported shape contracts stable: mono WAV decode can yield
            # (T, 1), but exported 1D waveforms are declared as (T).
            if component.shape and len(component.shape) == 1:
                node = fn.reshape(node, shape=component.shape)
        elif component.encoding == "mp4":
            sequence_length = int(component.shape[0]) if component.shape else None
            node = fn.decoders.video(raw_node, sequence_length=sequence_length)
            node = _layout_transpose(node, modality="video", layout=component.layout)
        else:
            raise ValueError(f"Unsupported WebDataset component encoding: {component.encoding}")
        processed.append(node)

    return tuple(processed)


def build_external_source_loader(
    *,
    dataset: Any,
    x_keys: Sequence[str],
    y_keys: Sequence[str],
    batch_size: int,
    num_threads: int,
    prefetch_factor: int | None,
    shuffle: bool,
    local_rank: int,
    global_rank: int,
    world_size: int,
) -> KeyedDaliIterator:
    """Build a DALI external-source loader from a Python iterable.

    Returns:
        A ``KeyedDaliIterator`` yielding ``(x, y)`` TensorDict pairs.
    """
    device_id = local_rank
    iterator = ExternalSampleIterator(
        dataset,
        x_keys=x_keys,
        y_keys=y_keys,
        num_shards=world_size,
        shard_id=global_rank,
        shuffle=shuffle,
    )
    pipeline = external_source_pipeline(
        external_source=iterator,
        num_outputs=len(x_keys) + len(y_keys) + 1,
        batch_size=batch_size,
        num_threads=max(1, num_threads),
        device_id=device_id,
        prefetch_queue_depth=prefetch_factor or 2,
    )
    pipeline.build()
    return KeyedDaliIterator(
        pipelines=[pipeline],
        x_keys=x_keys,
        y_keys=y_keys,
        metadata_labels=None,
        output_map=[*x_keys, *y_keys, INDEX_OUTPUT],
        size=iterator.n,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        last_batch_padded=True,
        auto_reset=True,
        prepare_first_batch=True,
    )


def build_native_file_loader(
    *,
    kind: str,
    files: list[str],
    x_key: str,
    metadata_labels: TensorDict | None,
    batch_size: int,
    num_threads: int,
    prefetch_factor: int | None,
    shuffle: bool,
    local_rank: int,
    global_rank: int,
    world_size: int,
    sample_layout: str | None = None,
    sample_rate: int = 16000,
    sequence_length: int | None = None,
) -> KeyedDaliIterator:
    """Build a DALI native file loader for the given modality.

    Returns:
        A ``KeyedDaliIterator`` yielding ``(x, y)`` TensorDict pairs.

    Raises:
        ValueError: If ``kind`` is unsupported or video lacks a sequence length.
    """
    device_id = local_rank
    pipeline_kwargs = {
        "files": files,
        "shuffle": shuffle,
        "shard_id": global_rank,
        "num_shards": world_size,
        "batch_size": batch_size,
        "num_threads": max(1, num_threads),
        "device_id": device_id,
        "prefetch_queue_depth": prefetch_factor or 2,
    }
    if kind == "audio":
        pipeline = audio_file_pipeline(
            target_sr=sample_rate,
            mono=sample_layout == "T",
            num_samples=sequence_length,
            layout=sample_layout,
            **pipeline_kwargs,
        )
    elif kind == "image":
        pipeline = image_file_pipeline(layout=sample_layout, **pipeline_kwargs)
    elif kind == "video":
        if sequence_length is None:
            raise ValueError("Video DALI loading requires a fixed sequence length")
        pipeline = video_file_pipeline(
            sequence_length=sequence_length,
            layout=sample_layout,
            reader_device="gpu" if device_id >= 0 else "cpu",
            **pipeline_kwargs,
        )
    elif kind == "text":
        pipeline = text_file_pipeline(**pipeline_kwargs)
    elif kind == "numpy":
        pipeline = numpy_file_pipeline(**pipeline_kwargs)
    else:
        raise ValueError(f"Unsupported native file DALI kind: {kind}")

    pipeline.build()
    return KeyedDaliIterator(
        pipelines=[pipeline],
        x_keys=[x_key],
        y_keys=[],
        metadata_labels=metadata_labels,
        output_map=[x_key, INDEX_OUTPUT],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        prepare_first_batch=True,
    )


def build_webdataset_loader(
    *,
    shard_paths: list[str],
    index_paths: list[str] | None,
    x_keys: Sequence[str],
    y_keys: Sequence[str],
    metadata_labels: TensorDict | None,
    batch_size: int,
    num_threads: int,
    prefetch_factor: int | None,
    shuffle: bool,
    local_rank: int,
    global_rank: int,
    world_size: int,
    components: list[WebDatasetComponentSpec],
) -> KeyedDaliIterator:
    """Build a DALI WebDataset loader from tar archive paths.

    Returns:
        A ``KeyedDaliIterator`` yielding ``(x, y)`` TensorDict pairs.
    """
    device_id = local_rank
    pipeline = webdataset_pipeline(
        paths=shard_paths,
        index_paths=index_paths,
        shuffle=shuffle,
        shard_id=global_rank,
        num_shards=world_size,
        components=components,
        batch_size=batch_size,
        num_threads=max(1, num_threads),
        device_id=device_id,
        prefetch_queue_depth=prefetch_factor or 2,
    )
    pipeline.build()
    return KeyedDaliIterator(
        pipelines=[pipeline],
        x_keys=x_keys,
        y_keys=y_keys,
        metadata_labels=metadata_labels,
        output_map=[component.key for component in components],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        prepare_first_batch=True,
    )
