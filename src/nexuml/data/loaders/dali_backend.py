"""DALI loader backend using native readers when possible."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, cast

import torch
from tensordict import TensorDict
from torch.utils.data import WeightedRandomSampler

from nexuml.data.exported import ExportedDataset

logger = logging.getLogger(__name__)

_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
_TEXT_SUFFIXES = {".md", ".text", ".txt"}


def _check_dali_available() -> None:
    if importlib.util.find_spec("nvidia.dali") is None:
        raise ImportError(
            "DALI backend requested but nvidia-dali is not installed. "
            "Install it with: pip install nexuml[dali]"
        )


def _torch_fallback(module: Any, dataset: Any, *, split: str, shuffle: bool, sampler=None):
    from nexuml.data.loaders.torch_backend import TorchLoaderBackend

    return TorchLoaderBackend().create_loader(
        module,
        dataset,
        split=split,
        shuffle=shuffle,
        sampler=sampler,
    )


def _get_root_dataset(dataset: Any) -> Any:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _get_meta(dataset: Any):
    if hasattr(dataset, "meta") and dataset.meta is not None:
        return dataset.meta

    root = _get_root_dataset(dataset)
    meta = getattr(root, "meta", None)
    if meta is None:
        return None

    if hasattr(dataset, "indices"):
        return meta.iloc[list(dataset.indices)].reset_index(drop=True)

    return meta


def _sample_contract(dataset: Any) -> tuple[list[str], str | None, int | None]:
    root = _get_root_dataset(dataset)
    explicit_keys = getattr(root, "dali_x_keys", None)
    if explicit_keys:
        return (
            list(explicit_keys),
            getattr(root, "dali_layout", None),
            getattr(root, "dali_sequence_length", None),
        )

    if len(dataset) == 0:
        raise ValueError("Cannot infer a DALI contract from an empty dataset")
    x_sample, _ = dataset[0]
    x_keys = list(x_sample.keys())
    if len(x_keys) != 1:
        raise ValueError(f"DALI currently requires a single x key per native source, got {x_keys}")
    x_key = x_keys[0]
    value = x_sample[x_key]
    modality = str(getattr(_get_root_dataset(dataset), "modality", "generic")).lower()
    layout = _infer_layout(value, modality)
    sequence_length = _infer_sequence_length(value, modality, layout)
    return x_keys, layout, sequence_length


def _infer_layout(tensor: torch.Tensor, modality: str) -> str | None:
    shape = tuple(int(dim) for dim in tensor.shape)
    if modality == "image" and len(shape) == 3:
        if shape[0] in {1, 3, 4}:
            return "CHW"
        if shape[-1] in {1, 3, 4}:
            return "HWC"
    if modality == "video" and len(shape) == 4:
        if shape[1] in {1, 3, 4}:
            return "TCHW"
        if shape[-1] in {1, 3, 4}:
            return "THWC"
    if modality == "audio" and len(shape) == 2:
        if shape[0] <= 8 and shape[1] > shape[0]:
            return "CT"
        return "TC"
    if modality == "audio" and len(shape) == 1:
        return "T"
    return None


def _infer_sequence_length(
    tensor: torch.Tensor,
    modality: str,
    layout: str | None,
) -> int | None:
    shape = tuple(int(dim) for dim in tensor.shape)
    if modality == "video" and len(shape) == 4:
        return shape[0]
    if modality != "audio":
        return None
    if layout in {"T", "TC"} and shape:
        return shape[0]
    if layout == "CT" and len(shape) >= 2:
        return shape[1]
    return None


def _get_local_rank() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return local_rank if torch.cuda.is_available() else -1


def _get_global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _build_metadata_labels(meta, label_names: list[str]) -> TensorDict | None:
    if meta is None or not label_names:
        return None
    data = {}
    for name in label_names:
        if name not in meta.columns:
            continue
        data[name] = torch.as_tensor(meta[name].tolist())
    if not data:
        return None
    return TensorDict(cast(Any, data), batch_size=[len(meta)])


def _merged_dataset_has_consistent_keys(root_dataset: Any) -> bool:
    sources = getattr(root_dataset, "meta_data_list", None)
    if not isinstance(sources, dict) or len(sources) <= 1:
        return True
    contracts = set()
    for dataset in sources.values():
        if len(dataset) == 0:
            continue
        x_sample, _ = dataset[0]
        contracts.add(tuple(sorted(x_sample.keys())))
    return len(contracts) <= 1


class DaliLoaderBackend:
    """DALI backend that prefers native readers and falls back deliberately."""

    def create_loader(
        self,
        module: Any,
        dataset: Any,
        *,
        split: str,
        shuffle: bool = False,
        sampler: WeightedRandomSampler | None = None,
    ) -> Any:
        _check_dali_available()

        from nexuml.data.loaders.dali_multimodal import (
            INDEX_OUTPUT,
            WebDatasetComponentSpec,
            build_external_source_loader,
            build_native_file_loader,
            build_webdataset_loader,
        )

        if sampler is not None:
            logger.info("Sampler requested; falling back to torch backend for %s", split)
            return _torch_fallback(module, dataset, split=split, shuffle=shuffle, sampler=sampler)

        root_dataset = _get_root_dataset(dataset)
        if getattr(root_dataset, "supports_dali_loader", True) is False:
            logger.info(
                "Dataset does not support native DALI loading; falling back to torch for %s", split
            )
            return _torch_fallback(module, dataset, split=split, shuffle=shuffle, sampler=sampler)

        if getattr(root_dataset, "data", None) is not None:
            logger.info("In-memory dataset; falling back to torch backend for %s", split)
            return _torch_fallback(module, dataset, split=split, shuffle=shuffle, sampler=sampler)

        if not _merged_dataset_has_consistent_keys(root_dataset):
            logger.info(
                "Merged dataset uses inconsistent x-key contracts; falling back to torch for %s",
                split,
            )
            return _torch_fallback(module, dataset, split=split, shuffle=shuffle, sampler=sampler)

        loader_spec = module.loader_spec
        num_threads = max(1, loader_spec.num_workers)
        prefetch_factor = loader_spec.prefetch_factor or 2
        local_rank = _get_local_rank()
        global_rank = _get_global_rank()
        world_size = _get_world_size()

        if isinstance(root_dataset, ExportedDataset):
            metadata_labels = _build_metadata_labels(
                _get_meta(dataset), list(root_dataset.meta_label_keys)
            )
            if root_dataset.backend == "webdataset":
                components: list[WebDatasetComponentSpec] = []
                for logical_key in list(root_dataset.x_keys) + list(root_dataset.file_label_keys):
                    stored_key = root_dataset._stored_x_keys.get(
                        logical_key
                    ) or root_dataset._stored_y_keys.get(logical_key)
                    if stored_key is None:
                        continue
                    spec = root_dataset.key_specs.get(stored_key, {})
                    storage = spec.get("storage", {})
                    member_ext = storage.get("member_ext")
                    if not member_ext:
                        continue
                    components.append(
                        WebDatasetComponentSpec(
                            key=logical_key,
                            member_ext=str(member_ext),
                            encoding=str(spec.get("encoding", "npy")),
                            layout=spec.get("layout"),
                            shape=spec.get("shape"),
                            dtype=spec.get("dtype"),
                            modality=root_dataset.modality
                            if logical_key in root_dataset.x_keys
                            else "generic",
                        )
                    )
                components.append(
                    WebDatasetComponentSpec(
                        key=INDEX_OUTPUT,
                        member_ext="__index.npy",
                        encoding="npy",
                        dtype="int64",
                    )
                )
                shard_paths = [
                    str(root_dataset.root / rel_path)
                    for rel_path in cast(list[str], root_dataset.extra.get("shards", []) or [])
                ]
                index_paths = [
                    str(root_dataset.root / rel_path)
                    for rel_path in cast(list[str], root_dataset.extra.get("index_paths", []) or [])
                ] or None
                loader = build_webdataset_loader(
                    shard_paths=shard_paths,
                    index_paths=index_paths,
                    x_keys=root_dataset.x_keys,
                    y_keys=root_dataset.file_label_keys,
                    metadata_labels=metadata_labels,
                    batch_size=loader_spec.batch_size,
                    num_threads=num_threads,
                    prefetch_factor=prefetch_factor,
                    shuffle=shuffle,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    world_size=world_size,
                    components=components,
                )
                loader.metadata = _get_meta(dataset)
                return loader

            if (
                root_dataset.backend == "numpy"
                and len(root_dataset.x_keys) == 1
                and not root_dataset.file_label_keys
            ):
                stored_key = root_dataset._stored_x_keys[root_dataset.x_keys[0]]
                files = [
                    str(root_dataset.root / "data" / stored_key / f"{int(idx):08d}.npy")
                    for idx in _get_meta(dataset)["export_index"].tolist()
                ]
                loader = build_native_file_loader(
                    kind="numpy",
                    files=files,
                    x_key=root_dataset.x_keys[0],
                    metadata_labels=metadata_labels,
                    batch_size=loader_spec.batch_size,
                    num_threads=num_threads,
                    prefetch_factor=prefetch_factor,
                    shuffle=shuffle,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    world_size=world_size,
                )
                loader.metadata = _get_meta(dataset)
                return loader

            return build_external_source_loader(
                dataset=dataset,
                x_keys=root_dataset.x_keys,
                y_keys=root_dataset.y_keys,
                batch_size=loader_spec.batch_size,
                num_threads=num_threads,
                prefetch_factor=prefetch_factor,
                shuffle=shuffle,
                local_rank=local_rank,
                global_rank=global_rank,
                world_size=world_size,
            )

        meta = _get_meta(dataset)
        if meta is None or "file" not in meta.columns or meta.empty:
            logger.info("No file metadata available; falling back to torch for %s", split)
            return _torch_fallback(module, dataset, split=split, shuffle=shuffle, sampler=sampler)

        try:
            x_keys, layout, sequence_length = _sample_contract(dataset)
        except Exception as exc:
            logger.info(
                "Could not infer DALI x-key contract (%s); falling back to torch for %s", exc, split
            )
            return _torch_fallback(module, dataset, split=split, shuffle=shuffle, sampler=sampler)

        x_key = x_keys[0]
        files = [str(path) for path in meta["file"].tolist()]
        label_names = list(getattr(root_dataset, "label_names", []))
        metadata_labels = _build_metadata_labels(meta, label_names)
        modality = str(getattr(root_dataset, "modality", "audio")).lower()
        suffix = Path(files[0]).suffix.lower()

        if suffix == ".npy":
            kind = "numpy"
        elif modality == "audio" or suffix in _AUDIO_SUFFIXES:
            kind = "audio"
        elif modality == "image" or suffix in _IMAGE_SUFFIXES:
            kind = "image"
        elif modality == "video" or suffix in _VIDEO_SUFFIXES:
            kind = "video"
        elif modality == "text" or suffix in _TEXT_SUFFIXES:
            kind = "text"
        else:
            kind = "audio"

        if kind == "video" and local_rank < 0:
            logger.info("DALI video reader requires CUDA; falling back to torch for %s", split)
            return _torch_fallback(module, dataset, split=split, shuffle=shuffle, sampler=sampler)

        loader = build_native_file_loader(
            kind=kind,
            files=files,
            x_key=x_key,
            metadata_labels=metadata_labels,
            batch_size=loader_spec.batch_size,
            num_threads=num_threads,
            prefetch_factor=prefetch_factor,
            shuffle=shuffle,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            sample_layout=layout,
            sample_rate=int(getattr(root_dataset, "sample_rate", 16000)),
            sequence_length=sequence_length,
        )
        loader.metadata = meta.reset_index(drop=True)
        return loader
