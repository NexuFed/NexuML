"""Windowed loader backend for tensor-shard exports."""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence, cast

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset, WeightedRandomSampler

from nexuml.data.export.tensor_shards import TensorShardsBackend

logger = logging.getLogger(__name__)


def _identity(value: Any) -> Any:
    """Return a pre-batched item unchanged."""
    return value


class _TensorShardWindowDataset(IterableDataset[tuple[TensorDict, TensorDict | None]]):
    """Stream pre-batched tensors from split-specific shard windows."""

    def __init__(
        self,
        *,
        dataset: Any,
        split: str,
        batch_size: int,
        shards_per_window: int,
        prefetch_windows: int,
        prefetch_workers: int,
        shuffle_shards: bool,
        shuffle_samples: bool,
        pin_memory: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        super().__init__()

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if shards_per_window <= 0:
            raise ValueError("shards_per_window must be positive")
        if shards_per_window == 1 and (shuffle_shards or shuffle_samples):
            raise ValueError("shards_per_window must be > 1 when shuffling is enabled")
        if prefetch_windows < 0:
            raise ValueError("prefetch_windows must be non-negative")
        if prefetch_windows == 0:
            prefetch_workers = 0
        elif prefetch_workers <= 0:
            raise ValueError("prefetch_workers must be positive when prefetch_windows > 0")

        if getattr(dataset, "backend", None) != "tensor_shards":
            raise TypeError(
                "The tensor_shards loader requires an ExportedDataset with backend='tensor_shards'"
            )
        if not hasattr(dataset, "root"):
            raise TypeError("The tensor_shards loader requires dataset.root")
        if getattr(dataset, "meta", None) is None:
            raise ValueError("The tensor_shards loader requires dataset metadata")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                raise NotImplementedError(
                    "Distributed tensor-shard partitioning is not implemented yet"
                )

        self.source_dataset = dataset
        self.meta = dataset.meta.reset_index(drop=True)
        self.root = Path(dataset.root)
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.shards_per_window = int(shards_per_window)
        self.prefetch_windows = int(prefetch_windows)
        self.prefetch_workers = int(prefetch_workers)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_samples = bool(shuffle_samples)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        manifest = TensorShardsBackend.load_manifest(self.root)
        all_entries = cast(list[dict[str, Any]], manifest["shards"])
        if self.split == "predict":
            self._entries = list(all_entries)
        else:
            self._entries = [entry for entry in all_entries if str(entry["split"]) == self.split]

        self._num_samples = sum(int(entry["num_samples"]) for entry in self._entries)
        if self._num_samples != len(self.meta):
            raise ValueError(
                f"Manifest contains {self._num_samples} samples for split "
                f"{self.split!r}, but the dataset metadata contains {len(self.meta)}"
            )

        if "export_index" not in self.meta.columns:
            raise ValueError("Tensor-shard metadata must contain an export_index column")

        export_indices = [int(value) for value in self.meta["export_index"].tolist()]
        self._export_to_local = {
            export_index: local_index for local_index, export_index in enumerate(export_indices)
        }
        if len(self._export_to_local) != len(export_indices):
            raise ValueError("Duplicate export_index values in dataset metadata")

        self._stored_x_keys = dict(dataset._stored_x_keys)
        self._stored_y_keys = dict(dataset._stored_y_keys)
        self._meta_label_keys = list(dataset.meta_label_keys)

    def __len__(self) -> int:
        if self.drop_last:
            return self._num_samples // self.batch_size
        return math.ceil(self._num_samples / self.batch_size)

    def __iter__(self) -> Iterator[tuple[TensorDict, TensorDict | None]]:
        epoch = self._epoch
        self._epoch += 1

        entries = list(self._entries)
        random_generator = random.Random(self.seed + epoch)
        if self.shuffle_shards:
            random_generator.shuffle(entries)

        windows = [
            entries[start : start + self.shards_per_window]
            for start in range(0, len(entries), self.shards_per_window)
        ]
        sample_generator = torch.Generator().manual_seed(self.seed + epoch)

        if self.prefetch_windows == 0:
            for window_entries in windows:
                features, export_indices = self._load_window(window_entries)
                yield from self._iter_batches(
                    features,
                    export_indices,
                    generator=sample_generator,
                )
            return

        pending: deque[Future[tuple[dict[str, torch.Tensor], torch.Tensor]]] = deque()
        next_window = 0
        executor = ThreadPoolExecutor(max_workers=self.prefetch_workers)

        try:
            while next_window < len(windows) and len(pending) < self.prefetch_windows + 1:
                pending.append(executor.submit(self._load_window, windows[next_window]))
                next_window += 1

            while pending:
                features, export_indices = pending.popleft().result()

                while next_window < len(windows) and len(pending) < self.prefetch_windows:
                    pending.append(executor.submit(self._load_window, windows[next_window]))
                    next_window += 1

                yield from self._iter_batches(
                    features,
                    export_indices,
                    generator=sample_generator,
                )
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def _load_window(
        self,
        entries: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        feature_chunks: dict[str, list[torch.Tensor]] = {}
        index_chunks: list[torch.Tensor] = []
        expected_keys: set[str] | None = None

        for entry in entries:
            shard = TensorShardsBackend.load_shard(self.root, entry)
            if str(shard["split"]) != str(entry["split"]):
                raise ValueError(
                    f"Shard split mismatch for {entry['path']}: "
                    f"{shard['split']!r} != {entry['split']!r}"
                )

            valid_count = int(shard["num_samples"])
            features = cast(dict[str, torch.Tensor], shard["features"])
            keys = set(features)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(f"Shard {entry['path']} contains inconsistent feature keys")

            indices = cast(torch.Tensor, shard["indices"])[:valid_count]
            if indices.numel() != valid_count or bool((indices < 0).any()):
                raise ValueError(f"Shard {entry['path']} contains invalid real indices")
            index_chunks.append(indices.to(dtype=torch.long, device="cpu"))

            for key, tensor in features.items():
                feature_chunks.setdefault(str(key), []).append(tensor[:valid_count])

        if not index_chunks:
            return {}, torch.empty(0, dtype=torch.long)

        merged = {key: torch.cat(chunks, dim=0) for key, chunks in feature_chunks.items()}
        merged_indices = torch.cat(index_chunks, dim=0)

        if self.pin_memory and torch.cuda.is_available():
            merged = {key: tensor.pin_memory() for key, tensor in merged.items()}
            merged_indices = merged_indices.pin_memory()

        return merged, merged_indices

    def _iter_batches(
        self,
        features: dict[str, torch.Tensor],
        export_indices: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> Iterator[tuple[TensorDict, TensorDict | None]]:
        window_size = int(export_indices.shape[0])
        if window_size == 0:
            return

        order = torch.randperm(window_size, generator=generator) if self.shuffle_samples else None
        stop = window_size - (window_size % self.batch_size) if self.drop_last else window_size

        for start in range(0, stop, self.batch_size):
            end = min(start + self.batch_size, stop)

            if order is None:
                batch_features = {key: tensor[start:end] for key, tensor in features.items()}
                batch_export_indices = export_indices[start:end]
            else:
                positions = order[start:end]
                batch_features = {
                    key: tensor.index_select(0, positions) for key, tensor in features.items()
                }
                batch_export_indices = export_indices.index_select(0, positions)

            local_indices = self._local_indices(batch_export_indices)
            batch_len = int(local_indices.shape[0])

            x_payload = {
                logical_key: batch_features[stored_key]
                for logical_key, stored_key in self._stored_x_keys.items()
            }
            x_payload["sample_index"] = local_indices
            x = TensorDict(x_payload, batch_size=[batch_len])

            y_payload = {
                logical_key: batch_features[stored_key]
                for logical_key, stored_key in self._stored_y_keys.items()
            }
            if self._meta_label_keys:
                rows = self.meta.iloc[local_indices.tolist()]
                for key in self._meta_label_keys:
                    tensors = [
                        self.source_dataset._label_to_tensor(value) for value in rows[key].tolist()
                    ]
                    y_payload[key] = torch.stack(tensors)

            y = TensorDict(y_payload, batch_size=[batch_len]) if y_payload else None
            yield x, y

    def _local_indices(self, export_indices: torch.Tensor) -> torch.Tensor:
        try:
            values = [
                self._export_to_local[int(export_index)] for export_index in export_indices.tolist()
            ]
        except KeyError as exc:
            raise KeyError(
                f"Export index {exc.args[0]} is not present in split {self.split!r} metadata"
            ) from exc
        return torch.tensor(values, dtype=torch.long)


class TensorShardWindowLoader(DataLoader):
    """PyTorch DataLoader that receives already-batched shard-window items."""

    def __init__(
        self,
        *,
        dataset: Any,
        split: str,
        batch_size: int,
        shards_per_window: int = 6,
        prefetch_windows: int = 2,
        prefetch_workers: int = 2,
        shuffle_shards: bool = False,
        shuffle_samples: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        window_dataset = _TensorShardWindowDataset(
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            shards_per_window=shards_per_window,
            prefetch_windows=prefetch_windows,
            prefetch_workers=prefetch_workers,
            shuffle_shards=shuffle_shards,
            shuffle_samples=shuffle_samples,
            pin_memory=pin_memory,
            drop_last=drop_last,
            seed=seed,
        )
        super().__init__(
            window_dataset,
            batch_size=None,
            num_workers=0,
            collate_fn=_identity,
            pin_memory=False,
        )


class TensorShardsLoaderBackend:
    """Create split-specific loaders for tensor-shard exports."""

    def create_loader(
        self,
        module: Any,
        dataset: Any,
        *,
        split: str,
        shuffle: bool = False,
        sampler: WeightedRandomSampler | None = None,
    ) -> TensorShardWindowLoader:
        if sampler is not None:
            raise ValueError("Weighted sampling is not supported by the tensor_shards loader")

        batch_size = module.loader_spec.batch_size
        if batch_size is None:
            raise ValueError("tensor_shards requires loader_spec.batch_size")

        if module.loader_spec.num_workers != 0:
            logger.warning(
                "tensor_shards manages prefetch internally; loader_spec.num_workers is ignored"
            )

        params = module.loader_spec.params
        return TensorShardWindowLoader(
            dataset=dataset,
            split=split,
            batch_size=int(batch_size),
            shards_per_window=int(params.get("shards_per_window", 6)),
            prefetch_windows=int(params.get("prefetch_windows", 2)),
            prefetch_workers=int(params.get("prefetch_workers", 2)),
            shuffle_shards=bool(params.get("shuffle_shards", shuffle)),
            shuffle_samples=bool(params.get("shuffle_samples", shuffle)),
            pin_memory=bool(params.get("pin_memory", False)),
            drop_last=bool(params.get("drop_last", False)),
            seed=int(params.get("seed", module.seed)),
        )
