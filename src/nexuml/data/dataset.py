"""Base dataset contracts and metadata-backed helpers for NexuML."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence, Sized
from typing import Any, Self, cast

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)

# Standard columns that are not labels.
_STANDARD_COLUMNS = {
    "file",
    "split",
    "dataset",
    "assignment",
    "basename",
    "modality",
    "source_dataset",
    "source_index",
    "source_type",
}

_KEEP_DATA = object()


class NexuDataset(torch.utils.data.Dataset):
    """Base dataset returning ``(x, y)`` TensorDict pairs.

    The base class supports two common patterns used throughout NexuML:

    - metadata-backed datasets via ``self.meta``
    - in-memory datasets via ``self.data``

    Subclasses can still override ``__getitem__`` entirely for bespoke behavior.
    """

    def __init__(
        self,
        meta: pd.DataFrame | None = None,
        label_names: list[str] | None = None,
        split_ratio: list[float] | None = None,
        do_split: bool = False,
        modality: str = "audio",
        data: Dataset | Sequence | torch.Tensor | np.ndarray | None = None,
    ):
        super().__init__()

        self.meta: pd.DataFrame | None = None
        self.modality = modality
        self.data = data
        self.split_ratio = split_ratio or [0.85, 0.10, 0.05]
        self.label_names: list[str] = list(label_names or [])
        self.num_classes: dict[str, int] = {}

        if meta is not None:
            self._set_meta(meta, label_names=label_names)

            if do_split:
                assert abs(sum(self.split_ratio) - 1.0) < 1e-6, "split_ratio must sum to 1"
                self.split_meta(self.split_ratio)

    def _set_meta(
        self,
        meta: pd.DataFrame,
        label_names: list[str] | None = None,
    ) -> None:
        self.meta = meta.reset_index(drop=True)

        if label_names is not None:
            self.label_names = list(label_names)
        elif not self.label_names:
            self.label_names = [
                column for column in self.meta.columns if column not in _STANDARD_COLUMNS
            ]
            if self.label_names:
                logger.warning(
                    "No label_names provided; using all non-standard columns: %s",
                    self.label_names,
                )

        self._update_num_classes()

    def _update_num_classes(self) -> None:
        self.num_classes = {}
        if self.meta is None:
            return

        for name in self.label_names:
            if name not in self.meta.columns:
                raise ValueError(
                    f"Label '{name}' not found in meta columns: {self.meta.columns.tolist()}"
                )

            value = self._first_label_value(self.meta[name])
            if value is None:
                self.num_classes[name] = 0
            elif isinstance(value, (list, np.ndarray)):
                self.num_classes[name] = len(value)
            else:
                self.num_classes[name] = int(self.meta[name].nunique())

    @staticmethod
    def _first_label_value(series: pd.Series):
        for value in series.tolist():
            if isinstance(value, (list, np.ndarray)):
                return value
            if not pd.isna(value):
                return value
        return None

    def clone_with_meta(
        self,
        meta: pd.DataFrame,
        data=_KEEP_DATA,
    ) -> Self:
        clone = copy.copy(self)
        clone._set_meta(meta, label_names=self.label_names)
        if data is not _KEEP_DATA:
            clone.data = cast(Any, data)
        return clone

    def _subset_data(self, indices: list[int]):
        if self.data is None:
            return None

        return self._subset_data_object(self.data, indices)

    @classmethod
    def _subset_data_object(cls, data, indices: list[int]):
        if isinstance(data, Dataset):
            return Subset(data, indices)
        if isinstance(data, torch.Tensor):
            return data[indices]
        if isinstance(data, np.ndarray):
            return data[indices]
        if isinstance(data, tuple):
            return tuple(cls._subset_data_object(part, indices) for part in data)
        if isinstance(data, list):
            return [data[i] for i in indices]

        try:
            return data[indices]
        except Exception:
            return [data[i] for i in indices]

    def take(self, indices: Sequence[int]) -> "NexuDataset":
        if self.meta is None:
            raise ValueError("take() requires a dataset with meta information")

        subset_indices = list(indices)
        subset_meta = self.meta.iloc[subset_indices].reset_index(drop=True)
        subset_data = self._subset_data(subset_indices)
        return self.clone_with_meta(subset_meta, data=subset_data)

    def split_meta(self, split_ratio: list[float]) -> None:
        """Convert 'fit' rows to train/val and 'all' rows to train/val/test."""
        if self.meta is None or "split" not in self.meta.columns:
            return

        meta = self.meta

        fit_mask = meta["split"] == "fit"
        if fit_mask.any():
            val_idx = meta[fit_mask].sample(frac=split_ratio[1]).index
            meta.loc[val_idx, "split"] = "val"
            meta.loc[meta["split"] == "fit", "split"] = "train"

        all_mask = meta["split"] == "all"
        if all_mask.any():
            val_idx = meta[all_mask].sample(frac=split_ratio[1]).index
            meta.loc[val_idx, "split"] = "val"
            all_mask = meta["split"] == "all"
            test_frac = split_ratio[2] / (split_ratio[0] + split_ratio[2])
            test_idx = meta[all_mask].sample(frac=test_frac).index
            meta.loc[test_idx, "split"] = "test"
            meta.loc[meta["split"] == "all", "split"] = "train"

        self.meta = meta.reset_index(drop=True)

    def get_split(self, split: str) -> "NexuDataset":
        """Return a view of this dataset filtered to a single split.

        Returns:
            A new :class:`NexuDataset` containing only the requested split.

        Raises:
            ValueError: If the dataset has no metadata.
        """
        if self.meta is None:
            raise ValueError("get_split() requires a dataset with meta information")

        indices = self.meta[self.meta["split"] == split].index.tolist()
        return self.take(indices)

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Load features for a single sample.

        Subclasses can override this to lazily load file-backed content.

        Returns:
            Feature tensor dictionary for the sample.
        """
        if self.data is not None:
            item = self.data[idx]
            if isinstance(item, (tuple, list)) and item:
                item = item[0]
            if isinstance(item, TensorDict):
                return item
            if isinstance(item, torch.Tensor):
                return TensorDict({"features": item}, batch_size=[])
            if isinstance(item, np.ndarray):
                return TensorDict({"features": torch.from_numpy(item)}, batch_size=[])
            if hasattr(item, "__array__"):
                return TensorDict(
                    {"features": torch.as_tensor(np.asarray(item))},
                    batch_size=[],
                )

        return TensorDict({}, batch_size=[])

    @staticmethod
    def _label_to_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, np.ndarray)):
            return torch.tensor(value, dtype=torch.float32)
        return torch.tensor(float(value), dtype=torch.float32)

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict | None:
        """Load labels for a single sample.

        By default labels are read from ``self.meta`` using ``self.label_names``.
        Subclasses can override this for file-backed or computed targets.

        Returns:
            Label tensor dictionary, or ``None`` if no labels are configured.
        """
        if not self.label_names:
            return None

        y_dict: dict[str, torch.Tensor] = {}
        for name in self.label_names:
            if name not in row.index:
                continue
            y_dict[name] = self._label_to_tensor(row[name])

        return TensorDict(y_dict, batch_size=[]) if y_dict else None  # ty: ignore[invalid-argument-type]

    def __len__(self) -> int:
        if self.meta is not None:
            return len(self.meta)
        if self.data is not None:
            if not isinstance(self.data, Sized):
                raise TypeError(f"data of type {type(self.data).__name__} does not support len()")
            return len(self.data)
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[TensorDict, TensorDict | None]:
        if self.meta is None:
            raise NotImplementedError

        row = self.meta.iloc[index]
        x = self.load_item(index, row)
        x["sample_index"] = torch.tensor(index, dtype=torch.long)
        y = self.load_labels(index, row)
        return x, y

    def download(self) -> None:
        """Override to implement dataset downloading."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement download()")
