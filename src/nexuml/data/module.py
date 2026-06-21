"""Lightning DataModule for NexuML datasets."""

from __future__ import annotations

import logging

import lightning as L
import torch
from torch.utils.data import Subset, WeightedRandomSampler, random_split

from nexuml.core.types import LoaderSpec
from nexuml.data.dataset import NexuDataset
from nexuml.data.loaders import get_loader_backend

logger = logging.getLogger(__name__)


class NexuDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping a ``NexuDataset``.

    Supports:
    - random splits or metadata-driven splits
    - weighted sampling for the native torch backend
    - backend-driven loader construction via ``LoaderSpec.backend``
    """

    def __init__(
        self,
        dataset: NexuDataset,
        loader_spec: LoaderSpec | None = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        split_by_column: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.loader_spec = loader_spec or LoaderSpec()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.split_by_column = split_by_column
        self._train_ds: NexuDataset | Subset | None = None
        self._val_ds: NexuDataset | Subset | None = None
        self._test_ds: NexuDataset | Subset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            return

        if self.split_by_column and getattr(self.dataset, "meta", None) is not None:
            self._setup_from_column()
        else:
            self._setup_random_split()

    def _setup_from_column(self) -> None:
        """Split using the ``split`` column in the dataset metadata."""
        meta = self.dataset.meta
        if meta is None:
            self._setup_random_split()
            return

        if hasattr(self.dataset, "get_split"):
            self._train_ds = self.dataset.get_split("train")
            self._val_ds = self.dataset.get_split("val")
            self._test_ds = self.dataset.get_split("test")
        else:
            train_idx = meta[meta["split"] == "train"].index.tolist()
            val_idx = meta[meta["split"] == "val"].index.tolist()
            test_idx = meta[meta["split"] == "test"].index.tolist()

            self._train_ds = Subset(self.dataset, train_idx)
            self._val_ds = Subset(self.dataset, val_idx)
            self._test_ds = Subset(self.dataset, test_idx)

        logger.info(
            "Column split - train: %d, val: %d, test: %d",
            len(self._train_ds),
            len(self._val_ds),
            len(self._test_ds),
        )

    def _setup_random_split(self) -> None:
        n = len(self.dataset)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)
        n_test = n - n_train - n_val

        generator = torch.Generator().manual_seed(self.seed)
        self._train_ds, self._val_ds, self._test_ds = random_split(
            self.dataset, [n_train, n_val, n_test], generator=generator
        )

    def _make_sampler(self, dataset_split) -> WeightedRandomSampler | None:
        """Build a ``WeightedRandomSampler`` from the first label column.

        Returns:
            A ``WeightedRandomSampler`` if weighted sampling is enabled and
            label metadata is available; otherwise ``None``.
        """
        if not self.loader_spec.weighted_sampling:
            return None
        if not getattr(self.dataset, "label_names", None):
            return None

        label_col = self.dataset.label_names[0]
        dataset_meta = getattr(self.dataset, "meta", None)

        if hasattr(dataset_split, "meta") and getattr(dataset_split, "meta", None) is not None:
            labels = dataset_split.meta[label_col].values
        elif isinstance(dataset_split, Subset) and dataset_meta is not None:
            labels = dataset_meta.iloc[dataset_split.indices][label_col].values
        else:
            return None

        try:
            label_ids = [int(float(label)) for label in labels]
        except (TypeError, ValueError):
            return None

        class_counts = torch.bincount(torch.tensor(label_ids))
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        sample_weights = class_weights[label_ids].tolist()
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    def _loader(self, dataset_split, split: str, shuffle: bool = False, sampler=None):
        backend = get_loader_backend(self.loader_spec.backend)
        return backend.create_loader(
            self,
            dataset_split,
            split=split,
            shuffle=shuffle,
            sampler=sampler,
        )

    def train_dataloader(self):
        sampler = self._make_sampler(self._train_ds)
        return self._loader(
            self._train_ds,
            split="train",
            shuffle=self.loader_spec.shuffle_train and sampler is None,
            sampler=sampler,
        )

    def val_dataloader(self):
        return self._loader(self._val_ds, split="val", shuffle=False)

    def test_dataloader(self):
        return self._loader(self._test_ds, split="test", shuffle=False)

    def predict_dataloader(self):
        return self._loader(self.dataset, split="predict", shuffle=False)
