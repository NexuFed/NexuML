"""Data loading utilities."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split

from nexuml.data.dataset import NexuDataset


def create_dataloaders(
    dataset: NexuDataset,
    batch_size: int = 64,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Split dataset and create train/val/test dataloaders.

    Returns:
        Dictionary mapping split names to ``DataLoader`` instances.
    """
    n = len(dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    def _collate(batch):
        xs, ys = zip(*batch)
        x_td = {k: torch.stack([x[k] for x in xs]) for k in xs[0].keys()}

        if ys[0] is not None:
            y_td = {k: torch.stack([y[k] for y in ys]) for k in ys[0].keys()}
        else:
            y_td = None

        from tensordict import TensorDict

        x_out = TensorDict(x_td, batch_size=[len(xs)])  # ty: ignore[invalid-argument-type]
        y_out = TensorDict(y_td, batch_size=[len(ys)]) if y_td else None  # ty: ignore[invalid-argument-type]
        return x_out, y_out

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate,
            drop_last=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
            drop_last=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
            drop_last=False,
        ),
    }
