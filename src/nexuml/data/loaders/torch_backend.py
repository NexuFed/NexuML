"""Native PyTorch DataLoader backend."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, WeightedRandomSampler


def _collate_tensordict(batch):
    """Collate ``(TensorDict, TensorDict | None)`` pairs into batches.

    Returns:
        Tuple of batched feature and label ``TensorDict``.
    """
    xs, ys = zip(*batch)
    x_td = {key: torch.stack([x[key] for x in xs]) for key in xs[0].keys()}
    x_out = TensorDict(x_td, batch_size=[len(xs)])  # ty: ignore[invalid-argument-type]

    if ys[0] is not None:
        y_td = {key: torch.stack([y[key] for y in ys]) for key in ys[0].keys()}
        y_out = TensorDict(y_td, batch_size=[len(ys)])  # ty: ignore[invalid-argument-type]
    else:
        y_out = None

    return x_out, y_out


class TorchLoaderBackend:
    """Native PyTorch dataloader backend."""

    def create_loader(
        self,
        module,
        dataset,
        *,
        split: str,
        shuffle: bool = False,
        sampler: WeightedRandomSampler | None = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=module.loader_spec.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=module.loader_spec.num_workers,
            collate_fn=_collate_tensordict,
            drop_last=False,
            persistent_workers=module.loader_spec.persistent_workers
            and module.loader_spec.num_workers > 0,
            prefetch_factor=module.loader_spec.prefetch_factor
            if module.loader_spec.num_workers > 0
            else None,
        )
