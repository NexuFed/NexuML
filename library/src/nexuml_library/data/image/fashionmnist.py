"""FashionMNIST dataset source for in-memory image experiments and benchmarks."""

from __future__ import annotations
from nexuml.core.discovery import data_source

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from tensordict import TensorDict

from nexuml.data.dataset import NexuDataset


@data_source("FashionMNISTDataset")
class FashionMNISTDataset(NexuDataset):
    """In-memory FashionMNIST dataset backed by ``self.data`` and metadata labels."""

    LABEL_NAMES = ["category"]

    def __init__(
        self,
        root: str | Path = "data/fashionmnist",
        train: bool = True,
        download: bool = False,
        data=None,
        targets: Sequence[int] | torch.Tensor | None = None,
        split: str | None = None,
    ):
        dataset_data = data
        dataset_targets = targets

        if dataset_data is None:
            from torchvision import datasets, transforms

            fashionmnist_dataset = datasets.FashionMNIST(
                root=str(root),
                train=train,
                download=download,
                transform=transforms.ToTensor(),
            )
            dataset_data = fashionmnist_dataset
            dataset_targets = getattr(fashionmnist_dataset, "targets", None)
        elif dataset_targets is None and hasattr(dataset_data, "targets"):
            dataset_targets = getattr(dataset_data, "targets")

        if dataset_targets is None:
            raise ValueError("FashionMNISTDataset requires targets when custom data is provided")

        if isinstance(dataset_targets, torch.Tensor):
            target_list = dataset_targets.tolist()
        else:
            target_list = list(dataset_targets)

        split_name = split or ("fit" if train else "test")
        meta = pd.DataFrame({"category": target_list, "split": [split_name] * len(target_list)})

        super().__init__(
            meta=meta,
            label_names=self.LABEL_NAMES,
            do_split=False,
            modality="image",
            data=dataset_data,
        )

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        x = super().load_item(idx, row)
        if "features" not in x.keys():
            return x

        features = x["features"]
        if not isinstance(features, torch.Tensor):
            return x
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if features.dtype != torch.float32:
            features = features.float()
        if torch.max(features) > 1.0:
            features = features / 255.0
        return TensorDict({"features": features}, batch_size=[])
