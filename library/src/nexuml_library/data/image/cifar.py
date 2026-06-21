"""CIFAR dataset sources for image classification experiments."""

from __future__ import annotations
from nexuml.core.discovery import data_source

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from tensordict import TensorDict

from nexuml.data.dataset import NexuDataset


@data_source("CIFAR10Dataset")
class CIFAR10Dataset(NexuDataset):
    """CIFAR-10 dataset backed by ``self.data`` and metadata labels."""

    LABEL_NAMES = ["class_labels"]
    NUM_CLASSES = 10

    def __init__(
        self,
        root: str | Path = "data/cifar10",
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

            cifar_dataset = datasets.CIFAR10(
                root=str(root),
                train=train,
                download=download,
                transform=transforms.ToTensor(),
            )
            dataset_data = cifar_dataset
            dataset_targets = getattr(cifar_dataset, "targets", None)
        elif dataset_targets is None and hasattr(dataset_data, "targets"):
            dataset_targets = getattr(dataset_data, "targets")

        if dataset_targets is None:
            raise ValueError("CIFAR10Dataset requires targets when custom data is provided")

        if isinstance(dataset_targets, torch.Tensor):
            target_list = dataset_targets.tolist()
        else:
            target_list = list(dataset_targets)

        split_name = split or ("fit" if train else "test")
        meta = pd.DataFrame({"class_labels": target_list, "split": [split_name] * len(target_list)})

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

        # Ensure shape is [3, 32, 32]
        if features.dim() == 3 and features.shape[0] != 3:
            # Handle [H, W, C] -> [C, H, W]
            if features.shape[-1] == 3:
                features = features.permute(2, 0, 1)

        if features.dtype != torch.float32:
            features = features.float()

        if torch.max(features) > 1.0:
            features = features / 255.0

        return TensorDict({"features": features}, batch_size=[])


@data_source("CIFAR100Dataset")
class CIFAR100Dataset(CIFAR10Dataset):
    """CIFAR-100 dataset backed by ``self.data`` and metadata labels."""

    NUM_CLASSES = 100

    def __init__(
        self,
        root: str | Path = "data/cifar100",
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

            cifar_dataset = datasets.CIFAR100(
                root=str(root),
                train=train,
                download=download,
                transform=transforms.ToTensor(),
            )
            dataset_data = cifar_dataset
            dataset_targets = getattr(cifar_dataset, "targets", None)
        elif dataset_targets is None and hasattr(dataset_data, "targets"):
            dataset_targets = getattr(dataset_data, "targets")

        if dataset_targets is None:
            raise ValueError("CIFAR100Dataset requires targets when custom data is provided")

        if isinstance(dataset_targets, torch.Tensor):
            target_list = dataset_targets.tolist()
        else:
            target_list = list(dataset_targets)

        split_name = split or ("fit" if train else "test")
        meta = pd.DataFrame({"class_labels": target_list, "split": [split_name] * len(target_list)})

        NexuDataset.__init__(
            self,
            meta=meta,
            label_names=["class_labels"],
            do_split=False,
            modality="image",
            data=dataset_data,
        )
