"""CIFAR data scenario fragments."""

from __future__ import annotations

from nexuml.core.types import DataSpec, DatasetSpec
from nexuml_library.scenarios.data.roots import resolve_data_root


def cifar10_data(
    root: str = "cifar10",
    download: bool = True,
) -> DataSpec:
    """Create a DataSpec for CIFAR-10 image classification.

    Returns:
        DataSpec: CIFAR-10 dataset specification.
    """
    resolved_root = resolve_data_root(root)
    return DataSpec(
        source_type="cifar10",
        datasets=[
            DatasetSpec(
                type_key="CIFAR10Dataset",
                params={"root": str(resolved_root), "train": True, "download": download},
                modality="image",
                split_type="all",
            ),
        ],
        input_shapes={"features": [3, 32, 32]},
        num_classes=10,
        feature_key="features",
    )


def cifar100_data(
    root: str = "cifar100",
    download: bool = True,
) -> DataSpec:
    """Create a DataSpec for CIFAR-100 image classification.

    Returns:
        DataSpec: CIFAR-100 dataset specification.
    """
    resolved_root = resolve_data_root(root)
    return DataSpec(
        source_type="cifar100",
        datasets=[
            DatasetSpec(
                type_key="CIFAR100Dataset",
                params={"root": str(resolved_root), "train": True, "download": download},
                modality="image",
                split_type="all",
            ),
        ],
        input_shapes={"features": [3, 32, 32]},
        num_classes=100,
        feature_key="features",
    )
