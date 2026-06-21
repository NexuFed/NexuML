"""MNIST and FashionMNIST ResNet classification scenarios."""

from __future__ import annotations
from nexuml.core.discovery import scenario

from nexuml.core.types import DataSpec, DatasetSpec, ScenarioSpec
from nexuml_library.scenarios.data.roots import resolve_data_root
from nexuml_library.scenarios.evaluation.base import classification_evaluation
from nexuml_library.scenarios.model.resnet_classifier import resnet_classifier
from nexuml_library.scenarios.training.defaults import default_training


def mnist_data(download: bool = True, root: str = "mnist") -> DataSpec:
    """Create a DataSpec for MNIST image classification.

    Returns:
        DataSpec: MNIST dataset specification with fit and test splits.
    """
    resolved_root = resolve_data_root(root)
    return DataSpec(
        source_type="mnist",
        datasets=[
            DatasetSpec(
                type_key="MNISTDataset",
                params={"root": str(resolved_root), "train": True, "download": download},
                modality="image",
                split_type="fit",
            ),
            DatasetSpec(
                type_key="MNISTDataset",
                params={"root": str(resolved_root), "train": False, "download": download},
                modality="image",
                split_type="test",
            ),
        ],
        input_shapes={"features": [1, 28, 28]},
        num_classes=10,
        feature_key="features",
    )


def fashionmnist_data(download: bool = True, root: str = "fashionmnist") -> DataSpec:
    """Create a DataSpec for FashionMNIST image classification.

    Returns:
        DataSpec: FashionMNIST dataset specification with fit and test splits.
    """
    resolved_root = resolve_data_root(root)
    return DataSpec(
        source_type="fashionmnist",
        datasets=[
            DatasetSpec(
                type_key="FashionMNISTDataset",
                params={"root": str(resolved_root), "train": True, "download": download},
                modality="image",
                split_type="fit",
            ),
            DatasetSpec(
                type_key="FashionMNISTDataset",
                params={"root": str(resolved_root), "train": False, "download": download},
                modality="image",
                split_type="test",
            ),
        ],
        input_shapes={"features": [1, 28, 28]},
        num_classes=10,
        feature_key="features",
    )


@scenario("mnist-resnet")
def mnist_resnet(
    download: bool = True,
    resnet_type: str = "resnet18",
    pretrained: bool = False,
    cifar_stem: bool = True,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    """MNIST image classification with ResNet backbone.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    return _image_resnet_scenario(
        name="mnist_resnet",
        data=mnist_data(download=download),
        label_key="digit",
        resnet_type=resnet_type,
        pretrained=pretrained,
        cifar_stem=cifar_stem,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
    )


@scenario("fashionmnist-resnet")
def fashionmnist_resnet(
    download: bool = True,
    resnet_type: str = "resnet18",
    pretrained: bool = False,
    cifar_stem: bool = True,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    """FashionMNIST image classification with ResNet backbone.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    return _image_resnet_scenario(
        name="fashionmnist_resnet",
        data=fashionmnist_data(download=download),
        label_key="category",
        resnet_type=resnet_type,
        pretrained=pretrained,
        cifar_stem=cifar_stem,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
    )


def _image_resnet_scenario(
    name: str,
    data: DataSpec,
    label_key: str,
    resnet_type: str,
    pretrained: bool,
    cifar_stem: bool,
    lr: float,
    batch_size: int,
    max_epochs: int,
) -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        pipeline=resnet_classifier(
            num_classes=10,
            resnet_type=resnet_type,
            pretrained=pretrained,
            cifar_stem=cifar_stem,
            label_key=label_key,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"classification_loss": 1.0},
            metric_keys=["accuracy", "f1"],
        ),
        data=data,
        evaluation=classification_evaluation(label_key=label_key, feature_key="embeddings"),
    )
