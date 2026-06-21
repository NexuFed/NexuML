"""Composed CIFAR + ResNet classification scenarios."""

from __future__ import annotations
from nexuml.core.discovery import scenario

from nexuml.core.types import ScenarioSpec
from nexuml_library.scenarios.data.cifar import cifar10_data, cifar100_data
from nexuml_library.scenarios.evaluation.base import (
    classification_evaluation,
)
from nexuml_library.scenarios.model.resnet_classifier import resnet_classifier
from nexuml_library.scenarios.training.defaults import default_training


@scenario("cifar-resnet")
def cifar_resnet(
    dataset: str = "cifar10",
    download: bool = True,
    resnet_type: str = "resnet18",
    pretrained: bool = False,
    cifar_stem: bool | None = None,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    """CIFAR image classification with ResNet backbone.

    Args:
        dataset: ``"cifar10"`` or ``"cifar100"``.
        download: Auto-download dataset if not present.
        resnet_type: ResNet variant (e.g. ``"resnet18"``).
        pretrained: Load pretrained ImageNet weights.
        cifar_stem: Use CIFAR-friendly stem. Defaults to ``True``
            when ``pretrained=False``.
        lr: Learning rate.
        batch_size: Training batch size.
        max_epochs: Maximum training epochs.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and
            evaluation.

    Raises:
        ValueError: If ``dataset`` is not ``"cifar10"`` or ``"cifar100"``.
    """
    if dataset == "cifar10":
        data = cifar10_data(
            download=download,
        )
        num_classes = 10
    elif dataset == "cifar100":
        data = cifar100_data(
            download=download,
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'cifar10' or 'cifar100'.")

    return ScenarioSpec(
        name="cifar_resnet",
        pipeline=resnet_classifier(
            num_classes=num_classes,
            resnet_type=resnet_type,
            pretrained=pretrained,
            cifar_stem=cifar_stem,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"classification_loss": 1.0},
            metric_keys=["accuracy", "f1"],
        ),
        data=data,
        evaluation=classification_evaluation(),
    )
