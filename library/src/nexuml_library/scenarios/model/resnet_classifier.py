"""ResNet classification model scenario fragments."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec


def resnet_classifier(
    num_classes: int = 10,
    resnet_type: str = "resnet18",
    pretrained: bool = False,
    cifar_stem: bool | None = None,
    label_key: str = "class_labels",
) -> PipelineSpec:
    """Create a PipelineSpec for a ResNet image classifier.

    Pipeline stages:
    - Encoder: ResNet backbone producing embeddings
    - Head: LatentClassificationHead producing logits
    - Loss: ClassificationLoss + ClassificationMetrics

    Returns:
        PipelineSpec: Pipeline with ResNet encoder, classification head and
            loss layers.
    """
    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    type_key="ResNet",
                    keys_in=["features"],
                    keys_out=["embeddings"],
                    params={
                        "resnet_type": resnet_type,
                        "pretrained": pretrained,
                        "cifar_stem": cifar_stem,
                    },
                ),
            ],
            "Head": [
                LayerSpec(
                    type_key="LatentClassificationHead",
                    keys_in=["embeddings"],
                    keys_out=["class_logits"],
                    params={"num_classes": num_classes},
                ),
            ],
            "Loss": [
                LayerSpec(
                    type_key="ClassificationLoss",
                    keys_in=["class_logits"],
                    keys_out=["classification_loss"],
                    params={
                        "loss_type": "cross_entropy",
                        "label_key": label_key,
                    },
                ),
                LayerSpec(
                    type_key="ClassificationMetrics",
                    keys_in=["class_logits"],
                    keys_out=["accuracy", "f1"],
                    params={
                        "label_key": label_key,
                        "metrics": ["accuracy", "f1"],
                    },
                ),
            ],
        }
    )
