"""ResNet classification model scenario fragments."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec
from nexuml_library.layers.head.classification_head import LatentClassificationHead
from nexuml_library.layers.loss.classification_loss import ClassificationLoss
from nexuml_library.layers.loss.classification_metrics import ClassificationMetrics
from nexuml_library.layers.model.resnet import ResNet


def resnet_classifier(
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
                    component=ResNet(
                        resnet_type=resnet_type,
                        pretrained=pretrained,
                        cifar_stem=cifar_stem,
                    ),
                    keys_in=["features"],
                    keys_out=["embeddings"],
                ),
            ],
            "Head": [
                LayerSpec(
                    component=LatentClassificationHead(),
                    keys_in=["embeddings"],
                    keys_out=["class_logits"],
                ),
            ],
            "Loss": [
                LayerSpec(
                    component=ClassificationLoss(loss_type="cross_entropy"),
                    keys_in=["class_logits"],
                    keys_out=["classification_loss"],
                    label_key=label_key,
                ),
                LayerSpec(
                    component=ClassificationMetrics(metrics=["accuracy", "f1"]),
                    keys_in=["class_logits"],
                    keys_out=["accuracy", "f1"],
                    label_key=label_key,
                ),
            ],
        }
    )
