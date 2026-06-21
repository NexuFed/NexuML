"""Convolutional autoencoder on AudioSet with a latent classification head."""

from __future__ import annotations
from nexuml.core.discovery import scenario

from nexuml.core.types import LayerSpec, ScenarioSpec
from nexuml_library.scenarios.data.audioset import audioset_data
from nexuml_library.scenarios.evaluation.base import classification_evaluation
from nexuml_library.scenarios.model.conv_ae import conv_ae_lmbe
from nexuml_library.scenarios.training.defaults import default_training


@scenario("audioset-conv-ae-clshead")
def audioset_conv_ae_clshead(
    data_root: str = "audioset_hf/full",
    download: bool = False,
    sample_rate: int = 16000,
    clip_num_samples: int = 160000,
    n_mels: int = 128,
    hop_length: int = 512,
    latent_dim: int = 64,
    channel_schedule: list[int] | None = None,
    num_classes: int = 527,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 10,
    validate_layout: bool = False,
) -> ScenarioSpec:
    """Conv AE trained on AudioSet waveforms with reconstruction + classification.

    A ``LatentClassificationHead`` is added on top of the encoder latent space,
    and both reconstruction and classification losses are optimized jointly.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    time_frames = clip_num_samples // hop_length
    pipeline = conv_ae_lmbe(
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        time_frames=time_frames,
        latent_dim=latent_dim,
        channel_schedule=channel_schedule,
    )
    pipeline.stages.setdefault("Heads", []).append(
        LayerSpec(
            type_key="LatentClassificationHead",
            keys_in=["latent"],
            keys_out=["class_logits"],
            params={"num_classes": num_classes},
        )
    )
    pipeline.stages["Loss"].extend(
        [
            LayerSpec(
                type_key="ClassificationLoss",
                keys_in=["class_logits"],
                keys_out=["classification_loss"],
                params={
                    "loss_type": "cross_entropy",
                    "label_key": "class",
                },
            ),
            LayerSpec(
                type_key="ClassificationMetrics",
                keys_in=["class_logits"],
                keys_out=["accuracy", "f1"],
                params={
                    "label_key": "class",
                    "metrics": ["accuracy", "f1"],
                },
            ),
        ]
    )

    return ScenarioSpec(
        name="audioset_conv_ae_clshead",
        pipeline=pipeline,
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0, "classification_loss": 1.0},
            metric_keys=["accuracy", "f1"],
        ),
        data=audioset_data(
            data_root=data_root,
            download=download,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            batch_size=batch_size,
            num_classes=num_classes,
            validate_layout=validate_layout,
        ),
        evaluation=classification_evaluation(label_key="class", feature_key="latent"),
    )
