"""Linear autoencoder model scenario fragments."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec
from nexuml_library.layers.feature.lmbe import LMBE
from nexuml_library.layers.head.anomaly_score import AnomalyScore
from nexuml_library.layers.head.classification_head import LatentClassificationHead
from nexuml_library.layers.head.regression_head import LatentRegressionHead
from nexuml_library.layers.loss.classification_loss import ClassificationLoss
from nexuml_library.layers.loss.classification_metrics import ClassificationMetrics
from nexuml_library.layers.loss.reconstruction_loss import ReconstructionLoss
from nexuml_library.layers.loss.regression_loss import RegressionLoss
from nexuml_library.layers.model.linear_encoder import LinearEncoder


def linear_ae_reconstruction(
    input_dim: int = 128,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    activation: str = "torch.nn.ReLU",
    feature_key: str = "features",
) -> PipelineSpec:
    """Create a PipelineSpec for a linear autoencoder with reconstruction loss.

    Returns:
        PipelineSpec: Pipeline with linear encoder, decoder and reconstruction
            loss layers.
    """
    hidden_dims = hidden_dims or [64, 32]
    decoder_hidden = list(reversed(hidden_dims))

    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    component=LinearEncoder(
                        hidden_dims=hidden_dims,
                        output_dim=latent_dim,
                        activation=activation,
                    ),
                    keys_in=[feature_key],
                    keys_out=["latent"],
                    meta_out={"output_dim": "latent_dim"},
                ),
            ],
            "Decoder": [
                LayerSpec(
                    component=LinearEncoder(
                        hidden_dims=decoder_hidden,
                        output_dim=input_dim,
                        activation=activation,
                    ),
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                ),
            ],
            "Loss": [
                LayerSpec(
                    component=ReconstructionLoss(),
                    keys_in=[feature_key, "reconstructed"],
                    keys_out=["reconstruction_loss"],
                ),
            ],
        }
    )


def linear_ae_anomaly_detection(
    input_dim: int = 128,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    activation: str = "torch.nn.ReLU",
    feature_key: str = "features",
    score_reduction: str = "mean",
) -> PipelineSpec:
    """Linear autoencoder with reconstruction loss and an anomaly score.

    Returns:
        PipelineSpec: Pipeline with linear encoder, decoder, reconstruction
            loss and anomaly score layers.
    """
    pipeline = linear_ae_reconstruction(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation=activation,
        feature_key=feature_key,
    )
    pipeline.stages["Loss"].append(
        LayerSpec(
            component=AnomalyScore(reduction=score_reduction),
            keys_in=[feature_key, "reconstructed"],
            keys_out=["anomaly_score"],
        ),
    )
    return pipeline


def linear_ae_lmbe(
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    time_frames: int = 128,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    activation: str = "torch.nn.ReLU",
    score_reduction: str = "mean",
) -> PipelineSpec:
    """Waveform -> LMBE -> linear autoencoder with an anomaly score.

    Returns:
        PipelineSpec: Full waveform-to-reconstruction pipeline with LMBE
            feature extraction and linear autoencoder.
    """
    ae = linear_ae_anomaly_detection(
        input_dim=n_mels * time_frames,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation=activation,
        feature_key="spectrogram",
        score_reduction=score_reduction,
    )
    stages = {
        "Features": [
            LayerSpec(
                component=LMBE(
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    to_db=True,
                    normalize=True,
                ),
                keys_in=["waveform"],
                keys_out=["spectrogram"],
            )
        ]
    }
    stages.update(ae.stages)
    return PipelineSpec(stages=stages)


def linear_ae_multiclass(
    input_dim: int = 128,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    activation: str = "torch.nn.ReLU",
) -> PipelineSpec:
    """Create a PipelineSpec for a linear AE with reconstruction + classification.

    Returns:
        PipelineSpec: Pipeline with encoder, decoder, classification head,
            reconstruction loss, classification loss and metrics layers.
    """
    hidden_dims = hidden_dims or [64, 32]
    decoder_hidden = list(reversed(hidden_dims))

    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    component=LinearEncoder(
                        hidden_dims=hidden_dims,
                        output_dim=latent_dim,
                        activation=activation,
                    ),
                    keys_in=["features"],
                    keys_out=["latent"],
                ),
            ],
            "Decoder": [
                LayerSpec(
                    component=LinearEncoder(
                        hidden_dims=decoder_hidden,
                        output_dim=input_dim,
                        activation=activation,
                    ),
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                ),
            ],
            "Heads": [
                LayerSpec(
                    component=LatentClassificationHead(),
                    keys_in=["latent"],
                    keys_out=["class_logits"],
                ),
            ],
            "Loss": [
                LayerSpec(
                    component=ReconstructionLoss(),
                    keys_in=["features", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                ),
                LayerSpec(
                    component=ClassificationLoss(loss_type="cross_entropy"),
                    keys_in=["class_logits"],
                    keys_out=["classification_loss"],
                    label_key="class_labels",
                ),
                LayerSpec(
                    component=ClassificationMetrics(metrics=["accuracy", "f1"]),
                    keys_in=["class_logits"],
                    keys_out=["accuracy", "f1"],
                    label_key="class_labels",
                ),
            ],
        }
    )


def linear_ae_multilabel(
    input_dim: int = 128,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    activation: str = "torch.nn.ReLU",
) -> PipelineSpec:
    """Create a PipelineSpec for a linear AE with reconstruction + multilabel classification.

    Returns:
        PipelineSpec: Pipeline with encoder, decoder, multilabel head,
            reconstruction loss and multilabel loss layers.
    """
    hidden_dims = hidden_dims or [64, 32]
    decoder_hidden = list(reversed(hidden_dims))

    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    component=LinearEncoder(
                        hidden_dims=hidden_dims,
                        output_dim=latent_dim,
                        activation=activation,
                    ),
                    keys_in=["features"],
                    keys_out=["latent"],
                ),
            ],
            "Decoder": [
                LayerSpec(
                    component=LinearEncoder(
                        hidden_dims=decoder_hidden,
                        output_dim=input_dim,
                        activation=activation,
                    ),
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                ),
            ],
            "Heads": [
                LayerSpec(
                    component=LatentClassificationHead(),
                    keys_in=["latent"],
                    keys_out=["multilabel_logits"],
                ),
            ],
            "Loss": [
                LayerSpec(
                    component=ReconstructionLoss(),
                    keys_in=["features", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                ),
                LayerSpec(
                    component=ClassificationLoss(loss_type="bce"),
                    keys_in=["multilabel_logits"],
                    keys_out=["multilabel_loss"],
                    label_key="multilabel_targets",
                ),
            ],
        }
    )


def linear_ae_regression(
    input_dim: int = 128,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    num_outputs: int = 3,
    activation: str = "torch.nn.ReLU",
) -> PipelineSpec:
    """Create a PipelineSpec for a linear AE with reconstruction + regression.

    Returns:
        PipelineSpec: Pipeline with encoder, decoder, regression head,
            reconstruction loss and regression loss layers.
    """
    hidden_dims = hidden_dims or [64, 32]
    decoder_hidden = list(reversed(hidden_dims))

    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    component=LinearEncoder(
                        hidden_dims=hidden_dims,
                        output_dim=latent_dim,
                        activation=activation,
                    ),
                    keys_in=["features"],
                    keys_out=["latent"],
                ),
            ],
            "Decoder": [
                LayerSpec(
                    component=LinearEncoder(
                        hidden_dims=decoder_hidden,
                        output_dim=input_dim,
                        activation=activation,
                    ),
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                ),
            ],
            "Heads": [
                LayerSpec(
                    component=LatentRegressionHead(num_outputs=num_outputs),
                    keys_in=["latent"],
                    keys_out=["regression_predictions"],
                ),
            ],
            "Loss": [
                LayerSpec(
                    component=ReconstructionLoss(),
                    keys_in=["features", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                ),
                LayerSpec(
                    component=RegressionLoss(),
                    keys_in=["regression_predictions"],
                    keys_out=["regression_loss"],
                    label_key="regression_targets",
                ),
            ],
        }
    )
