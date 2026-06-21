"""Linear autoencoder model scenario fragments."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec


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
                    type_key="LinearEncoder",
                    keys_in=[feature_key],
                    keys_out=["latent"],
                    params={
                        "hidden_dims": hidden_dims,
                        "output_dim": latent_dim,
                        "activation": activation,
                    },
                    meta_out={"output_dim": "latent_dim"},
                ),
            ],
            "Decoder": [
                LayerSpec(
                    type_key="LinearEncoder",
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    params={
                        "hidden_dims": decoder_hidden,
                        "output_dim": input_dim,
                        "activation": activation,
                    },
                ),
            ],
            "Loss": [
                LayerSpec(
                    type_key="ReconstructionLoss",
                    keys_in=[feature_key, "reconstructed"],
                    keys_out=["reconstruction_loss"],
                    params={},
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
            type_key="AnomalyScore",
            keys_in=[feature_key, "reconstructed"],
            keys_out=["anomaly_score"],
            params={"reduction": score_reduction},
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
                type_key="LMBE",
                keys_in=["waveform"],
                keys_out=["spectrogram"],
                params={
                    "sample_rate": sample_rate,
                    "n_mels": n_mels,
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "to_db": True,
                    "normalize": True,
                },
            )
        ]
    }
    stages.update(ae.stages)
    return PipelineSpec(stages=stages)


def linear_ae_multiclass(
    input_dim: int = 128,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    num_classes: int = 5,
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
                    type_key="LinearEncoder",
                    keys_in=["features"],
                    keys_out=["latent"],
                    params={
                        "hidden_dims": hidden_dims,
                        "output_dim": latent_dim,
                        "activation": activation,
                    },
                ),
            ],
            "Decoder": [
                LayerSpec(
                    type_key="LinearEncoder",
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    params={
                        "hidden_dims": decoder_hidden,
                        "output_dim": input_dim,
                        "activation": activation,
                    },
                ),
            ],
            "Heads": [
                LayerSpec(
                    type_key="LatentClassificationHead",
                    keys_in=["latent"],
                    keys_out=["class_logits"],
                    params={
                        "num_classes": num_classes,
                    },
                ),
            ],
            "Loss": [
                LayerSpec(
                    type_key="ReconstructionLoss",
                    keys_in=["features", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                    params={},
                ),
                LayerSpec(
                    type_key="ClassificationLoss",
                    keys_in=["class_logits"],
                    keys_out=["classification_loss"],
                    params={
                        "loss_type": "cross_entropy",
                        "label_key": "class_labels",
                    },
                ),
                LayerSpec(
                    type_key="ClassificationMetrics",
                    keys_in=["class_logits"],
                    keys_out=["accuracy", "f1"],
                    params={
                        "label_key": "class_labels",
                        "metrics": ["accuracy", "f1"],
                    },
                ),
            ],
        }
    )


def linear_ae_multilabel(
    input_dim: int = 128,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    num_classes: int = 5,
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
                    type_key="LinearEncoder",
                    keys_in=["features"],
                    keys_out=["latent"],
                    params={
                        "hidden_dims": hidden_dims,
                        "output_dim": latent_dim,
                        "activation": activation,
                    },
                ),
            ],
            "Decoder": [
                LayerSpec(
                    type_key="LinearEncoder",
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    params={
                        "hidden_dims": decoder_hidden,
                        "output_dim": input_dim,
                        "activation": activation,
                    },
                ),
            ],
            "Heads": [
                LayerSpec(
                    type_key="LatentClassificationHead",
                    keys_in=["latent"],
                    keys_out=["multilabel_logits"],
                    params={
                        "num_classes": num_classes,
                    },
                ),
            ],
            "Loss": [
                LayerSpec(
                    type_key="ReconstructionLoss",
                    keys_in=["features", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                    params={},
                ),
                LayerSpec(
                    type_key="ClassificationLoss",
                    keys_in=["multilabel_logits"],
                    keys_out=["multilabel_loss"],
                    params={
                        "loss_type": "bce",
                        "label_key": "multilabel_targets",
                    },
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
                    type_key="LinearEncoder",
                    keys_in=["features"],
                    keys_out=["latent"],
                    params={
                        "hidden_dims": hidden_dims,
                        "output_dim": latent_dim,
                        "activation": activation,
                    },
                ),
            ],
            "Decoder": [
                LayerSpec(
                    type_key="LinearEncoder",
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    params={
                        "hidden_dims": decoder_hidden,
                        "output_dim": input_dim,
                        "activation": activation,
                    },
                ),
            ],
            "Heads": [
                LayerSpec(
                    type_key="LatentRegressionHead",
                    keys_in=["latent"],
                    keys_out=["regression_predictions"],
                    params={
                        "num_outputs": num_outputs,
                    },
                ),
            ],
            "Loss": [
                LayerSpec(
                    type_key="ReconstructionLoss",
                    keys_in=["features", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                    params={},
                ),
                LayerSpec(
                    type_key="RegressionLoss",
                    keys_in=["regression_predictions"],
                    keys_out=["regression_loss"],
                    params={
                        "label_key": "regression_targets",
                    },
                ),
            ],
        }
    )
