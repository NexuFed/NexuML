"""Convolutional autoencoder model scenario fragments."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec
from nexuml_library.layers.feature.lmbe import LMBE
from nexuml_library.layers.head.anomaly_score import AnomalyScore
from nexuml_library.layers.loss.reconstruction_loss import ReconstructionLoss
from nexuml_library.layers.model.conv_autoencoder import (
    ConvolutionalDecoder,
    ConvolutionalEncoder,
    VariationalLatent,
)


def spectrogram_conv_ae(
    input_shape: tuple[int, int, int] = (1, 64, 64),
    latent_dim: int = 64,
    channel_schedule: list[int] | None = None,
    activation: str = "relu",
    score_reduction: str = "mean",
) -> PipelineSpec:
    """2D convolutional autoencoder with reconstruction loss and anomaly score.

    Returns:
        PipelineSpec: Pipeline with encoder, decoder, reconstruction loss and
            anomaly score layers.
    """
    channel_schedule = channel_schedule or [16, 32, 64]
    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    component=ConvolutionalEncoder(
                        output_dim=latent_dim,
                        channel_schedule=channel_schedule,
                        activation=activation,
                    ),
                    keys_in=["spectrogram"],
                    keys_out=["latent"],
                    meta_out={
                        "decoder_shape": "decoder_shape",
                    },
                ),
            ],
            "Decoder": [
                LayerSpec(
                    component=ConvolutionalDecoder(
                        output_shape=input_shape,
                        channel_schedule=channel_schedule,
                        activation=activation,
                    ),
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    meta_in={"decoder_shape": "decoder_shape"},
                ),
            ],
            "Loss": [
                LayerSpec(
                    component=ReconstructionLoss(),
                    keys_in=["spectrogram", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                ),
                LayerSpec(
                    component=AnomalyScore(reduction=score_reduction),
                    keys_in=["spectrogram", "reconstructed"],
                    keys_out=["anomaly_score"],
                ),
            ],
        }
    )


def spectrogram_conv_cvae(
    input_shape: tuple[int, int, int] = (1, 64, 64),
    encoder_dim: int = 128,
    latent_dim: int = 32,
    beta: float = 1.0,
    channel_schedule: list[int] | None = None,
    activation: str = "relu",
    score_reduction: str = "mean",
) -> PipelineSpec:
    """2D convolutional variational autoencoder with anomaly score.

    Returns:
        PipelineSpec: Pipeline with encoder, variational latent, decoder,
            reconstruction loss and anomaly score layers.
    """
    channel_schedule = channel_schedule or [16, 32, 64]
    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    component=ConvolutionalEncoder(
                        output_dim=encoder_dim,
                        channel_schedule=channel_schedule,
                        activation=activation,
                    ),
                    keys_in=["spectrogram"],
                    keys_out=["encoded"],
                    meta_out={"decoder_shape": "decoder_shape"},
                ),
                LayerSpec(
                    component=VariationalLatent(latent_dim=latent_dim, beta=beta),
                    keys_in=["encoded"],
                    keys_out=["latent", "latent_mu", "latent_logvar", "kl_loss"],
                ),
            ],
            "Decoder": [
                LayerSpec(
                    component=ConvolutionalDecoder(
                        output_shape=input_shape,
                        channel_schedule=channel_schedule,
                        activation=activation,
                    ),
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    meta_in={"decoder_shape": "decoder_shape"},
                ),
            ],
            "Loss": [
                LayerSpec(
                    component=ReconstructionLoss(),
                    keys_in=["spectrogram", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                ),
                LayerSpec(
                    component=AnomalyScore(reduction=score_reduction),
                    keys_in=["spectrogram", "reconstructed"],
                    keys_out=["anomaly_score"],
                ),
            ],
        }
    )


def conv_ae_lmbe(
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    time_frames: int = 128,
    latent_dim: int = 64,
    channel_schedule: list[int] | None = None,
    activation: str = "relu",
) -> PipelineSpec:
    """Waveform -> LMBE -> convolutional AE.

    Returns:
        PipelineSpec: Full waveform-to-reconstruction pipeline with LMBE
            feature extraction and convolutional autoencoder.
    """
    ae = spectrogram_conv_ae(
        input_shape=(1, n_mels, time_frames),
        latent_dim=latent_dim,
        channel_schedule=channel_schedule,
        activation=activation,
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


def conv_cvae_lmbe(
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    time_frames: int = 128,
    encoder_dim: int = 128,
    latent_dim: int = 32,
    beta: float = 1.0,
    channel_schedule: list[int] | None = None,
    activation: str = "relu",
) -> PipelineSpec:
    """Waveform -> LMBE -> convolutional CVAE.

    Returns:
        PipelineSpec: Full waveform-to-reconstruction pipeline with LMBE
            feature extraction and convolutional variational autoencoder.
    """
    cvae = spectrogram_conv_cvae(
        input_shape=(1, n_mels, time_frames),
        encoder_dim=encoder_dim,
        latent_dim=latent_dim,
        beta=beta,
        channel_schedule=channel_schedule,
        activation=activation,
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
    stages.update(cvae.stages)
    return PipelineSpec(stages=stages)
