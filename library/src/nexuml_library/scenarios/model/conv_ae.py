"""Convolutional autoencoder model scenario fragments."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec


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
                    type_key="ConvolutionalEncoder",
                    keys_in=["spectrogram"],
                    keys_out=["latent"],
                    params={
                        "output_dim": latent_dim,
                        "channel_schedule": channel_schedule,
                        "activation": activation,
                    },
                    meta_out={
                        "decoder_shape": "decoder_shape",
                    },
                ),
            ],
            "Decoder": [
                LayerSpec(
                    type_key="ConvolutionalDecoder",
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    params={
                        "output_shape": input_shape,
                        "channel_schedule": channel_schedule,
                        "activation": activation,
                    },
                    meta_in={"decoder_shape": "decoder_shape"},
                ),
            ],
            "Loss": [
                LayerSpec(
                    type_key="ReconstructionLoss",
                    keys_in=["spectrogram", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                    params={},
                ),
                LayerSpec(
                    type_key="AnomalyScore",
                    keys_in=["spectrogram", "reconstructed"],
                    keys_out=["anomaly_score"],
                    params={"reduction": score_reduction},
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
                    type_key="ConvolutionalEncoder",
                    keys_in=["spectrogram"],
                    keys_out=["encoded"],
                    params={
                        "output_dim": encoder_dim,
                        "channel_schedule": channel_schedule,
                        "activation": activation,
                    },
                    meta_out={"decoder_shape": "decoder_shape"},
                ),
                LayerSpec(
                    type_key="VariationalLatent",
                    keys_in=["encoded"],
                    keys_out=["latent", "latent_mu", "latent_logvar", "kl_loss"],
                    params={"latent_dim": latent_dim, "beta": beta},
                ),
            ],
            "Decoder": [
                LayerSpec(
                    type_key="ConvolutionalDecoder",
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    params={
                        "output_shape": input_shape,
                        "channel_schedule": channel_schedule,
                        "activation": activation,
                    },
                    meta_in={"decoder_shape": "decoder_shape"},
                ),
            ],
            "Loss": [
                LayerSpec(
                    type_key="ReconstructionLoss",
                    keys_in=["spectrogram", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                    params={},
                ),
                LayerSpec(
                    type_key="AnomalyScore",
                    keys_in=["spectrogram", "reconstructed"],
                    keys_out=["anomaly_score"],
                    params={"reduction": score_reduction},
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
    stages.update(cvae.stages)
    return PipelineSpec(stages=stages)
