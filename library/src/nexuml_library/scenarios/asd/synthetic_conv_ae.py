"""Synthetic convolutional autoencoder scenarios."""

from __future__ import annotations
from nexuml.core.discovery import scenario

from nexuml.core.types import ScenarioSpec, TargetSpec
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data
from nexuml_library.scenarios.model.conv_ae import spectrogram_conv_ae, spectrogram_conv_cvae
from nexuml_library.scenarios.training.defaults import default_training


@scenario("synthetic-conv-ae-anomaly")
def synthetic_conv_ae_anomaly(
    input_shape: tuple[int, int, int] = (1, 64, 64),
    num_samples: int = 160,
    anomaly_fraction: float = 0.2,
    latent_dim: int = 64,
    channel_schedule: list[int] | None = None,
    lr: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 2,
) -> ScenarioSpec:
    """Synthetic spectrogram convolutional AE anomaly workflow.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    return ScenarioSpec(
        name="synthetic_conv_ae_anomaly",
        pipeline=spectrogram_conv_ae(
            input_shape=input_shape,
            latent_dim=latent_dim,
            channel_schedule=channel_schedule,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        data=synthetic_vector_data(
            feature_shape=input_shape,
            feature_key="spectrogram",
            num_samples=num_samples,
            targets=[
                TargetSpec(
                    type="anomaly",
                    key="anomaly_label",
                    positive_fraction=anomaly_fraction,
                )
            ],
        ),
    )


@scenario("synthetic-conv-cvae-anomaly")
def synthetic_conv_cvae_anomaly(
    input_shape: tuple[int, int, int] = (1, 64, 64),
    num_samples: int = 160,
    anomaly_fraction: float = 0.2,
    encoder_dim: int = 128,
    latent_dim: int = 32,
    beta: float = 1e-3,
    channel_schedule: list[int] | None = None,
    lr: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 2,
) -> ScenarioSpec:
    """Synthetic spectrogram convolutional VAE anomaly workflow.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    return ScenarioSpec(
        name="synthetic_conv_cvae_anomaly",
        pipeline=spectrogram_conv_cvae(
            input_shape=input_shape,
            encoder_dim=encoder_dim,
            latent_dim=latent_dim,
            beta=beta,
            channel_schedule=channel_schedule,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0, "kl_loss": 1.0},
        ),
        data=synthetic_vector_data(
            feature_shape=input_shape,
            feature_key="spectrogram",
            num_samples=num_samples,
            targets=[
                TargetSpec(
                    type="anomaly",
                    key="anomaly_label",
                    positive_fraction=anomaly_fraction,
                )
            ],
        ),
    )
