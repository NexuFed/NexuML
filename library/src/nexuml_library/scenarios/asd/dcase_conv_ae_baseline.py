"""DCASE convolutional autoencoder scenarios."""

from __future__ import annotations
from nexuml.core.discovery import scenario

import math

from nexuml.core.types import ScenarioSpec
from nexuml_library.scenarios.data.dcase import MachineSpec, dcase_data
from nexuml_library.scenarios.evaluation.anomaly import anomaly_evaluation_spec
from nexuml_library.scenarios.model.conv_ae import conv_ae_lmbe, conv_cvae_lmbe
from nexuml_library.scenarios.training.defaults import default_training


@scenario("dcase-conv-ae")
def dcase_conv_ae(
    machine_types: list[str] | None = None,
    machine_specs: list[MachineSpec] | None = None,
    download: bool = False,
    n_mels: int = 128,
    hop_length: int = 512,
    clip_num_samples: int = 160000,
    latent_dim: int = 64,
    channel_schedule: list[int] | None = None,
    lr: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 25,
) -> ScenarioSpec:
    """DCASE convolutional autoencoder anomaly workflow.

    Machine year and data_type come from the built-in DCASE catalog when
    ``machine_types`` is given. Use ``machine_specs`` for explicit multi-year
    control. ``time_frames`` is computed from ``clip_num_samples / hop_length``.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    time_frames = math.ceil(clip_num_samples / hop_length)
    return ScenarioSpec(
        name="dcase_conv_ae",
        pipeline=conv_ae_lmbe(
            n_mels=n_mels,
            time_frames=time_frames,
            latent_dim=latent_dim,
            channel_schedule=channel_schedule,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        data=dcase_data(
            machine_types=machine_types,
            machine_specs=machine_specs,
            download=download,
            clip_num_samples=clip_num_samples,
            batch_size=batch_size,
        ),
        evaluation=anomaly_evaluation_spec(
            label_key="anomaly",
            output_score_key="anomaly_score",
        ),
    )


@scenario("dcase-conv-cvae")
def dcase_conv_cvae(
    machine_types: list[str] | None = None,
    machine_specs: list[MachineSpec] | None = None,
    download: bool = False,
    n_mels: int = 128,
    hop_length: int = 512,
    clip_num_samples: int = 160000,
    encoder_dim: int = 128,
    latent_dim: int = 32,
    beta: float = 1e-3,
    channel_schedule: list[int] | None = None,
    lr: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 25,
) -> ScenarioSpec:
    """DCASE convolutional variational autoencoder anomaly workflow.

    Machine year and data_type come from the built-in DCASE catalog when
    ``machine_types`` is given. ``time_frames`` is computed automatically.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    time_frames = math.ceil(clip_num_samples / hop_length)
    return ScenarioSpec(
        name="dcase_conv_cvae",
        pipeline=conv_cvae_lmbe(
            n_mels=n_mels,
            time_frames=time_frames,
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
        data=dcase_data(
            machine_types=machine_types,
            machine_specs=machine_specs,
            download=download,
            clip_num_samples=clip_num_samples,
            batch_size=batch_size,
        ),
        evaluation=anomaly_evaluation_spec(
            label_key="anomaly",
            output_score_key="anomaly_score",
        ),
    )
