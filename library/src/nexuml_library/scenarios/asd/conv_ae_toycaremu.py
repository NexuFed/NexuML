"""Convolutional autoencoder ASD scenario on the ToyCarEmu DCASE 2026 machine."""

from __future__ import annotations
from nexuml.core.discovery import scenario

import math

from nexuml.core.types import ScenarioSpec
from nexuml_library.scenarios.data.dcase import MachineSpec, dcase_data
from nexuml_library.scenarios.evaluation.anomaly import (
    anomaly_evaluation_spec,
    decision_rule_spec,
)
from nexuml_library.scenarios.model.conv_ae import conv_ae_lmbe
from nexuml_library.scenarios.training.defaults import default_training


@scenario("conv-ae-toycaremu")
def conv_ae_toycaremu(
    n_mels: int = 128,
    hop_length: int = 512,
    clip_num_samples: int = 160000,
    latent_dim: int = 64,
    channel_schedule: list[int] | None = None,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 25,
) -> ScenarioSpec:
    """Convolutional AE anomaly detector trained on DCASE 2026 ToyCarEmu.

    This is a narrow, single-machine didactic preset with an explicit decision
    rule in the loss stage.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    time_frames = math.ceil(clip_num_samples / hop_length)
    pipeline = conv_ae_lmbe(
        n_mels=n_mels,
        hop_length=hop_length,
        time_frames=time_frames,
        latent_dim=latent_dim,
        channel_schedule=channel_schedule,
    )
    pipeline.stages["Loss"].extend(
        decision_rule_spec(score_key="anomaly_score", decision_key="decision")
    )

    return ScenarioSpec(
        name="conv_ae_toycaremu",
        pipeline=pipeline,
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        data=dcase_data(
            machine_specs=[MachineSpec(machine_type="ToyCarEmu", year=2026, data_type="dev")],
            clip_num_samples=clip_num_samples,
            batch_size=batch_size,
        ),
        evaluation=anomaly_evaluation_spec(
            label_key="anomaly",
            output_score_key="anomaly_score",
            decision_key="decision",
        ),
    )
