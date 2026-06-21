"""Optuna tuning file for the convolutional AE ToyCarEmu ASD scenario."""

from __future__ import annotations

from nexuml.core.types import ScenarioSpec, TuningSpec
from nexuml_library.scenarios.asd.conv_ae_toycaremu import conv_ae_toycaremu

SEARCH_SPACE = {
    "channel_schedule": {
        "type": "categorical",
        "choices": [[16, 32, 64], [32, 64, 128], [16, 32, 64, 128]],
    },
    "latent_dim": {"type": "int", "low": 16, "high": 128},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "training.lr": {
        "type": "float",
        "low": 1e-4,
        "high": 1e-2,
        "log": True,
    },
}

TUNING_SPEC = TuningSpec(
    n_trials=20,
    directions=["maximize"],
    metric_key="anomaly_eval/omega",
    storage=".experiments/optuna/conv_ae_asd.log",
    prune=False,
)


def scenario() -> ScenarioSpec:
    """Return the default base scenario for tuning."""
    return build(
        channel_schedule=[16, 32, 64],
        latent_dim=64,
        batch_size=32,
    )


def build(channel_schedule: list[int], latent_dim: int, batch_size: int) -> ScenarioSpec:
    """Build a scenario from sampled architectural hyperparameters.

    Returns:
        ScenarioSpec: Assembled scenario with tuning spec attached.
    """
    scenario_spec = conv_ae_toycaremu(
        channel_schedule=channel_schedule,
        latent_dim=latent_dim,
        batch_size=batch_size,
        max_epochs=25,
    )
    scenario_spec.tuning = TUNING_SPEC
    return scenario_spec
