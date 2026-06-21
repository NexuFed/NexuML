"""Optuna tuning file for the linear AE ToyCarEmu ASD scenario."""

from __future__ import annotations

from nexuml.core.types import ScenarioSpec, TuningSpec
from nexuml_library.scenarios.asd.linear_ae_toycaremu import linear_ae_toycaremu

SEARCH_SPACE = {
    "hidden_dims": {
        "type": "categorical",
        "choices": [[64, 32], [128, 64], [128, 64, 32]],
    },
    "latent_dim": {"type": "int", "low": 4, "high": 32},
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
    storage=".experiments/optuna/linear_ae_asd.log",
    prune=False,
)


def scenario() -> ScenarioSpec:
    """Return the default base scenario for tuning."""
    return build(
        hidden_dims=[64, 32],
        latent_dim=8,
        batch_size=64,
    )


def build(hidden_dims: list[int], latent_dim: int, batch_size: int) -> ScenarioSpec:
    """Build a scenario from sampled architectural hyperparameters.

    Returns:
        ScenarioSpec: Assembled scenario with tuning spec attached.
    """
    scenario_spec = linear_ae_toycaremu(
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        batch_size=batch_size,
        max_epochs=25,
    )
    scenario_spec.tuning = TUNING_SPEC
    return scenario_spec
