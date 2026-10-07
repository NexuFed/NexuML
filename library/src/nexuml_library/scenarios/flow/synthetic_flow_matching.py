"""Synthetic reference scenario for Flow Matching."""

from __future__ import annotations

from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data
from nexuml_library.scenarios.model.flow_matching import mlp_flow_matching
from nexuml_library.scenarios.training.defaults import default_training


@scenario("synthetic-flow-matching")
def synthetic_flow_matching(
    feature_shape: tuple[int, ...] = (2,),
    num_samples: int = 2048,
    num_clusters: int = 4,
    hidden_dims: list[int] | None = None,
    time_embedding_dim: int = 16,
    noise_std: float = 1.0,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 20,
    seed: int = 42,
) -> ScenarioSpec:
    """Train a small continuous flow on clustered synthetic vectors.

    Returns:
        ScenarioSpec: Self-contained Flow Matching reference scenario.
    """
    return ScenarioSpec(
        name="synthetic_flow_matching",
        pipeline=mlp_flow_matching(
            hidden_dims=hidden_dims or [128, 128],
            time_embedding_dim=time_embedding_dim,
            noise_std=noise_std,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"flow_matching_loss": 1.0},
        ),
        data=synthetic_vector_data(
            feature_shape=feature_shape,
            num_samples=num_samples,
            num_clusters=num_clusters,
            seed=seed,
        ),
    )
