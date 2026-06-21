"""Composed scenarios combining data + model + training + evaluation."""

from __future__ import annotations
from nexuml.core.discovery import scenario

import math

from nexuml.core.types import ScenarioSpec, TargetSpec
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data
from nexuml_library.scenarios.evaluation.base import (
    classification_evaluation,
    reconstruction_evaluation,
    regression_evaluation,
)
from nexuml_library.scenarios.model.linear_ae import (
    linear_ae_multiclass,
    linear_ae_multilabel,
    linear_ae_reconstruction,
    linear_ae_regression,
)
from nexuml_library.scenarios.training.defaults import default_training


@scenario("synthetic-linear-ae-reconstruction")
def synthetic_linear_ae_reconstruction(
    feature_shape: tuple[int, ...] = (128,),
    num_samples: int = 1000,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    """Synthetic vector reconstruction with linear autoencoder.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    input_dim = math.prod(feature_shape)
    hidden_dims = hidden_dims or [64, 32]

    return ScenarioSpec(
        name="synthetic_linear_ae_reconstruction",
        pipeline=linear_ae_reconstruction(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        data=synthetic_vector_data(
            feature_shape=feature_shape,
            num_samples=num_samples,
        ),
        evaluation=reconstruction_evaluation(),
    )


@scenario("synthetic-linear-ae-multiclass")
def synthetic_linear_ae_multiclass(
    feature_shape: tuple[int, ...] = (128,),
    num_samples: int = 1000,
    num_classes: int = 5,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    """Synthetic vector reconstruction + multiclass classification.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    input_dim = math.prod(feature_shape)
    hidden_dims = hidden_dims or [64, 32]

    return ScenarioSpec(
        name="synthetic_linear_ae_multiclass",
        pipeline=linear_ae_multiclass(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_classes=num_classes,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0, "classification_loss": 1.0},
        ),
        data=synthetic_vector_data(
            feature_shape=feature_shape,
            num_samples=num_samples,
            num_clusters=num_classes,
            targets=[
                TargetSpec(type="multiclass", key="class_labels", num_classes=num_classes),
            ],
        ),
        evaluation=classification_evaluation(),
    )


@scenario("synthetic-linear-ae-multilabel")
def synthetic_linear_ae_multilabel(
    feature_shape: tuple[int, ...] = (128,),
    num_samples: int = 1000,
    num_classes: int = 5,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    """Synthetic vector reconstruction + multilabel classification.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    input_dim = math.prod(feature_shape)
    hidden_dims = hidden_dims or [64, 32]

    return ScenarioSpec(
        name="synthetic_linear_ae_multilabel",
        pipeline=linear_ae_multilabel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_classes=num_classes,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0, "multilabel_loss": 1.0},
        ),
        data=synthetic_vector_data(
            feature_shape=feature_shape,
            num_samples=num_samples,
            targets=[
                TargetSpec(type="multilabel", key="multilabel_targets", num_classes=num_classes),
            ],
        ),
        evaluation=classification_evaluation(),
    )


@scenario("synthetic-linear-ae-regression")
def synthetic_linear_ae_regression(
    feature_shape: tuple[int, ...] = (128,),
    num_samples: int = 1000,
    num_outputs: int = 3,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    """Synthetic vector reconstruction + regression.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    input_dim = math.prod(feature_shape)
    hidden_dims = hidden_dims or [64, 32]

    return ScenarioSpec(
        name="synthetic_linear_ae_regression",
        pipeline=linear_ae_regression(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_outputs=num_outputs,
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"reconstruction_loss": 1.0, "regression_loss": 1.0},
        ),
        data=synthetic_vector_data(
            feature_shape=feature_shape,
            num_samples=num_samples,
            targets=[
                TargetSpec(type="regression", key="regression_targets", num_outputs=num_outputs),
            ],
        ),
        evaluation=regression_evaluation(),
    )
