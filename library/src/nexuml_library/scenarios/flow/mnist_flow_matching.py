"""MNIST Flow Matching reference scenario."""

from __future__ import annotations

from nexuml.core.discovery import scenario
from nexuml.core.types import EvalAlgorithmSpec, EvaluationSpec, ScenarioSpec
from nexuml_library.evaluation.visualizers.flow import FlowVisualizer
from nexuml_library.scenarios.model.flow_matching import unet_flow_matching
from nexuml_library.scenarios.training.defaults import default_logging, default_training
from nexuml_library.scenarios.vision.mnist_resnet import mnist_data


@scenario("mnist-flow-matching")
def mnist_flow_matching(
    base_channels: int = 32,
    time_embedding_dim: int = 128,
    num_steps: int = 300,
    lr: float = 1e-4,
    batch_size: int = 128,
    max_epochs: int = 100,
    download: bool = True,
) -> ScenarioSpec:
    """Build an unconditional MNIST Flow Matching scenario.

    Returns:
        MNIST Flow Matching scenario.
    """
    return ScenarioSpec(
        name="mnist_flow_matching",
        pipeline=unet_flow_matching(
            base_channels=base_channels,
            time_embedding_dim=time_embedding_dim,
            num_steps=num_steps,
        ),
        evaluation=EvaluationSpec(
            algorithms=[
                EvalAlgorithmSpec(
                    algorithm=FlowVisualizer(
                        max_samples=64,
                        image_range="minus_one_one",
                    ),
                    feature_key="features",
                )
            ]
        ),
        logging=default_logging(
            experiment_name="Flow Matching",
            run_name="mnist-flow-matching",
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"flow_matching_loss": 1.0},
        ),
        data=mnist_data(download=download, normalize=True),
    )
