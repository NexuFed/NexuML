"""Research-only smoke recipe, using native NexuML training."""

import lightning as L
import torch

from nexuml import nn_module
from nexuml.core.discovery import scenario
from nexuml.core.types import (
    DataSpec,
    EvalAlgorithmSpec,
    EvaluationSpec,
    LayerSpec,
    PipelineSpec,
    ScenarioSpec,
    TargetSpec,
    TrainingSpec,
)

from agent_demo.components import PairedVectors, RootMeanSquaredError, SquaredError


@scenario("agent-demo-regression")
def regression() -> ScenarioSpec:
    """Compose a CPU research smoke with no final-test partition.

    Returns:
        A native NexuML scenario using user-owned data/loss definitions.
    """
    L.seed_everything(7, workers=True)
    return ScenarioSpec(
        name="agent-demo-regression",
        data=DataSpec(
            source=PairedVectors(),
            input_shapes={"features": [4]},
            targets=[TargetSpec(type="regression", key="target", num_outputs=4)],
            train_split=0.75,
            val_split=0.25,
            test_split=0.0,
        ),
        pipeline=PipelineSpec(
            stages={
                "predict": [
                    LayerSpec(
                        component=nn_module(torch.nn.Linear, in_features=4, out_features=4),
                        keys_in=["features"],
                        keys_out=["prediction"],
                    )
                ],
                "objective": [
                    LayerSpec(
                        component=SquaredError(),
                        keys_in=["prediction"],
                        keys_out=["mse"],
                        label_key="target",
                    )
                ],
            }
        ),
        training=TrainingSpec(
            loss_keys={"mse": 1.0},
            max_epochs=1,
            batch_size=8,
            accelerator="cpu",
            devices=1,
        ),
        evaluation=EvaluationSpec(
            algorithms=[
                EvalAlgorithmSpec(
                    algorithm=RootMeanSquaredError(),
                    feature_key="prediction",
                    label_key="target",
                    enabled=False,
                )
            ]
        ),
    )
