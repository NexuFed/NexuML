"""Declarative public SO-101 cloning: nexuml train --scenario-file this_file.py.

Set NEXUML_SO101_DATA_ROOT to the staged pinned LeRobot snapshot and
NEXUML_SO101_OUTPUT_ROOT to a new owned output directory. This is offline block
stacking, not the simulator-specific SO-101 joint-reach recipe.
"""

import os
from pathlib import Path

from nexuml.core.types import (
    CallbackSpec,
    DataSpec,
    DatasetSpec,
    ExportSpec,
    LayerSpec,
    LoaderSpec,
    LoggingSpec,
    PipelineSpec,
    ScenarioSpec,
    TargetSpec,
    TensorBoardSpec,
    TrainingSpec,
)
from nexuml_library.data.lerobot import LeRobotDataSource
from nexuml_library.layers.head.regression_head import LatentRegressionHead
from nexuml_library.layers.loss.regression_loss import RegressionLoss


def scenario() -> ScenarioSpec:
    """Declare a bounded state-to-command fit with episode-disjoint held-out data.

    Returns:
        An ordinary supervised GPU scenario, without a native policy contract.

    Raises:
        ValueError: If output paths are relative or would modify the dataset.
    """
    data = Path(os.environ["NEXUML_SO101_DATA_ROOT"])
    output = Path(os.environ["NEXUML_SO101_OUTPUT_ROOT"])
    if not data.is_absolute() or not output.is_absolute() or output.is_relative_to(data):
        raise ValueError("use absolute data/output roots; outputs must not modify the dataset")
    return ScenarioSpec(
        name="public-so101-block-stack-bc",
        data=DataSpec(
            datasets=[
                DatasetSpec(
                    source=LeRobotDataSource(
                        root=str(data),
                        repo_id="villekuosmanen/armnetbench_block_stack",
                        split_episodes={
                            "train": tuple(range(40)),
                            "val": tuple(range(40, 45)),
                            "test": tuple(range(45, 50)),
                        },
                    ),
                    split_type="keep",
                    modality="robot",
                )
            ],
            input_shapes={"observation.state": [6]},
            feature_key="observation.state",
            targets=[TargetSpec(type="regression", key="action", num_outputs=6)],
            loader=LoaderSpec(batch_size=32, num_workers=0),
        ),
        pipeline=PipelineSpec(
            stages={
                "Actor": [
                    LayerSpec(
                        component=LatentRegressionHead(num_outputs=6, hidden_dims=[]),
                        keys_in=["observation.state"],
                        keys_out=["predicted_action"],
                    )
                ],
                "Loss": [
                    LayerSpec(
                        component=RegressionLoss(),
                        keys_in=["predicted_action"],
                        keys_out=["imitation_loss"],
                        label_key="action",
                    )
                ],
            }
        ),
        training=TrainingSpec(
            max_steps=64,
            max_epochs=100,
            batch_size=32,
            lr=0.01,
            loss_keys={"imitation_loss": 1.0},
            accelerator="gpu",
            devices=1,
            seed=42,
            deterministic=True,
        ),
        logging=LoggingSpec(
            tensorboard=TensorBoardSpec(log_dir=str(output / "logs/tb")),
            diagram=None,
            temp_artifact_dir=str(output / "logs/staging"),
        ),
        callbacks=[
            CallbackSpec(
                factory="lightning.pytorch.callbacks:ModelCheckpoint",
                kwargs={
                    "dirpath": str(output / "checkpoints"),
                    "every_n_train_steps": 32,
                    "save_top_k": -1,
                    "save_last": True,
                    "filename": "step-{step}",
                },
            )
        ],
        exports=[ExportSpec(output=str(output / "model"))],
    )
