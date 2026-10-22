"""Native lithium-ion capacity forecasts using PatchFormer or RUL-Mamba.

Targets, splits and units match the single-step capacity forecasting task
of their source publications. Requires converted per-cell capacity CSVs;
this is not an RUL-to-EOL rollout or an exact paper benchmark reproduction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback, optimizer
from nexuml.core.discovery import scenario
from nexuml.core.types import (
    DataSpec,
    DatasetSpec,
    EvalAlgorithmSpec,
    EvaluationSpec,
    LayerSpec,
    LoaderSpec,
    PipelineSpec,
    ScenarioSpec,
)
from nexuml_library.data.maintenance.battery_capacity import BatteryCapacityWindows
from nexuml_library.evaluation.maintenance.battery_capacity import (
    BatteryCapacityEvaluator,
    BatteryCapacitySMAPE,
)
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_advanced import AdvancedRULForecaster
from nexuml_library.scenarios.training.defaults import default_logging, default_training


def _battery_capacity_scenario(
    kind: Literal["patchformer", "rul_mamba"],
) -> ScenarioSpec:
    """Forecast observed NASA battery capacity using its original Ah target.

    Args:
        kind: Paper architecture used for univariate future capacity prediction.

    Returns:
        Native NexuML capacity forecasting scenario using a held-out battery.
    """
    length, width, dropout, lr, batch_size = (
        (30, 30, 0.1, 1e-3, 16) if kind == "patchformer"
        else (16, 48, 0.0615, 0.0022, 16)
    )
    run_name = f"battery-nasa-{kind}-capacity-one-step"
    root = (os.environ.get("NEXUML_BATTERY_CAPACITY_ROOT") or "").strip()
    dataset = BatteryCapacityWindows(
        root=root or "data/battery/nasa",
        lookback=length,
        train_cells=["B0006", "B0007", "B0018"],
        test_cell="B0005",
        end_of_life_ah=1.4,
    )
    checkpoint_root = (
        (os.environ.get("NEXUML_RUL_CHECKPOINT_ROOT") or "").strip()
        or ".experiments/rul/checkpoints"
    )
    checkpoint_dir = Path(checkpoint_root).expanduser().resolve() / run_name
    return ScenarioSpec(
        name=run_name,
        data=DataSpec(
            datasets=[
                DatasetSpec(
                    source=dataset,
                    modality="sensor",
                    split_type="keep",
                )
            ],
            loader=LoaderSpec(num_workers=0, shuffle_train=True),
            feature_key="capacity_history",
            input_shapes={"capacity_history": [length, 1]},
        ),
        pipeline=PipelineSpec(
            stages={
                "RULModel": [
                    LayerSpec(
                        component=AdvancedRULForecaster(
                            architecture=kind,
                            hidden_dim=width,
                            dropout=dropout,
                            patch_size=5,
                            attention_heads=2,
                            decoder_depth=1,
                            initialization_seed=42,
                        ),
                        keys_in=["capacity_history"],
                        keys_out=["predicted_capacity"],
                    ),
                ],
                "RULLoss": [
                    LayerSpec(
                        component=BatteryCapacitySMAPE(),
                        keys_in=["predicted_capacity"],
                        keys_out=["capacity_loss"],
                        label_key="future_capacity",
                    )
                ],
            }
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=200,
            loss_keys={"capacity_loss": 1.0},
            metric_keys=[],
        ).model_copy(
            update={"optimizer": optimizer(torch.optim.Adam, lr=lr)}
        ),
        evaluation=EvaluationSpec(
            metrics=[],
            algorithms=[
                EvalAlgorithmSpec(
                    name="capacity_forecast",
                    algorithm=BatteryCapacityEvaluator(),
                    feature_key="predicted_capacity",
                    label_key="future_capacity",
                )
            ],
            test_result_metrics="all",
        ),
        logging=default_logging(
            experiment_name="NEX-233 Battery Capacity Forecasting",
            run_name=run_name,
            use_mlflow=False,
        ),
        callbacks=[
            callback(RULSeedOnFitStart, seed=42),
            callback(
                ModelCheckpoint,
                dirpath=str(checkpoint_dir),
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
            callback(EarlyStopping, monitor="val/loss", patience=20, mode="min"),
            callback(
                Goel2026RestoreBestWeights,
                output_file=f".experiments/rul/audits/{run_name}.checkpoint.json",
            ),
        ],
    )


@scenario("maintenance-battery-patchformer-nasa")
def maintenance_battery_patchformer_nasa() -> ScenarioSpec:
    """Train the PatchFormer DPAN/FAN on real next-cycle NASA battery capacities.

    Returns:
        Native NexuML capacity prediction experiment with battery B0005 held out.
    """
    return _battery_capacity_scenario("patchformer")


@scenario("maintenance-battery-rul-mamba-nasa")
def maintenance_battery_rul_mamba_nasa() -> ScenarioSpec:
    """Train RUL-Mamba FAN-SSM-GRN on observed NASA battery capacity sequences.

    Returns:
        Native NexuML capacity prediction experiment with B0005 held out.
    """
    return _battery_capacity_scenario("rul_mamba")
