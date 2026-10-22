"""Native DMM-JA scenario using genuine labeled industrial bimodal signals."""

from __future__ import annotations

import os
from pathlib import Path

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
from nexuml_library.data.maintenance.bimodal_rul import LabeledBimodalRULWindows
from nexuml_library.evaluation.maintenance.rul_evaluator import RULBenchmarkEvaluator
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026PlateauLR,
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_advanced import AdvancedRULForecaster
from nexuml_library.layers.model.rul_forecaster import RULRegressionObjective
from nexuml_library.scenarios.training.defaults import default_logging, default_training


@scenario("maintenance-rul-dmm-ja-bimodal")
def maintenance_rul_dmm_ja_bimodal() -> ScenarioSpec:
    """Apply DMM-JA to real temperature/load measurements with explicit RUL.

    The source must provide asset-disjoint splits and genuine RUL labels.
    ETDataset is not claimed to include these labels automatically.

    Returns:
        Native NexuML MSE-regression DMM-JA architecture scenario.
    """
    run_name = "industrial-dmm-ja-bimodal-labeled-rul"
    path = (
        (os.environ.get("NEXUML_RUL_BIMODAL_CSV") or "").strip()
        or "data/maintenance/bimodal_rul.csv"
    )
    dataset = LabeledBimodalRULWindows(
        path=path,
        window=24,
        normalizer_file=f".experiments/rul/prepared/{run_name}.normalizer.json",
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
                    source=dataset, modality="sensor", split_type="keep"
                )
            ],
            loader=LoaderSpec(num_workers=0, shuffle_train=True),
            feature_key="rul_history",
            input_shapes={"rul_history": [24, 2]},
        ),
        pipeline=PipelineSpec(
            stages={
                "RULModel": [
                    LayerSpec(
                        component=AdvancedRULForecaster(
                            architecture="dmm_ja",
                            hidden_dim=64,
                            dropout=0.15,
                            jump_threshold=2.5,
                            initialization_seed=42,
                        ),
                        keys_in=["rul_history"],
                        keys_out=["predicted_rul"],
                    )
                ],
                "RULLoss": [
                    LayerSpec(
                        component=RULRegressionObjective(objective="mse"),
                        keys_in=["predicted_rul"],
                        keys_out=["rul_loss"],
                        label_key="rul",
                    )
                ],
            }
        ),
        training=default_training(
            lr=2e-4,
            batch_size=64,
            max_epochs=700,
            loss_keys={"rul_loss": 1.0},
            metric_keys=[],
        ).model_copy(
            update={"optimizer": optimizer(torch.optim.Adam, lr=2e-4)}
        ),
        evaluation=EvaluationSpec(
            metrics=[],
            algorithms=[
                EvalAlgorithmSpec(
                    algorithm=RULBenchmarkEvaluator(target_scale=1.0),
                    feature_key="predicted_rul",
                    label_key="rul",
                )
            ],
            test_result_metrics="all",
        ),
        logging=default_logging(
            experiment_name="NEX-233 Labeled Industrial DMM-JA",
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
            callback(EarlyStopping, monitor="val/loss", patience=35, mode="min"),
            callback(Goel2026PlateauLR, factor=0.7, patience=5),
            callback(
                Goel2026RestoreBestWeights,
                output_file=f".experiments/rul/audits/{run_name}.checkpoint.json",
            ),
        ],
    )
