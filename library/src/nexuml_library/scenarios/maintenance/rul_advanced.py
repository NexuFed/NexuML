"""Native NexuML comparison scenarios for three advanced RUL architectures.

The papers address distinct signals/tasks. These scenarios fairly test their
architectural ingredients on C-MAPSS FD001; they are NOT publication-protocol
reproductions on ETDataset, CALCE, NASA battery or Oxford/TJU capacities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback, optimizer
from nexuml.core.discovery import scenario
from nexuml.core.types import EvalAlgorithmSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.evaluation.maintenance.rul_evaluator import RULBenchmarkEvaluator
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_advanced import AdvancedRULForecaster
from nexuml_library.scenarios.maintenance.rul import (
    _LU2025_SENSORS,
    maintenance_rul_benchmark,
)
from nexuml.core.types import LayerSpec


def _advanced_scenario(
    kind: Literal["dmm_ja", "patchformer", "rul_mamba"],
) -> ScenarioSpec:
    """Construct one paper-structured, direct-RUL architecture comparison.

    Args:
        kind: Architecture whose original modeling blocks are adapted to RUL.

    Returns:
        Isolated native NexuML FD001 scenario, including per-engine audit.

    Raises:
        ValueError: If native auto-download was replaced by an incompatible source.
    """
    config = {
        "dmm_ja": (24, 64, 0.15, 64, 700, 35, 2e-4),
        "patchformer": (30, 64, 0.10, 16, 150, 20, 1e-3),
        "rul_mamba": (16, 48, 0.0615, 16, 200, 20, 0.0022),
    }
    window, width, dropout, batch_size, epochs, patience, lr = config[kind]
    base = maintenance_rul_benchmark(
        architecture="lstm",
        objective="mse",
        cmapss_window=window,
        cmapss_sensor_ids=_LU2025_SENSORS,
        cmapss_profile=f"advanced-{kind}-fd001",
        cmapss_rul_cap=130.0,
        cmapss_padding_mode="zero",
        cmapss_train_full_windows_only=True,
        cmapss_normalize_targets=False,
        hidden_dim=width,
        depth=2,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=patience,
        seed=42,
    )
    source = base.data.datasets[0].source
    if not isinstance(source, AutoCMapSSRULWindows):
        raise ValueError("Advanced FD001 scenario needs native C-MAPSS auto source")
    run_name = f"rul-cmapss-FD001-{kind}-direct-rul-adaptation-seed42"
    root = (
        (os.environ.get("NEXUML_RUL_CHECKPOINT_ROOT") or "").strip()
        or ".experiments/rul/checkpoints"
    )
    checkpoint_dir = Path(root).expanduser().resolve() / run_name
    model = AdvancedRULForecaster(
        architecture=kind,
        hidden_dim=width,
        dropout=dropout,
        patch_size=5,
        attention_heads=2,
        decoder_depth=1,
        jump_threshold=2.5,
        initialization_seed=42,
    )
    model_stage = LayerSpec(
        component=model,
        keys_in=["rul_history"],
        keys_out=["predicted_rul"],
    )
    capped_evaluator = EvalAlgorithmSpec(
        name="rul_capped",
        algorithm=RULBenchmarkEvaluator(
            target_scale=1.0,
            include_nasa_score=True,
        ),
        feature_key="predicted_rul",
        label_key="rul",
    )
    raw_evaluator = EvalAlgorithmSpec(
        name="rul_official_raw",
        algorithm=RULBenchmarkEvaluator(
            target_scale=1.0,
            include_nasa_score=True,
        ),
        feature_key="predicted_rul",
        label_key="rul_raw",
    )
    audit = EvalAlgorithmSpec(
        name="engine_audit",
        algorithm=RULPerEngineAudit(
            artifact_path=source.artifact_path,
            output_file=f".experiments/rul/audits/{run_name}.csv",
        ),
        feature_key="predicted_rul",
        label_key="rul",
    )
    return base.model_copy(
        update={
            "name": run_name,
            "pipeline": PipelineSpec(
                stages={
                    "RULModel": [model_stage],
                    "RULLoss": base.pipeline.stages["RULLoss"],
                }
            ),
            "training": base.training.model_copy(
                update={"optimizer": optimizer(torch.optim.Adam, lr=lr)}
            ),
            "evaluation": base.evaluation.model_copy(
                update={"algorithms": [capped_evaluator, raw_evaluator, audit]}
            ),
            "logging": (
                base.logging.model_copy(update={"run_name": run_name})
                if base.logging is not None
                else None
            ),
            "callbacks": [
                callback(RULSeedOnFitStart, seed=42),
                callback(
                    ModelCheckpoint,
                    dirpath=str(checkpoint_dir),
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                ),
                callback(
                    EarlyStopping,
                    monitor="val/loss",
                    patience=patience,
                    mode="min",
                ),
                callback(
                    Goel2026RestoreBestWeights,
                    output_file=f".experiments/rul/audits/{run_name}.checkpoint.json",
                ),
            ],
        }
    )


@scenario("maintenance-rul-dmm-ja-adapted")
def maintenance_rul_dmm_ja_adapted() -> ScenarioSpec:
    """Compare DMM-JA architecture on C-MAPSS instead of ETDataset.

    Returns:
        Typed FD001 sensor-modal approximation to DMM-JA.
    """
    return _advanced_scenario("dmm_ja")


@scenario("maintenance-rul-patchformer-adapted")
def maintenance_rul_patchformer_adapted() -> ScenarioSpec:
    """Compare PatchFormer DPAN/FAN on C-MAPSS direct RUL, not battery capacity.

    Returns:
        Typed FD001 PatchFormer architecture adaptation.
    """
    return _advanced_scenario("patchformer")


@scenario("maintenance-rul-mamba-adapted")
def maintenance_rul_mamba_adapted() -> ScenarioSpec:
    """Compare RUL-Mamba FAN/GRN/SSM on C-MAPSS direct RUL labels.

    Returns:
        Typed FD001 RUL-Mamba architecture adaptation.
    """
    return _advanced_scenario("rul_mamba")
