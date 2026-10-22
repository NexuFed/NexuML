"""Paired controlled FD001/FD003 RUL generalization scenarios.

All four methods share 30 cycles, sensors, asset-disjoint splits, train-only
normalization, Adam optimizer, learning rate, batch size and validation protocol.
Architecture and the prespecified 2:1 asymmetric loss are the only differences.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback, optimizer
from nexuml.core.discovery import scenario
from nexuml.core.types import EvalAlgorithmSpec, LayerSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.evaluation.maintenance.rul_evaluator import RULBenchmarkEvaluator
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_advanced import AdvancedRULForecaster
from nexuml_library.layers.model.rul_paper_lstm import Goel2026LSTMForecaster
from nexuml_library.layers.model.rul_safety import RULAsymmetricSquaredLoss
from nexuml_library.scenarios.maintenance.rul import (
    _GOEL2026_FD003_SENSORS,
    _LU2025_SENSORS,
    maintenance_rul_benchmark,
)

_StudyKind = Literal["patchformer", "dmm_ja", "safety_lstm", "rul_mamba"]
_StudySubset = Literal["FD001", "FD003"]


def controlled_rul_study(
    kind: _StudyKind,
    subset: _StudySubset,
    seed: int,
) -> ScenarioSpec:
    """Build a shared-protocol, independently checkpointed RUL comparison.

    Args:
        kind: One of three main models or the 30-cycle Mamba diagnostic.
        subset: NASA FD001 or FD003, respecting the 14/15-sensor protocol.
        seed: Shared asset-split, initialization and data-loader seed.

    Returns:
        Native RUL experiment with paired preparation and raw/capped test audits.

    Raises:
        ValueError: If the baseline source is incompatible or a seed is invalid.
    """
    if seed < 0:
        raise ValueError("RUL generalization seed must be nonnegative")
    sensors = (
        _GOEL2026_FD003_SENSORS if subset == "FD003" else _LU2025_SENSORS
    )
    profile = f"controlled-w30-s{seed}"
    base = maintenance_rul_benchmark(
        architecture="lstm",
        objective="mse",
        cmapss_subset=subset,
        cmapss_window=30,
        cmapss_stride=1,
        cmapss_sensor_ids=sensors,
        cmapss_rul_cap=130.0,
        cmapss_profile=profile,
        cmapss_padding_mode="zero",
        cmapss_train_full_windows_only=True,
        cmapss_normalize_targets=False,
        hidden_dim=64,
        depth=2,
        dropout=0.1,
        lr=1e-3,
        batch_size=64,
        max_epochs=200,
        patience=20,
        seed=seed,
    )
    source = base.data.datasets[0].source
    if not isinstance(source, AutoCMapSSRULWindows):
        raise ValueError("Controlled RUL study requires the native C-MAPSS source")

    kind_name = kind.replace("_", "-")
    run = f"rul-{subset}-controlled-{kind_name}-seed{seed}"
    checkpoint_root = (
        (os.environ.get("NEXUML_RUL_CHECKPOINT_ROOT") or "").strip()
        or ".experiments/rul/checkpoints"
    )
    checkpoint_dir = Path(checkpoint_root).expanduser().resolve() / run

    if kind == "safety_lstm":
        model = Goel2026LSTMForecaster(
            hidden_dim=32, dropout=0.5, initialization_seed=seed
        )
        loss = LayerSpec(
            component=RULAsymmetricSquaredLoss(overestimate_weight=2.0),
            keys_in=["predicted_rul"],
            keys_out=["rul_loss"],
            label_key="rul",
        )
    else:
        model = AdvancedRULForecaster(
            architecture=kind,
            hidden_dim=64,
            dropout=0.1,
            patch_size=5,
            attention_heads=2,
            decoder_depth=1,
            jump_threshold=2.5,
            initialization_seed=seed,
        )
        loss = base.pipeline.stages["RULLoss"][0]

    model_layer = LayerSpec(
        component=model,
        keys_in=["rul_history"],
        keys_out=["predicted_rul"],
    )
    algorithms = [
        EvalAlgorithmSpec(
            name=f"rul_{label}",
            algorithm=RULBenchmarkEvaluator(
                target_scale=1.0, include_nasa_score=True
            ),
            feature_key="predicted_rul",
            label_key=target,
        )
        for label, target in (("capped", "rul"), ("official_raw", "rul_raw"))
    ]
    algorithms.append(
        EvalAlgorithmSpec(
            name="engine_audit",
            algorithm=RULPerEngineAudit(
                artifact_path=source.artifact_path,
                output_file=f".experiments/rul/audits/{run}.csv",
            ),
            feature_key="predicted_rul",
            label_key="rul",
        )
    )
    return base.model_copy(
        update={
            "name": run,
            "pipeline": PipelineSpec(
                stages={
                    "RULModel": [model_layer],
                    "RULLoss": [loss, *base.pipeline.stages["RULLoss"][1:]],
                }
            ),
            "training": base.training.model_copy(
                update={"optimizer": optimizer(torch.optim.Adam, lr=1e-3)}
            ),
            "evaluation": base.evaluation.model_copy(
                update={"algorithms": algorithms}
            ),
            "logging": (
                base.logging.model_copy(update={"run_name": run})
                if base.logging is not None
                else None
            ),
            "callbacks": [
                callback(RULSeedOnFitStart, seed=seed),
                callback(
                    ModelCheckpoint,
                    dirpath=str(checkpoint_dir),
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                ),
                callback(
                    EarlyStopping, monitor="val/loss", mode="min", patience=20
                ),
                callback(
                    Goel2026RestoreBestWeights,
                    output_file=f".experiments/rul/audits/{run}.checkpoint.json",
                ),
            ],
        }
    )


@scenario("maintenance-rul-controlled-fd001-patchformer-s42")
def controlled_fd001_patchformer_s42() -> ScenarioSpec:
    """Run controlled patchformer on FD001, seed 42.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("patchformer", "FD001", 42)


@scenario("maintenance-rul-controlled-fd001-dmm-ja-s42")
def controlled_fd001_dmm_ja_s42() -> ScenarioSpec:
    """Run controlled dmm-ja on FD001, seed 42.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("dmm_ja", "FD001", 42)


@scenario("maintenance-rul-controlled-fd001-safety-lstm-s42")
def controlled_fd001_safety_lstm_s42() -> ScenarioSpec:
    """Run controlled safety-lstm on FD001, seed 42.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("safety_lstm", "FD001", 42)


@scenario("maintenance-rul-controlled-fd001-rul-mamba-s42")
def controlled_fd001_rul_mamba_s42() -> ScenarioSpec:
    """Run controlled rul-mamba on FD001, seed 42.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("rul_mamba", "FD001", 42)


@scenario("maintenance-rul-controlled-fd001-patchformer-s43")
def controlled_fd001_patchformer_s43() -> ScenarioSpec:
    """Run controlled patchformer on FD001, seed 43.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("patchformer", "FD001", 43)


@scenario("maintenance-rul-controlled-fd001-dmm-ja-s43")
def controlled_fd001_dmm_ja_s43() -> ScenarioSpec:
    """Run controlled dmm-ja on FD001, seed 43.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("dmm_ja", "FD001", 43)


@scenario("maintenance-rul-controlled-fd001-safety-lstm-s43")
def controlled_fd001_safety_lstm_s43() -> ScenarioSpec:
    """Run controlled safety-lstm on FD001, seed 43.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("safety_lstm", "FD001", 43)


@scenario("maintenance-rul-controlled-fd001-rul-mamba-s43")
def controlled_fd001_rul_mamba_s43() -> ScenarioSpec:
    """Run controlled rul-mamba on FD001, seed 43.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("rul_mamba", "FD001", 43)


@scenario("maintenance-rul-controlled-fd001-patchformer-s44")
def controlled_fd001_patchformer_s44() -> ScenarioSpec:
    """Run controlled patchformer on FD001, seed 44.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("patchformer", "FD001", 44)


@scenario("maintenance-rul-controlled-fd001-dmm-ja-s44")
def controlled_fd001_dmm_ja_s44() -> ScenarioSpec:
    """Run controlled dmm-ja on FD001, seed 44.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("dmm_ja", "FD001", 44)


@scenario("maintenance-rul-controlled-fd001-safety-lstm-s44")
def controlled_fd001_safety_lstm_s44() -> ScenarioSpec:
    """Run controlled safety-lstm on FD001, seed 44.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("safety_lstm", "FD001", 44)


@scenario("maintenance-rul-controlled-fd001-rul-mamba-s44")
def controlled_fd001_rul_mamba_s44() -> ScenarioSpec:
    """Run controlled rul-mamba on FD001, seed 44.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("rul_mamba", "FD001", 44)


@scenario("maintenance-rul-controlled-fd003-patchformer-s42")
def controlled_fd003_patchformer_s42() -> ScenarioSpec:
    """Run controlled patchformer on FD003, seed 42.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("patchformer", "FD003", 42)


@scenario("maintenance-rul-controlled-fd003-dmm-ja-s42")
def controlled_fd003_dmm_ja_s42() -> ScenarioSpec:
    """Run controlled dmm-ja on FD003, seed 42.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("dmm_ja", "FD003", 42)


@scenario("maintenance-rul-controlled-fd003-safety-lstm-s42")
def controlled_fd003_safety_lstm_s42() -> ScenarioSpec:
    """Run controlled safety-lstm on FD003, seed 42.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("safety_lstm", "FD003", 42)


@scenario("maintenance-rul-controlled-fd003-rul-mamba-s42")
def controlled_fd003_rul_mamba_s42() -> ScenarioSpec:
    """Run controlled rul-mamba on FD003, seed 42.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("rul_mamba", "FD003", 42)


@scenario("maintenance-rul-controlled-fd003-patchformer-s43")
def controlled_fd003_patchformer_s43() -> ScenarioSpec:
    """Run controlled patchformer on FD003, seed 43.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("patchformer", "FD003", 43)


@scenario("maintenance-rul-controlled-fd003-dmm-ja-s43")
def controlled_fd003_dmm_ja_s43() -> ScenarioSpec:
    """Run controlled dmm-ja on FD003, seed 43.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("dmm_ja", "FD003", 43)


@scenario("maintenance-rul-controlled-fd003-safety-lstm-s43")
def controlled_fd003_safety_lstm_s43() -> ScenarioSpec:
    """Run controlled safety-lstm on FD003, seed 43.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("safety_lstm", "FD003", 43)


@scenario("maintenance-rul-controlled-fd003-rul-mamba-s43")
def controlled_fd003_rul_mamba_s43() -> ScenarioSpec:
    """Run controlled rul-mamba on FD003, seed 43.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("rul_mamba", "FD003", 43)


@scenario("maintenance-rul-controlled-fd003-patchformer-s44")
def controlled_fd003_patchformer_s44() -> ScenarioSpec:
    """Run controlled patchformer on FD003, seed 44.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("patchformer", "FD003", 44)


@scenario("maintenance-rul-controlled-fd003-dmm-ja-s44")
def controlled_fd003_dmm_ja_s44() -> ScenarioSpec:
    """Run controlled dmm-ja on FD003, seed 44.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("dmm_ja", "FD003", 44)


@scenario("maintenance-rul-controlled-fd003-safety-lstm-s44")
def controlled_fd003_safety_lstm_s44() -> ScenarioSpec:
    """Run controlled safety-lstm on FD003, seed 44.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("safety_lstm", "FD003", 44)


@scenario("maintenance-rul-controlled-fd003-rul-mamba-s44")
def controlled_fd003_rul_mamba_s44() -> ScenarioSpec:
    """Run controlled rul-mamba on FD003, seed 44.

    Returns:
        Native train/test scenario sharing the study's preparation protocol.
    """
    return controlled_rul_study("rul_mamba", "FD003", 44)
