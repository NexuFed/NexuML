"""Native PatchFormer/Flow Matching time-to-RUL calibration experiments.

Prerequisites are ordinary supervised PatchFormer and latent flow training
runs for the same subset/seed. This recipe trains a small independent MLP
calibration on original C-MAPSS training engines and selects val/loss only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback, optimizer
from nexuml.core.discovery import scenario
from nexuml.core.types import DatasetSpec, LayerSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows
from nexuml_library.data.maintenance.rul_flow_calibration import (
    AutoPatchFormerFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_flow_calibration import PatchFormerFlowRULHead
from nexuml_library.scenarios.maintenance.rul_controlled import controlled_rul_study


def patchformer_flow_rul_calibration(
    subset: Literal["FD001", "FD003"],
    seed: int,
    mode: Literal["flow_time", "hybrid"],
) -> ScenarioSpec:
    """Calibrate a held-out RUL forecast using frozen flow-path positions.

    Args:
        subset: FD001 or FD003 with its own supervised frozen model.
        seed: Match data split, frozen checkpoints and calibration RNG.
        mode: Flow time alone or additive correction to PatchFormer RUL.

    Returns:
        Native split-safe NexuML regression/evaluation/checkpoint scenario.

    Raises:
        ValueError: If the upstream baseline protocol is not the expected source.
    """
    base = controlled_rul_study("patchformer", subset, seed)
    original = base.data.datasets[0].source
    if not isinstance(original, AutoCMapSSRULWindows):
        raise ValueError("Flow-time calibration requires native C-MAPSS source")
    name = f"rul-{subset}-patchformer-flow-{mode.replace('_', '-')}-seed{seed}"
    latent_run = f"rul-{subset}-patchformer-latent-flow-seed{seed}"
    source = AutoPatchFormerFlowRULFeatures(
        subset=subset,
        seed=seed,
        root=original.root,
        prepared_rul_path=original.artifact_path,
        endpoint_artifact=str(
            Path(f".experiments/rul/prepared/{latent_run}-endpoints.pt").resolve()
        ),
        diagnostics_file=f".experiments/rul/audits/{name}.geometry.json",
    )
    data = base.data.model_copy(
        update={
            "datasets": [
                DatasetSpec(source=source, modality="sensor", split_type="keep")
            ],
            "input_shapes": {"flow_rul_features": [3]},
            "feature_key": "flow_rul_features",
        }
    )
    stages = {
        "RULModel": [
            LayerSpec(
                component=PatchFormerFlowRULHead(
                    mode=mode, hidden_dim=32, dropout=0.1,
                    initialization_seed=seed,
                ),
                keys_in=["flow_rul_features"],
                keys_out=["predicted_rul"],
            )
        ],
        "RULLoss": base.pipeline.stages["RULLoss"],
    }
    checkpoint_dir = Path(".experiments/rul/checkpoints").resolve() / name
    return base.model_copy(
        update={
            "name": name,
            "data": data,
            "pipeline": PipelineSpec(stages=stages),
            "training": base.training.model_copy(
                update={
                    "optimizer": optimizer(torch.optim.Adam, lr=1e-3),
                    "max_epochs": 150,
                    "batch_size": 64,
                }
            ),
            "evaluation": base.evaluation.model_copy(
                update={
                    "algorithms": [
                        entry.model_copy(
                            update={
                                "algorithm": entry.algorithm.model_copy(
                                    update={
                                        "output_file": (
                                            f".experiments/rul/audits/{name}.csv"
                                        )
                                    }
                                )
                            }
                        )
                        if entry.name == "engine_audit"
                        else entry
                        for entry in base.evaluation.algorithms
                    ]
                }
            ),
            "logging": (
                base.logging.model_copy(update={"run_name": name})
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
                    output_file=f".experiments/rul/audits/{name}.checkpoint.json",
                ),
            ],
        }
    )


@scenario("maintenance-rul-patchformer-flow-time-fd001-s42")
def patchformer_flow_time_fd001_s42() -> ScenarioSpec:
    """Calibrate flow-time RUL on FD001 with seed 42.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD001", 42, "flow_time")


@scenario("maintenance-rul-patchformer-hybrid-fd001-s42")
def patchformer_hybrid_fd001_s42() -> ScenarioSpec:
    """Calibrate hybrid RUL on FD001 with seed 42.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD001", 42, "hybrid")


@scenario("maintenance-rul-patchformer-flow-time-fd001-s43")
def patchformer_flow_time_fd001_s43() -> ScenarioSpec:
    """Calibrate flow-time RUL on FD001 with seed 43.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD001", 43, "flow_time")


@scenario("maintenance-rul-patchformer-hybrid-fd001-s43")
def patchformer_hybrid_fd001_s43() -> ScenarioSpec:
    """Calibrate hybrid RUL on FD001 with seed 43.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD001", 43, "hybrid")


@scenario("maintenance-rul-patchformer-flow-time-fd001-s44")
def patchformer_flow_time_fd001_s44() -> ScenarioSpec:
    """Calibrate flow-time RUL on FD001 with seed 44.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD001", 44, "flow_time")


@scenario("maintenance-rul-patchformer-hybrid-fd001-s44")
def patchformer_hybrid_fd001_s44() -> ScenarioSpec:
    """Calibrate hybrid RUL on FD001 with seed 44.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD001", 44, "hybrid")


@scenario("maintenance-rul-patchformer-flow-time-fd003-s42")
def patchformer_flow_time_fd003_s42() -> ScenarioSpec:
    """Calibrate flow-time RUL on FD003 with seed 42.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD003", 42, "flow_time")


@scenario("maintenance-rul-patchformer-hybrid-fd003-s42")
def patchformer_hybrid_fd003_s42() -> ScenarioSpec:
    """Calibrate hybrid RUL on FD003 with seed 42.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD003", 42, "hybrid")


@scenario("maintenance-rul-patchformer-flow-time-fd003-s43")
def patchformer_flow_time_fd003_s43() -> ScenarioSpec:
    """Calibrate flow-time RUL on FD003 with seed 43.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD003", 43, "flow_time")


@scenario("maintenance-rul-patchformer-hybrid-fd003-s43")
def patchformer_hybrid_fd003_s43() -> ScenarioSpec:
    """Calibrate hybrid RUL on FD003 with seed 43.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD003", 43, "hybrid")


@scenario("maintenance-rul-patchformer-flow-time-fd003-s44")
def patchformer_flow_time_fd003_s44() -> ScenarioSpec:
    """Calibrate flow-time RUL on FD003 with seed 44.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD003", 44, "flow_time")


@scenario("maintenance-rul-patchformer-hybrid-fd003-s44")
def patchformer_hybrid_fd003_s44() -> ScenarioSpec:
    """Calibrate hybrid RUL on FD003 with seed 44.

    Returns:
        Pretrained-encoder/flow NexuML RUL calibration scenario.
    """
    return patchformer_flow_rul_calibration("FD003", 44, "hybrid")
