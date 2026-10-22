"""Native flow-only and matched encoder-only RUL benchmarks on C-MAPSS.

Uses the existing pretrained PatchFormer *embedding* and (for flow-state)
the learned NexuML Flow vector field. Neither scenario accesses the former
PatchFormer RUL predictor's outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import DatasetSpec, EvalAlgorithmSpec, LayerSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.rul_flow_calibration import (
    AutoPatchFormerFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_flow_state import PatchFormerFlowStateRULHead
from nexuml_library.scenarios.maintenance.rul_flow_calibration import (
    patchformer_flow_rul_calibration,
)


def patchformer_flow_state_rul(
    subset: Literal["FD001", "FD003"],
    seed: int,
    kind: Literal["flow_state", "encoder_only"],
) -> ScenarioSpec:
    """Build a flow-driven supervised RUL head or its encoder-only control.

    Args:
        subset: C-MAPSS subset with asset-disjoint train/validation/test splits.
        seed: Upstream pretrained model and calibration initialization seed.
        kind: Frozen full-flow state or frozen encoder without flow features.

    Returns:
        Native NexuML regression recipe preserving official per-engine audits.
    """
    baseline = patchformer_flow_rul_calibration(subset, seed, "hybrid")
    run = f"rul-{subset}-patchformer-{kind.replace('_', '-')}-seed{seed}"
    source = baseline.data.datasets[0].source
    if not isinstance(source, AutoPatchFormerFlowRULFeatures):
        raise TypeError("Flow state requires native PatchFormer flow RUL features")
    feature_mode: Literal["flow_only", "encoder_only"] = (
        "flow_only" if kind == "flow_state" else "encoder_only"
    )
    data = baseline.data.model_copy(
        update={
            "datasets": [
                DatasetSpec(
                    source=source.model_copy(
                        update={
                            "feature_mode": feature_mode,
                            "diagnostics_file": (
                                f".experiments/rul/audits/{run}.geometry.json"
                            ),
                        }
                    ),
                    modality="sensor",
                    split_type="keep",
                )
            ],
            "input_shapes": {"flow_rul_features": [258]},
            "feature_key": "flow_rul_features",
        }
    )
    model = LayerSpec(
        component=PatchFormerFlowStateRULHead(
            hidden_dim=128, dropout=0.1, initialization_seed=seed
        ),
        keys_in=["flow_rul_features"],
        keys_out=["predicted_rul"],
    )
    audit = EvalAlgorithmSpec(
        name="engine_audit",
        algorithm=RULPerEngineAudit(
            artifact_path=source.prepared_rul_path,
            output_file=f".experiments/rul/audits/{run}.csv",
        ),
        feature_key="predicted_rul",
        label_key="rul",
    )
    checkpoint_dir = Path(".experiments/rul/checkpoints").resolve() / run
    return baseline.model_copy(
        update={
            "name": run,
            "data": data,
            "pipeline": PipelineSpec(
                stages={
                    "RULModel": [model],
                    "RULLoss": baseline.pipeline.stages["RULLoss"],
                }
            ),
            "evaluation": baseline.evaluation.model_copy(
                update={
                    "algorithms": [*baseline.evaluation.algorithms[:2], audit]
                }
            ),
            "logging": (
                baseline.logging.model_copy(update={"run_name": run})
                if baseline.logging is not None
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


@scenario("maintenance-rul-patchformer-flow-state-fd001-s42")
def patchformer_flow_state_fd001_s42() -> ScenarioSpec:
    """Run flow-state supervised RUL on FD001 seed 42.

    Returns:
        Pretrained-encoder standalone regression recipe with flow_state inputs.
    """
    return patchformer_flow_state_rul("FD001", 42, "flow_state")


@scenario("maintenance-rul-patchformer-encoder-only-fd001-s42")
def patchformer_encoder_only_fd001_s42() -> ScenarioSpec:
    """Run encoder-only supervised RUL on FD001 seed 42.

    Returns:
        Pretrained-encoder standalone regression recipe with encoder_only inputs.
    """
    return patchformer_flow_state_rul("FD001", 42, "encoder_only")


@scenario("maintenance-rul-patchformer-flow-state-fd001-s43")
def patchformer_flow_state_fd001_s43() -> ScenarioSpec:
    """Run flow-state supervised RUL on FD001 seed 43.

    Returns:
        Pretrained-encoder standalone regression recipe with flow_state inputs.
    """
    return patchformer_flow_state_rul("FD001", 43, "flow_state")


@scenario("maintenance-rul-patchformer-encoder-only-fd001-s43")
def patchformer_encoder_only_fd001_s43() -> ScenarioSpec:
    """Run encoder-only supervised RUL on FD001 seed 43.

    Returns:
        Pretrained-encoder standalone regression recipe with encoder_only inputs.
    """
    return patchformer_flow_state_rul("FD001", 43, "encoder_only")


@scenario("maintenance-rul-patchformer-flow-state-fd001-s44")
def patchformer_flow_state_fd001_s44() -> ScenarioSpec:
    """Run flow-state supervised RUL on FD001 seed 44.

    Returns:
        Pretrained-encoder standalone regression recipe with flow_state inputs.
    """
    return patchformer_flow_state_rul("FD001", 44, "flow_state")


@scenario("maintenance-rul-patchformer-encoder-only-fd001-s44")
def patchformer_encoder_only_fd001_s44() -> ScenarioSpec:
    """Run encoder-only supervised RUL on FD001 seed 44.

    Returns:
        Pretrained-encoder standalone regression recipe with encoder_only inputs.
    """
    return patchformer_flow_state_rul("FD001", 44, "encoder_only")


@scenario("maintenance-rul-patchformer-flow-state-fd003-s42")
def patchformer_flow_state_fd003_s42() -> ScenarioSpec:
    """Run flow-state supervised RUL on FD003 seed 42.

    Returns:
        Pretrained-encoder standalone regression recipe with flow_state inputs.
    """
    return patchformer_flow_state_rul("FD003", 42, "flow_state")


@scenario("maintenance-rul-patchformer-encoder-only-fd003-s42")
def patchformer_encoder_only_fd003_s42() -> ScenarioSpec:
    """Run encoder-only supervised RUL on FD003 seed 42.

    Returns:
        Pretrained-encoder standalone regression recipe with encoder_only inputs.
    """
    return patchformer_flow_state_rul("FD003", 42, "encoder_only")


@scenario("maintenance-rul-patchformer-flow-state-fd003-s43")
def patchformer_flow_state_fd003_s43() -> ScenarioSpec:
    """Run flow-state supervised RUL on FD003 seed 43.

    Returns:
        Pretrained-encoder standalone regression recipe with flow_state inputs.
    """
    return patchformer_flow_state_rul("FD003", 43, "flow_state")


@scenario("maintenance-rul-patchformer-encoder-only-fd003-s43")
def patchformer_encoder_only_fd003_s43() -> ScenarioSpec:
    """Run encoder-only supervised RUL on FD003 seed 43.

    Returns:
        Pretrained-encoder standalone regression recipe with encoder_only inputs.
    """
    return patchformer_flow_state_rul("FD003", 43, "encoder_only")


@scenario("maintenance-rul-patchformer-flow-state-fd003-s44")
def patchformer_flow_state_fd003_s44() -> ScenarioSpec:
    """Run flow-state supervised RUL on FD003 seed 44.

    Returns:
        Pretrained-encoder standalone regression recipe with flow_state inputs.
    """
    return patchformer_flow_state_rul("FD003", 44, "flow_state")


@scenario("maintenance-rul-patchformer-encoder-only-fd003-s44")
def patchformer_encoder_only_fd003_s44() -> ScenarioSpec:
    """Run encoder-only supervised RUL on FD003 seed 44.

    Returns:
        Pretrained-encoder standalone regression recipe with encoder_only inputs.
    """
    return patchformer_flow_state_rul("FD003", 44, "encoder_only")
