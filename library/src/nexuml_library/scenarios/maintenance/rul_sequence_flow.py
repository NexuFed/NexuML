"""Native RUL evaluations of full-sequence Transformer/Conformer/CfC flows.

Every flow-aware temporal readout has a matched encoder-only counterpart
using the same full DPAN tokens, 256-input head, optimizer and label protocol.
All supervised RUL loss is applied only to the new temporal attention head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback, optimizer
from nexuml.core.discovery import scenario
from nexuml.core.types import (
    DatasetSpec,
    EvalAlgorithmSpec,
    LayerSpec,
    PipelineSpec,
    ScenarioSpec,
)
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows
from nexuml_library.data.maintenance.rul_sequence_flow import (
    AutoPatchFormerSequenceFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.evaluation.maintenance.rul_sequence_flow_checkpoint import (
    SequenceArchitecture,
    sequence_flow_run,
)
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_controlled import controlled_rul_study


def patchformer_sequence_flow_rul(
    subset: Literal["FD001", "FD003"],
    seed: int,
    architecture: SequenceArchitecture,
    use_flow: bool,
) -> ScenarioSpec:
    """Train a full-sequence RUL readout with or without the frozen Flow.

    Args:
        subset: Engine-disjoint FD001 or FD003 benchmark.
        seed: Original PatchFormer/flow and new head initialization seed.
        architecture: Matching Transformer, Conformer or CfC flow checkpoint.
        use_flow: Add frozen sequence-flow displacements when True.

    Returns:
        Native RUL dataset, identical temporal readout and official test audit.

    Raises:
        ValueError: If the upstream original C-MAPSS protocol is unavailable.
    """
    base = controlled_rul_study("patchformer", subset, seed)
    original = base.data.datasets[0].source
    if not isinstance(original, AutoCMapSSRULWindows):
        raise ValueError("Sequence RUL requires controlled C-MAPSS source")
    flow_run = sequence_flow_run(subset, seed, architecture)
    arm = "flow" if use_flow else "encoder"
    run = f"rul-{subset}-patchformer-sequence-{architecture}-{arm}-rul-seed{seed}"
    endpoint_file = Path(
        f".experiments/rul/prepared/{flow_run}-endpoints.pt"
    ).resolve()
    source = AutoPatchFormerSequenceFlowRULFeatures(
        subset=subset,
        seed=seed,
        architecture=architecture,
        use_flow=use_flow,
        root=original.root,
        prepared_rul_path=original.artifact_path,
        endpoint_artifact=str(endpoint_file),
        diagnostics_file=f".experiments/rul/audits/{run}.geometry.json",
        encoding_batch_size=32,
    )
    data = base.data.model_copy(
        update={
            "datasets": [
                DatasetSpec(source=source, modality="sensor", split_type="keep")
            ],
            "input_shapes": {"sequence_rul_features": [30, 256]},
            "feature_key": "sequence_rul_features",
        }
    )
    stage = LayerSpec(
        component=PatchFormerSequenceFlowRULHead(
            hidden_dim=128,
            attention_heads=4,
            dropout=0.1,
            initialization_seed=seed,
        ),
        keys_in=["sequence_rul_features"],
        keys_out=["predicted_rul"],
    )
    audit = EvalAlgorithmSpec(
        name="engine_audit",
        algorithm=RULPerEngineAudit(
            artifact_path=original.artifact_path,
            output_file=f".experiments/rul/audits/{run}.csv",
        ),
        feature_key="predicted_rul",
        label_key="rul",
    )
    checkpoint_dir = Path(".experiments/rul/checkpoints").resolve() / run
    return base.model_copy(
        update={
            "name": run,
            "data": data,
            "pipeline": PipelineSpec(
                stages={
                    "RULModel": [stage],
                    "RULLoss": base.pipeline.stages["RULLoss"],
                }
            ),
            "training": base.training.model_copy(
                update={
                    "optimizer": optimizer(torch.optim.Adam, lr=1e-3),
                    "max_epochs": 150,
                    "batch_size": 64,
                }
            ),
            "evaluation": base.evaluation.model_copy(
                update={
                    "algorithms": [*base.evaluation.algorithms[:2], audit]
                }
            ),
            "logging": (
                base.logging.model_copy(update={"run_name": run})
                if base.logging is not None else None
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
                    EarlyStopping,
                    monitor="val/loss",
                    mode="min",
                    patience=20,
                ),
                callback(
                    Goel2026RestoreBestWeights,
                    output_file=f".experiments/rul/audits/{run}.checkpoint.json",
                ),
            ],
        }
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-flow-fd001-s42")
def patchformer_sequence_transformer_flow_fd001_s42() -> ScenarioSpec:
    """Evaluate transformer flow-aware sequence RUL on FD001 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 42, "transformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-encoder-fd001-s42")
def patchformer_sequence_transformer_encoder_fd001_s42() -> ScenarioSpec:
    """Evaluate transformer encoder-aware sequence RUL on FD001 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 42, "transformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-flow-fd001-s42")
def patchformer_sequence_conformer_flow_fd001_s42() -> ScenarioSpec:
    """Evaluate conformer flow-aware sequence RUL on FD001 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 42, "conformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-encoder-fd001-s42")
def patchformer_sequence_conformer_encoder_fd001_s42() -> ScenarioSpec:
    """Evaluate conformer encoder-aware sequence RUL on FD001 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 42, "conformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-flow-fd001-s42")
def patchformer_sequence_cfc_flow_fd001_s42() -> ScenarioSpec:
    """Evaluate cfc flow-aware sequence RUL on FD001 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 42, "cfc", true
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-encoder-fd001-s42")
def patchformer_sequence_cfc_encoder_fd001_s42() -> ScenarioSpec:
    """Evaluate cfc encoder-aware sequence RUL on FD001 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 42, "cfc", false
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-flow-fd001-s43")
def patchformer_sequence_transformer_flow_fd001_s43() -> ScenarioSpec:
    """Evaluate transformer flow-aware sequence RUL on FD001 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 43, "transformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-encoder-fd001-s43")
def patchformer_sequence_transformer_encoder_fd001_s43() -> ScenarioSpec:
    """Evaluate transformer encoder-aware sequence RUL on FD001 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 43, "transformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-flow-fd001-s43")
def patchformer_sequence_conformer_flow_fd001_s43() -> ScenarioSpec:
    """Evaluate conformer flow-aware sequence RUL on FD001 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 43, "conformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-encoder-fd001-s43")
def patchformer_sequence_conformer_encoder_fd001_s43() -> ScenarioSpec:
    """Evaluate conformer encoder-aware sequence RUL on FD001 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 43, "conformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-flow-fd001-s43")
def patchformer_sequence_cfc_flow_fd001_s43() -> ScenarioSpec:
    """Evaluate cfc flow-aware sequence RUL on FD001 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 43, "cfc", true
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-encoder-fd001-s43")
def patchformer_sequence_cfc_encoder_fd001_s43() -> ScenarioSpec:
    """Evaluate cfc encoder-aware sequence RUL on FD001 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 43, "cfc", false
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-flow-fd001-s44")
def patchformer_sequence_transformer_flow_fd001_s44() -> ScenarioSpec:
    """Evaluate transformer flow-aware sequence RUL on FD001 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 44, "transformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-encoder-fd001-s44")
def patchformer_sequence_transformer_encoder_fd001_s44() -> ScenarioSpec:
    """Evaluate transformer encoder-aware sequence RUL on FD001 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 44, "transformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-flow-fd001-s44")
def patchformer_sequence_conformer_flow_fd001_s44() -> ScenarioSpec:
    """Evaluate conformer flow-aware sequence RUL on FD001 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 44, "conformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-encoder-fd001-s44")
def patchformer_sequence_conformer_encoder_fd001_s44() -> ScenarioSpec:
    """Evaluate conformer encoder-aware sequence RUL on FD001 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 44, "conformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-flow-fd001-s44")
def patchformer_sequence_cfc_flow_fd001_s44() -> ScenarioSpec:
    """Evaluate cfc flow-aware sequence RUL on FD001 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 44, "cfc", true
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-encoder-fd001-s44")
def patchformer_sequence_cfc_encoder_fd001_s44() -> ScenarioSpec:
    """Evaluate cfc encoder-aware sequence RUL on FD001 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD001", 44, "cfc", false
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-flow-fd003-s42")
def patchformer_sequence_transformer_flow_fd003_s42() -> ScenarioSpec:
    """Evaluate transformer flow-aware sequence RUL on FD003 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 42, "transformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-encoder-fd003-s42")
def patchformer_sequence_transformer_encoder_fd003_s42() -> ScenarioSpec:
    """Evaluate transformer encoder-aware sequence RUL on FD003 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 42, "transformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-flow-fd003-s42")
def patchformer_sequence_conformer_flow_fd003_s42() -> ScenarioSpec:
    """Evaluate conformer flow-aware sequence RUL on FD003 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 42, "conformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-encoder-fd003-s42")
def patchformer_sequence_conformer_encoder_fd003_s42() -> ScenarioSpec:
    """Evaluate conformer encoder-aware sequence RUL on FD003 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 42, "conformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-flow-fd003-s42")
def patchformer_sequence_cfc_flow_fd003_s42() -> ScenarioSpec:
    """Evaluate cfc flow-aware sequence RUL on FD003 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 42, "cfc", true
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-encoder-fd003-s42")
def patchformer_sequence_cfc_encoder_fd003_s42() -> ScenarioSpec:
    """Evaluate cfc encoder-aware sequence RUL on FD003 seed 42.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 42, "cfc", false
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-flow-fd003-s43")
def patchformer_sequence_transformer_flow_fd003_s43() -> ScenarioSpec:
    """Evaluate transformer flow-aware sequence RUL on FD003 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 43, "transformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-encoder-fd003-s43")
def patchformer_sequence_transformer_encoder_fd003_s43() -> ScenarioSpec:
    """Evaluate transformer encoder-aware sequence RUL on FD003 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 43, "transformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-flow-fd003-s43")
def patchformer_sequence_conformer_flow_fd003_s43() -> ScenarioSpec:
    """Evaluate conformer flow-aware sequence RUL on FD003 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 43, "conformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-encoder-fd003-s43")
def patchformer_sequence_conformer_encoder_fd003_s43() -> ScenarioSpec:
    """Evaluate conformer encoder-aware sequence RUL on FD003 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 43, "conformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-flow-fd003-s43")
def patchformer_sequence_cfc_flow_fd003_s43() -> ScenarioSpec:
    """Evaluate cfc flow-aware sequence RUL on FD003 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 43, "cfc", true
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-encoder-fd003-s43")
def patchformer_sequence_cfc_encoder_fd003_s43() -> ScenarioSpec:
    """Evaluate cfc encoder-aware sequence RUL on FD003 seed 43.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 43, "cfc", false
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-flow-fd003-s44")
def patchformer_sequence_transformer_flow_fd003_s44() -> ScenarioSpec:
    """Evaluate transformer flow-aware sequence RUL on FD003 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 44, "transformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-encoder-fd003-s44")
def patchformer_sequence_transformer_encoder_fd003_s44() -> ScenarioSpec:
    """Evaluate transformer encoder-aware sequence RUL on FD003 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 44, "transformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-flow-fd003-s44")
def patchformer_sequence_conformer_flow_fd003_s44() -> ScenarioSpec:
    """Evaluate conformer flow-aware sequence RUL on FD003 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 44, "conformer", true
    )


@scenario("maintenance-rul-patchformer-sequence-conformer-encoder-fd003-s44")
def patchformer_sequence_conformer_encoder_fd003_s44() -> ScenarioSpec:
    """Evaluate conformer encoder-aware sequence RUL on FD003 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 44, "conformer", false
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-flow-fd003-s44")
def patchformer_sequence_cfc_flow_fd003_s44() -> ScenarioSpec:
    """Evaluate cfc flow-aware sequence RUL on FD003 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 44, "cfc", true
    )


@scenario("maintenance-rul-patchformer-sequence-cfc-encoder-fd003-s44")
def patchformer_sequence_cfc_encoder_fd003_s44() -> ScenarioSpec:
    """Evaluate cfc encoder-aware sequence RUL on FD003 seed 44.

    Returns:
        Official-label held-out RUL benchmark with matched full-sequence readout.
    """
    return patchformer_sequence_flow_rul(
        "FD003", 44, "cfc", false
    )
