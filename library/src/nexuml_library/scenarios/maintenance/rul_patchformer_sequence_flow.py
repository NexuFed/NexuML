"""PatchFormer full-sequence latent Flow Matching architecture comparisons.

All variants preserve [batch,30,128] frozen DPAN tokens, use the identical
healthy/faulty pairing and training protocol, and select validation SWD only.
The former PatchFormer FAN/RUL regression head is never called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import DatasetSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.patchformer_flow import AutoPatchFormerFlowEndpoints
from nexuml_library.evaluation.maintenance.ablation_validation import (
    DistributionalTransportValidation,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.flow import SequenceFlow
from nexuml_library.scenarios.maintenance.rul_patchformer_flow import (
    patchformer_embedding_flow,
)
from nexuml_library.scenarios.training.defaults import default_logging


def patchformer_sequence_flow(
    subset: Literal["FD001", "FD003"] = "FD001",
    seed: int = 42,
    architecture: Literal["transformer", "conformer", "cfc"] = "transformer",
) -> ScenarioSpec:
    """Train sequence-preserving FM using frozen full PatchFormer DPAN features.

    Args:
        subset: Asset-disjoint NASA FD001 or FD003.
        seed: Matched upstream PatchFormer and stochastic training seed.
        architecture: Transformer, Conformer or closed-form recurrent CfC.

    Returns:
        Registered-compatible, SWD-selected NexuML sequential flow scenario.

    Raises:
        ValueError: If the base scenario does not use frozen PatchFormer pairs.
    """
    base = patchformer_embedding_flow(subset, seed)
    source = base.data.datasets[0].source
    if not isinstance(source, AutoPatchFormerFlowEndpoints):
        raise ValueError("Sequence flow requires frozen PatchFormer endpoint data")
    run = f"rul-{subset}-patchformer-sequence-{architecture}-flow-seed{seed}"
    prepared = Path(
        f".experiments/rul/prepared/{run}-endpoints.pt"
    ).resolve()
    data = base.data.model_copy(
        update={
            "datasets": [
                DatasetSpec(
                    source=source.model_copy(
                        update={
                            "sequence_embeddings": True,
                            "output_file": str(prepared),
                        }
                    ),
                    modality="sensor",
                    split_type="keep",
                )
            ],
            "input_shapes": {
                "healthy_spectrum": [30, 128],
                "faulty_spectrum": [30, 128],
            },
        }
    )
    stages = dict(base.pipeline.stages)
    flow_layer = stages["Flow"][0]
    stages["Flow"] = [
        flow_layer.model_copy(
            update={
                "component": SequenceFlow(
                    architecture=architecture,
                    model_dim=128,
                    depth=2,
                    attention_heads=4,
                    conv_kernel=7,
                    dropout=0.0,
                    num_steps=50,
                    initialization_seed=seed,
                ),
            }
        )
    ]
    checkpoint_dir = Path(".experiments/rul/checkpoints").resolve() / run
    return base.model_copy(
        update={
            "name": run,
            "data": data,
            "pipeline": PipelineSpec(stages=stages),
            "logging": default_logging(
                experiment_name="NEX-233 PatchFormer sequential latent FM",
                run_name=run,
                use_mlflow=False,
            ),
            "callbacks": [
                callback(RULSeedOnFitStart, seed=seed),
                callback(
                    DistributionalTransportValidation,
                    prepared_file=str(prepared),
                    batch_size=32,
                    projections=128,
                    seed=seed,
                ),
                callback(
                    ModelCheckpoint,
                    dirpath=str(checkpoint_dir),
                    monitor="val/transport_swd",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                ),
                callback(
                    EarlyStopping,
                    monitor="val/transport_swd",
                    mode="min",
                    patience=25,
                ),
            ],
        }
    )


@scenario("maintenance-rul-patchformer-sequence-transformer-fd001-s42")
def patchformer_sequence_transformer_fd001_s42() -> ScenarioSpec:
    """Train sequence transformer FM on FD001 with seed 42.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 42, "transformer")


@scenario("maintenance-rul-patchformer-sequence-conformer-fd001-s42")
def patchformer_sequence_conformer_fd001_s42() -> ScenarioSpec:
    """Train sequence conformer FM on FD001 with seed 42.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 42, "conformer")


@scenario("maintenance-rul-patchformer-sequence-cfc-fd001-s42")
def patchformer_sequence_cfc_fd001_s42() -> ScenarioSpec:
    """Train sequence cfc FM on FD001 with seed 42.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 42, "cfc")


@scenario("maintenance-rul-patchformer-sequence-transformer-fd001-s43")
def patchformer_sequence_transformer_fd001_s43() -> ScenarioSpec:
    """Train sequence transformer FM on FD001 with seed 43.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 43, "transformer")


@scenario("maintenance-rul-patchformer-sequence-conformer-fd001-s43")
def patchformer_sequence_conformer_fd001_s43() -> ScenarioSpec:
    """Train sequence conformer FM on FD001 with seed 43.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 43, "conformer")


@scenario("maintenance-rul-patchformer-sequence-cfc-fd001-s43")
def patchformer_sequence_cfc_fd001_s43() -> ScenarioSpec:
    """Train sequence cfc FM on FD001 with seed 43.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 43, "cfc")


@scenario("maintenance-rul-patchformer-sequence-transformer-fd001-s44")
def patchformer_sequence_transformer_fd001_s44() -> ScenarioSpec:
    """Train sequence transformer FM on FD001 with seed 44.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 44, "transformer")


@scenario("maintenance-rul-patchformer-sequence-conformer-fd001-s44")
def patchformer_sequence_conformer_fd001_s44() -> ScenarioSpec:
    """Train sequence conformer FM on FD001 with seed 44.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 44, "conformer")


@scenario("maintenance-rul-patchformer-sequence-cfc-fd001-s44")
def patchformer_sequence_cfc_fd001_s44() -> ScenarioSpec:
    """Train sequence cfc FM on FD001 with seed 44.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD001", 44, "cfc")


@scenario("maintenance-rul-patchformer-sequence-transformer-fd003-s42")
def patchformer_sequence_transformer_fd003_s42() -> ScenarioSpec:
    """Train sequence transformer FM on FD003 with seed 42.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 42, "transformer")


@scenario("maintenance-rul-patchformer-sequence-conformer-fd003-s42")
def patchformer_sequence_conformer_fd003_s42() -> ScenarioSpec:
    """Train sequence conformer FM on FD003 with seed 42.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 42, "conformer")


@scenario("maintenance-rul-patchformer-sequence-cfc-fd003-s42")
def patchformer_sequence_cfc_fd003_s42() -> ScenarioSpec:
    """Train sequence cfc FM on FD003 with seed 42.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 42, "cfc")


@scenario("maintenance-rul-patchformer-sequence-transformer-fd003-s43")
def patchformer_sequence_transformer_fd003_s43() -> ScenarioSpec:
    """Train sequence transformer FM on FD003 with seed 43.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 43, "transformer")


@scenario("maintenance-rul-patchformer-sequence-conformer-fd003-s43")
def patchformer_sequence_conformer_fd003_s43() -> ScenarioSpec:
    """Train sequence conformer FM on FD003 with seed 43.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 43, "conformer")


@scenario("maintenance-rul-patchformer-sequence-cfc-fd003-s43")
def patchformer_sequence_cfc_fd003_s43() -> ScenarioSpec:
    """Train sequence cfc FM on FD003 with seed 43.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 43, "cfc")


@scenario("maintenance-rul-patchformer-sequence-transformer-fd003-s44")
def patchformer_sequence_transformer_fd003_s44() -> ScenarioSpec:
    """Train sequence transformer FM on FD003 with seed 44.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 44, "transformer")


@scenario("maintenance-rul-patchformer-sequence-conformer-fd003-s44")
def patchformer_sequence_conformer_fd003_s44() -> ScenarioSpec:
    """Train sequence conformer FM on FD003 with seed 44.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 44, "conformer")


@scenario("maintenance-rul-patchformer-sequence-cfc-fd003-s44")
def patchformer_sequence_cfc_fd003_s44() -> ScenarioSpec:
    """Train sequence cfc FM on FD003 with seed 44.

    Returns:
        Official-split-native, frozen-DPAN full-sequence transport experiment.
    """
    return patchformer_sequence_flow("FD003", 44, "cfc")
