"""Native PatchFormer-embedding Flow Matching from healthy to degraded RUL windows.

This stage is *not* a supervised RUL improvement claim. The frozen PatchFormer
was supervised upstream; here the NEX-196 vector field learns independent
healthy/high-RUL to degraded/low-RUL transport in its normalized latent space.
Validation uses pairing-invariant SWD, not official test-engine RUL labels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import DatasetSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.patchformer_flow import (
    AutoPatchFormerFlowEndpoints,
)
from nexuml_library.evaluation.maintenance.ablation_validation import (
    DistributionalTransportValidation,
)
from nexuml_library.scenarios.flow.audio_health_to_fault import (
    audio_health_to_fault_flow,
)
from nexuml_library.scenarios.training.defaults import default_logging


def patchformer_embedding_flow(
    subset: Literal["FD001", "FD003"] = "FD001",
    seed: int = 42,
    *,
    checkpoint: str | Path | None = None,
) -> ScenarioSpec:
    """Train the existing NexuML flow on frozen PatchFormer DPAN embeddings.

    Args:
        subset: Official engine-disjoint NASA C-MAPSS FD001 or FD003.
        seed: Paired study split/feature sampling/training seed.
        checkpoint: Trusted, trained PatchFormer best validation checkpoint.

    Returns:
        Typed Flow Matching pipeline with independent SWD-selected checkpoints.

    Raises:
        ValueError: If split seed is negative.
    """
    if seed < 0:
        raise ValueError("Flow Matching seed must be nonnegative")
    run = f"rul-{subset}-patchformer-latent-flow-seed{seed}"
    prepared = Path(
        f".experiments/rul/prepared/cmapss-{subset}-seed{seed}-"
        f"controlled-w30-s{seed}.pt"
    ).resolve()
    endpoint_file = Path(
        f".experiments/rul/prepared/{run}-endpoints.pt"
    ).resolve()
    root = (
        (os.environ.get("NEXUML_CMAPSS_ROOT") or "").strip()
        or "data/cmapss"
    )
    selected_checkpoint = str(
        checkpoint
        if checkpoint is not None
        else (os.environ.get("NEXUML_PATCHFORMER_CHECKPOINT") or "").strip()
    )
    source = AutoPatchFormerFlowEndpoints(
        subset=subset,
        seed=seed,
        root=root,
        checkpoint_path=selected_checkpoint,
        prepared_rul_path=str(prepared),
        output_file=str(endpoint_file),
        healthy_min_rul=100.0,
        faulty_max_rul=30.0,
        train_pairs=512,
        val_pairs=128,
    )
    hidden_dims = [256, 256]
    batch_size = 32
    base = audio_health_to_fault_flow(
        data_roots=["unused-patchformer-latent-input"],
        hidden_dims=hidden_dims,
        num_steps=50,
        lr=3e-4,
        batch_size=batch_size,
        max_epochs=250,
        num_workers=0,
        seed=seed,
    )
    stages = {
        name: layers
        for name, layers in base.pipeline.stages.items()
        if name != "AudioFeatures"
    }
    data = base.data.model_copy(
        update={
            "datasets": [
                DatasetSpec(source=source, modality="sensor", split_type="keep")
            ],
            "input_shapes": {
                "healthy_spectrum": [128],
                "faulty_spectrum": [128],
            },
        }
    )
    checkpoint_root = (
        (os.environ.get("NEXUML_RUL_CHECKPOINT_ROOT") or "").strip()
        or ".experiments/rul/checkpoints"
    )
    checkpoints = Path(checkpoint_root).expanduser().resolve() / run
    return base.model_copy(
        update={
            "name": run,
            "data": data,
            "pipeline": PipelineSpec(stages=stages),
            "logging": default_logging(
                experiment_name="NEX-233 PatchFormer latent FM",
                run_name=run,
                use_mlflow=False,
            ),
            "callbacks": [
                callback(
                    DistributionalTransportValidation,
                    prepared_file=str(endpoint_file),
                    batch_size=64,
                    projections=128,
                    seed=seed,
                ),
                callback(
                    ModelCheckpoint,
                    dirpath=str(checkpoints),
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


@scenario("maintenance-rul-patchformer-flow-fd001-s42")
def patchformer_flow_fd001_s42() -> ScenarioSpec:
    """Train frozen PatchFormer latent FM on FD001 with seed 42.

    Returns:
        Registered native NexuML flow experiment with frozen encoder.
    """
    return patchformer_embedding_flow("FD001", 42)


@scenario("maintenance-rul-patchformer-flow-fd001-s43")
def patchformer_flow_fd001_s43() -> ScenarioSpec:
    """Train frozen PatchFormer latent FM on FD001 with seed 43.

    Returns:
        Registered native NexuML flow experiment with frozen encoder.
    """
    return patchformer_embedding_flow("FD001", 43)


@scenario("maintenance-rul-patchformer-flow-fd001-s44")
def patchformer_flow_fd001_s44() -> ScenarioSpec:
    """Train frozen PatchFormer latent FM on FD001 with seed 44.

    Returns:
        Registered native NexuML flow experiment with frozen encoder.
    """
    return patchformer_embedding_flow("FD001", 44)


@scenario("maintenance-rul-patchformer-flow-fd003-s42")
def patchformer_flow_fd003_s42() -> ScenarioSpec:
    """Train frozen PatchFormer latent FM on FD003 with seed 42.

    Returns:
        Registered native NexuML flow experiment with frozen encoder.
    """
    return patchformer_embedding_flow("FD003", 42)


@scenario("maintenance-rul-patchformer-flow-fd003-s43")
def patchformer_flow_fd003_s43() -> ScenarioSpec:
    """Train frozen PatchFormer latent FM on FD003 with seed 43.

    Returns:
        Registered native NexuML flow experiment with frozen encoder.
    """
    return patchformer_embedding_flow("FD003", 43)


@scenario("maintenance-rul-patchformer-flow-fd003-s44")
def patchformer_flow_fd003_s44() -> ScenarioSpec:
    """Train frozen PatchFormer latent FM on FD003 with seed 44.

    Returns:
        Registered native NexuML flow experiment with frozen encoder.
    """
    return patchformer_embedding_flow("FD003", 44)
