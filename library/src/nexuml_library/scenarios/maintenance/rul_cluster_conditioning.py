"""NEX-233 FD003: soft healthy/failure cluster conditions with no future leakage.

Four paired experiments: unchanged all_no_trend reference, healthy gate,
failure gate, and both gates. All fitting/selection uses training/validation
engines and the original seed-42 pretrained PatchFormer. Test is held out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import DatasetSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.rul_cluster_conditions import (
    AutoClusterConditionedCMapSSRULWindows,
)
from nexuml_library.data.maintenance.rul_enhanced import (
    AutoPairedCMapSSRULWindows,
)
from nexuml_library.evaluation.maintenance.rul_generalization_diagnostics import (
    SequenceRULGeneralizationDiagnostics,
)
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_cluster_auxiliary_loss import (
    RULClusterAuxiliaryLoss,
)
from nexuml_library.layers.model.rul_cluster_conditioned_cfc import (
    ClusterConditionedCfCRULHead,
)
from nexuml_library.layers.model.rul_enhanced_cfc import EnhancedCfCRULHead
from nexuml_library.scenarios.maintenance.rul_enhanced_cfc_followup import (
    followup_study,
)

ClusterArm = Literal["reference", "working", "failure", "both"]
ARMS: tuple[ClusterArm, ...] = ("reference", "working", "failure", "both")
SEEDS = (42, 43, 44)
ROOT = ".experiments/rul/enhanced-cfc/cluster-conditioned/fd003"
CLUSTER_PATH = ".experiments/rul/prepared/cmapss-FD003-seed42-cfc-conditional-clusters-v1.pt"


def cluster_conditioning_study(arm: ClusterArm, seed: int) -> ScenarioSpec:
    """Build one independently checkpointed FD003 soft-condition experiment.

    Args:
        arm: Unchanged reference, working-only, failure-only or both gates.
        seed: Matched independent CfC head/training seed.

    Returns:
        Native NexuML scenario with isolated artifacts and pure validation RUL loss.

    Raises:
        ValueError: If the experiment is outside the prespecified 4 x 3 matrix.
    """
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("Unsupported FD003 cluster-conditioning arm or seed")
    parent = followup_study("FD003", "all_no_trend", seed)
    run = f"rul-FD003-enhanced-cfc-cluster-{arm}-s{seed}"
    source = parent.data.datasets[0].source
    if not isinstance(source, AutoPairedCMapSSRULWindows):
        raise ValueError("Expected original engine-disjoint FD003 source")
    model_layer = parent.pipeline.stages["RULModel"][0]
    model = model_layer.component
    if not isinstance(model, EnhancedCfCRULHead):
        raise ValueError("Expected prior all_no_trend CfC model")

    stages = dict(parent.pipeline.stages)
    data = parent.data
    if arm != "reference":
        conditioned_source = AutoClusterConditionedCMapSSRULWindows(
            subset="FD003",
            seed=42,
            root=source.root,
            artifact_path=source.artifact_path,
            cluster_artifact=CLUSTER_PATH,
            working_clusters=3,
            failure_clusters=2,
            endpoint_windows=4,
        )
        data = data.model_copy(
            update={
                "datasets": [
                    DatasetSpec(source=conditioned_source, modality="sensor", split_type="keep")
                ],
            }
        )
        conditioned_head = ClusterConditionedCfCRULHead(
            subset="FD003",
            upstream_seed=42,
            initialization_seed=seed,
            pretrained_encoder=True,
            train_encoder=model.train_encoder,
            sensor_graph=model.sensor_graph,
            trend=False,
            multiscale=model.multiscale,
            segmentation=model.segmentation,
            trajectory=model.trajectory,
            dropout=model.dropout,
            condition=arm,
            cluster_artifact=CLUSTER_PATH,
            working_clusters=3,
            failure_clusters=2,
            temperature=1.0,
        )
        stages["RULModel"] = [model_layer.model_copy(update={"component": conditioned_head})]
        old_loss = stages["RULLoss"][0]
        loss = RULClusterAuxiliaryLoss(
            consistency_weight=0.1,
            working_weight=5.0 if arm in ("working", "both") else 0.0,
            failure_weight=5.0 if arm in ("failure", "both") else 0.0,
        )
        stages["RULLoss"] = [
            old_loss.model_copy(update={"component": loss}),
            *stages["RULLoss"][1:],
        ]

    callbacks = list(parent.callbacks)
    callbacks[0] = callback(RULSeedOnFitStart, seed=seed)
    callbacks[1] = callback(
        SequenceRULGeneralizationDiagnostics, output_file=f"{ROOT}/{run}.curves.csv"
    )
    callbacks[2] = callback(
        ModelCheckpoint,
        dirpath=str(Path(".experiments/rul/checkpoints").resolve() / run),
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks[3] = callback(
        EarlyStopping,
        monitor="val/loss",
        mode="min",
        patience=20,
    )
    callbacks[4] = callback(
        Goel2026RestoreBestWeights,
        output_file=f".experiments/rul/audits/{run}.checkpoint.json",
    )
    algorithms = list(parent.evaluation.algorithms)
    engine_audit = algorithms[-1]
    algorithms[-1] = engine_audit.model_copy(
        update={
            "algorithm": engine_audit.algorithm.model_copy(
                update={"output_file": f".experiments/rul/audits/{run}.csv"}
            )
        }
    )
    return parent.model_copy(
        update={
            "name": run,
            "data": data,
            "pipeline": PipelineSpec(stages=stages),
            "callbacks": callbacks,
            "evaluation": parent.evaluation.model_copy(update={"algorithms": algorithms}),
            "logging": (
                parent.logging.model_copy(update={"run_name": run})
                if parent.logging is not None
                else None
            ),
        }
    )


@scenario("maintenance-rul-cfc-cluster-fd003-reference-s42")
def cluster_fd003_reference_s42() -> ScenarioSpec:
    """Train FD003 CfC reference conditioning with paired seed 42.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("reference", 42)


@scenario("maintenance-rul-cfc-cluster-fd003-reference-s43")
def cluster_fd003_reference_s43() -> ScenarioSpec:
    """Train FD003 CfC reference conditioning with paired seed 43.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("reference", 43)


@scenario("maintenance-rul-cfc-cluster-fd003-reference-s44")
def cluster_fd003_reference_s44() -> ScenarioSpec:
    """Train FD003 CfC reference conditioning with paired seed 44.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("reference", 44)


@scenario("maintenance-rul-cfc-cluster-fd003-working-s42")
def cluster_fd003_working_s42() -> ScenarioSpec:
    """Train FD003 CfC working conditioning with paired seed 42.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("working", 42)


@scenario("maintenance-rul-cfc-cluster-fd003-working-s43")
def cluster_fd003_working_s43() -> ScenarioSpec:
    """Train FD003 CfC working conditioning with paired seed 43.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("working", 43)


@scenario("maintenance-rul-cfc-cluster-fd003-working-s44")
def cluster_fd003_working_s44() -> ScenarioSpec:
    """Train FD003 CfC working conditioning with paired seed 44.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("working", 44)


@scenario("maintenance-rul-cfc-cluster-fd003-failure-s42")
def cluster_fd003_failure_s42() -> ScenarioSpec:
    """Train FD003 CfC failure conditioning with paired seed 42.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("failure", 42)


@scenario("maintenance-rul-cfc-cluster-fd003-failure-s43")
def cluster_fd003_failure_s43() -> ScenarioSpec:
    """Train FD003 CfC failure conditioning with paired seed 43.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("failure", 43)


@scenario("maintenance-rul-cfc-cluster-fd003-failure-s44")
def cluster_fd003_failure_s44() -> ScenarioSpec:
    """Train FD003 CfC failure conditioning with paired seed 44.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("failure", 44)


@scenario("maintenance-rul-cfc-cluster-fd003-both-s42")
def cluster_fd003_both_s42() -> ScenarioSpec:
    """Train FD003 CfC both conditioning with paired seed 42.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("both", 42)


@scenario("maintenance-rul-cfc-cluster-fd003-both-s43")
def cluster_fd003_both_s43() -> ScenarioSpec:
    """Train FD003 CfC both conditioning with paired seed 43.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("both", 43)


@scenario("maintenance-rul-cfc-cluster-fd003-both-s44")
def cluster_fd003_both_s44() -> ScenarioSpec:
    """Train FD003 CfC both conditioning with paired seed 44.

    Returns:
        Native leakage-safe soft-conditioning experiment.
    """
    return cluster_conditioning_study("both", 44)
