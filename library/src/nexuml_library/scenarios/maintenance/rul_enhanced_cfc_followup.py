"""NEX-233: independent 10-seed reliability, leave-one-out and FD001 replication.

Reuse the exact existing CfC head and training setup. Only controlled
enhancement toggles, head/train RNG seed, and run-specific artifacts differ.
The FD001 replication trains its own domain-specific model; it does not
evaluate FD003 weights on FD001 sensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import PipelineSpec, ScenarioSpec
from nexuml_library.evaluation.maintenance.rul_generalization_diagnostics import (
    SequenceRULGeneralizationDiagnostics,
)
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_enhanced_cfc import EnhancedCfCRULHead
from nexuml_library.layers.model.rul_trajectory_loss import (
    RULTrajectoryConsistencyLoss,
)
from nexuml_library.scenarios.maintenance.rul_enhanced_cfc import (
    enhanced_cfc_rul_study,
)

FollowupConfiguration = Literal[
    "baseline",
    "all_no_trend",
    "without_finetune",
    "without_graph",
    "without_multiscale",
    "without_segmentation",
    "without_trajectory",
]
VALIDATION_ARMS: tuple[FollowupConfiguration, ...] = (
    "baseline", "all_no_trend",
)
LEAVE_ONE_OUT: tuple[FollowupConfiguration, ...] = (
    "without_finetune",
    "without_graph",
    "without_multiscale",
    "without_segmentation",
    "without_trajectory",
)
RELIABILITY_SEEDS = tuple(range(42, 52))
ABLATION_SEEDS = (42, 43, 44)
TRANSFER_SEEDS = (42, 43, 44)
ROOT = ".experiments/rul/enhanced-cfc/followup"


def followup_run_name(
    subset: Literal["FD001", "FD003"], configuration: FollowupConfiguration, seed: int
) -> str:
    """Name one independently fitted trial without touching prior checkpoints.

    Returns:
        Experiment-specific run prefix for audits and checkpoints.
    """
    return f"rul-{subset}-enhanced-cfc-followup-{configuration}-s{seed}"


def followup_study(
    subset: Literal["FD001", "FD003"],
    configuration: FollowupConfiguration,
    seed: int,
) -> ScenarioSpec:
    """Build one native, independently checkpointed follow-up experiment.

    Args:
        subset: Original FD003 study or separately trained FD001 replication.
        configuration: Baseline, full no-trend, or one additional factor removed.
        seed: Readout model initialization and fitting RNG seed.

    Returns:
        Native NexuML scenario with isolated callbacks and artifacts.

    Raises:
        ValueError: If the prespecified experiment matrix is violated.
    """
    if subset == "FD001":
        valid = configuration in VALIDATION_ARMS and seed in TRANSFER_SEEDS
    else:
        valid = (
            configuration in VALIDATION_ARMS and seed in RELIABILITY_SEEDS
        ) or (configuration in LEAVE_ONE_OUT and seed in ABLATION_SEEDS)
    if not valid:
        raise ValueError("Unsupported CfC follow-up subset/configuration/seed")

    # The parent constructor exposes only the original screening seeds.
    # Build an identical definition and replace both reproducibility seeds,
    # without modifying any upstream seed-42 source or split.
    parent = "baseline" if configuration == "baseline" else "all_no_trend"
    base = enhanced_cfc_rul_study(parent, 42, subset)
    model_layer = base.pipeline.stages["RULModel"][0]
    model = model_layer.component
    if not isinstance(model, EnhancedCfCRULHead):
        raise ValueError("Expected enhanced CfC model component")
    switches: dict[str, bool | int] = {"initialization_seed": seed}
    omitted = configuration.removeprefix("without_")
    if configuration in LEAVE_ONE_OUT:
        field = "train_encoder" if omitted == "finetune" else omitted
        switches[field] = False
    new_head = model.model_copy(update=switches)
    new_model_layer = model_layer.model_copy(update={"component": new_head})

    loss_layer = base.pipeline.stages["RULLoss"][0]
    loss = loss_layer.component
    if not isinstance(loss, RULTrajectoryConsistencyLoss):
        raise ValueError("Expected trajectory consistency loss")
    if configuration == "without_trajectory":
        loss = loss.model_copy(update={"consistency_weight": 0.0})
    new_loss_layer = loss_layer.model_copy(update={"component": loss})
    stages = dict(base.pipeline.stages)
    stages["RULModel"] = [new_model_layer]
    stages["RULLoss"] = [new_loss_layer, *stages["RULLoss"][1:]]

    run = followup_run_name(subset, configuration, seed)
    output = f"{ROOT}/{subset.lower()}"
    callbacks = list(base.callbacks)
    callbacks[0] = callback(RULSeedOnFitStart, seed=seed)
    callbacks[1] = callback(
        SequenceRULGeneralizationDiagnostics,
        output_file=f"{output}/{run}.curves.csv",
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
        EarlyStopping, monitor="val/loss", mode="min", patience=20,
    )
    callbacks[4] = callback(
        Goel2026RestoreBestWeights,
        output_file=f".experiments/rul/audits/{run}.checkpoint.json",
    )
    algorithms = list(base.evaluation.algorithms)
    engine_audit = algorithms[-1]
    algorithms[-1] = engine_audit.model_copy(
        update={
            "algorithm": engine_audit.algorithm.model_copy(
                update={"output_file": f".experiments/rul/audits/{run}.csv"}
            )
        }
    )
    return base.model_copy(
        update={
            "name": run,
            "pipeline": PipelineSpec(stages=stages),
            "callbacks": callbacks,
            "evaluation": base.evaluation.model_copy(update={"algorithms": algorithms}),
            "logging": (
                base.logging.model_copy(update={"run_name": run})
                if base.logging is not None
                else None
            ),
        }
    )


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s42")
def followup_fd003_baseline_s42() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s42")
def followup_fd003_all_no_trend_s42() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s43")
def followup_fd003_baseline_s43() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s43")
def followup_fd003_all_no_trend_s43() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s44")
def followup_fd003_baseline_s44() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 44)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s44")
def followup_fd003_all_no_trend_s44() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 44)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s45")
def followup_fd003_baseline_s45() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 45 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 45)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s45")
def followup_fd003_all_no_trend_s45() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 45 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 45)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s46")
def followup_fd003_baseline_s46() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 46 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 46)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s46")
def followup_fd003_all_no_trend_s46() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 46 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 46)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s47")
def followup_fd003_baseline_s47() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 47 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 47)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s47")
def followup_fd003_all_no_trend_s47() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 47 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 47)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s48")
def followup_fd003_baseline_s48() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 48 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 48)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s48")
def followup_fd003_all_no_trend_s48() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 48 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 48)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s49")
def followup_fd003_baseline_s49() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 49 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 49)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s49")
def followup_fd003_all_no_trend_s49() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 49 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 49)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s50")
def followup_fd003_baseline_s50() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 50 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 50)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s50")
def followup_fd003_all_no_trend_s50() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 50 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 50)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-baseline-s51")
def followup_fd003_baseline_s51() -> ScenarioSpec:
    """Run the independently selected FD003 baseline seed 51 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "baseline", 51)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-all_no_trend-s51")
def followup_fd003_all_no_trend_s51() -> ScenarioSpec:
    """Run the independently selected FD003 all_no_trend seed 51 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "all_no_trend", 51)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_finetune-s42")
def followup_fd003_without_finetune_s42() -> ScenarioSpec:
    """Run the independently selected FD003 without_finetune seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_finetune", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_finetune-s43")
def followup_fd003_without_finetune_s43() -> ScenarioSpec:
    """Run the independently selected FD003 without_finetune seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_finetune", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_finetune-s44")
def followup_fd003_without_finetune_s44() -> ScenarioSpec:
    """Run the independently selected FD003 without_finetune seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_finetune", 44)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_graph-s42")
def followup_fd003_without_graph_s42() -> ScenarioSpec:
    """Run the independently selected FD003 without_graph seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_graph", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_graph-s43")
def followup_fd003_without_graph_s43() -> ScenarioSpec:
    """Run the independently selected FD003 without_graph seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_graph", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_graph-s44")
def followup_fd003_without_graph_s44() -> ScenarioSpec:
    """Run the independently selected FD003 without_graph seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_graph", 44)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_multiscale-s42")
def followup_fd003_without_multiscale_s42() -> ScenarioSpec:
    """Run the independently selected FD003 without_multiscale seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_multiscale", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_multiscale-s43")
def followup_fd003_without_multiscale_s43() -> ScenarioSpec:
    """Run the independently selected FD003 without_multiscale seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_multiscale", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_multiscale-s44")
def followup_fd003_without_multiscale_s44() -> ScenarioSpec:
    """Run the independently selected FD003 without_multiscale seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_multiscale", 44)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_segmentation-s42")
def followup_fd003_without_segmentation_s42() -> ScenarioSpec:
    """Run the independently selected FD003 without_segmentation seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_segmentation", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_segmentation-s43")
def followup_fd003_without_segmentation_s43() -> ScenarioSpec:
    """Run the independently selected FD003 without_segmentation seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_segmentation", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_segmentation-s44")
def followup_fd003_without_segmentation_s44() -> ScenarioSpec:
    """Run the independently selected FD003 without_segmentation seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_segmentation", 44)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_trajectory-s42")
def followup_fd003_without_trajectory_s42() -> ScenarioSpec:
    """Run the independently selected FD003 without_trajectory seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_trajectory", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_trajectory-s43")
def followup_fd003_without_trajectory_s43() -> ScenarioSpec:
    """Run the independently selected FD003 without_trajectory seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_trajectory", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd003-without_trajectory-s44")
def followup_fd003_without_trajectory_s44() -> ScenarioSpec:
    """Run the independently selected FD003 without_trajectory seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD003", "without_trajectory", 44)


@scenario("maintenance-rul-enhanced-cfc-followup-fd001-baseline-s42")
def followup_fd001_baseline_s42() -> ScenarioSpec:
    """Run the independently selected FD001 baseline seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD001", "baseline", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd001-baseline-s43")
def followup_fd001_baseline_s43() -> ScenarioSpec:
    """Run the independently selected FD001 baseline seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD001", "baseline", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd001-baseline-s44")
def followup_fd001_baseline_s44() -> ScenarioSpec:
    """Run the independently selected FD001 baseline seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD001", "baseline", 44)


@scenario("maintenance-rul-enhanced-cfc-followup-fd001-all_no_trend-s42")
def followup_fd001_all_no_trend_s42() -> ScenarioSpec:
    """Run the independently selected FD001 all_no_trend seed 42 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD001", "all_no_trend", 42)


@scenario("maintenance-rul-enhanced-cfc-followup-fd001-all_no_trend-s43")
def followup_fd001_all_no_trend_s43() -> ScenarioSpec:
    """Run the independently selected FD001 all_no_trend seed 43 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD001", "all_no_trend", 43)


@scenario("maintenance-rul-enhanced-cfc-followup-fd001-all_no_trend-s44")
def followup_fd001_all_no_trend_s44() -> ScenarioSpec:
    """Run the independently selected FD001 all_no_trend seed 44 study.

    Returns:
        Native RUL training and validation scenario.
    """
    return followup_study("FD001", "all_no_trend", 44)


