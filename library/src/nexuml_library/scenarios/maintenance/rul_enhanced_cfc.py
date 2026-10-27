"""Controlled FD003 CfC RUL improvements and grouped on/off architecture ablation.

One-factor additions and leave-one-out ablations are intentionally preferable
to inferring independent contributions from only a single all-on comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import DatasetSpec, LayerSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.rul_enhanced import (
    AutoPairedCMapSSRULWindows,
)
from nexuml_library.data.maintenance.rul_sequence_flow import (
    AutoPatchFormerSequenceFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_composite_comparison import (
    RULCompositeAblationComparison,
)
from nexuml_library.evaluation.maintenance.rul_generalization_diagnostics import (
    SequenceRULGeneralizationDiagnostics,
)
from nexuml_library.evaluation.maintenance.rul_mc_conformal import (
    RULMCConformalDiagnostics,
)
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_enhanced_cfc import EnhancedCfCRULHead
from nexuml_library.layers.model.rul_trajectory_loss import (
    RULTrajectoryConsistencyLoss,
)
from nexuml_library.scenarios.maintenance.rul_sequence_flow import (
    patchformer_sequence_flow_rul,
)

# Only mechanisms that change point estimation belong in this matrix.
# MC dropout/conformal calibration is evaluated on the SAME fitted model.
Option = Literal[
    "finetune", "graph", "trend", "multiscale", "segmentation", "trajectory",
]
OPTIONS: tuple[Option, ...] = (
    "finetune", "graph", "trend", "multiscale", "segmentation", "trajectory",
)
Ablation = Literal[
    "baseline",
    "finetune", "graph", "trend", "multiscale", "segmentation", "trajectory",
    "all", "all_no_finetune", "all_no_graph", "all_no_trend",
    "all_no_multiscale", "all_no_segmentation", "all_no_trajectory",
]
CONFIGURATIONS: tuple[Ablation, ...] = (
    "baseline", "finetune", "graph", "trend", "multiscale", "segmentation",
    "trajectory", "all", "all_no_finetune", "all_no_graph",
    "all_no_trend", "all_no_multiscale", "all_no_segmentation",
    "all_no_trajectory",
)


def enabled_options(configuration: Ablation) -> frozenset[Option]:
    """Resolve one-factor and leave-one-out enhancement comparisons.

    Args:
        configuration: Fully specified ablation configuration.

    Returns:
        Enabled optional mechanisms, independent of head seed.

    Raises:
        ValueError: If the configuration is unknown.
    """
    if configuration == "baseline":
        return frozenset()
    if configuration == "all":
        return frozenset(OPTIONS)
    if configuration.startswith("all_no_"):
        omitted = configuration.removeprefix("all_no_")
        return frozenset(option for option in OPTIONS if option != omitted)
    for option in OPTIONS:
        if configuration == option:
            return frozenset((option,))
    raise ValueError(f"Unknown CfC enhancement configuration: {configuration}")


def enhanced_cfc_rul_study(
    configuration: Ablation,
    head_seed: int = 42,
    subset: Literal["FD001", "FD003"] = "FD003",
) -> ScenarioSpec:
    """Compare every enhancement under identical FD003 CfC readout conditions.

    Args:
        configuration: Baseline, one-factor, all-on, or leave-one-out variant.
        head_seed: Independent readout seed, holding upstream seed 42 fixed.
        subset: FD003 confirmatory-style benchmark or FD001 exploratory control.

    Returns:
        Native NexuML scenario with raw sensor features and optional objectives.

    Raises:
        ValueError: If the seed or selected configuration is unregistered.
    """
    if head_seed not in (42, 43, 44):
        raise ValueError("Enhanced CfC study requires head seed 42, 43 or 44")
    if configuration not in CONFIGURATIONS:
        raise ValueError("Enhanced CfC configuration is not registered")
    options = enabled_options(configuration)
    base = patchformer_sequence_flow_rul(subset, 42, "cfc", False)
    run = f"rul-{subset}-enhanced-cfc-{configuration}-s{head_seed}"
    original = base.data.datasets[0].source
    if not isinstance(original, AutoPatchFormerSequenceFlowRULFeatures):
        raise ValueError("Enhanced CfC requires controlled native PatchFormer source")
    raw = AutoPairedCMapSSRULWindows(
        subset=subset,
        seed=42,
        root=original.root,
        artifact_path=original.prepared_rul_path,
    )
    channels = 15 if subset == "FD003" else 14
    data = base.data.model_copy(
        update={
            "datasets": [DatasetSpec(source=raw, modality="sensor", split_type="keep")],
            "input_shapes": {
                "rul_history": [30, channels],
                "previous_rul_history": [30, channels],
                "pair_mask": [1],
            },
            "feature_key": "rul_history",
        }
    )
    head = EnhancedCfCRULHead(
        subset=subset,
        upstream_seed=42,
        initialization_seed=head_seed,
        train_encoder="finetune" in options,
        sensor_graph="graph" in options,
        trend="trend" in options,
        multiscale="multiscale" in options,
        segmentation="segmentation" in options,
        trajectory="trajectory" in options,
        dropout=0.1,
    )
    stages = {
        "RULModel": [
            LayerSpec(
                component=head,
                keys_in=["rul_history"],
                keys_out=["predicted_rul", "previous_predicted_rul"],
            )
        ],
        "RULLoss": [
            LayerSpec(
                component=RULTrajectoryConsistencyLoss(
                    consistency_weight=0.1 if "trajectory" in options else 0.0
                ),
                keys_in=["predicted_rul"],
                keys_out=["rul_loss"],
                label_key="rul",
            )
        ],
    }
    callbacks = list(base.callbacks)
    callbacks[0] = callbacks[0].model_copy(update={"kwargs": {"seed": head_seed}})
    curves = f".experiments/rul/enhanced-cfc/{subset.lower()}/{run}.curves.csv"
    callbacks.insert(
        1,
        callback(SequenceRULGeneralizationDiagnostics, output_file=curves),
    )
    callbacks[2] = callbacks[2].model_copy(
        update={
            "kwargs": {
                "dirpath": str(Path(".experiments/rul/checkpoints").resolve() / run),
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": True,
            }
        }
    )
    callbacks[-1] = callbacks[-1].model_copy(
        update={"kwargs": {"output_file": f".experiments/rul/audits/{run}.checkpoint.json"}}
    )
    if configuration == "all":
        # Calibration observes the SAME all-on fitted model. It changes no
        # loss, head, training randomness or checkpoint selection.
        callbacks.append(
            callback(
                RULMCConformalDiagnostics,
                output_file=(
                    f".experiments/rul/enhanced-cfc/{subset.lower()}/{run}.uncertainty.json"
                ),
                stochastic_samples=20,
                alpha=0.1,
            )
        )
    evaluations = list(base.evaluation.algorithms)
    evaluations[-1] = evaluations[-1].model_copy(
        update={
            "algorithm": evaluations[-1].algorithm.model_copy(
                update={"output_file": f".experiments/rul/audits/{run}.csv"}
            )
        }
    )
    return base.model_copy(
        update={
            "name": run,
            "data": data,
            "pipeline": PipelineSpec(stages=stages),
            "training": base.training.model_copy(update={"lr": 1e-3}),
            "evaluation": base.evaluation.model_copy(update={"algorithms": evaluations}),
            "logging": (
                base.logging.model_copy(update={"run_name": run})
                if base.logging is not None
                else None
            ),
            "callbacks": callbacks,
        }
    )


@scenario("maintenance-rul-enhanced-cfc-fd003-baseline-s42")
def enhanced_cfc_baseline_s42() -> ScenarioSpec:
    """Train FD003 CfC with baseline enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("baseline", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-finetune-s42")
def enhanced_cfc_finetune_s42() -> ScenarioSpec:
    """Train FD003 CfC with finetune enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("finetune", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-graph-s42")
def enhanced_cfc_graph_s42() -> ScenarioSpec:
    """Train FD003 CfC with graph enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("graph", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-trend-s42")
def enhanced_cfc_trend_s42() -> ScenarioSpec:
    """Train FD003 CfC with trend enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("trend", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-multiscale-s42")
def enhanced_cfc_multiscale_s42() -> ScenarioSpec:
    """Train FD003 CfC with multiscale enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("multiscale", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-segmentation-s42")
def enhanced_cfc_segmentation_s42() -> ScenarioSpec:
    """Train FD003 CfC with segmentation enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("segmentation", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-trajectory-s42")
def enhanced_cfc_trajectory_s42() -> ScenarioSpec:
    """Train FD003 CfC with trajectory enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("trajectory", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all-s42")
def enhanced_cfc_all_s42() -> ScenarioSpec:
    """Train FD003 CfC with all enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_finetune-s42")
def enhanced_cfc_all_no_finetune_s42() -> ScenarioSpec:
    """Train FD003 CfC with all_no_finetune enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_finetune", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_graph-s42")
def enhanced_cfc_all_no_graph_s42() -> ScenarioSpec:
    """Train FD003 CfC with all_no_graph enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_graph", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_trend-s42")
def enhanced_cfc_all_no_trend_s42() -> ScenarioSpec:
    """Train FD003 CfC with all_no_trend enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_trend", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_multiscale-s42")
def enhanced_cfc_all_no_multiscale_s42() -> ScenarioSpec:
    """Train FD003 CfC with all_no_multiscale enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_multiscale", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_segmentation-s42")
def enhanced_cfc_all_no_segmentation_s42() -> ScenarioSpec:
    """Train FD003 CfC with all_no_segmentation enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_segmentation", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_trajectory-s42")
def enhanced_cfc_all_no_trajectory_s42() -> ScenarioSpec:
    """Train FD003 CfC with all_no_trajectory enhancements, readout seed 42.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_trajectory", 42, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-baseline-s43")
def enhanced_cfc_baseline_s43() -> ScenarioSpec:
    """Train FD003 CfC with baseline enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("baseline", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-finetune-s43")
def enhanced_cfc_finetune_s43() -> ScenarioSpec:
    """Train FD003 CfC with finetune enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("finetune", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-graph-s43")
def enhanced_cfc_graph_s43() -> ScenarioSpec:
    """Train FD003 CfC with graph enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("graph", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-trend-s43")
def enhanced_cfc_trend_s43() -> ScenarioSpec:
    """Train FD003 CfC with trend enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("trend", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-multiscale-s43")
def enhanced_cfc_multiscale_s43() -> ScenarioSpec:
    """Train FD003 CfC with multiscale enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("multiscale", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-segmentation-s43")
def enhanced_cfc_segmentation_s43() -> ScenarioSpec:
    """Train FD003 CfC with segmentation enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("segmentation", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-trajectory-s43")
def enhanced_cfc_trajectory_s43() -> ScenarioSpec:
    """Train FD003 CfC with trajectory enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("trajectory", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all-s43")
def enhanced_cfc_all_s43() -> ScenarioSpec:
    """Train FD003 CfC with all enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_finetune-s43")
def enhanced_cfc_all_no_finetune_s43() -> ScenarioSpec:
    """Train FD003 CfC with all_no_finetune enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_finetune", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_graph-s43")
def enhanced_cfc_all_no_graph_s43() -> ScenarioSpec:
    """Train FD003 CfC with all_no_graph enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_graph", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_trend-s43")
def enhanced_cfc_all_no_trend_s43() -> ScenarioSpec:
    """Train FD003 CfC with all_no_trend enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_trend", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_multiscale-s43")
def enhanced_cfc_all_no_multiscale_s43() -> ScenarioSpec:
    """Train FD003 CfC with all_no_multiscale enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_multiscale", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_segmentation-s43")
def enhanced_cfc_all_no_segmentation_s43() -> ScenarioSpec:
    """Train FD003 CfC with all_no_segmentation enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_segmentation", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_trajectory-s43")
def enhanced_cfc_all_no_trajectory_s43() -> ScenarioSpec:
    """Train FD003 CfC with all_no_trajectory enhancements, readout seed 43.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_trajectory", 43, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-baseline-s44")
def enhanced_cfc_baseline_s44() -> ScenarioSpec:
    """Train FD003 CfC with baseline enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("baseline", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-finetune-s44")
def enhanced_cfc_finetune_s44() -> ScenarioSpec:
    """Train FD003 CfC with finetune enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("finetune", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-graph-s44")
def enhanced_cfc_graph_s44() -> ScenarioSpec:
    """Train FD003 CfC with graph enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("graph", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-trend-s44")
def enhanced_cfc_trend_s44() -> ScenarioSpec:
    """Train FD003 CfC with trend enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("trend", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-multiscale-s44")
def enhanced_cfc_multiscale_s44() -> ScenarioSpec:
    """Train FD003 CfC with multiscale enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("multiscale", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-segmentation-s44")
def enhanced_cfc_segmentation_s44() -> ScenarioSpec:
    """Train FD003 CfC with segmentation enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("segmentation", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-trajectory-s44")
def enhanced_cfc_trajectory_s44() -> ScenarioSpec:
    """Train FD003 CfC with trajectory enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("trajectory", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all-s44")
def enhanced_cfc_all_s44() -> ScenarioSpec:
    """Train FD003 CfC with all enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_finetune-s44")
def enhanced_cfc_all_no_finetune_s44() -> ScenarioSpec:
    """Train FD003 CfC with all_no_finetune enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_finetune", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_graph-s44")
def enhanced_cfc_all_no_graph_s44() -> ScenarioSpec:
    """Train FD003 CfC with all_no_graph enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_graph", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_trend-s44")
def enhanced_cfc_all_no_trend_s44() -> ScenarioSpec:
    """Train FD003 CfC with all_no_trend enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_trend", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_multiscale-s44")
def enhanced_cfc_all_no_multiscale_s44() -> ScenarioSpec:
    """Train FD003 CfC with all_no_multiscale enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_multiscale", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_segmentation-s44")
def enhanced_cfc_all_no_segmentation_s44() -> ScenarioSpec:
    """Train FD003 CfC with all_no_segmentation enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_segmentation", 44, "FD003")


@scenario("maintenance-rul-enhanced-cfc-fd003-all_no_trajectory-s44")
def enhanced_cfc_all_no_trajectory_s44() -> ScenarioSpec:
    """Train FD003 CfC with all_no_trajectory enhancements, readout seed 44.

    Returns:
        Native benchmark for one registered architectural intervention.
    """
    return enhanced_cfc_rul_study("all_no_trajectory", 44, "FD003")


def enhanced_cfc_native_comparison(
    head_seed: int = 42,
) -> ScenarioSpec:
    """Compare all 14 FD003 CfC modeling variants inside one NexuML pipeline.

    Like `maintenance-rul-all`, this composite runs all branches under
    one optimizer and aggregate validation checkpoint. It exposes an
    individual validation MSE for every branch but is **not** equivalent
    to 16 independently restored model training runs.

    Args:
        head_seed: Seed of matched CfC readout initializations and data loading.

    Returns:
        Native composite ScenarioSpec with disjoint head parameters and losses.

    Raises:
        ValueError: If the requested seed is not a declared comparison seed.
        TypeError: If a comparison variant does not provide the expected
            enhanced CfC head or trajectory MSE loss.
    """
    if head_seed not in (42, 43, 44):
        raise ValueError("Enhanced CfC composite seed must be 42, 43 or 44")
    base = enhanced_cfc_rul_study("baseline", head_seed)
    run = f"rul-FD003-enhanced-cfc-compare-six-s{head_seed}"
    heads: list[LayerSpec] = []
    objectives: list[LayerSpec] = []
    loss_weights: dict[str, float] = {}
    for configuration in CONFIGURATIONS:
        variant = enhanced_cfc_rul_study(configuration, head_seed)
        head = variant.pipeline.stages["RULModel"][0].component
        objective = variant.pipeline.stages["RULLoss"][0].component
        if not isinstance(head, EnhancedCfCRULHead):
            raise TypeError("Composite comparison requires an enhanced CfC head")
        if not isinstance(objective, RULTrajectoryConsistencyLoss):
            raise TypeError("Composite comparison requires trajectory MSE loss")
        current_key = f"predicted_rul_{configuration}"
        previous_key = f"previous_predicted_rul_{configuration}"
        loss_key = f"rul_loss_{configuration}"
        heads.append(
            LayerSpec(
                component=head,
                keys_in=["rul_history"],
                keys_out=[current_key, previous_key],
            )
        )
        objectives.append(
            LayerSpec(
                component=objective.model_copy(update={"previous_key": previous_key}),
                keys_in=[current_key],
                keys_out=[loss_key],
                label_key="rul",
            )
        )
        loss_weights[loss_key] = 1.0 / len(CONFIGURATIONS)
    curves = f".experiments/rul/enhanced-cfc/fd003/{run}.validation.csv"
    checkpoint_dir = Path(".experiments/rul/checkpoints").resolve() / run
    callbacks = [
        callback(RULSeedOnFitStart, seed=head_seed),
        callback(
            RULCompositeAblationComparison,
            output_file=curves,
            seed=head_seed,
        ),
        callback(
            ModelCheckpoint,
            dirpath=str(checkpoint_dir),
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        callback(EarlyStopping, monitor="val/loss", mode="min", patience=20),
        callback(
            Goel2026RestoreBestWeights,
            output_file=f".experiments/rul/audits/{run}.checkpoint.json",
        ),
    ]
    return base.model_copy(
        update={
            "name": run,
            "pipeline": PipelineSpec(stages={"RULModels": heads, "RULLoss": objectives}),
            "training": base.training.model_copy(
                update={
                    "loss_keys": loss_weights,
                    "batch_size": 16,
                }
            ),
            "evaluation": base.evaluation.model_copy(update={"algorithms": []}),
            "logging": (
                base.logging.model_copy(update={"run_name": run})
                if base.logging is not None
                else None
            ),
            "callbacks": callbacks,
        }
    )


@scenario("maintenance-rul-enhanced-cfc-fd003-compare-six")
def enhanced_cfc_compare() -> ScenarioSpec:
    """Train and compare the six modeling interventions in one native run.

    Returns:
        Simultaneous matched-ablation scenario for head seed 42.
    """
    return enhanced_cfc_native_comparison(42)


@scenario("maintenance-rul-enhanced-cfc-fd003-compare-six-s43")
def enhanced_cfc_compare_s43() -> ScenarioSpec:
    """Compare the six FD003 modeling interventions with readout seed 43.

    Returns:
        Second native matched-ablation seed.
    """
    return enhanced_cfc_native_comparison(43)


@scenario("maintenance-rul-enhanced-cfc-fd003-compare-six-s44")
def enhanced_cfc_compare_s44() -> ScenarioSpec:
    """Compare the six FD003 modeling interventions with readout seed 44.

    Returns:
        Third native matched-ablation seed.
    """
    return enhanced_cfc_native_comparison(44)
