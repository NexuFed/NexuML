"""Independent validation-only confirmation of the NEX-233 CfC screening winners.

Nine native scenarios each train one architecture, select its own minimum
validation-loss checkpoint and preserve separate experiment/audit artifacts.
The FD003 seed-42 upstream PatchFormer and data split remain shared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec
from nexuml_library.evaluation.maintenance.rul_generalization_diagnostics import (
    SequenceRULGeneralizationDiagnostics,
)
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
    Goel2026RestoreBestWeights,
)
from nexuml_library.scenarios.maintenance.rul_enhanced_cfc import (
    enhanced_cfc_rul_study,
)

Confirmation = Literal["baseline", "finetune", "all_no_trend"]
CONFIRMATIONS: tuple[Confirmation, ...] = (
    "baseline",
    "finetune",
    "all_no_trend",
)
CONFIRMATION_SEEDS = (42, 43, 44)
CONFIRMATION_ROOT = ".experiments/rul/enhanced-cfc/fd003/confirmation"


def independent_cfc_confirmation(configuration: Confirmation, seed: int) -> ScenarioSpec:
    """Build an isolated, individually selected training run for one variant.

    Args:
        configuration: Screening candidate chosen before this confirmation.
        seed: Readout initialization/training seed; upstream remains seed 42.

    Returns:
        Native NexuML scenario with unique output files and checkpoint.

    Raises:
        ValueError: If configuration or seed is not a prespecified choice.
    """
    if configuration not in CONFIRMATIONS or seed not in CONFIRMATION_SEEDS:
        raise ValueError("Unexpected independent CfC confirmation configuration/seed")
    base = enhanced_cfc_rul_study(configuration, seed, "FD003")
    run = f"rul-FD003-enhanced-cfc-confirm-{configuration}-s{seed}"
    callbacks = list(base.callbacks)
    # Replace only artifact-related callbacks; retain the existing seed,
    # optimizer, early stopping, dataset, head and loss definitions.
    callbacks[1] = callback(
        SequenceRULGeneralizationDiagnostics,
        output_file=f"{CONFIRMATION_ROOT}/{run}.curves.csv",
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
    algorithms = list(base.evaluation.algorithms)
    audit = algorithms[-1]
    algorithms[-1] = audit.model_copy(
        update={
            "algorithm": audit.algorithm.model_copy(
                update={"output_file": f".experiments/rul/audits/{run}.csv"}
            )
        }
    )
    return base.model_copy(
        update={
            "name": run,
            "callbacks": callbacks,
            "evaluation": base.evaluation.model_copy(update={"algorithms": algorithms}),
            "logging": (
                base.logging.model_copy(update={"run_name": run})
                if base.logging is not None
                else None
            ),
        }
    )


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-baseline-s42")
def independent_baseline_s42() -> ScenarioSpec:
    """Independently train FD003 baseline with seed 42.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("baseline", 42)


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-baseline-s43")
def independent_baseline_s43() -> ScenarioSpec:
    """Independently train FD003 baseline with seed 43.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("baseline", 43)


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-baseline-s44")
def independent_baseline_s44() -> ScenarioSpec:
    """Independently train FD003 baseline with seed 44.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("baseline", 44)


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-finetune-s42")
def independent_finetune_s42() -> ScenarioSpec:
    """Independently train FD003 finetune with seed 42.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("finetune", 42)


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-finetune-s43")
def independent_finetune_s43() -> ScenarioSpec:
    """Independently train FD003 finetune with seed 43.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("finetune", 43)


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-finetune-s44")
def independent_finetune_s44() -> ScenarioSpec:
    """Independently train FD003 finetune with seed 44.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("finetune", 44)


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-all_no_trend-s42")
def independent_all_no_trend_s42() -> ScenarioSpec:
    """Independently train FD003 all_no_trend with seed 42.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("all_no_trend", 42)


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-all_no_trend-s43")
def independent_all_no_trend_s43() -> ScenarioSpec:
    """Independently train FD003 all_no_trend with seed 43.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("all_no_trend", 43)


@scenario("maintenance-rul-enhanced-cfc-confirm-fd003-all_no_trend-s44")
def independent_all_no_trend_s44() -> ScenarioSpec:
    """Independently train FD003 all_no_trend with seed 44.

    Returns:
        One validation-selected native NexuML RUL experiment.
    """
    return independent_cfc_confirmation("all_no_trend", 44)
