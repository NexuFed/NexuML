"""Validation-only comparison of native composite CfC RUL ablation branches.

The combined study shares a NexuML optimizer and fit lifecycle. Each branch
has disjoint parameters and its own validation-MSE trace, but is NOT an
independently checkpoint-selected trial.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.trainer.states import TrainerFn


def summarize_composite_seeds(root: Path) -> dict[str, Any]:
    """Aggregate three comparable six-factor composite runs without test data.

    Args:
        root: Directory with the three native per-seed comparison JSON files.

    Returns:
        Validation-only per-arm statistics and paired seed-level effects.

    Raises:
        ValueError: If seeds, configurations or metric provenance disagree.
        FileNotFoundError: If any predeclared seed report is absent.
    """
    from nexuml_library.scenarios.maintenance.rul_enhanced_cfc import (
        CONFIGURATIONS,
        OPTIONS,
    )

    seeds = (42, 43, 44)
    scores: dict[str, dict[int, float]] = {
        name: {} for name in CONFIGURATIONS
    }
    for seed in seeds:
        file = root / f"rul-FD003-enhanced-cfc-compare-six-s{seed}.validation.json"
        report = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"Malformed composite report for seed {seed}")
        if (
            report.get("study_version") != "six-model-factors-v2"
            or report.get("subset") != "FD003"
            or report.get("head_seed") != seed
            or report.get("configurations") != list(CONFIGURATIONS)
            or report.get("test_predictions_read") is not False
            or report.get("optimizer_shared") is not True
        ):
            raise ValueError(f"Incompatible six-factor composite report: {file}")
        ranking = report.get("ranking")
        if not isinstance(ranking, list) or len(ranking) != len(CONFIGURATIONS):
            raise ValueError(f"Incomplete composite ranking for seed {seed}")
        for entry in ranking:
            if not isinstance(entry, dict):
                raise ValueError(f"Malformed ranked entry for seed {seed}")
            name = entry.get("configuration")
            if not isinstance(name, str) or name not in scores:
                raise ValueError(f"Unknown composite configuration for seed {seed}")
            if seed in scores[name]:
                raise ValueError(f"Duplicate composite configuration {name}")
            value = entry.get("best_observed_val_mse")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid validation MSE for {name}, seed {seed}")
            scores[name][seed] = float(value)
    if any(len(per_seed) != len(seeds) for per_seed in scores.values()):
        raise ValueError("Missing composite configuration scores")
    mean_scores: dict[str, float] = {
        name: statistics.mean(per_seed.values())
        for name, per_seed in scores.items()
    }
    effects: dict[str, dict[str, Any]] = {}
    for option in OPTIONS:
        omitted = next(
            (name for name in CONFIGURATIONS if name == f"all_no_{option}"),
            None,
        )
        if omitted is None:
            raise ValueError(f"Missing leave-one-out configuration: {option}")
        standalone = [
            scores[option][seed] - scores["baseline"][seed]
            for seed in seeds
        ]
        full = [
            scores["all"][seed] - scores[omitted][seed]
            for seed in seeds
        ]
        effects[option] = {
            "single_minus_baseline_mean": statistics.mean(standalone),
            "single_minus_baseline_by_seed": dict(zip(seeds, standalone)),
            "all_minus_leave_one_out_mean": statistics.mean(full),
            "all_minus_leave_one_out_by_seed": dict(zip(seeds, full)),
        }
    return {
        "study": "NEX-233 six-factor native composite CfC comparison across seeds",
        "study_version": "six-model-factors-v2",
        "subset": "FD003",
        "head_seeds": list(seeds),
        "optimizer_shared_within_each_seed": True,
        "independently_restored_arm_checkpoints": False,
        "point_estimation_only": True,
        "test_predictions_read": False,
        "ranking": [
            {
                "configuration": name,
                "validation_mse_mean": mean_scores[name],
                "validation_mse_std": statistics.stdev(scores[name].values()),
                "validation_mse_by_seed": {
                    str(seed): scores[name][seed] for seed in seeds
                },
            }
            for name in sorted(CONFIGURATIONS, key=lambda name: mean_scores[name])
        ],
        "paired_effects": effects,
        "caveat": (
            "Scores are each branch's best observed fitting-epoch validation MSE "
            "within one shared-optimizer composite, not independent model selection. "
            "FD003 has previously been inspected, so this is exploratory."
        ),
    }


class RULCompositeAblationComparison(L.Callback):
    """Collect validation MSE per branch without inspecting test predictions."""

    def __init__(self, output_file: str, seed: int) -> None:
        super().__init__()
        self.output_file = Path(output_file)
        self.seed = seed
        self._rows: list[dict[str, float | int]] = []

    def on_fit_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Clear old validation results when a new NexuML fitting run starts."""
        self._rows.clear()
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.unlink(missing_ok=True)
        self.output_file.with_suffix(".json").unlink(missing_ok=True)

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Record all per-arm validation losses for a fitting epoch.

        Raises:
            ValueError: If a branch's native validation metric is missing.
        """
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return
        from nexuml_library.scenarios.maintenance.rul_enhanced_cfc import (
            CONFIGURATIONS,
        )

        values: dict[str, float | int] = {"epoch": int(trainer.current_epoch)}
        for configuration in CONFIGURATIONS:
            key = f"val/rul_loss_{configuration}"
            raw = trainer.callback_metrics.get(key)
            if raw is None:
                raise ValueError(f"Composite CfC validation metric missing: {key}")
            value = float(raw.detach().cpu()) if hasattr(raw, "detach") else float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid validation MSE for {configuration}")
            values[configuration] = value
        self._rows.append(values)
        with self.output_file.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("epoch", *CONFIGURATIONS)
            )
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(values)

    def on_fit_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Summarize per-arm best observed validation MSE, never test labels.

        Raises:
            ValueError: If no complete fitting validation epoch was observed.
        """
        from nexuml_library.scenarios.maintenance.rul_enhanced_cfc import (
            Ablation,
            CONFIGURATIONS,
            OPTIONS,
            enabled_options,
        )

        if not self._rows:
            raise ValueError("No fitting-epoch composite CfC validation metrics")
        best: dict[Ablation, dict[str, float | int]] = {}
        for configuration in CONFIGURATIONS:
            record = min(
                self._rows, key=lambda row: float(row[configuration])
            )
            best[configuration] = {
                "best_observed_val_mse": float(record[configuration]),
                "best_epoch": int(record["epoch"]),
            }
        effects = {}
        for option in OPTIONS:
            omitted = next(
                (name for name in CONFIGURATIONS if name == f"all_no_{option}"),
                None,
            )
            if omitted is None:
                raise ValueError(f"Missing leave-one-out CfC configuration: {option}")
            effects[option] = {
                "single_minus_baseline_mse": (
                    float(best[option]["best_observed_val_mse"])
                    - float(best["baseline"]["best_observed_val_mse"])
                ),
                "all_minus_leave_one_out_mse": (
                    float(best["all"]["best_observed_val_mse"])
                    - float(best[omitted]["best_observed_val_mse"])
                ),
            }
        ranked = sorted(
            CONFIGURATIONS,
            key=lambda name: float(best[name]["best_observed_val_mse"]),
        )
        ranking: list[dict[str, Any]] = [
            {
                "configuration": name,
                "enabled": sorted(enabled_options(name)),
                **best[name],
            }
            for name in ranked
        ]
        report: dict[str, Any] = {
            "study": "NEX-233 NexuML native composite CfC enhancement comparison",
            "study_version": "six-model-factors-v2",
            "point_estimation_only": True,
            "subset": "FD003",
            "head_seed": self.seed,
            "configurations": list(CONFIGURATIONS),
            "fit_epochs": len(self._rows),
            "test_predictions_read": False,
            "optimizer_shared": True,
            "early_stopping_shared": True,
            "best_per_arm_epoch_is_observational": True,
            "independently_restored_arm_checkpoints": False,
            "ranking": ranking,
            "effects": effects,
            "caveat": (
                "The 14 branches share one optimizer and aggregate early stopping. "
                "Their individual minima can occur at different epochs. "
                "This is an economical matched simultaneous ablation, not a "
                "replacement for separately trained and checkpoint-selected runs."
            ),
        }
        self.output_file.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Produce a cross-seed report as soon as all three native runs exist.
        root = self.output_file.parent
        paths = [
            root / f"rul-FD003-enhanced-cfc-compare-six-s{seed}.validation.json"
            for seed in (42, 43, 44)
        ]
        if all(path.is_file() for path in paths):
            combined = summarize_composite_seeds(root)
            (root / "compare-six-seeds-42-43-44.summary.json").write_text(
                json.dumps(combined, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
