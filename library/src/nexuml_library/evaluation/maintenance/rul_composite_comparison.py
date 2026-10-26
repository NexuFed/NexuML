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

from nexuml_library.scenarios.maintenance.rul_enhanced_cfc import (
    CONFIGURATIONS,
    OPTIONS,
    enabled_options,
)


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
        if not self._rows:
            raise ValueError("No fitting-epoch composite CfC validation metrics")
        best: dict[str, dict[str, float | int]] = {}
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
            effects[option] = {
                "single_minus_baseline_mse": (
                    float(best[option]["best_observed_val_mse"])
                    - float(best["baseline"]["best_observed_val_mse"])
                ),
                "all_minus_leave_one_out_mse": (
                    float(best["all"]["best_observed_val_mse"])
                    - float(best[f"all_no_{option}"]["best_observed_val_mse"])
                ),
            }
        ranking = sorted(
            (
                {
                    "configuration": name,
                    "enabled": sorted(enabled_options(name)),
                    **metrics,
                }
                for name, metrics in best.items()
            ),
            key=lambda item: float(item["best_observed_val_mse"]),
        )
        report: dict[str, Any] = {
            "study": "NEX-233 NexuML native composite CfC enhancement comparison",
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
                "The 16 branches share one optimizer and aggregate early stopping. "
                "Their individual minima can occur at different epochs. "
                "This is an economical matched simultaneous ablation, not a "
                "replacement for separately trained and checkpoint-selected runs."
            ),
        }
        self.output_file.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
