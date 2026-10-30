"""Validation-only aggregation of independently restored FD003 CfC candidates.

Read each run's native best-checkpoint audit and its deterministic
train/validation evaluation-mode curve. Never use test labels or test metrics
to rank candidates. An incomplete or inconsistent campaign is rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

CONFIGURATIONS = ("baseline", "finetune", "all_no_trend")
SEEDS = (42, 43, 44)


def _finite_float(value: object, *, label: str) -> float:
    """Validate one finite nonnegative reported metric.

    Returns:
        Finite floating-point metric.

    Raises:
        ValueError: If the value is missing, malformed or negative.
    """
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {label}: {result!r}")
    return result


def _read_run(
    root: Path, configuration: str, seed: int
) -> dict[str, Any]:
    """Read the epoch at which this run's own val/loss checkpoint was saved.

    Returns:
        Validated run score, generalization diagnostics and timing metadata.

    Raises:
        FileNotFoundError: If any mandatory checkpoint evidence is absent.
        ValueError: If artifacts disagree on checkpoint identity or epochs.
    """
    run = f"rul-FD003-enhanced-cfc-confirm-{configuration}-s{seed}"
    rul_root = root.parents[2]
    audit_path = rul_root / "audits" / f"{run}.checkpoint.json"
    curve_path = root / f"{run}.curves.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("selection") != "best minimum val/loss"
        or audit.get("loaded_strict_model_state") is not True
    ):
        raise ValueError(f"Untrusted or incomplete best-checkpoint audit: {audit_path}")
    path = audit.get("best_model_path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Missing checkpoint reference in {audit_path}")
    selected = Path(path)
    if selected.parent.name != run or selected.name == "last.ckpt":
        raise ValueError(f"Unexpected checkpoint selection for {run}: {selected}")
    # The audit contains a pod-internal absolute path. Verify the identical
    # run and filename on the host's independently mounted Prisma volume.
    checkpoint = rul_root / "checkpoints" / run / selected.name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing validation-selected checkpoint: {checkpoint}")
    epoch = audit.get("checkpoint_epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError(f"Invalid selected checkpoint epoch for {run}: {epoch!r}")
    checkpoint_loss = _finite_float(
        audit.get("best_model_score"), label=f"{run} best val/loss"
    )
    with curve_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("epoch") == str(epoch)]
    if len(matches) != 1:
        raise ValueError(
            f"{run}: expected one evaluation-mode record at checkpoint epoch {epoch}"
        )
    row = matches[0]
    for field in ("train_windows", "val_windows", "val_engine_count"):
        value = row.get(field)
        if value is None or not value.isdigit() or int(value) <= 0:
            raise ValueError(f"{run}: missing {field} in selected validation curve")
    score = _finite_float(row.get("val_eval_mse"), label=f"{run} val_eval_mse")
    train_score = _finite_float(
        row.get("train_eval_mse"), label=f"{run} train_eval_mse"
    )
    timing_file = root / f"{configuration.replace('_', '-')}-s{seed}.execution.json"
    timing: dict[str, Any] | None = None
    if timing_file.is_file():
        loaded = json.loads(timing_file.read_text(encoding="utf-8"))
        if (
            not isinstance(loaded, dict)
            or loaded.get("configuration") != configuration
            or loaded.get("seed") != seed
            or loaded.get("timing_includes_fit_validate_test") is not True
        ):
            raise ValueError(f"Incompatible execution metadata: {timing_file}")
        timing = {
            "train_command_wall_seconds": _finite_float(
                loaded.get("train_command_wall_seconds"), label=str(timing_file)
            ),
            "kubernetes_node": loaded.get("kubernetes_node"),
        }
    return {
        "run": run,
        "configuration": configuration,
        "seed": seed,
        "checkpoint_epoch": epoch,
        "best_checkpoint_filename": selected.name,
        "checkpoint_val_loss": checkpoint_loss,
        "selected_val_eval_mse": score,
        "selected_train_eval_mse": train_score,
        "generalization_gap_mse": score - train_score,
        "worst_validation_group_mse": _finite_float(
            row.get("val_group_max_mse"), label=f"{run} group maximum"
        ),
        "validation_windows": int(row["val_windows"]),
        "training_windows": int(row["train_windows"]),
        "validation_engines": int(row["val_engine_count"]),
        "execution": timing,
    }


def summarize_independent_confirmation(root: Path) -> dict[str, Any]:
    """Summarize all nine separately checkpointed RUL runs.

    Args:
        root: Shared Prisma confirmation output directory.

    Returns:
        Per-seed validation scores, paired effects and engineering diagnostics.

    Raises:
        FileNotFoundError: If any run has not completed successfully.
        ValueError: If evidence is inconsistent or nonfinite.
    """
    runs: list[dict[str, Any]] = [
        _read_run(root, configuration, seed)
        for configuration in CONFIGURATIONS
        for seed in SEEDS
    ]
    shapes = {
        (
            run["training_windows"], run["validation_windows"],
            run["validation_engines"]
        )
        for run in runs
    }
    if len(shapes) != 1:
        raise ValueError("Independent confirmation runs do not share data split sizes")
    scores = {
        configuration: {
            seed: next(
                float(run["selected_val_eval_mse"]) for run in runs
                if run["configuration"] == configuration and run["seed"] == seed
            )
            for seed in SEEDS
        }
        for configuration in CONFIGURATIONS
    }
    ranked = sorted(
        CONFIGURATIONS, key=lambda configuration: statistics.mean(
            scores[configuration].values()
        )
    )
    ranking = [
        {
            "configuration": configuration,
            "selected_checkpoint_val_eval_mse_mean": statistics.mean(
                scores[configuration].values()
            ),
            "selected_checkpoint_val_eval_mse_std": statistics.stdev(
                scores[configuration].values()
            ),
            "by_seed": {str(seed): scores[configuration][seed] for seed in SEEDS},
        }
        for configuration in ranked
    ]
    differences = {
        configuration: {
            "vs_baseline_mean": statistics.mean(
                scores[configuration][seed] - scores["baseline"][seed]
                for seed in SEEDS
            ),
            "vs_baseline_by_seed": {
                str(seed): scores[configuration][seed] - scores["baseline"][seed]
                for seed in SEEDS
            },
        }
        for configuration in CONFIGURATIONS
        if configuration != "baseline"
    }
    return {
        "study": "NEX-233 independent native CfC confirmation",
        "study_version": "independent-checkpoint-v1",
        "subset": "FD003",
        "upstream_encoder_seed": 42,
        "head_seeds": list(SEEDS),
        "configurations": list(CONFIGURATIONS),
        "independent_optimizers": True,
        "independent_best_validation_checkpoints": True,
        "test_predictions_read": False,
        "metric": "evaluation-mode validation window MSE at each own val/loss-best checkpoint",
        "ranking": ranking,
        "paired_effects": differences,
        "runs": runs,
        "caveat": (
            "Exploratory: FD003 test metrics were previously inspected. "
            "All readout seeds share one upstream encoder and engine split. "
            "Job timing includes fit, final validate and test and depends on node hardware; "
            "do not interpret three seeds as independent test populations."
        ),
    }


def main() -> None:
    """Generate an auditable JSON report after all nine runs finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(".experiments/rul/enhanced-cfc/fd003/confirmation"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = summarize_independent_confirmation(args.root)
    output = args.output or args.root / "independent-confirmation-summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Validation-only independent confirmation: {output}")
    for item in result["ranking"]:
        print(
            f"{item['configuration']}: "
            f"{item['selected_checkpoint_val_eval_mse_mean']:.4f} "
            f"+/- {item['selected_checkpoint_val_eval_mse_std']:.4f}"
        )


if __name__ == "__main__":
    main()
