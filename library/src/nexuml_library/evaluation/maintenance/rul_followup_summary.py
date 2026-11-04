"""Validation-only report for the NEX-233 independent CfC follow-up campaign.

The report uses only selected-checkpoint audits, deterministic validation
curves and Job completion metadata. It never reads test predictions/scores.
FD001 is a separately trained benchmark, not FD003-to-FD001 transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

ARMS = ("baseline", "all_no_trend")
OMISSIONS = (
    "without_finetune", "without_graph", "without_multiscale",
    "without_segmentation", "without_trajectory",
)
RELIABILITY_SEEDS = tuple(range(42, 52))
ABLATION_SEEDS = (42, 43, 44)
FD001_SEEDS = (42, 43, 44)
IMAGE = "ghcr.io/nexufed/nexuml:sha-f8d5490-cuda12.8.1"
_CAMPAIGNS = ("reliability", "ablation", "fd001")
_T95_DF9 = 2.2621571627409915


def _metric(value: object, *, description: str) -> float:
    """Parse and validate a finite nonnegative RUL error metric.

    Returns:
        Nonnegative finite metric.

    Raises:
        ValueError: For missing, nonnumeric, infinite or negative values.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Invalid {description}: {value!r}")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid {description}: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {description}: {result!r}")
    return result


def _run_name(subset: str, configuration: str, seed: int) -> str:
    return f"rul-{subset}-enhanced-cfc-followup-{configuration}-s{seed}"


def _read_run(
    root: Path, subset: str, configuration: str, seed: int, campaign: str
) -> dict[str, Any]:
    """Read one complete best-checkpoint validation result.

    Returns:
        Audited per-seed training and validation evidence.

    Raises:
        FileNotFoundError: If any mandatory result is missing.
        ValueError: If checkpoint identity, labels or metadata are inconsistent.
    """
    run = _run_name(subset, configuration, seed)
    subset_root = root / subset.lower()
    rul_root = root.parents[1]
    audit_path = rul_root / "audits" / f"{run}.checkpoint.json"
    checkpoint_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(checkpoint_audit, dict)
        or checkpoint_audit.get("selection") != "best minimum val/loss"
        or checkpoint_audit.get("loaded_strict_model_state") is not True
    ):
        raise ValueError(f"Untrusted checkpoint audit: {audit_path}")
    selected_name = checkpoint_audit.get("best_model_path")
    epoch = checkpoint_audit.get("checkpoint_epoch")
    if (
        not isinstance(selected_name, str)
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 0
    ):
        raise ValueError(f"Invalid best-checkpoint metadata: {run}")
    selected = Path(selected_name)
    checkpoint = rul_root / "checkpoints" / run / selected.name
    if (
        selected.parent.name != run
        or not selected.name.endswith(".ckpt")
        or selected.name == "last.ckpt"
        or not checkpoint.is_file()
    ):
        raise ValueError(f"Checkpoint missing or not selected for {run}: {checkpoint}")
    recorded_loss = _metric(
        checkpoint_audit.get("best_model_score"), description=f"{run} checkpoint loss"
    )
    with (subset_root / f"{run}.curves.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    matching = [row for row in rows if row.get("epoch") == str(epoch)]
    if len(matching) != 1:
        raise ValueError(f"Missing/duplicate validation curve at epoch {epoch}: {run}")
    selected_row = matching[0]
    val = _metric(selected_row.get("val_eval_mse"), description=f"{run} val MSE")
    train = _metric(selected_row.get("train_eval_mse"), description=f"{run} train MSE")
    worst = _metric(
        selected_row.get("val_group_max_mse"), description=f"{run} worst group"
    )
    if abs(recorded_loss - val) > 0.1:
        raise ValueError(f"Selected val/loss disagrees with validation curve: {run}")
    shapes = []
    for field in ("train_windows", "val_windows", "val_engine_count"):
        n = selected_row.get(field)
        if not isinstance(n, str) or not n.isdecimal() or int(n) <= 0:
            raise ValueError(f"Invalid {field} in validation diagnostics: {run}")
        shapes.append(int(n))

    marker_path = subset_root / f"{configuration.replace('_', '-')}-s{seed}.execution.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    digest = marker.get("source_sha256") if isinstance(marker, dict) else None
    if (
        not isinstance(marker, dict)
        or marker.get("subset") != subset
        or marker.get("configuration") != configuration
        or marker.get("campaign") != campaign
        or marker.get("seed") != seed
        or marker.get("container_image") != IMAGE
        or marker.get("timing_includes_fit_validate_test") is not True
        or not isinstance(digest, str)
        or re.fullmatch(r"[a-f0-9]{64}", digest) is None
    ):
        raise ValueError(f"Inconsistent completion marker: {marker_path}")
    wall_seconds = _metric(
        marker.get("train_command_wall_seconds"), description=f"{run} runtime"
    )
    return {
        "subset": subset,
        "campaign": campaign,
        "configuration": configuration,
        "seed": seed,
        "run": run,
        "checkpoint_epoch": epoch,
        "checkpoint_val_loss": recorded_loss,
        "selected_val_eval_mse": val,
        "selected_train_eval_mse": train,
        "generalization_gap_mse": val - train,
        "worst_validation_group_mse": worst,
        "training_windows": shapes[0],
        "validation_windows": shapes[1],
        "validation_engines": shapes[2],
        "source_sha256": digest,
        "container_image": IMAGE,
        "kubernetes_node": marker.get("kubernetes_node"),
        "train_command_wall_seconds": wall_seconds,
    }


def _summarize_arm(
    records: list[dict[str, Any]], configuration: str, seeds: tuple[int, ...]
) -> dict[str, Any]:
    vals = {
        seed: next(
            float(row["selected_val_eval_mse"])
            for row in records
            if row["configuration"] == configuration and row["seed"] == seed
        )
        for seed in seeds
    }
    scores = list(vals.values())
    return {
        "configuration": configuration,
        "validation_mse_mean": statistics.mean(scores),
        "validation_mse_std": statistics.stdev(scores),
        "by_seed": {str(seed): vals[seed] for seed in seeds},
    }


def _paired(
    records: list[dict[str, Any]],
    reference: str,
    candidate: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Measure paired candidate-minus-reference validation differences.

    Returns:
        Mean, sample SD, per-seed differences and improvement count.
    """
    data = {
        seed: {
            row["configuration"]: float(row["selected_val_eval_mse"])
            for row in records
            if row["seed"] == seed and row["configuration"] in (reference, candidate)
        }
        for seed in seeds
    }
    diffs = [data[seed][candidate] - data[seed][reference] for seed in seeds]
    mean = statistics.mean(diffs)
    reference_mean = statistics.mean(data[seed][reference] for seed in seeds)
    report: dict[str, Any] = {
        "reference": reference,
        "candidate": candidate,
        "candidate_minus_reference_mean": mean,
        "candidate_minus_reference_std": statistics.stdev(diffs),
        "relative_mean_change_pct": 100.0 * mean / reference_mean,
        "improved_seeds": sum(diff < 0 for diff in diffs),
        "total_seeds": len(seeds),
        "by_seed": {str(seed): diff for seed, diff in zip(seeds, diffs, strict=True)},
    }
    if len(seeds) == 10:
        margin = _T95_DF9 * statistics.stdev(diffs) / math.sqrt(len(seeds))
        report["paired_mean_95pct_t_interval"] = [mean - margin, mean + margin]
        report["interval_note"] = (
            "Exploratory t interval over readout seeds only; "
            "shared encoder and engine split, not independent dataset populations."
        )
    return report


def summarize_followup(root: Path, campaign: str = "all") -> dict[str, Any]:
    """Validate and aggregate a complete predefined follow-up campaign.

    Args:
        root: Prisma directory .experiments/rul/enhanced-cfc/followup.
        campaign: reliability, ablation, fd001, or all.

    Returns:
        Validation-only paired-effect results, reproducibility and provenance.

    Raises:
        ValueError: If campaign selection, artifacts, splits or sources disagree.
        FileNotFoundError: If an experiment did not complete.
    """
    if campaign not in (*_CAMPAIGNS, "all"):
        raise ValueError(f"Unknown follow-up campaign: {campaign}")
    selected = _CAMPAIGNS if campaign == "all" else (campaign,)
    read: dict[tuple[str, str, int], dict[str, Any]] = {}

    def load(subset: str, config: str, seed: int, phase: str) -> None:
        key = (subset, config, seed)
        if key not in read:
            read[key] = _read_run(root, subset, config, seed, phase)

    if "reliability" in selected:
        for config in ARMS:
            for seed in RELIABILITY_SEEDS:
                load("FD003", config, seed, "reliability")
    if "ablation" in selected:
        for seed in ABLATION_SEEDS:
            load("FD003", "all_no_trend", seed, "reliability")
        for config in OMISSIONS:
            for seed in ABLATION_SEEDS:
                load("FD003", config, seed, "ablation")
    if "fd001" in selected:
        for config in ARMS:
            for seed in FD001_SEEDS:
                load("FD001", config, seed, "fd001")
    rows = list(read.values())
    if len({row["source_sha256"] for row in rows}) != 1:
        raise ValueError("Follow-up runs were trained from different staged sources")
    for subset in ("FD003", "FD001"):
        shapes = {
            (row["training_windows"], row["validation_windows"], row["validation_engines"])
            for row in rows
            if row["subset"] == subset
        }
        if len(shapes) > 1:
            raise ValueError(f"Different training/validation engine splits in {subset}")

    analysis: dict[str, Any] = {}
    if "reliability" in selected:
        fd003 = [row for row in rows if row["subset"] == "FD003"]
        analysis["reliability"] = {
            "seeds": list(RELIABILITY_SEEDS),
            "ranking": sorted(
                [_summarize_arm(fd003, config, RELIABILITY_SEEDS) for config in ARMS],
                key=lambda item: item["validation_mse_mean"],
            ),
            "paired_effect": _paired(
                fd003, "baseline", "all_no_trend", RELIABILITY_SEEDS
            ),
        }
    if "ablation" in selected:
        fd003 = [row for row in rows if row["subset"] == "FD003"]
        analysis["ablation"] = {
            "seeds": list(ABLATION_SEEDS),
            "full_reference": "all_no_trend",
            "leave_one_out": {
                config: _paired(fd003, "all_no_trend", config, ABLATION_SEEDS)
                for config in OMISSIONS
            },
        }
    if "fd001" in selected:
        fd001 = [row for row in rows if row["subset"] == "FD001"]
        analysis["fd001"] = {
            "seeds": list(FD001_SEEDS),
            "benchmark": (
                "Separately trained FD001 models with a different machine regime. "
                "Not a transfer evaluation of FD003-fitted weights."
            ),
            "ranking": sorted(
                [_summarize_arm(fd001, config, FD001_SEEDS) for config in ARMS],
                key=lambda item: item["validation_mse_mean"],
            ),
            "paired_effect": _paired(
                fd001, "baseline", "all_no_trend", FD001_SEEDS
            ),
        }
    return {
        "study": "NEX-233 independent CfC reliability, factor ablation and FD001 replication",
        "protocol": "NEX-233-followup-v1",
        "campaign": campaign,
        "source_sha256": rows[0]["source_sha256"],
        "container_image": IMAGE,
        "upstream_encoder_seed": 42,
        "checkpoint_selection": "minimum validation val/loss per independent training Job",
        "metric": "evaluation-mode validation window MSE at selected checkpoint epoch",
        "test_predictions_read": False,
        "analyses": analysis,
        "runs": rows,
        "caveats": [
            "FD003 was used in exploratory screening; this is not a fresh test study.",
            "Seeds vary only head initialization and fitting RNG; upstream seed remains 42.",
            "FD001 is independently trained on a different subset, not FD003 weight transfer.",
            "The interval is not a confidence interval over independent engine populations.",
            "Runtime includes training, final validation and test and is node dependent.",
        ],
    }


def main() -> None:
    """Write one machine-readable summary after all requested Jobs complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(".experiments/rul/enhanced-cfc/followup"),
    )
    parser.add_argument(
        "--campaign", choices=(*_CAMPAIGNS, "all"), default="all",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = summarize_followup(args.root, args.campaign)
    output = args.output or args.root / f"{args.campaign}-summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Validation-only NEX-233 {args.campaign}: {output}")
    for group, details in result["analyses"].items():
        if group == "ablation":
            print("ablation: individual leave-one-out effects in summary JSON")
        else:
            for row in details["ranking"]:
                print(f"{group}/{row['configuration']}: {row['validation_mse_mean']:.4f}")


if __name__ == "__main__":
    main()
