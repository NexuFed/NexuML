"""Paired FD001/FD003 RUL study summary from native NexuML audit outputs.

This reporting utility does not fit models or select hyperparameters.
It treats Goel's publication values as descriptive references only because
the authors' exact held-out label convention remains unspecified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

_PUBLISHED: dict[tuple[str, str], dict[str, float]] = {
    ("FD001", "lstm"): {"rmse": 14.931, "nasa": 410.97},
    ("FD001", "cnn"): {"rmse": 16.971, "nasa": 365.0},
    ("FD003", "lstm"): {"rmse": 14.204, "nasa": 409.1},
    ("FD003", "cnn"): {"rmse": 15.678, "nasa": 647.6},
}
_FIELDS = ("predicted_rul", "true_rul_capped", "true_rul_raw")


def _audit_filename(subset: str, architecture: str, kind: str, seed: int) -> str:
    """Construct the scenario-defined immutable engine audit basename.

    Returns:
        Relative CSV basename for one paired FD001/FD003 run.
    """
    prefix = f"rul-cmapss-{subset}-{architecture}-mse-seed{seed}"
    suffix = (
        f"goel2026-{architecture}-protocol-raw"
        if kind == "goel"
        else f"safety-{architecture}-asymmetric-w2"
    )
    return f"{prefix}-{suffix}.csv"


def _read_engine_audit(path: Path) -> dict[str, dict[str, float]]:
    """Load a complete official test audit while rejecting invalid residuals.

    Returns:
        Mapping of the 100 official engines to predictions and both target laws.

    Raises:
        FileNotFoundError: If a required experiment has not been run.
        ValueError: If a CSV has missing, duplicated, or inconsistent observations.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment audit: {path}")
    records: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(_FIELDS + ("asset_id",)).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required RUL columns in {path}")
        for row in reader:
            engine = row["asset_id"]
            if not engine.startswith("test:") or engine in records:
                raise ValueError(f"Unexpected or duplicate test engine: {engine}")
            entry = {field: float(row[field]) for field in _FIELDS}
            if not all(math.isfinite(value) for value in entry.values()):
                raise ValueError(f"Non-finite audit value for {engine}")
            records[engine] = entry
    if len(records) != 100:
        raise ValueError(f"Expected 100 official engines in {path}, got {len(records)}")
    return records


def _metrics(records: dict[str, dict[str, float]], label: str) -> dict[str, float]:
    """Calculate sample-weighted RUL metrics in original cycle units.

    Returns:
        RMSE, MAE, positive-error mean, and summed NASA asymmetric score.
    """
    errors = [
        row["predicted_rul"] - row[f"true_rul_{label}"]
        for row in records.values()
    ]
    return {
        "rmse": math.sqrt(statistics.mean(error * error for error in errors)),
        "mae": statistics.mean(abs(error) for error in errors),
        "late_bias": statistics.mean(max(error, 0.0) for error in errors),
        "nasa": sum(
            math.expm1(error / 10.0 if error >= 0 else -error / 13.0)
            for error in errors
        ),
    }


def summarize_reproducibility(
    audit_root: Path,
    *,
    seeds: tuple[int, ...] = (42, 43, 44),
    subsets: tuple[str, ...] = ("FD001", "FD003"),
    architectures: tuple[str, ...] = ("lstm", "cnn"),
) -> dict[str, object]:
    """Aggregate paired test residuals without tuning on official test sets.

    Returns:
        Per-seed results and descriptive mean, standard deviation and sign counts.

    Raises:
        ValueError: If seeds are missing or test engines/targets are inconsistent.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify unique experiment seeds")
    if any(subset not in ("FD001", "FD003") for subset in subsets):
        raise ValueError("Only FD001/FD003 have paper-matched Goel protocols")
    if any(arch not in ("lstm", "cnn") for arch in architectures):
        raise ValueError("Expected Goel LSTM or CNN architecture")
    comparisons: list[dict[str, object]] = []
    for subset in subsets:
        for architecture in architectures:
            pairs: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
            reference_targets: dict[str, tuple[float, float]] | None = None
            for seed in seeds:
                original = _read_engine_audit(
                    audit_root / _audit_filename(subset, architecture, "goel", seed)
                )
                safety = _read_engine_audit(
                    audit_root / _audit_filename(subset, architecture, "safety", seed)
                )
                if original.keys() != safety.keys():
                    raise ValueError("Paired Goel and safety experiments use different engines")
                for engine in original:
                    targets = (
                        original[engine]["true_rul_capped"],
                        original[engine]["true_rul_raw"],
                    )
                    counterpart = (
                        safety[engine]["true_rul_capped"],
                        safety[engine]["true_rul_raw"],
                    )
                    if targets != counterpart:
                        raise ValueError(f"Paired test labels disagree for {engine}")
                    if reference_targets is not None and (
                        engine not in reference_targets
                        or reference_targets[engine] != targets
                    ):
                        raise ValueError(f"Different official test targets for {engine}")
                if reference_targets is None:
                    reference_targets = {
                        engine: (
                            data["true_rul_capped"],
                            data["true_rul_raw"],
                        )
                        for engine, data in original.items()
                    }
                pairs[seed] = {
                    label: {
                        "goel": _metrics(original, label),
                        "safety": _metrics(safety, label),
                    }
                    for label in ("capped", "raw")
                }
            for label in ("capped", "raw"):
                per_seed = []
                for seed in seeds:
                    models = pairs[seed][label]
                    changes = {
                        metric: models["safety"][metric] - models["goel"][metric]
                        for metric in ("rmse", "mae", "late_bias", "nasa")
                    }
                    per_seed.append(
                        {
                            "seed": seed,
                            "goel": models["goel"],
                            "safety": models["safety"],
                            "delta_safety_minus_goel": changes,
                        }
                    )
                aggregates = {}
                for metric in ("rmse", "mae", "late_bias", "nasa"):
                    values = [
                        entry["delta_safety_minus_goel"][metric] for entry in per_seed
                    ]
                    aggregates[metric] = {
                        "mean": statistics.mean(values),
                        "stdev": statistics.stdev(values) if len(values) > 1 else None,
                        "improved_seeds": sum(value < 0 for value in values),
                        "seed_count": len(values),
                    }
                comparisons.append(
                    {
                        "subset": subset,
                        "architecture": architecture,
                        "test_label_convention": label,
                        "paper_reference_unverified_target_convention": _PUBLISHED[
                            (subset, architecture)
                        ],
                        "per_seed": per_seed,
                        "paired_delta_summary": aggregates,
                    }
                )
    return {
        "study": "NEX-233 paired Goel MSE versus safety asymmetric MSE",
        "paper": "Goel et al. (2026), arXiv:2604.27234v1, Table III",
        "source": "Only native NexuML per-engine test audit CSVs",
        "warning": (
            "Test data were inspected before this safety method was designed; "
            "results are exploratory and cannot establish a confirmatory "
            "independent performance gain. Three seeds and shared test engines "
            "do not provide independent test replicates. Published test-label "
            "capping convention is unspecified."
        ),
        "seeds": list(seeds),
        "comparisons": comparisons,
    }


def main() -> None:
    """Create a post-training paired comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    report = summarize_reproducibility(
        args.audit_root,
        seeds=tuple(args.seeds),
    )
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Paired RUL study saved to {args.output}")


if __name__ == "__main__":
    main()
