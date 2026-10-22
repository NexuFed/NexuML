"""Post-training paired FD001/FD003 PatchFormer and safety RUL study report.

All metrics use the 100 official test engines, but paired seeds are training
replicates and not independent test sets. No hyperparameter selection occurs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from nexuml_library.evaluation.maintenance.rul_repro_summary import (
    _metrics,
    _read_engine_audit,
)

_KINDS = ("patchformer", "dmm_ja", "safety_lstm", "rul_mamba")
_METRICS = ("rmse", "mae", "late_bias", "nasa")


def summarize_controlled_study(
    audit_root: Path,
    *,
    subsets: tuple[str, ...] = ("FD001", "FD003"),
    seeds: tuple[int, ...] = (42, 43, 44),
) -> dict[str, Any]:
    """Compare identical-engine paired study predictions on raw/capped labels.

    Args:
        audit_root: Directory containing native per-engine audit CSVs.
        subsets: Official subsets to compare.
        seeds: Identical training seeds for all four architectures.

    Returns:
        Per-seed metrics and descriptive paired change statistics.

    Raises:
        ValueError: If experiment identifiers or test targets are inconsistent.
        FileNotFoundError: If any required model run is missing.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify distinct, nonempty training seeds")
    if not subsets or any(subset not in ("FD001", "FD003") for subset in subsets):
        raise ValueError("Controlled study supports FD001 and FD003")
    comparisons: list[dict[str, Any]] = []
    for subset in subsets:
        per_seed: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
        canonical_targets: dict[str, tuple[float, float]] | None = None
        for seed in seeds:
            scores: dict[str, dict[str, dict[str, float]]] = {}
            for kind in _KINDS:
                filename = f"rul-{subset}-controlled-{kind.replace('_', '-')}-seed{seed}.csv"
                rows = _read_engine_audit(audit_root / filename)
                targets = {
                    engine: (values["true_rul_capped"], values["true_rul_raw"])
                    for engine, values in rows.items()
                }
                if canonical_targets is None:
                    canonical_targets = targets
                if canonical_targets != targets:
                    raise ValueError(
                        f"Controlled experiment {filename} has different test targets"
                    )
                scores[kind] = {
                    label: _metrics(rows, label) for label in ("capped", "raw")
                }
            per_seed[seed] = scores
        for label in ("capped", "raw"):
            seed_metrics: list[dict[str, Any]] = []
            for seed in seeds:
                seed_metrics.append(
                    {
                        "seed": seed,
                        "models": {
                            kind: per_seed[seed][kind][label]
                            for kind in _KINDS
                        },
                    }
                )
            aggregates: dict[str, dict[str, dict[str, float | int | None]]] = {}
            for kind in _KINDS:
                aggregate: dict[str, dict[str, float | int | None]] = {}
                for metric in _METRICS:
                    values = [per_seed[seed][kind][label][metric] for seed in seeds]
                    differences = [
                        per_seed[seed][kind][label][metric]
                        - per_seed[seed]["safety_lstm"][label][metric]
                        for seed in seeds
                    ]
                    aggregate[metric] = {
                        "mean": statistics.mean(values),
                        "std_across_seeds": (
                            statistics.stdev(values) if len(values) > 1 else None
                        ),
                        "mean_difference_vs_safety_lstm": statistics.mean(differences),
                        "seeds_better_than_safety_lstm": sum(
                            value < 0 for value in differences
                        ),
                        "seed_count": len(seeds),
                    }
                aggregates[kind] = aggregate
            comparisons.append(
                {
                    "subset": subset,
                    "target_labels": label,
                    "per_seed": seed_metrics,
                    "aggregates": aggregates,
                }
            )
    return {
        "study": "NEX-233 controlled 30-cycle PatchFormer generalization",
        "caveat": (
            "FD001 test errors motivated this study; FD001 remains exploratory. "
            "Do not tune on FD003 holdout results. Multiple seeds share test engines."
        ),
        "training_seeds": list(seeds),
        "common_protocol": {
            "window": 30,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "max_epochs": 200,
            "validation_patience": 20,
            "test_size": 100,
        },
        "comparisons": comparisons,
    }


def main() -> None:
    """Write a machine-readable, post-training model-comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".experiments/rul/controlled/study-summary.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    summary = summarize_controlled_study(
        args.audit_root,
        seeds=tuple(args.seeds),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Controlled RUL summary saved to {args.output}")


if __name__ == "__main__":
    main()
