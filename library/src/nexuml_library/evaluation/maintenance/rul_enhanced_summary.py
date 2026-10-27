"""Validation-only main-effect and leave-one-out CfC RUL enhancement comparisons."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Literal

from nexuml_library.evaluation.maintenance.rul_generalization_summary import (
    _read_history,
)
from nexuml_library.scenarios.maintenance.rul_enhanced_cfc import (
    CONFIGURATIONS,
    OPTIONS,
    enabled_options,
)


def summarize_enhanced_cfc(
    root: Path,
    audit_root: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    subset: Literal["FD001", "FD003"] = "FD003",
) -> dict[str, Any]:
    """Compare identical readout seeds and engine splits on validation only.

    Args:
        root: Native NexuML evaluation-mode training and validation histories.
        audit_root: Native best-checkpoint provenance audits.
        seeds: Predeclared independent head initialization seeds.
        subset: Matching C-MAPSS controlled benchmark.

    Returns:
        Per-configuration validation errors and paired on/off improvements.

    Raises:
        ValueError: If run records, checkpoint selections, or assets disagree.
    """
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Enhanced CfC seeds must be distinct")
    if any(seed not in (42, 43, 44) for seed in seeds):
        raise ValueError("Enhanced CfC seeds must be 42, 43 or 44")
    records: dict[str, Any] = {}
    validation_engines: set[tuple[str, int]] | None = None
    for configuration in CONFIGURATIONS:
        for seed in seeds:
            run = f"rul-{subset}-enhanced-cfc-{configuration}-s{seed}"
            history, legacy = _read_history(root / f"{run}.curves.csv")
            best = min(history, key=lambda item: item["val_eval_mse"])
            stage = json.loads(
                (root / f"{run}.curves.validation_engines.json").read_text(
                    encoding="utf-8"
                )
            )
            engines = stage["validation_engines"]
            identities = {
                (str(item["asset_id"]), int(item["windows"])) for item in engines
            }
            if not identities or len(identities) != len(engines):
                raise ValueError(f"Invalid validation engines for {run}")
            if validation_engines is None:
                validation_engines = identities
            if identities != validation_engines:
                raise ValueError("CfC enhancement runs have different validation engines")
            if (
                int(stage["best_epoch"]) != int(best["epoch"])
                or stage.get("test_labels_used") is not False
            ):
                raise ValueError(f"Invalid validation selection in {run}")
            manifest = json.loads(
                (audit_root / f"{run}.checkpoint.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("checkpoint_epoch") != int(best["epoch"])
                or manifest.get("loaded_strict_model_state") is not True
            ):
                raise ValueError(f"Nonmatching best checkpoint in {run}")
            records[f"{configuration}_s{seed}"] = {
                "best_epoch": int(best["epoch"]),
                "best_val_mse": best["val_eval_mse"],
                "best_train_mse": best["train_eval_mse"],
                "train_val_gap": best["mse_gap"],
                "last_fitting_epoch": int(history[-1]["epoch"]),
                "post_best_val_mse_increase": (
                    history[-1]["val_eval_mse"] - best["val_eval_mse"]
                ),
                "validation_engine_mean_mse": statistics.mean(
                    float(item["mse"]) for item in engines
                ),
                "legacy_postfit_validation_excluded": legacy,
            }
    comparison: list[dict[str, Any]] = []
    scores_by_configuration: dict[str, dict[int, float]] = {}
    for configuration in CONFIGURATIONS:
        scores = [
            float(records[f"{configuration}_s{seed}"]["best_val_mse"])
            for seed in seeds
        ]
        by_seed = dict(zip(seeds, scores))
        scores_by_configuration[configuration] = by_seed
        comparison.append({
            "configuration": configuration,
            "enabled": sorted(enabled_options(configuration)),
            "val_mse_mean": statistics.mean(scores),
            "val_mse_std": statistics.stdev(scores) if len(scores) > 1 else None,
            "val_mse_by_head_seed": {
                str(seed): score for seed, score in by_seed.items()
            },
        })
    effects: dict[str, Any] = {}
    baseline_scores = scores_by_configuration["baseline"]
    all_scores = scores_by_configuration["all"]
    for option in OPTIONS:
        enabled_scores = scores_by_configuration[option]
        omitted_scores = scores_by_configuration[f"all_no_{option}"]
        one_delta = [
            enabled_scores[seed] - baseline_scores[seed] for seed in seeds
        ]
        omit_delta = [
            all_scores[seed] - omitted_scores[seed] for seed in seeds
        ]
        effects[option] = {
            "single_minus_baseline_mean": statistics.mean(one_delta),
            "single_minus_baseline_by_seed": one_delta,
            "all_minus_leave_one_out_mean": statistics.mean(omit_delta),
            "all_minus_leave_one_out_by_seed": omit_delta,
            "point_estimate_changes_expected": True,
        }
    ranking = sorted(comparison, key=lambda row: float(row["val_mse_mean"]))
    return {
        "study": "NEX-233 enhanced PatchFormer-CfC architecture ablation",
        "subset": subset,
        "upstream_patchformer_seed": 42,
        "head_seeds": list(seeds),
        "selection": "best fitting-epoch validation-window MSE",
        "test_predictions_read": False,
        "modes": "baseline + 6 singles + all-on + 6 leave-one-out",
        "calibration_note": (
            "MC dropout and conformal intervals are separate post-fit diagnostics "
            "on the all-on model, excluded from point-estimator ablations. "
            "Calibration uses validation engines also used for checkpoint "
            "selection, so nominal conformal coverage is not guaranteed."
        ),
        "ranking": ranking,
        "effects": effects,
        "per_run": records,
    }


def main() -> None:
    """Write a complete predeclared validation-only CfC enhancement report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD003"), default="FD003")
    parser.add_argument("--head-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    subset: Literal["FD001", "FD003"] = args.subset
    root = args.root or Path(".experiments/rul/enhanced-cfc") / subset.lower()
    output = args.output or root / "summary.json"
    report = summarize_enhanced_cfc(root, args.audit_root, tuple(args.head_seeds), subset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Enhanced CfC validation-only comparison written to {output}")


if __name__ == "__main__":
    main()
