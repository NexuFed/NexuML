"""Validation-only review of CfC temporal RUL overfitting ablations.

Summarizes matched flow and encoder-only runs without reading the official
test predictions. Use for model selection before any held-out confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

_VARIANTS = ("w128_d010", "w128_d030", "w064_d010", "w064_d030")
_ARMS = ("encoder", "flow")


def _read_history(path: Path) -> list[dict[str, float]]:
    """Read the full-epoch evaluation-mode training/validation history.

    Returns:
        Strictly increasing epoch-indexed, finite MSE records.

    Raises:
        ValueError: If the history is absent, duplicated or malformed.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if not rows or any(
        row["train_eval_mse"] < 0
        or row["val_eval_mse"] < 0
        or not all(
            float("-inf") < item < float("inf") for item in row.values()
        )
        for row in rows
    ):
        raise ValueError(f"Invalid full-epoch RUL history: {path}")
    if any(
        int(row["epoch"]) != idx for idx, row in enumerate(rows)
    ):
        raise ValueError(f"Nonconsecutive RUL diagnostic epochs: {path}")
    return rows


def summarize_cfc_generalization(
    root: Path, audit_root: Path, head_seeds: tuple[int, ...]
) -> dict[str, Any]:
    """Compare all requested seeds using validation only.

    Args:
        root: Full-epoch diagnostics CSV/JSON directory.
        audit_root: Upstream provenance JSON directory.
        head_seeds: Matched readout initialization seeds.

    Returns:
        Per-run curves and paired mean/std validation metrics, with best setting.

    Raises:
        ValueError: If paired checkpoints or validation asset provenance differ.
    """
    results: dict[str, Any] = {}
    provenance: set[tuple[str, str]] = set()
    observed_engines: set[tuple[str, int]] | None = None
    for head_seed in head_seeds:
        seed_key = str(head_seed)
        runs: dict[str, Any] = {}
        for arm in _ARMS:
            for variant in _VARIANTS:
                name = (
                    f"rul-FD001-cfc-generalization-{arm}-{variant}"
                    f"-head-s{head_seed}"
                )
                rows = _read_history(root / f"{name}.curves.csv")
                best = min(rows, key=lambda row: row["val_eval_mse"])
                last = rows[-1]
                engines = json.loads(
                    (root / f"{name}.curves.validation_engines.json").read_text(
                        encoding="utf-8"
                    )
                )
                if engines["best_epoch"] != int(best["epoch"]):
                    raise ValueError(f"Best epoch mismatch in {name}")
                values = engines["validation_engines"]
                engine_set = {
                    (str(row["asset_id"]), int(row["windows"])) for row in values
                }
                if not engine_set or len(engine_set) != len(values):
                    raise ValueError(f"Invalid validation engines in {name}")
                if observed_engines is None:
                    observed_engines = engine_set
                if observed_engines != engine_set:
                    raise ValueError(
                        "Generalization runs use different validation engines"
                    )
                provenance_path = audit_root / f"{name}.geometry.json"
                info = json.loads(provenance_path.read_text(encoding="utf-8"))
                if (
                    info.get("subset") != "FD001"
                    or info.get("seed") != 42
                    or info.get("architecture") != "cfc"
                    or info.get("use_flow") != (arm == "flow")
                    or info.get("test_labels_used_in_feature_fitting") is not False
                ):
                    raise ValueError(
                        f"Invalid frozen generalization provenance: {name}"
                    )
                provenance.add((
                    str(info["patchformer_checkpoint_sha256"]),
                    str(info["endpoint_artifact_sha256"]),
                ))
                key = f"{arm}_{variant}"
                runs[key] = {
                    "best_epoch": int(best["epoch"]),
                    "stopped_after_epoch": int(last["epoch"]),
                    "best_train_eval_mse": best["train_eval_mse"],
                    "best_val_eval_mse": best["val_eval_mse"],
                    "best_mse_gap": best["mse_gap"],
                    "final_train_eval_mse": last["train_eval_mse"],
                    "final_val_eval_mse": last["val_eval_mse"],
                    "val_mse_increase_after_best": (
                        last["val_eval_mse"] - best["val_eval_mse"]
                    ),
                    "worst_validation_engines": sorted(
                        values, key=lambda row: row["mse"], reverse=True
                    )[:5],
                    "curves_file": str(root / f"{name}.curves.csv"),
                }
        results[seed_key] = runs
    if len(provenance) != 1:
        raise ValueError("Generalization variants use inconsistent frozen input")
    ranked: list[dict[str, Any]] = []
    for variant in _VARIANTS:
        flow = [
            results[str(seed)][f"flow_{variant}"]["best_val_eval_mse"]
            for seed in head_seeds
        ]
        encoder = [
            results[str(seed)][f"encoder_{variant}"]["best_val_eval_mse"]
            for seed in head_seeds
        ]
        ranked.append({
            "variant": variant,
            "flow_val_mse_mean": statistics.mean(flow),
            "flow_val_mse_std": statistics.stdev(flow) if len(flow) > 1 else None,
            "encoder_val_mse_mean": statistics.mean(encoder),
            "encoder_val_mse_std": (
                statistics.stdev(encoder) if len(encoder) > 1 else None
            ),
            "flow_minus_encoder_val_mse_mean": (
                statistics.mean(flow) - statistics.mean(encoder)
            ),
        })
    ranked.sort(key=lambda row: row["flow_val_mse_mean"])
    return {
        "study": "NEX-233 CfC RUL generalization / regularization",
        "upstream_subset": "FD001",
        "upstream_seed": 42,
        "head_seeds": list(head_seeds),
        "selection_metric": "mean validation-window MSE",
        "test_predictions_read": False,
        "ranking": ranked,
        "preferred_on_validation": ranked[0]["variant"],
        "per_seed": results,
        "note": (
            "Train/validation MSE use dropout-disabled evaluation on entire "
            "engine-disjoint splits. A rising post-best validation MSE as "
            "training MSE falls supports overfitting. Held-out test metrics "
            "are deliberately excluded from this model-selection summary."
        ),
    }


def main() -> None:
    """Produce the paired validation-only CfC regularization summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(".experiments/rul/generalization")
    )
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument("--head-seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--output", type=Path,
        default=Path(".experiments/rul/generalization/summary.json"),
    )
    args = parser.parse_args()
    summary = summarize_cfc_generalization(
        args.root, args.audit_root, tuple(args.head_seeds)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"CfC generalization summary written to {args.output}")


if __name__ == "__main__":
    main()
