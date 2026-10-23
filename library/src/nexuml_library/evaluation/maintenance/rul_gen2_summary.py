"""Validation-only analysis of frozen-CfC RUL optimizer and engine-loss studies."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from nexuml_library.evaluation.maintenance.rul_generalization_summary import (
    _read_history,
)

_ARMS = ("encoder", "flow")
_DROPOUTS = ("d010", "d030")
_OPTIMIZERS = ("adam_1e3", "adamw_3e4", "adamw_1e4")
_WEIGHTINGS = ("window", "engine")


def summarize_cfc_gen2(
    root: Path, audit_root: Path, seeds: tuple[int, ...]
) -> dict[str, Any]:
    """Rank complete paired experiments exclusively by validation data.

    Args:
        root: Native per-epoch diagnostics and validation-engine partition files.
        audit_root: Immutable-source geometry/provenance JSON files.
        seeds: Readout-only independent initialization seeds to compare.

    Returns:
        Matched validation MSE, worst-group stress and generalization-gap report.

    Raises:
        ValueError: If train/test boundaries or frozen checkpoint hashes disagree.
    """
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide distinct trained readout seeds")
    runs: dict[str, Any] = {}
    frozen_encoders: set[str] = set()
    frozen_flows: set[str] = set()
    endpoint_digests: set[str] = set()
    validation_engines: set[tuple[str, int]] | None = None
    for seed in seeds:
        for arm in _ARMS:
            for dropout in _DROPOUTS:
                for opt in _OPTIMIZERS:
                    for weighting in _WEIGHTINGS:
                        key = f"{arm}_{dropout}_{opt}_{weighting}_s{seed}"
                        name = (
                            f"rul-FD001-cfc-gen2-{arm}-{dropout}-{opt}-"
                            f"{weighting}-head-s{seed}"
                        )
                        rows, legacy = _read_history(root / f"{name}.curves.csv")
                        best = min(rows, key=lambda item: item["val_eval_mse"])
                        last = rows[-1]
                        report = json.loads(
                            (root / f"{name}.curves.validation_engines.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        if int(report["best_epoch"]) != int(best["epoch"]):
                            raise ValueError(f"Best validation epoch mismatch: {name}")
                        engines = report["validation_engines"]
                        identities = {
                            (str(item["asset_id"]), int(item["windows"]))
                            for item in engines
                        }
                        if not identities or len(identities) != len(engines):
                            raise ValueError(f"Invalid held-out validation engines: {name}")
                        if validation_engines is None:
                            validation_engines = identities
                        if identities != validation_engines:
                            raise ValueError("Validation engine partitions differ across runs")
                        groups = report["validation_engine_groups"]
                        if len(groups) != 4 or sum(
                            int(group["engines"]) for group in groups
                        ) != len(engines):
                            raise ValueError(f"Incomplete validation stress groups: {name}")
                        info = json.loads(
                            (audit_root / f"{name}.geometry.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        if (
                            info.get("subset") != "FD001"
                            or info.get("seed") != 42
                            or info.get("architecture") != "cfc"
                            or info.get("use_flow") is not (arm == "flow")
                            or info.get("engine_balanced") is not (weighting == "engine")
                            or info.get("test_labels_used_in_feature_fitting") is not False
                        ):
                            raise ValueError(f"Invalid frozen-input provenance: {name}")
                        frozen_encoders.add(
                            str(info["patchformer_checkpoint_sha256"])
                        )
                        endpoint_digests.add(str(info["endpoint_artifact_sha256"]))
                        if arm == "flow":
                            frozen_flows.add(str(info["flow_checkpoint_sha256"]))
                        runs[key] = {
                            "best_epoch": int(best["epoch"]),
                            "last_fitting_epoch": int(last["epoch"]),
                            "best_train_eval_mse": best["train_eval_mse"],
                            "best_val_eval_mse": best["val_eval_mse"],
                            "best_mse_gap": best["mse_gap"],
                            "final_train_eval_mse": last["train_eval_mse"],
                            "final_val_eval_mse": last["val_eval_mse"],
                            "val_increase_after_best": (
                                last["val_eval_mse"] - best["val_eval_mse"]
                            ),
                            "val_worst_group_mse": max(
                                float(group["window_mse"]) for group in groups
                            ),
                            "val_group_std_mse": statistics.stdev(
                                float(group["window_mse"]) for group in groups
                            ),
                            "val_engine_mean_mse": statistics.mean(
                                float(item["mse"]) for item in engines
                            ),
                            "legacy_postfit_validation_excluded": legacy,
                        }
    if (
        len(frozen_encoders) != 1
        or len(endpoint_digests) != 1
        or len(frozen_flows) != 1
        or "None" in frozen_flows
    ):
        raise ValueError("Matched CfC study must use identical upstream checkpoints")
    ranking = []
    for dropout in _DROPOUTS:
        for opt in _OPTIMIZERS:
            for weighting in _WEIGHTINGS:
                stem = f"{dropout}_{opt}_{weighting}"
                flow = [runs[f"flow_{stem}_s{seed}"] for seed in seeds]
                control = [runs[f"encoder_{stem}_s{seed}"] for seed in seeds]
                flow_val = [float(row["best_val_eval_mse"]) for row in flow]
                control_val = [float(row["best_val_eval_mse"]) for row in control]
                paired_delta = [
                    flow_val[i] - control_val[i] for i in range(len(seeds))
                ]
                ranking.append({
                    "configuration": stem,
                    "flow_val_mse_mean": statistics.mean(flow_val),
                    "flow_val_mse_std": (
                        statistics.stdev(flow_val) if len(flow_val) > 1 else None
                    ),
                    "encoder_val_mse_mean": statistics.mean(control_val),
                    "paired_flow_minus_encoder_mean": statistics.mean(paired_delta),
                    "flow_worst_group_mse_mean": statistics.mean(
                        float(row["val_worst_group_mse"]) for row in flow
                    ),
                    "flow_engine_mean_mse_mean": statistics.mean(
                        float(row["val_engine_mean_mse"]) for row in flow
                    ),
                    "flow_post_best_mse_increase_mean": statistics.mean(
                        float(row["val_increase_after_best"]) for row in flow
                    ),
                })
    ranking.sort(key=lambda row: row["flow_val_mse_mean"])
    return {
        "study": "NEX-233 frozen CfC RUL: AdamW and engine-equal supervision",
        "subset": "FD001",
        "frozen_upstream_seed": 42,
        "head_seeds": list(seeds),
        "selection": "mean best fitting-epoch, window-weighted validation MSE",
        "test_predictions_read": False,
        "validation_groups": (
            "Four disjoint stress partitions of existing held-out validation engines; "
            "not independently retrained folds"
        ),
        "best_validation_configuration": ranking[0]["configuration"],
        "ranking": ranking,
        "per_run": runs,
    }


def main() -> None:
    """Generate reproducible paired validation-only generalization comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--root", type=Path, default=Path(".experiments/rul/gen2")
    )
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(".experiments/rul/gen2/summary.json"),
    )
    args = parser.parse_args()
    summary = summarize_cfc_gen2(
        args.root, args.audit_root, tuple(args.head_seeds)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validation-only CfC optimizer study written to {args.output}")


if __name__ == "__main__":
    main()
