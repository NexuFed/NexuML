"""Paired official C-MAPSS engine-level uncertainty for flow-state RUL.

Reports per-engine deltas and paired engine bootstrap intervals. Repeated
training seeds do not constitute independent test engine populations, and
bootstrap intervals are descriptive rather than a confirmatory significance
test on the already inspected FD001 holdout.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from nexuml_library.evaluation.maintenance.rul_flow_ablation_summary import (
    _geometry_report,
)
from nexuml_library.evaluation.maintenance.rul_repro_summary import (
    _metrics,
    _read_engine_audit,
)

_ARMS = ("encoder_only", "flow_state")
_METRICS = ("rmse", "mae", "late_bias", "nasa")


def _percentile(values: list[float], q: float) -> float:
    """Compute a deterministic linearly interpolated empirical percentile.

    Returns:
        Requested interpolated order statistic.
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    fraction = position - lower
    return ordered[lower] + fraction * (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    )


def paired_flow_state_summary(
    root: Path,
    subset: str,
    seed: int,
    *,
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Compare identical official engines under flow-state and encoder-only MLPs.

    Args:
        root: Native NexuML per-engine audit and geometry directory.
        subset: FD001 or FD003.
        seed: Matching upstream checkpoint and training seed.
        bootstrap_repeats: Deterministic paired resamples of official engines.
        bootstrap_seed: Random seed for descriptive uncertainty intervals.

    Returns:
        Original per-arm metrics, engine-level changes and bootstrap intervals.

    Raises:
        ValueError: If model identities, engine labels, or settings disagree.
    """
    if subset not in ("FD001", "FD003") or bootstrap_repeats < 1:
        raise ValueError("Use FD001/FD003 and a positive bootstrap count")
    base_names = {
        arm: f"rul-{subset}-patchformer-{arm.replace('_', '-')}-seed{seed}"
        for arm in _ARMS
    }
    geometry = {
        arm: _geometry_report(root, run)
        for arm, run in base_names.items()
    }
    if (
        geometry["flow_state"].get("flow_checkpoint_sha256") in (None, "")
        or geometry["encoder_only"].get("flow_checkpoint_sha256") is not None
        or any(
            report.get("subset") != subset
            or report.get("seed") != seed
            or report.get("test_labels_used_in_feature_fitting") is not False
            or report.get("uses_patchformer_regression_head") is not False
            for report in geometry.values()
        )
        or (
            geometry["flow_state"].get("patchformer_checkpoint_sha256")
            != geometry["encoder_only"].get("patchformer_checkpoint_sha256")
        )
        or (
            geometry["flow_state"].get("patchformer_upstream_run")
            != geometry["encoder_only"].get("patchformer_upstream_run")
        )
    ):
        raise ValueError("Flow-state comparison requires identical pretrained encoder")
    baseline_run = str(geometry["flow_state"]["patchformer_upstream_run"])
    names = {"baseline": baseline_run, **base_names}
    audits = {
        arm: _read_engine_audit(root / f"{run}.csv")
        for arm, run in names.items()
    }
    engines = sorted(audits["baseline"])
    truth = {
        asset: (
            audits["baseline"][asset]["true_rul_capped"],
            audits["baseline"][asset]["true_rul_raw"],
        )
        for asset in engines
    }
    if any(
        set(rows) != set(engines)
        or any(
            (record["true_rul_capped"], record["true_rul_raw"]) != truth[asset]
            for asset, record in rows.items()
        )
        for rows in audits.values()
    ):
        raise ValueError("Flow-state arms have mismatched official test labels")
    generator = random.Random(bootstrap_seed)
    resamples = [
        generator.choices(engines, k=len(engines))
        for _ in range(bootstrap_repeats)
    ]
    results: dict[str, Any] = {}
    for target in ("capped", "raw"):
        metrics = {
            arm: _metrics(rows, target) for arm, rows in audits.items()
        }
        differences: dict[str, Any] = {}
        for control in ("encoder_only", "baseline"):
            draws: dict[str, list[float]] = {key: [] for key in _METRICS}
            for sampled in resamples:
                replicate = {
                    arm: {
                        str(i): audits[arm][engine]
                        for i, engine in enumerate(sampled)
                    }
                    for arm in ("flow_state", control)
                }
                flow_scores = _metrics(replicate["flow_state"], target)
                control_scores = _metrics(replicate[control], target)
                for key in _METRICS:
                    draws[key].append(flow_scores[key] - control_scores[key])
            differences[control] = {
                key: {
                    "observed_delta": metrics["flow_state"][key] - metrics[control][key],
                    "paired_engine_bootstrap_95": [
                        _percentile(draws[key], 0.025),
                        _percentile(draws[key], 0.975),
                    ],
                    "bootstrap_mean": statistics.mean(draws[key]),
                }
                for key in _METRICS
            }
        results[target] = {
            "models": metrics,
            "flow_state_minus_controls": differences,
            "per_engine_delta_squared_error_vs_encoder": {
                engine: (
                    (
                        audits["flow_state"][engine]["predicted_rul"]
                        - truth[engine][0 if target == "capped" else 1]
                    ) ** 2
                    - (
                        audits["encoder_only"][engine]["predicted_rul"]
                        - truth[engine][0 if target == "capped" else 1]
                    ) ** 2
                )
                for engine in engines
            },
        }
    return {
        "study": "NEX-233 standalone Flow-state RUL versus encoder-only",
        "subset": subset,
        "training_seed": seed,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_repeats": bootstrap_repeats,
        "official_test_engine_count": len(engines),
        "frozen_encoder_sha256": geometry["flow_state"][
            "patchformer_checkpoint_sha256"
        ],
        "frozen_flow_sha256": geometry["flow_state"]["flow_checkpoint_sha256"],
        "caveat": (
            "Paired bootstrap resamples the same official test engines; FD001 "
            "already informed development and should not support fresh "
            "confirmatory significance claims."
        ),
        "results": results,
    }


def main() -> None:
    """Write an auditable, post-training paired-engine RUL report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD003"), default="FD001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(".experiments/rul/audits/flow-state-comparison.json"),
    )
    args = parser.parse_args()
    report = paired_flow_state_summary(
        args.audit_root,
        args.subset,
        args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Paired Flow-state RUL report: {args.output}")


if __name__ == "__main__":
    main()
