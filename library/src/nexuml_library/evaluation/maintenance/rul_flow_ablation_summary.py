"""Paired RUL ablation summary with checkpoint identity and engine checks.

Compares original PatchFormer, baseline-only residual calibration, flow-time,
flow-geometry and hybrid calibration. This report does not select models or
tune parameters using the official FD001/FD003 test labels.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.evaluation.maintenance.rul_repro_summary import (
    _metrics,
    _read_engine_audit,
)

_ARMS = ("baseline", "baseline_only", "flow_time", "flow_geometry", "hybrid")
_CALIBRATIONS = ("baseline_only", "flow_time", "flow_geometry", "hybrid")
_METRICS = ("rmse", "mae", "late_bias", "nasa")


def _run_name(subset: str, seed: int, arm: str) -> str:
    """Generate the native NexuML run identifier.

    Returns:
        Audit basename from the registered scenario.
    """
    if arm == "baseline_only":
        return f"rul-{subset}-patchformer-calibration-baseline-only-seed{seed}"
    return f"rul-{subset}-patchformer-flow-{arm.replace('_', '-')}-seed{seed}"


def _geometry_report(root: Path, run: str) -> dict[str, Any]:
    """Read immutable feature and checkpoint provenance for one calibration.

    Returns:
        Native geometry diagnostic manifest.

    Raises:
        ValueError: If an existing manifest does not contain required provenance.
    """
    path = root / f"{run}.geometry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid calibration geometry manifest: {path}")
    return payload


def _restore_legacy_upstream_provenance(
    reports: dict[str, dict[str, Any]],
    subset: str,
    seed: int,
    *,
    audit_root: Path,
) -> None:
    """Validate upstream identity for pre-existing calibration diagnostics.

    Older flow-time/hybrid geometry reports included the PatchFormer SHA but
    not the run name. Recover the run from the actual frozen endpoint artifact,
    never by guessing from seed or by loading test labels.

    Args:
        reports: Native per-arm geometry records to validate and augment.
        subset: C-MAPSS subset for the matching frozen flow.
        seed: Train/validation and initialization seed.
        audit_root: Local audit root alongside the prepared feature directory.

    Raises:
        ValueError: If old provenance contradicts the immutable flow artifact.
        FileNotFoundError: If the matching source artifact is unavailable.
    """
    if all(report.get("patchformer_upstream_run") for report in reports.values()):
        return
    base = audit_root.parent
    flow_run = f"rul-{subset}-patchformer-latent-flow-seed{seed}"
    endpoint_file = base / "prepared" / f"{flow_run}-endpoints.pt"
    endpoint = read_prepared_artifact(endpoint_file)
    settings = endpoint["settings"]
    if (
        settings.get("subset") != subset
        or settings.get("seed") != seed
        or settings.get("held_out_test_engine_labels_used") is not False
    ):
        raise ValueError("Historical flow endpoints do not match requested study")
    upstream = settings.get("frozen_patchformer_upstream_run")
    fingerprint = settings.get("frozen_patchformer_checkpoint_sha256")
    if not isinstance(upstream, str) or not isinstance(fingerprint, str):
        raise ValueError("Historical flow endpoints lack model provenance")
    for report in reports.values():
        if report.get("patchformer_checkpoint_sha256") != fingerprint:
            raise ValueError("Historical flow checkpoint PatchFormer identity mismatch")
        if not report.get("patchformer_upstream_run"):
            report["patchformer_upstream_run"] = upstream


def summarize_flow_ablation(
    audit_root: Path,
    *,
    subsets: tuple[str, ...] = ("FD001", "FD003"),
    seeds: tuple[int, ...] = (42, 43, 44),
) -> dict[str, Any]:
    """Summarize five paired held-out RUL arms with strict upstream identity.

    Args:
        audit_root: Native RUL audit CSV and geometry manifest directory.
        subsets: NASA subsets in this prespecified study.
        seeds: Repeated train/validation splits and initialization seeds.

    Returns:
        Five-arm per-seed results and signed paired improvements.

    Raises:
        ValueError: If a matched experiment has different test or model identity.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and nonempty")
    if not subsets or any(s not in ("FD001", "FD003") for s in subsets):
        raise ValueError("Only official FD001 and FD003 are supported")
    comparisons: list[dict[str, Any]] = []
    for subset in subsets:
        observed: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
        canonical_targets: dict[str, tuple[float, float]] | None = None
        for seed in seeds:
            reports = {
                arm: _geometry_report(audit_root, _run_name(subset, seed, arm))
                for arm in _CALIBRATIONS
            }
            _restore_legacy_upstream_provenance(
                reports, subset, seed, audit_root=audit_root
            )
            backbone_runs = {
                str(row.get("patchformer_upstream_run"))
                for row in reports.values()
            }
            backbone_hashes = {
                str(row.get("patchformer_checkpoint_sha256"))
                for row in reports.values()
            }
            if (
                len(backbone_runs) != 1
                or len(backbone_hashes) != 1
                or "None" in backbone_hashes
                or "None" in backbone_runs
            ):
                raise ValueError("Ablation arms use different frozen PatchFormer weights")
            original_run = next(iter(backbone_runs))
            flow_hashes = {
                str(reports[arm].get("flow_checkpoint_sha256"))
                for arm in ("flow_time", "flow_geometry", "hybrid")
            }
            if len(flow_hashes) != 1 or "None" in flow_hashes:
                raise ValueError("Flow ablation arms use different frozen Flow weights")
            if reports["baseline_only"].get("flow_checkpoint_sha256") is not None:
                raise ValueError("Baseline-only ablation must not load Flow weights")
            if any(
                report.get("subset") != subset
                or report.get("seed") != seed
                or report.get("test_labels_used_in_feature_fitting") is not False
                for report in reports.values()
            ):
                raise ValueError("Ablation data provenance or split mismatch")
            scores: dict[str, dict[str, dict[str, float]]] = {}
            for arm in _ARMS:
                run = original_run if arm == "baseline" else _run_name(subset, seed, arm)
                rows = _read_engine_audit(audit_root / f"{run}.csv")
                targets = {
                    engine: (data["true_rul_capped"], data["true_rul_raw"])
                    for engine, data in rows.items()
                }
                if canonical_targets is None:
                    canonical_targets = targets
                if targets != canonical_targets:
                    raise ValueError(
                        f"Official test labels/engines mismatch for {subset}: {run}"
                    )
                scores[arm] = {
                    label: _metrics(rows, label) for label in ("capped", "raw")
                }
            observed[seed] = scores
        for label in ("capped", "raw"):
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                models = {arm: observed[seed][arm][label] for arm in _ARMS}
                per_seed.append(
                    {
                        "seed": seed,
                        "models": models,
                        "differences_vs_patchformer": {
                            arm: {
                                metric: models[arm][metric] - models["baseline"][metric]
                                for metric in _METRICS
                            }
                            for arm in _CALIBRATIONS
                        },
                        "hybrid_minus_baseline_only": {
                            metric: models["hybrid"][metric]
                            - models["baseline_only"][metric]
                            for metric in _METRICS
                        },
                    }
                )
            summaries: dict[str, dict[str, dict[str, float | int | None]]] = {}
            for arm in _CALIBRATIONS:
                arm_summary: dict[str, dict[str, float | int | None]] = {}
                for metric in _METRICS:
                    values = [observed[s][arm][label][metric] for s in seeds]
                    deltas = [
                        observed[s][arm][label][metric]
                        - observed[s]["baseline"][label][metric]
                        for s in seeds
                    ]
                    arm_summary[metric] = {
                        "mean": statistics.mean(values),
                        "std_across_seeds": (
                            statistics.stdev(values) if len(values) > 1 else None
                        ),
                        "mean_change_vs_patchformer": statistics.mean(deltas),
                        "improved_seeds_vs_patchformer": sum(d < 0 for d in deltas),
                    }
                summaries[arm] = arm_summary
            direct: dict[str, dict[str, float | int]] = {}
            for metric in _METRICS:
                deltas = [
                    observed[s]["hybrid"][label][metric]
                    - observed[s]["baseline_only"][label][metric]
                    for s in seeds
                ]
                direct[metric] = {
                    "mean_hybrid_minus_baseline_only": statistics.mean(deltas),
                    "seeds_hybrid_better": sum(value < 0 for value in deltas),
                    "seed_count": len(seeds),
                }
            comparisons.append(
                {
                    "subset": subset,
                    "target_labels": label,
                    "per_seed": per_seed,
                    "model_summary": summaries,
                    "flow_specific_comparison": direct,
                }
            )
    return {
        "study": "NEX-233 PatchFormer/Flow RUL information ablation",
        "arms": list(_ARMS),
        "seeds": list(seeds),
        "caveat": (
            "FD001 official test residuals informed this study; it is exploratory. "
            "FD003 must not be reused for model tuning and confirmatory evaluation. "
            "Seeds share the same held-out engine population."
        ),
        "comparisons": comparisons,
    }


def main() -> None:
    """Create the native post-training five-arm RUL comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".experiments/rul/controlled/flow-ablation-summary.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    report = summarize_flow_ablation(args.audit_root, seeds=tuple(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Flow-specific ablation report saved to {args.output}")


if __name__ == "__main__":
    main()
