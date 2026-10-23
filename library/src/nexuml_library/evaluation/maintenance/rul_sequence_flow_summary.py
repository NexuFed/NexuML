"""Paired full-sequence Flow-versus-encoder RUL audit for all three backbones.

All scores use identical engine labels and a shared frozen PatchFormer.
The test set is summarized post hoc, never used to select upstream models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from nexuml_library.evaluation.maintenance.rul_repro_summary import (
    _metrics,
    _read_engine_audit,
)

_ARCHITECTURES = ("transformer", "conformer", "cfc")
_METRICS = ("rmse", "mae", "late_bias", "nasa")


def summarize_sequence_flow_rul(
    audit_root: Path,
    subset: Literal["FD001", "FD003"],
    seed: int,
) -> dict[str, Any]:
    """Compare six paired frozen-encoder full-token RUL models and PatchFormer.

    Args:
        audit_root: Native NexuML per-engine and feature-provenance output folder.
        subset: FD001 or FD003.
        seed: Matching original encoder, flow checkpoint and readout seed.

    Returns:
        Capped/raw performance, pairwise deltas and frozen checkpoint checks.

    Raises:
        ValueError: If official engine labels or frozen encoder identity mismatch.
    """
    reports: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, dict[str, float]]] = {}
    original_run: str | None = None
    upstream_sha: str | None = None
    for architecture in _ARCHITECTURES:
        for use_flow in (False, True):
            arm = "flow" if use_flow else "encoder"
            identifier = f"{architecture}_{arm}"
            run = (
                f"rul-{subset}-patchformer-sequence-{architecture}-"
                f"{arm}-rul-seed{seed}"
            )
            path = audit_root / f"{run}.geometry.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(report, dict)
                or report.get("subset") != subset
                or report.get("seed") != seed
                or report.get("architecture") != architecture
                or report.get("use_flow") is not use_flow
                or report.get("feature_shape") != [30, 256]
                or report.get("test_labels_used_in_feature_fitting") is not False
                or report.get("uses_patchformer_regression_head") is not False
            ):
                raise ValueError(f"Invalid sequence RUL provenance: {run}")
            sha = report.get("patchformer_checkpoint_sha256")
            upstream = report.get("patchformer_upstream_run")
            if not isinstance(sha, str) or not isinstance(upstream, str):
                raise ValueError(f"Missing frozen encoder identity: {run}")
            if upstream_sha is None:
                upstream_sha, original_run = sha, upstream
            if upstream_sha != sha or upstream != original_run:
                raise ValueError("Sequence RUL models use different PatchFormer encoders")
            if use_flow and not isinstance(
                report.get("flow_checkpoint_sha256"), str
            ):
                raise ValueError(f"Flow-aware run lacks frozen flow identity: {run}")
            if not use_flow and report.get("flow_checkpoint_sha256") is not None:
                raise ValueError(f"Encoder-only run loaded a flow checkpoint: {run}")
            reports[identifier] = report
            audits[identifier] = _read_engine_audit(audit_root / f"{run}.csv")
        encoder_info = reports[f"{architecture}_encoder"]
        flow_info = reports[f"{architecture}_flow"]
        if (
            encoder_info["endpoint_artifact_sha256"]
            != flow_info["endpoint_artifact_sha256"]
        ):
            raise ValueError(f"Mismatched frozen endpoint statistics: {architecture}")
    if original_run is None:
        raise ValueError("No sequence RUL models to summarize")
    audits["patchformer"] = _read_engine_audit(audit_root / f"{original_run}.csv")
    labels = {
        engine: (entry["true_rul_capped"], entry["true_rul_raw"])
        for engine, entry in audits["patchformer"].items()
    }
    for run, records in audits.items():
        if {
            engine: (entry["true_rul_capped"], entry["true_rul_raw"])
            for engine, entry in records.items()
        } != labels:
            raise ValueError(f"Mismatched official test engines or RUL labels: {run}")
    comparisons: dict[str, Any] = {}
    for label in ("capped", "raw"):
        scores = {name: _metrics(rows, label) for name, rows in audits.items()}
        differences = {}
        for architecture in _ARCHITECTURES:
            flow_score = scores[f"{architecture}_flow"]
            encoder_score = scores[f"{architecture}_encoder"]
            differences[architecture] = {
                "flow_minus_encoder": {
                    key: flow_score[key] - encoder_score[key]
                    for key in _METRICS
                },
                "flow_minus_patchformer": {
                    key: flow_score[key] - scores["patchformer"][key]
                    for key in _METRICS
                },
                "flow_improved_engine_count_mse": sum(
                    (
                        audits[f"{architecture}_flow"][engine]["predicted_rul"]
                        - labels[engine][0 if label == "capped" else 1]
                    ) ** 2
                    < (
                        audits[f"{architecture}_encoder"][engine]["predicted_rul"]
                        - labels[engine][0 if label == "capped" else 1]
                    ) ** 2
                    for engine in labels
                ),
            }
        comparisons[label] = {"models": scores, "comparisons": differences}
    return {
        "study": "NEX-233 frozen full-DPAN sequence Flow-specific RUL value",
        "subset": subset,
        "seed": seed,
        "official_test_engine_count": len(labels),
        "frozen_patchformer_upstream_run": original_run,
        "frozen_patchformer_checkpoint_sha256": upstream_sha,
        "comparisons": comparisons,
        "caveat": (
            "FD001 results are exploratory because its official test set has "
            "already informed development; avoid FD003 test-set tuning."
        ),
    }


def main() -> None:
    """Write the official matched full-sequence RUL comparison JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD003"), default="FD001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(".experiments/rul/audits/sequence-flow-rul-summary.json"),
    )
    args = parser.parse_args()
    result = summarize_sequence_flow_rul(args.audit_root, args.subset, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Paired sequential RUL comparison written to {args.output}")


if __name__ == "__main__":
    main()
