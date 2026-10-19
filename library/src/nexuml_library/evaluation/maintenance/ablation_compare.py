"""Compare independent and OT flow coupling on identical prepared features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_SETTINGS_KEYS = (
    "frontend",
    "scaling",
    "seed",
    "model_name",
    "feature_dim",
    "machine_type",
    "noise_db",
    "train_machine_ids",
    "val_machine_ids",
    "test_machine_ids",
    "clip_num_samples",
    "normalization_fit_split",
)


def _signature(report: dict[str, Any]) -> str:
    """Read the vector-field architecture signature from checkpoint provenance.

    Returns:
        The MLP-width suffix of this run.

    Raises:
        ValueError: If the run name lacks its expected architecture signature.
    """
    name = Path(report["best_checkpoint"]).parent.name
    if "-mlp-" not in name:
        raise ValueError(f"Missing MLP architecture signature in {name}")
    return name.rsplit("-mlp-", 1)[1]


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize descriptive training and held-out validation SWD.

    Returns:
        Relative SWD improvement for best/last and machine-wise train metrics.
    """
    train_best = report["training_best"]
    train_last = report["training_last"]
    val_best = report["validation_best"]
    val_last = report["validation_last"]
    return {
        "best_checkpoint": report["best_checkpoint"],
        "training_best_relative_reduction": train_best["swd_relative_reduction"],
        "training_last_relative_reduction": train_last["swd_relative_reduction"],
        "validation_best_relative_reduction": val_best["swd_relative_reduction"],
        "validation_last_relative_reduction": val_last["swd_relative_reduction"],
        "training_best_by_machine": train_best["by_machine"],
        "training_last_by_machine": train_last["by_machine"],
        "validation_improves_over_no_transport": val_best["swd_relative_reduction"] > 0,
    }


def compare_couplings(
    independent: dict[str, Any], ot: dict[str, Any]
) -> dict[str, Any]:
    """Compare coupling on a common frontend, scaler, architecture and split.

    Only best checkpoints selected by validation SWD are compared. All
    reported improvements are dimensionless relative to no transport.

    Args:
        independent: Evaluation JSON for independently paired training data.
        ot: Evaluation JSON for OT-coupled training data.

    Returns:
        Pairwise summary of training and validation transport improvements.

    Raises:
        ValueError: If reports have incompatible provenance or unsealed test.
    """
    left = independent["prepared_settings"]
    right = ot["prepared_settings"]
    if left["coupling"] != "independent" or right["coupling"] != "ot":
        raise ValueError("Expected independent and ot prepared coupling, in that order")
    if any(left.get(key) != right.get(key) for key in _SETTINGS_KEYS):
        raise ValueError("Couplings must share frontend, scaling, seed and machine splits")
    if _signature(independent) != _signature(ot):
        raise ValueError("Couplings must share the same trained MLP architecture")
    if not independent["test_is_sealed"] or not ot["test_is_sealed"]:
        raise ValueError("Do not compare couplings after unsealing the test partition")

    for split in ("training_best", "validation_best"):
        initial = independent[split]["swd_healthy_to_faulty"]
        other = ot[split]["swd_healthy_to_faulty"]
        if abs(initial - other) > 1e-6 * max(1.0, abs(initial)):
            raise ValueError(f"Incompatible {split} baseline distribution")

    independent_summary = _summary(independent)
    ot_summary = _summary(ot)
    return {
        "frontend": left["frontend"],
        "scaling": left["scaling"],
        "seed": left["seed"],
        "architecture": _signature(independent),
        "independent": independent_summary,
        "ot": ot_summary,
        "ot_minus_independent_validation_relative_reduction": (
            ot_summary["validation_best_relative_reduction"]
            - independent_summary["validation_best_relative_reduction"]
        ),
        "test_is_sealed": True,
    }


def main() -> None:
    """Compare two independently evaluated training runs without opening test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-report", type=Path, required=True)
    parser.add_argument("--ot-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    independent = json.loads(args.independent_report.read_text())
    ot = json.loads(args.ot_report.read_text())
    result = compare_couplings(independent, ot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Validation-best SWD improvement (independent): "
        f"{result['independent']['validation_best_relative_reduction']:.2%}"
    )
    print(
        "Validation-best SWD improvement (OT): "
        f"{result['ot']['validation_best_relative_reduction']:.2%}"
    )
    print(
        "OT minus independent: "
        f"{result['ot_minus_independent_validation_relative_reduction']:+.2%}"
    )
    print(f"Comparison written to {args.output}")


if __name__ == "__main__":
    main()
