"""Check paired independent/OT reports using only distributional diagnostics."""

from __future__ import annotations

from typing import Any

import pytest

from nexuml_library.evaluation.maintenance.ablation_compare import compare_couplings


def _report(coupling: str, validation_gain: float) -> dict[str, Any]:
    """Construct a sealed report from matched synthetic experimental settings.

    Returns:
        Report containing machine-disjoint provenance and SWD diagnostics.
    """
    return {
        "best_checkpoint": (
            f"/tmp/experiments/fisher-{coupling}-standardized-seed42-mlp-128x128/"
            "epoch=00-step=000016.ckpt"
        ),
        "prepared_settings": {
            "coupling": coupling,
            "frontend": "fisher",
            "scaling": "standardized",
            "seed": 42,
            "model_name": "jiangab/FISHER-small-0723",
            "feature_dim": 1536,
            "machine_type": "fan",
            "noise_db": 6,
            "train_machine_ids": ["id_00", "id_02"],
            "val_machine_ids": ["id_04"],
            "test_machine_ids": ["id_06"],
            "clip_num_samples": 16384,
            "normalization_fit_split": "train",
        },
        "training_best": {
            "swd_healthy_to_faulty": 0.6,
            "swd_relative_reduction": 0.3,
            "by_machine": {
                "fan:id_00": {"swd_relative_reduction": 0.35},
                "fan:id_02": {"swd_relative_reduction": 0.25},
            },
        },
        "training_last": {
            "swd_relative_reduction": 0.5,
            "by_machine": {},
        },
        "validation_best": {
            "swd_healthy_to_faulty": 0.4,
            "swd_relative_reduction": validation_gain,
        },
        "validation_last": {"swd_relative_reduction": validation_gain - 0.05},
        "test_is_sealed": True,
    }


def test_compare_couplings_reports_validation_advantage_and_train_gap() -> None:
    """Compare only relative distributional improvements with matched provenance."""
    result = compare_couplings(_report("independent", 0.1), _report("ot", -0.02))
    assert result["ot_minus_independent_validation_relative_reduction"] == pytest.approx(
        -0.12
    )
    assert result["independent"]["validation_improves_over_no_transport"]
    assert not result["ot"]["validation_improves_over_no_transport"]
    assert "fan:id_00" in result["ot"]["training_best_by_machine"]
    assert result["test_is_sealed"]


def test_compare_couplings_rejects_leakage_and_mismatched_splits() -> None:
    """Reject unsealed or unmatched experiments before comparison."""
    independent = _report("independent", 0.1)
    ot = _report("ot", 0.2)
    ot["test_is_sealed"] = False
    with pytest.raises(ValueError, match="unsealing"):
        compare_couplings(independent, ot)

    ot["test_is_sealed"] = True
    ot["prepared_settings"]["val_machine_ids"] = ["id_06"]
    with pytest.raises(ValueError, match="share frontend"):
        compare_couplings(independent, ot)


def test_compare_couplings_rejects_different_baselines_or_capacity() -> None:
    """Compare the same endpoint distributions and MLP architecture only."""
    independent = _report("independent", 0.1)
    ot = _report("ot", 0.2)
    ot["validation_best"]["swd_healthy_to_faulty"] = 0.6
    with pytest.raises(ValueError, match="baseline distribution"):
        compare_couplings(independent, ot)

    ot["validation_best"]["swd_healthy_to_faulty"] = 0.4
    ot["best_checkpoint"] = ot["best_checkpoint"].replace("128x128", "512x512")
    with pytest.raises(ValueError, match="MLP architecture"):
        compare_couplings(independent, ot)
