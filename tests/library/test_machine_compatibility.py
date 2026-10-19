"""Offline diagnostic tests for train/validation machine distribution shift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from nexuml_library.evaluation.maintenance.machine_compatibility import (
    diagnose_machine_compatibility,
)


def _artifact(tmp_path: Path) -> Path:
    """Create synthetic machine-specific healthy/faulty shifts with sealed test.

    Returns:
        Prepared artifact containing two training and one validation machine.
    """
    features: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for name, split, center, shift in (
        ("fan:id_00", "train", (0.0, 0.0), (1.0, 0.0)),
        ("fan:id_02", "train", (0.3, 0.0), (1.0, 0.0)),
        ("fan:id_04", "val", (4.0, 0.0), (0.0, 2.0)),
        ("fan:id_06", "test", (100.0, 50.0), (10.0, 10.0)),
    ):
        for i in range(8):
            healthy = torch.tensor([center[0] + i / 30, center[1] + i / 40])
            faulty = healthy + torch.tensor(shift)
            features.append(torch.stack((healthy, faulty)))
            rows.append({"group_id": name, "split": split, "paired": 1})
    file = tmp_path / "fisher-independent-standardized-seed42.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": torch.stack(features),
            "rows": rows,
            "settings": {
                "frontend": "fisher",
                "scaling": "standardized",
                "coupling": "independent",
                "seed": 42,
                "feature_dim": 2,
                "model_name": "synthetic",
            },
        },
        file,
    )
    return file


def test_machine_compatibility_detects_different_fault_shifts(tmp_path: Path) -> None:
    """Matching training directions differ from held-out validation direction."""
    report = diagnose_machine_compatibility(
        _artifact(tmp_path), projections=16, bootstrap=8, seed=42
    )
    assert report["test_is_sealed"]
    assert set(report["machines"]) == {"fan:id_00", "fan:id_02", "fan:id_04"}
    assert "fan:id_06" not in json.dumps(report)
    assert report["healthy_swd"]["labels"] == [
        "fan:id_00",
        "fan:id_02",
        "fan:id_04",
    ]
    assert len(report["healthy_swd"]["pairs"]) == 3
    assert len(report["faulty_swd"]["pairs"]) == 3
    directions = report["mean_fault_direction_cosine"]["matrix"]
    assert directions[0][1] == pytest.approx(1.0, abs=1e-6)
    assert directions[0][2] == pytest.approx(0.0, abs=1e-6)
    assert directions[1][2] == pytest.approx(0.0, abs=1e-6)

    clusters = report["healthy_cluster_hypotheses"]
    assignments = clusters["exploratory_two_group_assignment"]
    assert assignments["fan:id_00"] == assignments["fan:id_02"]
    assert assignments["fan:id_04"] != assignments["fan:id_00"]
    assert not clusters["usable_for_conditional_training"]
    assert clusters["assignment_includes_validation_healthy_features"]
    transfer = report["mean_shift_transfer_to_validation"]["fan:id_04"]
    assert set(transfer["train_derived_shifts"]) == {
        "fan:id_00",
        "fan:id_02",
        "train_pooled",
    }
    assert transfer["no_transport_swd"] > 0
    assert all(row["available"] for row in report["healthy_split_half_swd"].values())


def test_machine_diagnostic_is_pairing_invariant_and_never_uses_test(
    tmp_path: Path,
) -> None:
    """Permuting faulty partners and changing sealed test cannot alter report."""
    file = _artifact(tmp_path)
    before = diagnose_machine_compatibility(
        file, projections=16, bootstrap=8, seed=7
    )
    payload = torch.load(file, map_location="cpu", weights_only=True)
    features = payload["features"]
    for start in (0, 8):
        features[start : start + 8, 1] = features[
            start : start + 8, 1
        ].flip(0)
    features[24:] = 9999.0
    torch.save(payload, file)
    after = diagnose_machine_compatibility(
        file, projections=16, bootstrap=8, seed=7
    )
    assert before == after


def test_machine_diagnostic_rejects_invalid_projection_or_bootstrap(
    tmp_path: Path,
) -> None:
    """Enforce reproducible nontrivial resampling controls."""
    file = _artifact(tmp_path)
    with pytest.raises(ValueError, match="projections"):
        diagnose_machine_compatibility(file, projections=0)
    with pytest.raises(ValueError, match="bootstrap"):
        diagnose_machine_compatibility(file, bootstrap=1)


def test_machine_diagnostic_renders_optional_plots(tmp_path: Path) -> None:
    """Generate train-fit PCA without indexing the sealed test rows."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    from nexuml_library.evaluation.maintenance.machine_compatibility_plots import (
        plot_machine_compatibility,
    )

    file = _artifact(tmp_path)
    report = diagnose_machine_compatibility(
        file, projections=8, bootstrap=4, seed=42
    )
    output = tmp_path / "plots"
    plot_machine_compatibility(report, file, output)
    assert {path.name for path in output.glob("*.png")} == {
        "healthy_swd.png",
        "faulty_swd.png",
        "fault_direction_cosine.png",
        "healthy_train_fitted_pca.png",
    }
