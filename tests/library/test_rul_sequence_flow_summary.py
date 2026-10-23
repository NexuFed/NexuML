"""Paired official sequence-Flow RUL audit comparisons and provenance guards."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from nexuml_library.evaluation.maintenance.rul_sequence_flow_summary import (
    summarize_sequence_flow_rul,
)


def _audit(root: Path, run: str, error: float) -> None:
    """Write matched exact official RUL rows for a fabricated controlled run."""
    with (root / f"{run}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["asset_id", "predicted_rul", "true_rul_capped", "true_rul_raw"]
        )
        for engine in range(1, 101):
            label = float(30 + engine % 70)
            writer.writerow([f"test:FD003:{engine}", label + error, label, label])


def test_all_three_flow_specific_rul_comparisons_are_paired(
    tmp_path: Path,
) -> None:
    """All six matched heads and unmodified PatchFormer must share engines."""
    upstream = "rul-FD003-controlled-patchformer-seed42"
    _audit(tmp_path, upstream, 4.0)
    for architecture in ("transformer", "conformer", "cfc"):
        for use_flow in (False, True):
            arm = "flow" if use_flow else "encoder"
            run = (
                f"rul-FD003-patchformer-sequence-{architecture}-"
                f"{arm}-rul-seed42"
            )
            _audit(tmp_path, run, 2.0 if use_flow else 3.0)
            (tmp_path / f"{run}.geometry.json").write_text(
                json.dumps(
                    {
                        "subset": "FD003",
                        "seed": 42,
                        "architecture": architecture,
                        "use_flow": use_flow,
                        "feature_shape": [30, 256],
                        "test_labels_used_in_feature_fitting": False,
                        "uses_patchformer_regression_head": False,
                        "patchformer_upstream_run": upstream,
                        "patchformer_checkpoint_sha256": "shared-encoder",
                        "endpoint_artifact_sha256": f"{architecture}-endpoints",
                        "flow_checkpoint_sha256": (
                            f"{architecture}-flow" if use_flow else None
                        ),
                    }
                ),
                encoding="utf-8",
            )
    report = summarize_sequence_flow_rul(tmp_path, "FD003", 42)
    assert report["official_test_engine_count"] == 100
    for label in ("capped", "raw"):
        scored = report["comparisons"][label]["comparisons"]
        for architecture in ("transformer", "conformer", "cfc"):
            difference = scored[architecture]
            assert difference["flow_minus_encoder"]["rmse"] == pytest.approx(-1)
            assert difference["flow_improved_engine_count_mse"] == 100
    geometry = tmp_path / (
        "rul-FD003-patchformer-sequence-cfc-flow-rul-seed42.geometry.json"
    )
    corrupted = json.loads(geometry.read_text())
    corrupted["patchformer_checkpoint_sha256"] = "incompatible-encoder"
    geometry.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="different PatchFormer encoders"):
        summarize_sequence_flow_rul(tmp_path, "FD003", 42)
