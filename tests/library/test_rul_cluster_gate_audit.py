"""Offline, test-sealed NEX-233 cluster-gate checkpoint audit contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from nexuml_library.evaluation.maintenance.rul_cluster_gate_audit import (
    _EngineGateStats,
    _balanced_windows,
    _model_state,
    _selected_checkpoint,
    _summarize,
)


def test_engine_balanced_sampling_never_selects_test(tmp_path: Path) -> None:
    """Audit samples only train/validation engines and preserves endpoints."""
    rows = []
    for split, n_engines in (("train", 80), ("val", 20), ("test", 100)):
        for engine in range(n_engines):
            for step in range(10):
                rows.append({
                    "split": split,
                    "asset_id": f"{split}:FD003:{engine}",
                    "step": step,
                })
    sampled = _balanced_windows(rows, 3)
    assert set(sampled) == {"train", "val"}
    assert len(sampled["train"]) == 80
    assert len(sampled["val"]) == 20
    for split in ("train", "val"):
        for engine, indices in sampled[split].items():
            assert len(indices) == 3
            assert rows[indices[0]]["step"] == 0
            assert rows[indices[-1]]["step"] == 9
            assert all(rows[index]["asset_id"] == engine for index in indices)
            assert all(rows[index]["split"] != "test" for index in indices)
    with pytest.raises(ValueError, match="80 train"):
        _balanced_windows(rows[: 79 * 10], 3)


def test_gate_summary_balances_engines_and_masks_validation_teachers() -> None:
    """Engine summary is not dominated by windows from a long-lived engine."""
    a = _EngineGateStats.empty(2)
    b = _EngineGateStats.empty(2)
    for _ in range(8):
        a.append(torch.tensor([0.95, 0.05]), torch.tensor([1.0, 0.0]), torch.tensor(2.0))
    b.append(torch.tensor([0.1, 0.9]), torch.tensor([0.0, 2.0]), torch.tensor(2.0))
    data = {"train:FD003:1": a, "train:FD003:2": b}
    labels = {"train:FD003:1": [0, 1], "train:FD003:2": [1, 0]}
    train = _summarize(data, labels, "working")
    assert train["engines"] == 2
    assert train["sampled_windows"] == 9
    assert train["hard_occupancy_engines"] == [1, 1]
    assert train["hard_occupancy_windows"] == [8, 1]
    assert train["soft_occupancy_engine_mean"] == pytest.approx([0.525, 0.475])
    assert train["engine_teacher_accuracy"] == 1.0
    assert train["majority_teacher_accuracy"] == 0.5
    assert 0.0 <= train["engine_ece_10bins"] <= 1.0
    assert train["mean_projection_to_base_l2"] == pytest.approx(0.75)
    val = _summarize(data, None, "working")
    assert "engine_teacher_accuracy" not in val
    assert "teacher_engine_occupancy" not in val
    with pytest.raises(ValueError, match="Training labels"):
        _summarize(data, {"train:FD003:1": [0, 1]}, "working")


def test_gate_summary_rejects_nonfinite_probabilities() -> None:
    """Invalid model logits must fail rather than contaminate audit reports."""
    item = _EngineGateStats.empty(3)
    with pytest.raises(ValueError, match="Invalid checkpoint gate output"):
        item.append(torch.tensor([float("nan"), 0.0, 1.0]), torch.ones(64), torch.tensor(1.0))
    with pytest.raises(ValueError, match="Invalid checkpoint gate output"):
        item.append(torch.tensor([0.5, 0.5, 0.0]), torch.ones(64), torch.tensor(0.0))


def test_checkpoint_selection_remaps_pod_path_and_requires_matching_epoch(tmp_path: Path) -> None:
    """A best model path saved inside a Pod resolves to Prisma's native root."""
    run = "rul-FD003-enhanced-cfc-cluster-working-s42"
    directory = tmp_path / "rul/checkpoints" / run
    directory.mkdir(parents=True)
    checkpoint = directory / "epoch=7-step=123.ckpt"
    checkpoint.write_bytes(b"immutable model placeholder")
    audit = tmp_path / "rul/audits" / f"{run}.checkpoint.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({
        "selection": "best minimum val/loss",
        "checkpoint_epoch": 7,
        "best_model_path": f"/tmp/nexuml-run/.experiments/rul/checkpoints/{run}/{checkpoint.name}",
    }), encoding="utf-8")
    curve = (
        tmp_path
        / "rul/enhanced-cfc/cluster-conditioned/fd003"
        / f"{run}.curves.csv"
    )
    curve.parent.mkdir(parents=True)
    with curve.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "train_eval_mse", "val_eval_mse", "mse_gap", "val_group_max_mse"
        ])
        writer.writeheader()
        writer.writerow({
            "epoch": 7, "train_eval_mse": 100, "val_eval_mse": 120,
            "mse_gap": 20, "val_group_max_mse": 160,
        })
    selected, epoch, values = _selected_checkpoint(tmp_path, "working", 42)
    assert selected == checkpoint
    assert epoch == 7
    assert values["val_eval_mse"] == 120.0
    audit.write_text(json.dumps({
        "selection": "best minimum val/loss",
        "checkpoint_epoch": 7,
        "best_model_path": "/tmp/nexuml-run/.experiments/rul/checkpoints/wrong-run/last.ckpt",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid validation-selected"):
        _selected_checkpoint(tmp_path, "working", 42)


def test_checkpoint_head_identity_requires_correct_gates(tmp_path: Path) -> None:
    """Do not silently audit a reference or wrong-arm checkpoint as conditioned."""
    path = tmp_path / "trusted.ckpt"
    root = "pipeline.stages.RULModel.RULModel."
    state = {
        root + "encoder.embedding.0.weight": torch.zeros(64, 1),
        root + "reference_mean": torch.zeros(30, 128),
        root + "reference_std": torch.ones(30, 128),
        root + "working_gate.1.weight": torch.ones(64, 128),
    }
    torch.save({"epoch": 7, "state_dict": state}, path)
    assert "working_gate.1.weight" in _model_state(path, "working", 7)
    with pytest.raises(ValueError, match="gates"):
        _model_state(path, "failure", 7)
    with pytest.raises(ValueError, match="gates"):
        _model_state(path, "reference", 7)
    with pytest.raises(ValueError, match="Invalid selected Lightning checkpoint"):
        _model_state(path, "working", 5)
