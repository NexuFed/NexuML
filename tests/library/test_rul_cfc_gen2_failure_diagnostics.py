"""Tests for frozen Gen2 per-engine failure geometry and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nexuml.core.components import LayerBuildContext
from nexuml_library.evaluation.maintenance.rul_gen2_failure_diagnostics import (
    _asym_score,
    _diagnostic_geometry,
    _head_from_checkpoint,
    _run_name,
)
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)


def test_gen2_failure_geometry_uses_training_stage_axis() -> None:
    """A displacement toward train low-RUL features has positive projection."""
    training = torch.zeros(4, 30, 256)
    training[2:, :, 0] = 2.0
    targets = torch.tensor([[130.0], [110.0], [10.0], [20.0]])
    testing = torch.zeros(1, 30, 256)
    testing[:, :, 0] = 1.0
    testing[:, :, 128] = 0.5
    geometry = _diagnostic_geometry(testing, training, targets)
    assert geometry["delta_toward_low_rul"].tolist() == pytest.approx([0.5])
    assert geometry["baseline_degradation_projection"].tolist() == pytest.approx(
        [1.0]
    )
    assert geometry["transported_degradation_projection"].tolist() == pytest.approx(
        [1.5]
    )
    assert geometry["train_delta_rms_percentile"].tolist() == pytest.approx(
        [1.0]
    )
    nearest = geometry["nearest_train_pooled_embedding_distance"].tolist()
    assert nearest == pytest.approx([1.0])


def test_gen2_failure_geometry_rejects_unusable_training_stages() -> None:
    """Reference construction must reject a missing stage or wrong shape."""
    training = torch.zeros(2, 30, 256)
    features = torch.zeros(1, 30, 256)
    with pytest.raises(ValueError, match="both high- and low-RUL"):
        _diagnostic_geometry(training, training, torch.full((2, 1), 130.0))
    with pytest.raises(ValueError, match="aligned"):
        _diagnostic_geometry(features[:, :, :128], training, torch.zeros(2, 1))


def test_gen2_failure_score_has_asymmetric_error_cost() -> None:
    """Overestimating RUL is costlier than symmetric underestimation."""
    assert _asym_score(100.0, 100.0) == pytest.approx(0.0)
    assert _asym_score(120.0, 100.0) > _asym_score(80.0, 100.0)
    assert _run_name("flow", 44) == (
        "rul-FD001-cfc-gen2-flow-d010-adam_1e3-window-head-s44"
    )


def test_gen2_failure_head_restore_is_strict(tmp_path: Path) -> None:
    """Native head checkpoint keys must exactly and uniquely reconstruct weights."""
    context = LayerBuildContext(
        input_sizes={"sequence_rul_features": (30, 256)},
        keys_in=["sequence_rul_features"],
        keys_out=["predicted_rul"],
    )
    head = PatchFormerSequenceFlowRULHead(
        hidden_dim=64, attention_heads=4, dropout=0.1, initialization_seed=43
    ).build(context)
    head.eval()
    path = tmp_path / "head.ckpt"
    weights = {
        f"pipeline.RULModel.layer.{key}": value
        for key, value in head.state_dict().items()
    }
    torch.save({"state_dict": weights}, path)
    restored = _head_from_checkpoint(path, 43)
    sample = torch.randn(2, 30, 256)
    with torch.no_grad():
        actual = restored.forward_tensor(sample)
        assert torch.allclose(head.forward_tensor(sample), actual)
    duplicate = next(iter(head.state_dict()))
    weights[f"duplicate.{duplicate}"] = head.state_dict()[duplicate].clone()
    torch.save({"state_dict": weights}, path)
    with pytest.raises(ValueError, match="ambiguous"):
        _head_from_checkpoint(path, 43)
