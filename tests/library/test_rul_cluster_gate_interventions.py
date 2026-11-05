"""Checkpoint intervention contracts for completed FD003 cluster-conditioning runs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch

from nexuml_library.evaluation.maintenance.rul_cluster_gate_interventions import (
    Arms,
    _ReplayCfC,
    _age_bucket,
    _counterfactual_predictions,
    _donor_engines,
    _enabled_gates,
    _interventions,
    _score_prediction,
    _swap_engine_memberships,
    _validation_index,
    _verify_original_replay,
)


def _synthetic_rows() -> Iterator[dict[str, Any]]:
    """Yield interleaved synthetic training, validation and official test metadata.

    Yields:
        Rows with the original FD003 split design.
    """
    for step in range(4):
        for split, engines in (("train", 80), ("val", 20), ("test", 100)):
            for number in range(engines):
                yield {
                    "split": split,
                    "asset_id": f"{split}:FD003:{number}",
                    "step": step,
                }


def test_interventions_only_target_trained_gates() -> None:
    """A one-gate checkpoint cannot be ablated as an unavailable gate."""
    assert _enabled_gates("reference") == ()
    assert _enabled_gates("working") == ("working",)
    assert _enabled_gates("failure") == ("failure",)
    assert _enabled_gates("both") == ("working", "failure")
    assert _interventions("reference") == ("original",)
    assert "zero_failure" not in _interventions("working")
    assert "zero_working" not in _interventions("failure")
    assert "zero_both" in _interventions("both")
    assert "shuffle_memberships" in _interventions("both")


def test_validation_selection_is_exhaustive_and_test_sealed() -> None:
    """Intervention selects every val window and no train or official test rows."""
    rows = list(_synthetic_rows())
    selected, per_engine = _validation_index(rows)
    assert len(per_engine) == 20
    assert len(selected) == 80
    assert set(selected) == {i for i, row in enumerate(rows) if row["split"] == "val"}
    assert all(rows[index]["split"] == "val" for index in selected)
    for positions in per_engine.values():
        assert len(positions) == 4
        original = [selected[position] for position in positions]
        assert [rows[index]["step"] for index in original] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="20"):
        _validation_index([row for row in rows if row["asset_id"] != "val:FD003:19"])


def test_shuffle_is_deterministic_bijective_and_age_matched() -> None:
    """Shuffle preserves soft memberships but never retains an engine's own vector."""
    engines = ["a", "b", "c"]
    donors = _donor_engines(engines, 2026)
    assert donors == _donor_engines(engines, 2026)
    assert set(donors.values()) == set(engines)
    assert all(engine != donor for engine, donor in donors.items())
    positions = {"a": [0, 1, 2], "b": [3, 4], "c": [5, 6, 7, 8]}
    probability = torch.tensor(
        [[0.01], [0.02], [0.03], [0.11], [0.12],
         [0.21], [0.22], [0.23], [0.24]]
    )
    swapped = _swap_engine_memberships(probability, positions, donors)
    for engine, indices in positions.items():
        source = positions[donors[engine]]
        for rank, index in enumerate(indices):
            donor_rank = round(rank * (len(source) - 1) / (len(indices) - 1))
            assert swapped[index].item() == pytest.approx(
                probability[source[donor_rank]].item()
            )
            assert swapped[index].item() != probability[index].item()
    assert torch.equal(probability[:, 0], torch.tensor(
        [0.01, 0.02, 0.03, 0.11, 0.12, 0.21, 0.22, 0.23, 0.24]
    ))
    with pytest.raises(ValueError, match="derangement|original engine"):
        _swap_engine_memberships(probability, positions, {engine: engine for engine in engines})


def test_age_buckets_use_full_engine_lifetime() -> None:
    """Relative age is a post-hoc annotation, never an input to the gate."""
    assert [_age_bucket(rank, 7) for rank in range(7)] == [
        "early", "early", "middle", "middle", "late", "late", "late",
    ]
    with pytest.raises(ValueError, match="lifespan"):
        _age_bucket(3, 2)


def test_paired_metrics_preserve_engine_macro_and_window_weighting() -> None:
    """Macro-engine metrics do not let long engines dominate the sample mean."""
    observed = torch.tensor([[1.0], [1.0], [1.0], [3.0]])
    targets = torch.zeros_like(observed)
    summary, by_engine = _score_prediction(
        observed, targets, {"a": [0, 1, 2], "b": [3]}
    )
    assert summary["window_mse"] == pytest.approx(3.0)
    assert summary["engine_macro_mse"] == pytest.approx(5.0)
    assert summary["worst_engine_mse"] == pytest.approx(9.0)
    assert by_engine == {"a": pytest.approx(1.0), "b": pytest.approx(9.0)}


def test_native_checkpoint_reproduction_is_required() -> None:
    """Mismatched MSE fails instead of allowing misleading counterfactual tables."""
    _verify_original_replay(140.0, 140.01, "working-s42")
    with pytest.raises(ValueError, match="no interventions are valid"):
        _verify_original_replay(140.0, 145.0, "working-s42")


@pytest.mark.parametrize("arm", ["reference", "working", "failure", "both"])
def test_replay_strict_checkpoint_and_zero_projection(
    arm: Arms, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconstructed heads honor original, ablated and strictly saved model weights."""
    # Bootstrap an architecture to construct a synthetic, complete checkpoint.
    # A second instance must then strictly load the real PyTorch state_dict.
    with monkeypatch.context() as patcher:
        patcher.setattr(_ReplayCfC, "load_state_dict", lambda self, *a, **kw: None)
        prototype = _ReplayCfC(arm, {})
    state = prototype.state_dict()
    model = _ReplayCfC(arm, state).eval()
    encoded = torch.randn(2, 30, 64)
    probs = {
        kind: torch.full(
            (2, 3 if kind == "working" else 2),
            1.0 / (3 if kind == "working" else 2),
        )
        for kind in model.enabled
    }
    with torch.inference_mode():
        baseline = model.predict_from_encoded(
            encoded, probs, intervention="original"
        )
        assert baseline.shape == (2, 1)
        if arm == "reference":
            with pytest.raises(ValueError, match="Unsupported intervention"):
                model.predict_from_encoded(
                    encoded, probs, intervention="zero_both"
                )
        else:
            for kind in model.enabled:
                projector = (
                    model.working_projection if kind == "working"
                    else model.failure_projection
                )
                assert projector is not None
                projector.weight.zero_()
            zeroed = model.predict_from_encoded(
                encoded, probs, intervention="original"
            )
            intervention = (
                "zero_both" if arm == "both" else
                "zero_working" if arm == "working" else "zero_failure"
            )
            ablated = model.predict_from_encoded(
                encoded, probs, intervention=intervention
            )
            torch.testing.assert_close(zeroed, ablated, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Missing key"):
        _ReplayCfC(arm, {"reference_mean": state["reference_mean"]})


def test_counterfactual_prediction_batches_do_not_mutate_memberships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated gate interventions never overwrite the original probabilities."""
    with monkeypatch.context() as patcher:
        patcher.setattr(_ReplayCfC, "load_state_dict", lambda self, *a, **kw: None)
        prototype = _ReplayCfC("working", {})
    model = _ReplayCfC("working", prototype.state_dict()).eval()
    encoded = torch.randn(5, 30, 64)
    original = {"working": torch.softmax(torch.randn(5, 3), dim=-1)}
    shuffled = {"working": original["working"].flip(0)}
    preserved = original["working"].clone()
    with torch.inference_mode():
        full = _counterfactual_predictions(
            model, encoded, original, shuffled,
            "original", device=torch.device("cpu"), batch_size=2,
        )
        ablated = _counterfactual_predictions(
            model, encoded, original, shuffled,
            "zero_working", device=torch.device("cpu"), batch_size=2,
        )
        shuffled_prediction = _counterfactual_predictions(
            model, encoded, original, shuffled,
            "shuffle_memberships", device=torch.device("cpu"), batch_size=2,
        )
    assert full.shape == ablated.shape == shuffled_prediction.shape == (5, 1)
    torch.testing.assert_close(original["working"], preserved)
