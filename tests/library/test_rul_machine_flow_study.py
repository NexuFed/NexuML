"""Contracts for leakage-sealed within-engine conditional flow RUL evaluation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from nexuml_library.evaluation.maintenance.rul_machine_flow_study import (
    MachineTrajectory,
    RULCalibration,
    _evenly_spaced,
    _index_engine_rows,
    _prefix_count,
    _score,
    _transition_pairs,
    _validate_prepared_settings,
    estimate_rul_from_dynamics,
    fit_dynamics,
    fit_rul_calibration,
    run_study,
)
from nexuml_library.layers.model.rul_machine_flow_dynamics import (
    MachineFutureDynamics,
)


def _trace(
    engine: str,
    split: str,
    *,
    size: int = 70,
    shift: float = 0.0,
) -> MachineTrajectory:
    """Construct an engine whose latent degradation advances with cycles.

    Returns:
        A chronological synthetic machine.
    """
    time = torch.arange(size, dtype=torch.float32)
    latent = torch.zeros(size, 128)
    latent[:, 0] = time / 50 + shift
    latent[:, 1] = torch.sin(time / 10)
    target = (size - 1 - time).clamp(max=130).reshape(-1, 1)
    return MachineTrajectory(
        engine=engine,
        split="train" if split == "train" else "val",
        steps=torch.arange(size),
        latent=latent,
        rul=target,
    )


def test_engine_index_preserves_train_val_and_never_returns_test() -> None:
    """Only original 80/20 engine groups are available to the study."""
    rows: list[dict[str, Any]] = []
    for split, engines in (("train", 80), ("val", 20), ("test", 100)):
        for number in range(engines):
            for cycle in range(3):
                rows.append({
                    "split": split,
                    "asset_id": f"{split}:FD003:{number}",
                    "step": cycle,
                })
    indices = _index_engine_rows(rows)
    assert set(indices) == {"train", "val"}
    assert len(indices["train"]) == 80
    assert len(indices["val"]) == 20
    assert all(
        rows[index]["split"] == split
        for split in indices
        for machine in indices[split].values()
        for index in machine
    )
    with pytest.raises(ValueError, match="80 train and 20 val"):
        _index_engine_rows([row for row in rows if row["asset_id"] != "val:FD003:19"])


def test_observed_prefix_keeps_future_transitions_out() -> None:
    """No training horizon may cross the last observed engine cycle."""
    observed = torch.arange(70, dtype=torch.long)
    count = _prefix_count(len(observed), 0.7)
    assert count == 49
    before, after, horizon = _transition_pairs(observed[:count])
    assert bool((after < count).all())
    assert bool((before < after).all())
    torch.testing.assert_close(
        (observed[after] - observed[before]).float().reshape(-1, 1), horizon
    )
    assert (before == count - 1).sum() == 0
    with pytest.raises(ValueError, match="too-short|40 observations"):
        _prefix_count(30, 0.7)
    with pytest.raises(ValueError, match="no|No chronological"):
        _transition_pairs(torch.tensor([0, 100, 200]))


def test_retrospective_protocol_requires_existing_train_targets() -> None:
    """Prepared data must retain physical-cycle labels, not scaled targets."""
    expected = {
        "subset": "FD003", "seed": 42,
        "window": 30, "stride": 1, "rul_cap": 130.0,
        "target_scale": 1.0,
        "normalize_targets": False, "train_full_windows_only": True,
    }
    _validate_prepared_settings(expected)
    with pytest.raises(ValueError, match="unnormalized"):
        _validate_prepared_settings({**expected, "normalize_targets": True})
    with pytest.raises(ValueError, match="unnormalized"):
        _validate_prepared_settings({**expected, "target_scale": 130.0})


def test_same_capacity_and_boundary_for_both_forecasting_models() -> None:
    """Flow interpolation is distinct from the physical forecast horizon."""
    direct = MachineFutureDynamics(method="direct_future", dimension=128)
    flow = MachineFutureDynamics(method="flow", dimension=128)
    assert direct.parameter_count() == flow.parameter_count()
    start = torch.randn(5, 128)
    horizon = torch.tensor([[0.0], [1.0], [5.0], [10.0], [20.0]])
    for model in (direct, flow):
        result = model.forecast(start, horizon, integration_steps=4)
        assert result.shape == start.shape
        torch.testing.assert_close(result[0], start[0], rtol=0, atol=0)
        loss = model.training_loss(
            start,
            start + torch.ones_like(start) * 0.01,
            torch.ones(5, 1),
        )
        assert bool(torch.isfinite(loss))
        loss.backward()
    with pytest.raises(ValueError, match="positive"):
        flow.training_loss(start, start, torch.zeros(5, 1))


def test_train_dynamics_does_not_use_machine_rul_labels() -> None:
    """Identical prefix histories yield identical updates despite other labels."""
    original = _trace("val:FD003:1", "val")
    altered = dataclasses.replace(original, rul=original.rul.flip(0).clone())
    a, a_loss = fit_dynamics(
        method="direct_future",
        trace=original,
        prefix_count=49,
        seed=42,
        device=torch.device("cpu"),
        train_steps=3,
        batch_size=8,
        learning_rate=1e-3,
    )
    b, b_loss = fit_dynamics(
        method="direct_future",
        trace=altered,
        prefix_count=49,
        seed=42,
        device=torch.device("cpu"),
        train_steps=3,
        batch_size=8,
        learning_rate=1e-3,
    )
    assert a_loss == pytest.approx(b_loss)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


def test_calibration_is_only_on_completed_training_engines() -> None:
    """A supervised decoder can be learned without target validation labels."""
    training = {
        f"train:FD003:{i}": _trace(
            f"train:FD003:{i}", "train", shift=0.05 * i
        )
        for i in range(80)
    }
    decoder = fit_rul_calibration(training)
    prediction = decoder.capped(_trace("val:FD003:4", "val").latent)
    assert prediction.shape == (70,)
    assert bool((prediction >= 0).all())
    assert bool((prediction <= 130).all())
    with pytest.raises(ValueError, match="80 training engines"):
        fit_rul_calibration(dict(list(training.items())[:-1]))


def test_forecast_to_rul_calibration_detects_nonprogression() -> None:
    """Rate conversion uses physical cycles, not FM interpolation position."""
    weights = torch.zeros(128)
    weights[0] = 1
    decoder = RULCalibration(weights=weights, bias=torch.tensor(0.0))
    current = torch.zeros(128)
    current[0] = 40.0
    forecast = {}
    for k in (5, 10, 20):
        point = current.clone()
        point[0] = 40.0 - 2.0 * k
        forecast[k] = point
    prediction, slope, censored = estimate_rul_from_dynamics(
        decoder, current, forecast
    )
    assert prediction == pytest.approx(20)
    assert slope == pytest.approx(-2)
    assert not censored
    prediction, slope, censored = estimate_rul_from_dynamics(
        decoder, current, {5: current, 10: current}
    )
    assert prediction == 130
    assert slope == pytest.approx(0)
    assert censored


def test_sampling_and_metrics_do_not_require_future_labels_in_fit() -> None:
    """Engine-balanced train sampling and physical RUL scoring have fixed units."""
    assert _evenly_spaced(9, 3) == [0, 4, 8]
    assert len(set(_evenly_spaced(100, 64))) == 64
    assert _score([-10.0, 10.0]) == pytest.approx((10.0, 10.0))
    with pytest.raises(ValueError, match="positive|two"):
        _evenly_spaced(1, 64)


def test_causal_study_smoke_writes_guarded_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end synthetic prefix adaptation reports no official test results."""
    from nexuml_library.evaluation.maintenance import rul_machine_flow_study

    train = {
        f"train:FD003:{i}": _trace(
            f"train:FD003:{i}", "train", shift=i * 0.02
        )
        for i in range(80)
    }
    val = {
        f"val:FD003:{i}": _trace(
            f"val:FD003:{i}", "val", shift=i * 0.02
        )
        for i in range(20)
    }
    monkeypatch.setattr(
        rul_machine_flow_study,
        "load_trajectories",
        lambda *args, **kwargs: (
            train, val, {"test_observations_used": "false", "fixture": "synthetic"}
        ),
    )
    # The fitting API must receive a genuinely truncated trajectory and
    # zero real target-machine labels, not merely a cutoff into full data.
    actual_fit = rul_machine_flow_study.fit_dynamics

    def checked_fit(**kwargs: Any) -> tuple[MachineFutureDynamics, float]:
        observed = kwargs["trace"]
        assert len(observed.steps) == kwargs["prefix_count"]
        assert len(observed.steps) == 49
        assert bool((observed.rul == 0).all())
        return actual_fit(**kwargs)

    monkeypatch.setattr(rul_machine_flow_study, "fit_dynamics", checked_fit)
    # Avoid excessive CPU threading overhead in this tiny deterministic test.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = run_study(
            tmp_path,
            device=torch.device("cpu"),
            mode="causal",
            train_steps=2,
            batch_size=8,
            max_engines=1,
        )
    finally:
        torch.set_num_threads(old_threads)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["test_used"] is False
    assert manifest["results"]["causal"]["machines"] == 1
    assert manifest["results"]["causal"]["target_val_rul_labels_in_fit"] is False
    assert manifest["results"]["causal"]["validation_future_windows_in_fit"] is False
    assert (output.parent / "causal_per_machine.csv").is_file()
    assert (output.parent / "causal_forecast_horizons.csv").is_file()
    assert (output.parent / "causal_summary.json").is_file()
    assert set(manifest["results"]["causal"]["results"]) == {
        "direct", "nonflow", "flow"
    }
