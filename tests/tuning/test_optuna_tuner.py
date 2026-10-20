"""Fast, training-free regression tests for nexuml.tuning.optuna_tuner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexuml.core.types import TuningSpec
from nexuml.tuning.optuna_tuner import _resolve_search_space, build_objective, tune


class FakeTrial:
    """Duck-typed Optuna trial that always returns deterministic values."""

    def __init__(self, number: int = 0) -> None:
        self.number = number
        self.params: dict[str, Any] = {}

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        self.params[name] = low
        return low

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        self.params[name] = low
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        value = choices[0]
        self.params[name] = value
        return value


def _fake_train_result(logged_metrics: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        trainer=SimpleNamespace(logged_metrics=logged_metrics),
        eval_algorithm_results={},
    )


def test_resolve_search_space_scalar_override():
    trial = FakeTrial()
    search_space = {"training.lr": {"type": "float", "low": 1e-5, "high": 1e-2}}

    scalar, arch = _resolve_search_space(trial, search_space)

    assert scalar == {"training.lr": 1e-5}
    assert arch == {}


def test_resolve_search_space_categorical():
    trial = FakeTrial()
    search_space = {"training.batch_size": {"type": "categorical", "choices": [32, 64, 128]}}

    scalar, arch = _resolve_search_space(trial, search_space)

    assert scalar == {}
    assert arch == {"training": {"batch_size": 32}}


def test_resolve_search_space_conditional_branch():
    trial = FakeTrial()
    search_space = {
        "encoder_type": {
            "type": "categorical",
            "choices": ["conv", "linear"],
            "when": {"conv": {"encode.kernel_size": {"type": "int", "low": 3, "high": 7}}},
        },
    }

    scalar, arch = _resolve_search_space(trial, search_space)

    assert scalar == {}
    assert arch == {"encoder_type": "conv", "encode": {"kernel_size": 3}}


def test_resolve_search_space_derived():
    trial = FakeTrial()
    search_space = {
        "lr": {"type": "float", "low": 0.001, "high": 0.001},
        "use_high_lr": {"derived": "lr == 0.001"},
    }

    scalar, arch = _resolve_search_space(trial, search_space)

    assert scalar == {}
    assert arch == {"lr": 0.001, "use_high_lr": True}


def test_build_objective_rebuilds_scenario_per_trial(vector_scenario_spec, monkeypatch):
    seen: list[Any] = []

    def fake_train(scenario, registry=None, enable_progress_bar=False, run_name=None):
        seen.append(scenario)
        return _fake_train_result({"val/loss": 0.5})

    monkeypatch.setattr("nexuml.training.lightning.train", fake_train)

    objective = build_objective(
        vector_scenario_spec,
        {"training.lr": {"type": "float", "low": 0.0, "high": 0.0}},
        metric_key="val/loss",
    )

    objective(FakeTrial(number=0))
    objective(FakeTrial(number=1))

    assert len(seen) == 2
    assert seen[0] is not seen[1]
    assert seen[0] is not vector_scenario_spec
    assert vector_scenario_spec.training.lr == 1e-3


def test_build_objective_missing_metric_lists_available(vector_scenario_spec, monkeypatch):
    def fake_train(scenario, registry=None, enable_progress_bar=False, run_name=None):
        return _fake_train_result({"val/loss": 0.5, "val/acc": 0.9})

    monkeypatch.setattr("nexuml.training.lightning.train", fake_train)

    objective = build_objective(vector_scenario_spec, {}, metric_key="val/missing")

    with pytest.raises(ValueError) as excinfo:
        objective(FakeTrial())

    assert "val/loss" in str(excinfo.value)
    assert "val/acc" in str(excinfo.value)


@pytest.mark.parametrize("directions", [["minimize"], ["maximize"], ["minimize", "maximize"]])
def test_tune_forwards_directions_unchanged(
    vector_scenario_spec, tmp_path, monkeypatch, directions
):
    optuna = pytest.importorskip("optuna")
    calls = []
    study = SimpleNamespace(
        optimize=lambda objective, n_trials: None,
        best_trial=SimpleNamespace(number=0, params={}, value=0.5),
        trials=[],
    )

    def create_study(**kwargs):
        calls.append(kwargs)
        return study

    monkeypatch.setattr(optuna, "create_study", create_study)
    spec = TuningSpec(directions=directions, n_trials=1, storage=str(tmp_path / "study.log"))
    assert tune(vector_scenario_spec, {}, spec) is study
    if len(directions) == 1:
        assert calls[0]["direction"] == directions[0]
        assert "directions" not in calls[0]
    else:
        assert calls[0]["directions"] == directions
        assert "direction" not in calls[0]
    assert spec.directions == directions


@pytest.mark.parametrize("directions", [[], ["invalid"], ["MINIMIZE"], ["minimize", "invalid"]])
def test_tune_preserves_optuna_direction_errors(vector_scenario_spec, tmp_path, directions):
    optuna = pytest.importorskip("optuna")
    with pytest.raises(ValueError) as expected:
        if len(directions) == 1:
            optuna.create_study(direction=directions[0])
        else:
            optuna.create_study(directions=directions)

    spec = TuningSpec(directions=directions, storage=str(tmp_path / "study.log"))
    with pytest.raises(ValueError) as actual:
        tune(vector_scenario_spec, {}, spec)
    assert str(actual.value) == str(expected.value)
