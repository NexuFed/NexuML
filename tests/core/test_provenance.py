"""Pure-logic tests for nexuml.core.provenance."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nexuml.core.provenance import (
    _best_trial,
    _git_state,
    snapshot_scenario_file_run,
)
from nexuml.core.scenario_loader import LoadedScenarioFile
from nexuml.core.types import ScenarioSpec


def test_git_state_includes_commit_and_branch():
    state = _git_state(Path.cwd())
    assert "commit" in state
    assert "branch" in state
    assert "status" in state
    assert state["commit"] is not None


def test_best_trial_returns_none_for_none_study():
    assert _best_trial(None) is None


def test_best_trial_returns_none_when_study_has_no_best():
    class BadStudy:
        @property
        def best_trial(self):
            raise ValueError("no best")

    assert _best_trial(BadStudy()) is None


def test_best_trial_extracts_trial_data():
    trial = SimpleNamespace(number=7, value=0.42, params={"lr": 0.01})
    study = SimpleNamespace(best_trial=trial)
    best = _best_trial(study)
    assert best == {"number": 7, "value": 0.42, "params": {"lr": 0.01}}


def test_snapshot_writes_expected_files(tmp_path):
    scenario = ScenarioSpec(name="prov_test")
    loaded = LoadedScenarioFile(
        path=tmp_path / "scenario.py",
        scenario=scenario,
        source="# scenario",
    )
    out = snapshot_scenario_file_run(
        loaded,
        artifact_dir=tmp_path / "artifacts",
        command="test",
        command_args={"foo": "bar"},
    )
    assert out is not None
    assert (out / "scenario.py").read_text() == "# scenario"
    assert (out / "metadata.json").exists()
    assert (out / "git.json").exists()
    assert (out / "command.json").exists()
    assert (out / "resolved_scenario.yaml").exists()


def test_snapshot_returns_none_when_artifact_dir_none():
    loaded = LoadedScenarioFile(
        path=Path.cwd() / "scenario.py",
        scenario=ScenarioSpec(name="x"),
        source="",
    )
    assert snapshot_scenario_file_run(loaded, None, command="test") is None
