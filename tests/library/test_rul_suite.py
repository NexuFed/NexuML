"""NEX-233 registered RUL suite orchestration tests, without model retraining."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from nexuml_library.evaluation.maintenance.rul_suite import (
    RUL_SUITE_SCENARIOS,
    run_rul_scenario_suite,
)


def test_suite_uses_only_native_commands_and_isolates_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each dataset-model pair gets its own resolve/build/train and checkpoints."""
    first, second = tmp_path / "cmapss.pt", tmp_path / "xjtu.pt"
    first.touch()
    second.touch()
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        calls.append((command, env.copy()))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_rul_scenario_suite(
        [first, second],
        tmp_path / "suite",
        scenarios=RUL_SUITE_SCENARIOS[:2],
        max_epochs=1,
    )

    assert len(result["runs"]) == 4
    assert all(run["status"] == "completed" for run in result["runs"])
    assert len(calls) == 12
    for record, group in zip(result["runs"], range(0, len(calls), 3), strict=True):
        resolve, build, train = calls[group : group + 3]
        assert resolve[0][:3] == ["nexuml", "resolve", record["scenario"]]
        assert resolve[0][3:] == ["--output", record["config"]]
        assert build[0] == ["nexuml", "build", record["config"]]
        assert train[0] == [
            "nexuml",
            "train",
            "--config",
            record["config"],
            "--max-epochs",
            "1",
        ]
        for _, env in (resolve, build, train):
            assert env["NEXUML_RUL_PREPARED_FILE"] == record["prepared_file"]
            assert env["NEXUML_RUL_CHECKPOINT_ROOT"] == record["checkpoint_root"]
    saved = json.loads((tmp_path / "suite" / "manifest.json").read_text())
    assert saved["runs"] == result["runs"]
    assert len({run["checkpoint_root"] for run in saved["runs"]}) == 4


def test_suite_refuses_invalid_input_and_repeated_output(tmp_path: Path) -> None:
    """Do not start training for missing data or colliding suite outputs."""
    artifact = tmp_path / "prepared.pt"
    with pytest.raises(FileNotFoundError):
        run_rul_scenario_suite([artifact], tmp_path / "suite")
    artifact.touch()
    with pytest.raises(ValueError):
        run_rul_scenario_suite([artifact, artifact], tmp_path / "suite")
    with pytest.raises(ValueError):
        run_rul_scenario_suite([artifact], tmp_path / "suite", scenarios=["unknown"])
    (tmp_path / "suite").mkdir()
    with pytest.raises(FileExistsError):
        run_rul_scenario_suite([artifact], tmp_path / "suite")


def test_suite_records_failure_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed NexuML training command cannot be reported as a completed suite."""
    artifact = tmp_path / "prepared.pt"
    artifact.touch()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 2 if command[1] == "train" else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="NexuML train failed"):
        run_rul_scenario_suite(
            [artifact],
            tmp_path / "suite",
            scenarios=RUL_SUITE_SCENARIOS[:2],
        )
    saved = json.loads((tmp_path / "suite" / "manifest.json").read_text())
    assert len(calls) == 3
    assert saved["runs"][0]["status"] == "failed"
    assert saved["runs"][0]["failed_phase"] == "train"
    assert saved["runs"][0]["returncode"] == 2
    assert saved["runs"][1]["status"] == "pending"
