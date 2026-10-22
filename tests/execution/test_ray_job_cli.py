from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from nexuml.cli import main
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import RayJobExecutionSpec, ScenarioSpec

runner = CliRunner()


@pytest.fixture
def submit_only(monkeypatch):
    import nexuml.execution as execution
    import nexuml.training.lightning as lightning
    import nexuml.core.compiler as compiler

    submitted = []

    def submit(scenario):
        submitted.append(scenario)
        return "raysubmit_cli"

    monkeypatch.setattr(execution, "submit_ray_job", submit)
    monkeypatch.setattr(
        lightning, "NexuSession", lambda **_: pytest.fail("submitter built a local session")
    )
    monkeypatch.setattr(compiler, "compile", lambda *_: pytest.fail("submitter compiled locally"))
    return submitted


@pytest.mark.parametrize("source", ["name", "config", "file"])
def test_train_submits_all_scenario_sources_with_overrides(
    submit_only, tmp_path, monkeypatch, source
):
    scenario = ScenarioSpec(name="submit-cli", execution=RayJobExecutionSpec())
    args = ["train"]
    if source == "name":
        monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)
        args += ["submit-cli"]
    elif source == "config":
        path = tmp_path / "scenario.yaml"
        ResolvedConfig.from_scenario(scenario).save(path)
        args += ["--config", str(path)]
    else:
        path = tmp_path / "scenario.py"
        path.write_text(
            "from nexuml.core.types import RayJobExecutionSpec, ScenarioSpec\n"
            "def scenario():\n"
            "    return ScenarioSpec(name='submit-cli', execution=RayJobExecutionSpec())\n"
        )
        args += ["--scenario-file", str(path), "--artifact-dir", str(tmp_path / "receipt")]

    result = runner.invoke(
        main.app,
        [
            *args,
            "--max-epochs",
            "5",
            "-O",
            "training.max_epochs=2",
            "-O",
            "execution.entrypoint_num_gpus=1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(submit_only) == 1
    assert submit_only[0].training.max_epochs == 2
    assert submit_only[0].execution.entrypoint_num_gpus == 1
    assert "Ray job submitted: raysubmit_cli" in result.output
    assert "Training complete" not in result.output
    assert "Ray training complete" not in result.output
    if source == "file":
        receipt = tmp_path / "receipt"
        saved = ResolvedConfig.load(receipt / "resolved_scenario.yaml")
        assert saved.training.max_epochs == 2
        metadata = json.loads((receipt / "command.json").read_text())["args"]
        assert metadata["execution"] == "ray_job"
        assert metadata["submission_id"] == "raysubmit_cli"


def test_ray_job_checkpoint_flag_fails_before_submission(submit_only, monkeypatch):
    scenario = ScenarioSpec(name="resume", execution=RayJobExecutionSpec())
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)

    result = runner.invoke(main.app, ["train", "resume", "--trainer-checkpoint", "local.ckpt"])

    assert result.exit_code == 1
    assert "does not transport --trainer-checkpoint" in result.output
    assert not submit_only


def test_ray_job_cli_does_not_fall_back_to_local_training_on_failure(submit_only, monkeypatch):
    import nexuml.execution as execution

    scenario = ScenarioSpec(name="failure", execution=RayJobExecutionSpec())
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)

    def fail(_):
        raise RuntimeError("Jobs server unavailable")

    monkeypatch.setattr(execution, "submit_ray_job", fail)
    result = runner.invoke(main.app, ["train", "failure"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "Ray job submitted" not in result.output
    assert not submit_only


def test_ray_job_appears_in_backend_catalog():
    result = runner.invoke(main.app, ["backend", "list", "training"])
    assert result.exit_code == 0, result.output
    assert "ray_job" in result.output


def test_cli_selects_ray_job_without_changing_the_local_scenario(submit_only):
    result = runner.invoke(
        main.app,
        [
            "train",
            "--scenario-file",
            "scripts/isaaclab/cartpole.py",
            "-O",
            'execution={"kind":"ray_job","entrypoint_num_gpus":1}',
            "-O",
            "execution.verify=/certs/ca.pem",
            "-O",
            "interaction.environment.viewer.port=8081",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(submit_only) == 1
    scenario = submit_only[0]
    assert isinstance(scenario.execution, RayJobExecutionSpec)
    assert scenario.execution.entrypoint_num_gpus == 1
    assert scenario.execution.verify == "/certs/ca.pem"
    assert scenario.interaction.environment.viewer.port == 8081


@pytest.mark.parametrize(
    "execution",
    [
        {"kind": "ray_job", "entrypoint_num_gpus": -1},
        {"kind": "ray"},
    ],
)
def test_invalid_execution_override_is_rejected_before_dispatch(submit_only, execution):
    result = runner.invoke(
        main.app,
        [
            "train",
            "--scenario-file",
            "scripts/isaaclab/cartpole.py",
            "-O",
            f"execution={json.dumps(execution)}",
        ],
    )
    assert result.exit_code == 1
    assert "Override error" in result.output
    assert not submit_only
