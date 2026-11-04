from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nexuml.cli import main
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import ScenarioSpec, TrainingSpec
from nexuml.execution.definitions import RayJobsSDKExecution

runner = CliRunner()


@pytest.fixture
def submit_only(monkeypatch):
    import nexuml.execution as execution
    import nexuml.training.lightning as lightning
    import nexuml.core.compiler as compiler

    submitted = []

    def submit(scenario, **kwargs):
        submitted.append(scenario)
        return "raysubmit_cli"

    monkeypatch.setattr(execution, "run", submit)
    monkeypatch.setattr(
        lightning, "NexuSession", lambda **_: pytest.fail("submitter built a local session")
    )
    monkeypatch.setattr(compiler, "compile", lambda *_: pytest.fail("submitter compiled locally"))
    return submitted


@pytest.mark.parametrize("source", ["name", "config", "file"])
def test_train_submits_all_scenario_sources_with_overrides(
    submit_only, tmp_path, monkeypatch, source
):
    scenario = ScenarioSpec(name="submit-cli", execution=RayJobsSDKExecution())
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
            "from nexuml.core.types import ScenarioSpec\n"
            "from nexuml.execution.definitions import RayJobsSDKExecution\n"
            "def scenario():\n"
            "    return ScenarioSpec(name='submit-cli', execution=RayJobsSDKExecution())\n"
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
    text = " ".join(result.output.split())
    assert "backend: ray-jobs-sdk; operation: train" in text
    execution = submit_only[0].execution
    assert f"GPUs={execution.entrypoint_num_gpus}; CPUs={execution.entrypoint_num_cpus}" in text
    assert "RAM bytes=unspecified" in text
    assert "Configured epoch budget: 2" in text
    assert "phase: unknown until observed in job logs" in text
    assert "Training complete" not in result.output
    assert "Ray training complete" not in result.output
    if source == "file":
        receipt = tmp_path / "receipt"
        saved = ResolvedConfig.load(receipt / "resolved_scenario.yaml")
        assert isinstance(saved.training, TrainingSpec)
        assert saved.training.max_epochs == 2
        metadata = json.loads((receipt / "command.json").read_text())["args"]
        assert metadata["execution"] == "ray-jobs-sdk"
        assert metadata["submission_id"] == "raysubmit_cli"


def test_ray_job_checkpoint_path_is_not_read_before_submission(submit_only, monkeypatch):
    import nexuml.execution as execution

    scenario = ScenarioSpec(name="resume", execution=RayJobsSDKExecution())
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)
    monkeypatch.setattr(
        "nexuml.training.lightning.load_scenario_from_trainer_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("read worker checkpoint locally"),
    )
    calls = []
    monkeypatch.setattr(
        execution, "run", lambda spec, **kwargs: calls.append((spec, kwargs)) or "remote_resume"
    )

    path = "/worker only/partial ; 'checkpoint.ckpt"
    result = runner.invoke(main.app, ["train", "resume", "--trainer-checkpoint", path])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert str(calls[0][1]["trainer_checkpoint"]) == path
    assert "remote_resume" in result.output
    assert "saved semantics and output paths" in result.output
    assert "Worker train input:" in result.output


def test_ray_job_cli_does_not_fall_back_to_local_training_on_failure(submit_only, monkeypatch):
    import nexuml.execution as execution

    scenario = ScenarioSpec(name="failure", execution=RayJobsSDKExecution())
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)

    def fail(_, **kwargs):
        raise RuntimeError("Jobs server unavailable")

    monkeypatch.setattr(execution, "run", fail)
    result = runner.invoke(main.app, ["train", "failure"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "Ray job submitted" not in result.output
    assert not submit_only


def test_ray_job_appears_in_backend_catalog():
    result = runner.invoke(main.app, ["backend", "list", "execution"])
    assert result.exit_code == 0, result.output
    assert "ray-jobs-sdk" in result.output


def test_submission_summary_reports_rl_budget_without_dumping_execution_secrets(
    submit_only, monkeypatch
):
    from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo

    scenario = cartpole_ppo(total_frames=64, frames_per_batch=16)
    scenario.name = "literal [bold]scenario[/bold]"
    scenario.execution = RayJobsSDKExecution(
        entrypoint_num_cpus=2,
        entrypoint_memory=123456,
        entrypoint_resources={"pool_gpu_workers": 1},
        runtime_env={"env_vars": {"PRIVATE_TOKEN": "do-not-print-this"}},
    )
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)
    result = runner.invoke(main.app, ["train", "summary"])
    assert result.exit_code == 0, (result.output, result.exception)
    text = " ".join(result.output.split())
    assert "literal [bold]scenario[/bold]" in text
    assert "Configured physical frame budget: 64" in text
    assert "Requested training device: cpu" in text
    assert "RAM bytes=123456" in text and "pool_gpu_workers" in text
    assert "Configured epoch budget" not in text
    assert "do-not-print-this" not in text and "PRIVATE_TOKEN" not in text
    assert "Actual training device" not in text


@pytest.mark.parametrize("training_device", ["cpu", "gpu"])
def test_worker_completion_does_not_relabel_post_teardown_cpu_as_training_device(
    monkeypatch, training_device
):
    from types import SimpleNamespace
    import nexuml.execution as execution

    scenario = ScenarioSpec(name="finished", logging=None)
    scenario.training.accelerator = training_device
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)
    session = SimpleNamespace(
        lightning_module=SimpleNamespace(
            frames=0, device="cpu", fit_device="cuda:0" if training_device == "gpu" else "cpu"
        ),
        trainer=SimpleNamespace(global_step=0),
    )
    monkeypatch.setattr(execution, "run", lambda *_args, **_kwargs: session)
    result = runner.invoke(main.app, ["train", "finished"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Module device after completion: cpu" in result.output
    expected_fit_device = "cuda:0" if training_device == "gpu" else "cpu"
    assert f"Fit device observed at start: {expected_fit_device}" in result.output
    assert "Actual training device:" not in result.output
    assert "Physical training frames: 0" in result.output
    assert "Optimizer steps: 0" in result.output
    assert "Ray job submitted" not in result.output


@pytest.mark.parametrize("purposes", [("train",), ("eval",), ()])
@pytest.mark.parametrize("frames", [0, 64])
def test_finished_viewer_notice_uses_completed_scenario_without_enabling_replay(
    monkeypatch, purposes, frames
):
    from types import SimpleNamespace
    import nexuml.execution as execution

    scenario = ScenarioSpec(name="finished", logging=None)
    viewer = SimpleNamespace(purposes=purposes)
    completed = SimpleNamespace(
        interaction=SimpleNamespace(environment=SimpleNamespace(viewer=viewer))
    )
    module = SimpleNamespace(frames=frames, scenario=completed, device="cpu")
    result = SimpleNamespace(lightning_module=module)
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)
    calls = []
    monkeypatch.setattr(
        execution, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or result
    )
    output = runner.invoke(main.app, ["train", "finished"])
    assert output.exit_code == 0, (output.output, output.exception)
    text = " ".join(output.output.split())
    assert ("Configured training viewer: run finished" in text) == ("train" in purposes)
    if "train" in purposes:
        assert "not stalled training" in text
        assert "separate evaluation" in text and "explicit evaluation-viewer opt-in" in text
        assert "approved private worker route" in text
    assert len(calls) == 1
    assert viewer.purposes == purposes
    assert "http://" not in text and "port-forward" not in text
    assert (
        "Worker-local evaluation:" not in text
    )  # No produced package means no invented replay path.


@pytest.mark.parametrize("operation", ["train", "evaluate", "export"])
def test_worker_identity_uses_ray_metadata_without_printing_runtime_secrets(
    operation, monkeypatch, capsys
):
    monkeypatch.setenv(
        "RAY_JOB_CONFIG_JSON_ENV_VAR",
        json.dumps(
            {
                "metadata": {"job_submission_id": "actual-job-id", "private": "metadata-secret"},
                "runtime_env": {"env_vars": {"TOKEN": "runtime-secret"}},
            }
        ),
    )
    monkeypatch.setattr(main.socket, "gethostname", lambda: "allocated-worker")
    monkeypatch.setattr(main.os, "getpid", lambda: 123)
    main._print_worker_identity(operation)
    text = " ".join(capsys.readouterr().out.split())
    assert "Ray submission ID: actual-job-id" in text
    assert f"Worker operation: {operation}; host: allocated-worker; PID: 123" in text
    assert "metadata-secret" not in text and "runtime-secret" not in text
    assert "TOKEN" not in text and "private" not in text


@pytest.mark.parametrize(
    "metadata",
    ["broken json", "null", "[]", '{"metadata": []}', '{"metadata":{"job_submission_id":123}}'],
)
def test_worker_identity_marks_unavailable_metadata_without_failing(metadata, monkeypatch, capsys):
    monkeypatch.setenv("RAY_JOB_CONFIG_JSON_ENV_VAR", metadata)
    main._print_worker_identity("train")
    assert "Ray submission ID: unavailable" in capsys.readouterr().out


def test_local_identity_does_not_invent_a_ray_job(monkeypatch, capsys):
    monkeypatch.delenv("RAY_JOB_CONFIG_JSON_ENV_VAR", raising=False)
    main._print_worker_identity("train")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("width", [80, 1000])
def test_completion_commands_use_only_real_checkpoints_and_quote_worker_paths(
    tmp_path, monkeypatch, width
):
    from types import SimpleNamespace
    from rich.console import Console
    import nexuml.execution as execution

    checkpoint = tmp_path / ("saved " * 20 + "; 'checkpoint.ckpt")
    checkpoint.write_bytes(b"fixture only: never loaded")
    package = tmp_path / ("policy " * 20 + "; 'directory")
    package.mkdir()
    metadata = package / "metadata.json"
    metadata.write_text("{}")
    callbacks = [
        SimpleNamespace(last_model_path=str(checkpoint), best_model_path=str(checkpoint)),
        SimpleNamespace(last_model_path=str(tmp_path / "missing.ckpt")),
    ]
    scenario = ScenarioSpec(name="finished", logging=None)
    session = SimpleNamespace(
        lightning_module=SimpleNamespace(device="cpu", fit_device="cuda:0"),
        trainer=SimpleNamespace(checkpoint_callbacks=callbacks),
        artifacts=[{"kind": "train_package", "path": str(metadata)}],
    )
    monkeypatch.setattr(main, "console", Console(width=width, no_color=True))
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)
    monkeypatch.setattr(execution, "run", lambda *_args, **_kwargs: session)
    result = runner.invoke(main.app, ["train", "finished"])
    assert result.exit_code == 0, (result.output, result.exception)
    resume = [
        line.partition("Worker-local resume: ")[2]
        for line in result.output.splitlines()
        if "Worker-local resume: " in line
    ]
    evaluate = [
        line.partition("Worker-local evaluation: ")[2]
        for line in result.output.splitlines()
        if "Worker-local evaluation: " in line
    ]
    assert len(resume) == len(evaluate) == 1
    assert shlex.split(resume[0]) == ["nexuml", "train", "--trainer-checkpoint", str(checkpoint)]
    assert shlex.split(evaluate[0]) == ["nexuml", "evaluate", str(package), "--device", "cuda:0"]
    assert "missing.ckpt" not in result.output
    assert "From the submitter, select a trusted ray-jobs-sdk config" in " ".join(
        result.output.split()
    )


def test_cli_selects_ray_job_without_changing_the_local_scenario(submit_only):
    result = runner.invoke(
        main.app,
        [
            "train",
            "--scenario-file",
            "scripts/isaaclab/cartpole.py",
            "-O",
            'execution={"type":"ray-jobs-sdk","version":"1","params":'
            '{"entrypoint_num_gpus":1,"setup_command":["uv","sync","--locked"],'
            '"py_executable":".venv/bin/python"}}',
            "-O",
            "execution.verify=/certs/ca.pem",
            "-O",
            "interaction.environment.viewer.port=8081",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(submit_only) == 1
    scenario = submit_only[0]
    assert isinstance(scenario.execution, RayJobsSDKExecution)
    assert scenario.execution.entrypoint_num_gpus == 1
    assert scenario.execution.verify == "/certs/ca.pem"
    assert scenario.execution.setup_command == ["uv", "sync", "--locked"]
    assert scenario.execution.py_executable == ".venv/bin/python"
    assert scenario.interaction.environment.viewer.port == 8081


@pytest.mark.parametrize(
    "execution",
    [
        {"kind": "ray_job", "entrypoint_num_gpus": -1},
        {"kind": "ray_job", "setup_command": []},
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


@pytest.mark.parametrize("operation", ["evaluate", "export"])
@pytest.mark.parametrize("source", ["config", "file"])
def test_remote_lifecycle_selects_execution_before_worker_inputs(
    operation, source, submit_only, tmp_path, monkeypatch
):
    import nexuml.execution as execution

    scenario = ScenarioSpec(name="lifecycle", execution=RayJobsSDKExecution())
    if source == "config":
        path = tmp_path / "job.yaml"
        ResolvedConfig.from_scenario(scenario).save(path)
        selector = ["--config", str(path)]
    else:
        path = tmp_path / "job.py"
        path.write_text(
            "from nexuml.core.types import ScenarioSpec\n"
            "from nexuml.execution.definitions import RayJobsSDKExecution\n"
            "def scenario():\n"
            "    return ScenarioSpec(name='lifecycle', execution=RayJobsSDKExecution())\n"
        )
        selector = ["--scenario-file", str(path)]
    calls = []
    monkeypatch.setattr(
        execution, "run", lambda spec, **kwargs: calls.append((spec, kwargs)) or "lifecycle_id"
    )
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package",
        lambda *_args, **_kwargs: pytest.fail("opened worker package locally"),
    )
    worker_input = "/worker only/input ; 'data"
    args = (
        [
            "evaluate",
            worker_input,
            *selector,
            "--device",
            "cuda:0",
            "-O",
            "policy_evaluation.episodes=2",
        ]
        if operation == "evaluate"
        else ["export", *selector, "--checkpoint", worker_input, "--output", "/worker/policy"]
    )
    result = runner.invoke(main.app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(calls) == 1
    selected, kwargs = calls[0]
    assert isinstance(selected.execution, RayJobsSDKExecution)
    assert kwargs["operation"] == operation
    key = "model_package" if operation == "evaluate" else "checkpoint"
    assert str(kwargs[key]) == worker_input
    if operation == "evaluate":
        assert kwargs["overrides"] == ("policy_evaluation.episodes=2",)
        assert kwargs["device"] == "cuda:0"
    assert "Ray job submitted: lifecycle_id" in result.output
    text = " ".join(result.output.split())
    assert f"operation: {operation}" in text
    assert f"Worker {operation} input:" in text
    assert "Configured epoch budget" not in text
    assert "setup/learning phase: unknown" in text
    if operation == "export":
        assert "Requested worker export output (not yet produced): /worker/policy" in text
    assert "Training complete" not in result.output


@pytest.mark.parametrize("operation", ["evaluate", "export"])
def test_lifecycle_rejects_competing_selectors_before_dispatch(operation, submit_only):
    args = [operation, *(["/worker/policy"] if operation == "evaluate" else [])]
    result = runner.invoke(main.app, [*args, "--config", "one.yaml", "--scenario-file", "two.py"])
    assert result.exit_code == 1
    assert "Provide only one" in result.output
    assert not submit_only


def test_remote_export_requires_explicit_output(submit_only, tmp_path, monkeypatch):
    from nexuml.execution.service import run

    monkeypatch.setattr("nexuml.execution.run", run)
    scenario = ScenarioSpec(name="export", execution=RayJobsSDKExecution())
    path = tmp_path / "job.yaml"
    ResolvedConfig.from_scenario(scenario).save(path)
    result = runner.invoke(
        main.app, ["export", "--config", str(path), "--checkpoint", "/worker/model.ckpt"]
    )
    assert result.exit_code != 0
    assert "output" in str(result.exception).lower()
    assert not submit_only


def test_local_export_preserves_default_output(monkeypatch):
    from nexuml.execution.definitions import LocalExecution

    calls = []
    monkeypatch.setattr(main, "_export_local", lambda spec, **kw: calls.append(kw))
    LocalExecution().run(ScenarioSpec(name="local"), operation="export", output=None)
    assert calls[0]["output"] == Path("exported_model")
