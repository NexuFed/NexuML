from __future__ import annotations

import builtins
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexuml.core.config import ResolvedConfig
from nexuml.core.serialization import lower_model
from nexuml.core.types import ExportSpec, RayExecutionSpec, RayJobExecutionSpec, ScenarioSpec
from nexuml.execution import submit_ray_job
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_learning_ppo


@pytest.fixture
def jobs_client(monkeypatch):
    sdk = pytest.importorskip("ray.job_submission")
    calls = SimpleNamespace(connection=None, submission=None, error=None)

    class Client:
        def __init__(self, address, **kwargs):
            calls.connection = {"address": address, **kwargs}

        def submit_job(self, **kwargs):
            calls.submission = kwargs
            if calls.error:
                raise calls.error
            return "raysubmit_test"

    monkeypatch.setattr(sdk, "JobSubmissionClient", Client)
    return calls


def test_ray_job_config_roundtrip_allows_rl_without_enabling_ray_train():
    scenario = isaac_cartpole_learning_ppo()
    scenario.execution = RayJobExecutionSpec(
        address="https://jobs.example.test",
        verify="/certs/ca.pem",
        py_executable="uv run --locked --extra isaac --extra ray python",
        runtime_env={"working_dir": ".", "env_vars": {"UV_CACHE_DIR": "/mnt/data/uv-cache"}},
        entrypoint_num_gpus=1,
        entrypoint_resources={"pool_gpu_workers": 1},
    )

    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())

    assert isinstance(restored.execution, RayJobExecutionSpec)
    assert lower_model(restored) == lower_model(scenario)
    with pytest.raises(ValueError, match="Ray learner placement"):
        ScenarioSpec(
            **{
                **{key: getattr(scenario, key) for key in ScenarioSpec.model_fields},
                "execution": RayExecutionSpec(),
            }
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entrypoint_num_gpus": -1},
        {"entrypoint_num_cpus": float("nan")},
        {"entrypoint_num_gpus": float("inf")},
        {"entrypoint_memory": -1},
        {"entrypoint_resources": {"GPU-pool": 0}},
        {"entrypoint_resources": {"GPU-pool": float("nan")}},
        {"entrypoint_resources": {" ": 1}},
        {"py_executable": ""},
        {"verify": ""},
    ],
)
def test_ray_job_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RayJobExecutionSpec(**kwargs)


def test_submission_preserves_native_sdk_options_and_caller(jobs_client, monkeypatch):
    monkeypatch.setattr("ray.init", lambda **_: pytest.fail("Jobs must not call ray.init"))
    execution = RayJobExecutionSpec(
        address="https://jobs.example.test",
        verify="/certs/ca.pem",
        py_executable="uv run --locked --python 3.12 --extra isaac --extra ray python",
        runtime_env={
            "working_dir": ".",
            "excludes": ["private/"],
            "env_vars": {"UV_CACHE_DIR": "/mnt/data/uv-cache", "TOKEN": "submitter-only"},
        },
        entrypoint_num_cpus=2,
        entrypoint_num_gpus=1,
        entrypoint_memory=4096,
        entrypoint_resources={"pool_gpu_workers": 1},
    )
    scenario = ScenarioSpec(name="remote", execution=execution)
    before = lower_model(scenario)

    assert submit_ray_job(scenario) == "raysubmit_test"

    assert jobs_client.connection == {"address": execution.address, "verify": execution.verify}
    kwargs = jobs_client.submission
    assert kwargs["runtime_env"] == execution.runtime_env
    assert kwargs["entrypoint_num_cpus"] == 2
    assert kwargs["entrypoint_num_gpus"] == 1
    assert kwargs["entrypoint_memory"] == 4096
    assert kwargs["entrypoint_resources"] == {"pool_gpu_workers": 1}
    command = shlex.split(kwargs["entrypoint"])
    assert command[:-2] == shlex.split(execution.py_executable)
    assert command[-2] == "-c"
    assert "submitter-only" not in command[-1]
    assert "https://jobs.example.test" not in command[-1]
    assert lower_model(scenario) == before


@pytest.mark.parametrize("exit_code", [0, 7])
def test_remote_bootstrap_uses_local_cli_and_cleans_config(jobs_client, monkeypatch, exit_code):
    from nexuml.cli import main

    scenario = isaac_cartpole_learning_ppo(total_frames=128, frames_per_batch=64, num_envs=2)
    scenario.name = "quoted ' \" ; $(touch should-not-exist)\nscenario"
    scenario.exports = [ExportSpec(output="/mnt/data/run/policy")]
    scenario.execution = RayJobExecutionSpec()
    expected = lower_model(scenario)
    expected["execution"] = {"kind": "local"}
    paths = []

    def remote_cli(args):
        assert args[:2] == ["train", "--config"]
        path = Path(args[2])
        paths.append(path)
        restored = ResolvedConfig.load(path)
        assert lower_model(restored) == expected
        raise SystemExit(exit_code)

    monkeypatch.setattr(main, "app", remote_cli)
    submit_ray_job(scenario)
    code = shlex.split(jobs_client.submission["entrypoint"])[-1]
    with pytest.raises(SystemExit) as error:
        exec(compile(code, "<ray-job-bootstrap>", "exec"), {})
    assert error.value.code == exit_code
    assert paths and not paths[0].parent.exists()


def test_job_submission_uses_sdk_environment_defaults(jobs_client):
    submit_ray_job(ScenarioSpec(name="defaults", execution=RayJobExecutionSpec()))
    assert jobs_client.connection == {"address": None, "verify": True}
    assert jobs_client.submission["runtime_env"] == {"working_dir": "."}


def test_job_submission_failure_propagates_without_retry(jobs_client):
    jobs_client.error = RuntimeError("upload rejected")
    with pytest.raises(RuntimeError, match="upload rejected"):
        submit_ray_job(ScenarioSpec(name="failure", execution=RayJobExecutionSpec()))


@pytest.mark.parametrize("command", [" ", "python '"])
def test_bad_python_command_fails_before_connecting(jobs_client, command):
    scenario = ScenarioSpec(
        name="bad-command", execution=RayJobExecutionSpec(py_executable=command)
    )
    with pytest.raises(ValueError):
        submit_ray_job(scenario)
    assert jobs_client.connection is None


def test_submission_revalidates_mutated_execution_before_upload(jobs_client):
    scenario = ScenarioSpec(name="invalid-override", execution=RayJobExecutionSpec())
    scenario.execution.entrypoint_num_gpus = -1
    with pytest.raises(ValueError):
        submit_ray_job(scenario)
    assert jobs_client.connection is None


def test_jobs_dependency_is_lazy_and_missing_extra_is_actionable(monkeypatch):
    original_import = builtins.__import__

    def without_jobs(name, *args, **kwargs):
        if name == "ray.job_submission":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_jobs)
    with pytest.raises(RuntimeError, match=r"nexuml\[ray\]"):
        submit_ray_job(ScenarioSpec(name="missing-extra", execution=RayJobExecutionSpec()))


def test_submission_rejects_another_execution_kind():
    with pytest.raises(ValueError, match="ray_job"):
        submit_ray_job(ScenarioSpec(name="local"))


def test_generated_entrypoint_trains_and_exports_in_fresh_process(jobs_client, tmp_path):
    from nexuml_library.scenarios.asd.synthetic_linear_ae import synthetic_linear_ae_reconstruction

    scenario = synthetic_linear_ae_reconstruction(
        feature_shape=(4,),
        num_samples=32,
        hidden_dims=[4],
        latent_dim=2,
        batch_size=8,
        max_epochs=1,
    )
    scenario.training.accelerator = "cpu"
    scenario.training.devices = 1
    scenario.evaluation.algorithms = []
    output = tmp_path / "package"
    scenario.exports = [ExportSpec(output=str(output))]
    scenario.execution = RayJobExecutionSpec(py_executable=shlex.quote(sys.executable))
    submit_ray_job(scenario)

    result = subprocess.run(
        shlex.split(jobs_client.submission["entrypoint"]),
        cwd=tmp_path,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NEXUML_LOGS_ROOT": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Training complete!" in result.stdout
    assert "Ray job submitted" not in result.stdout
    restored = ResolvedConfig.load(output / "resolved_config.yaml")
    assert restored.execution.kind == "local"
    assert restored.training.max_epochs == 1
