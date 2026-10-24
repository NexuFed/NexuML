from __future__ import annotations

import builtins
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from nexuml.core.config import ResolvedConfig
from nexuml.core.serialization import lower_component, lower_model, restore_component
from nexuml.core.types import ExportSpec, ScenarioSpec, TrainingSpec
from nexuml.execution import submit_ray_job
from nexuml.execution.definitions import (
    LocalExecution,
    RayClusterExecution,
    RayJobExecution,
    RayJobsSDKExecution,
)
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_learning_ppo


@pytest.fixture
def jobs_client(monkeypatch):
    calls = SimpleNamespace(connection=None, submission=None, error=None, count=0)

    class Client:
        def __init__(self, address, **kwargs):
            calls.connection = {"address": address, **kwargs}

        def submit_job(self, **kwargs):
            calls.count += 1
            calls.submission = kwargs
            if calls.error:
                raise calls.error
            return kwargs.get("submission_id", "raysubmit_test")

    monkeypatch.setitem(
        sys.modules, "ray.job_submission", SimpleNamespace(JobSubmissionClient=Client)
    )
    return calls


def test_ray_job_config_roundtrip_allows_rl_without_enabling_ray_train():
    scenario = isaac_cartpole_learning_ppo()
    scenario.execution = RayJobsSDKExecution(
        address="https://jobs.example.test",
        verify="/certs/ca.pem",
        setup_command=["uv", "sync", "--locked", "--extra", "isaac", "--extra", "ray"],
        py_executable=".venv/bin/python",
        runtime_env={"working_dir": ".", "env_vars": {"UV_CACHE_DIR": "/mnt/data/uv-cache"}},
        entrypoint_num_gpus=1,
        entrypoint_resources={"pool_gpu_workers": 1},
    )

    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())

    assert isinstance(restored.execution, RayJobsSDKExecution)
    assert lower_model(restored) == lower_model(scenario)
    with pytest.raises(ValueError, match="Ray learner placement"):
        ScenarioSpec.model_validate(
            {
                **{key: getattr(scenario, key) for key in ScenarioSpec.model_fields},
                "execution": RayClusterExecution(),
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
        {"setup_command": []},
        {"setup_command": [""]},
        {"setup_command": [" "]},
        {"setup_command": ["python", "bad\x00argument"]},
        {"setup_command": "uv sync --locked"},
        {"runtime_env": {"env_vars": []}},
        {"runtime_env": {"env_vars": {"UV_PROJECT_ENVIRONMENT": ""}}},
    ],
)
def test_ray_job_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RayJobsSDKExecution(**kwargs)


def test_submission_preserves_native_sdk_options_and_caller(jobs_client, monkeypatch):
    execution = RayJobsSDKExecution(
        address="https://jobs.example.test",
        verify="/certs/ca.pem",
        setup_command=None,
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
    assert kwargs["runtime_env"] is not execution.runtime_env
    assert kwargs["runtime_env"]["excludes"] is not execution.runtime_env["excludes"]
    assert kwargs["entrypoint_resources"] is not execution.entrypoint_resources
    assert kwargs["entrypoint_num_cpus"] == 2
    assert kwargs["entrypoint_num_gpus"] == 1
    assert kwargs["entrypoint_memory"] == 4096
    assert kwargs["entrypoint_resources"] == {"pool_gpu_workers": 1}
    assert "submission_id" not in kwargs
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
    scenario.execution = RayJobsSDKExecution()
    expected = lower_model(scenario)
    expected["execution"] = {"type": "local", "version": "1", "params": {}}
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
    submit_ray_job(ScenarioSpec(name="defaults", execution=RayJobsSDKExecution()))
    assert jobs_client.connection == {"address": None, "verify": True}
    assert jobs_client.submission["entrypoint_num_gpus"] == 1
    assert jobs_client.submission["entrypoint_resources"] == {}
    assert "submission_id" not in jobs_client.submission
    env = jobs_client.submission["runtime_env"]["env_vars"]
    assert env["UV_CACHE_DIR"] == "/tmp/ray/uv-cache"
    assert "{run_id}" not in env["UV_PROJECT_ENVIRONMENT"]
    assert jobs_client.submission["entrypoint"].startswith("uv sync --locked && exec ")


def test_reusing_default_spec_creates_a_private_environment_for_each_job(jobs_client, monkeypatch):
    run_ids = iter(["a" * 32, "b" * 32])
    monkeypatch.setattr(
        "nexuml.execution.ray_job.uuid4", lambda: SimpleNamespace(hex=next(run_ids))
    )
    scenario = ScenarioSpec(
        name="defaults",
        execution=RayJobsSDKExecution(
            runtime_env={"working_dir": ".", "env_vars": {"ATTEMPT": "{run_id}"}}
        ),
    )
    before = lower_model(scenario)
    paths = []
    for _ in range(2):
        submit_ray_job(scenario)
        submission = jobs_client.submission
        env = submission["runtime_env"]["env_vars"]
        paths.append(env["UV_PROJECT_ENVIRONMENT"])
        assert Path(paths[-1]).parent == Path("/tmp/ray/nexuml-envs")
        assert len(Path(paths[-1]).name) == 32
        assert env["ATTEMPT"] == Path(paths[-1]).name
        command = shlex.split(submission["entrypoint"])
        assert command[command.index("exec") + 1] == paths[-1] + "/bin/python"
        assert env["UV_LINK_MODE"] == "hardlink"
    assert paths[0] != paths[1]
    assert lower_model(scenario) == before


def test_prepared_environment_opt_out_reaches_sdk_without_uv_setup(jobs_client):
    scenario = ScenarioSpec(name="prepared", execution=RayJobsSDKExecution(setup_command=None))
    submit_ray_job(scenario)
    assert jobs_client.submission["runtime_env"] == {"working_dir": "."}
    assert shlex.split(jobs_client.submission["entrypoint"])[:2] == ["python", "-c"]


def test_job_submission_failure_propagates_without_retry(jobs_client):
    jobs_client.error = RuntimeError("upload rejected")
    with pytest.raises(RuntimeError, match="upload rejected"):
        submit_ray_job(ScenarioSpec(name="failure", execution=RayJobsSDKExecution()))


@pytest.mark.parametrize("command", [" ", "python '"])
def test_bad_python_command_fails_before_connecting(jobs_client, command):
    scenario = ScenarioSpec(
        name="bad-command", execution=RayJobsSDKExecution(py_executable=command)
    )
    with pytest.raises(ValueError):
        submit_ray_job(scenario)
    assert jobs_client.connection is None


def test_submission_revalidates_unchecked_model_copy_before_upload(jobs_client):
    execution = RayJobsSDKExecution().model_copy(update={"entrypoint_num_gpus": -1})
    scenario = ScenarioSpec(name="invalid-override", execution=execution)
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
        submit_ray_job(ScenarioSpec(name="missing-extra", execution=RayJobsSDKExecution()))


def test_submission_rejects_another_execution_kind():
    with pytest.raises(ValueError, match="ray-jobs-sdk"):
        submit_ray_job(ScenarioSpec(name="local"))


@pytest.mark.parametrize("with_setup", [False, True])
def test_generated_entrypoint_trains_and_exports_in_fresh_process(
    jobs_client, tmp_path, with_setup
):
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
    setup_marker = tmp_path / "setup-finished"
    scenario.execution = RayJobsSDKExecution(
        setup_command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(setup_marker)!r}).touch()",
        ]
        if with_setup
        else None,
        py_executable=shlex.quote(sys.executable),
    )
    submit_ray_job(scenario)

    result = subprocess.run(
        ["/bin/sh", "-c", jobs_client.submission["entrypoint"]],
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
    assert isinstance(restored.execution, LocalExecution)
    assert isinstance(restored.training, TrainingSpec)
    assert restored.training.max_epochs == 1
    assert setup_marker.exists() == with_setup


@pytest.mark.parametrize("setup_exit,python_exit", [(0, 0), (0, 7), (23, 0), (141, 0)])
def test_setup_finishes_before_python_exec_and_preserves_arguments_and_exit_codes(
    jobs_client, tmp_path, setup_exit, python_exit
):
    setup_marker = tmp_path / "setup.json"
    python_marker = tmp_path / "python.json"
    target = tmp_path / "environment with spaces" / "python"
    target_program = (
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        f"Path({str(python_marker)!r}).write_text(json.dumps("
        '{"pid":os.getpid(),"argv":sys.argv[1:]}))\n'
        f"raise SystemExit({python_exit})\n"
    )
    setup_program = (
        "import json,sys\nfrom pathlib import Path\n"
        f"Path({str(setup_marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"if {setup_exit}: raise SystemExit({setup_exit})\n"
        f"target=Path({str(target)!r})\ntarget.parent.mkdir()\n"
        f"target.write_text({target_program!r})\ntarget.chmod(0o755)\n"
    )
    arguments = ["argument with spaces", "quoted 'value'", "", "$(touch INJECTED); echo bad\n"]
    scenario = ScenarioSpec(
        name="setup-order",
        execution=RayJobsSDKExecution(
            setup_command=[sys.executable, "-c", setup_program, *arguments],
            py_executable=shlex.quote(str(target)),
        ),
    )
    before = lower_model(scenario)
    submit_ray_job(scenario)
    assert lower_model(scenario) == before

    with subprocess.Popen(
        ["/bin/sh", "-c", jobs_client.submission["entrypoint"]],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == (setup_exit or python_exit), stdout + stderr
    assert json.loads(setup_marker.read_text()) == arguments
    assert not (tmp_path / "INJECTED").exists()
    assert python_marker.exists() == (setup_exit == 0)
    if setup_exit == 0:
        launched = json.loads(python_marker.read_text())
        assert launched["pid"] == process.pid  # The waiting shell execs Python directly.
        assert launched["argv"][0] == "-c"
        assert "setup-order" in launched["argv"][1]
        assert "setup_command" not in launched["argv"][1]


def test_mutated_setup_command_is_revalidated_before_connecting(jobs_client):
    execution = RayJobsSDKExecution()
    scenario = ScenarioSpec(name="invalid-setup", execution=execution)
    assert execution.setup_command is not None
    execution.setup_command.clear()
    with pytest.raises(ValueError):
        submit_ray_job(scenario)
    assert jobs_client.connection is None


def test_registered_defaults_are_frozen_and_do_not_mutate_constructor_input():
    from pydantic import ValidationError

    supplied = {
        "runtime_env": {
            "working_dir": "relative/project",
            "env_vars": {
                "UV_PROJECT_ENVIRONMENT": "/worker/private venv/{run_id}",
                "UV_CACHE_DIR": "/worker/cache",
                "UV_HTTP_TIMEOUT": "900",
            },
        }
    }
    before = copy.deepcopy(supplied)
    execution = RayJobsSDKExecution.model_validate(supplied)
    assert supplied == before
    assert execution.setup_command == ["uv", "sync", "--locked"]
    assert execution.entrypoint_num_gpus == 1
    assert execution.entrypoint_num_cpus == 0
    assert execution.entrypoint_memory is None
    assert execution.entrypoint_resources == {}
    assert shlex.split(execution.py_executable) == [
        "/worker/private venv/{run_id}/bin/python",
        "-u",
        "-X",
        "faulthandler",
    ]
    assert execution.runtime_env["env_vars"] == {
        **supplied["runtime_env"]["env_vars"],
        "UV_LINK_MODE": "hardlink",
        "UV_CONCURRENT_DOWNLOADS": "4",
    }
    with pytest.raises(ValidationError, match="frozen"):
        execution.py_executable = "different-python"
    assert RayJobsSDKExecution(py_executable="/prepared/python").py_executable == "/prepared/python"
    assert RayJobsSDKExecution(setup_command=None).runtime_env == {"working_dir": "."}
    assert RayJobsSDKExecution(setup_command=None).py_executable == "python"


def test_registered_sdk_roundtrip_and_distinct_kuberay_identity():
    execution = RayJobsSDKExecution()
    lowered = lower_component(execution)
    assert lowered["type"] == "ray-jobs-sdk"
    assert lowered["version"] == "1"
    assert "kind" not in lowered["params"]
    assert restore_component(kind="execution_backend", value=lowered) == execution
    assert lower_component(RayJobExecution())["type"] == "ray-job"
    assert execution.capabilities["resume"] and execution.capabilities["artifacts"]
    assert not any(execution.capabilities[key] for key in ("status", "logs", "cancellation"))
    with pytest.raises(ValueError, match="type/version/params"):
        ResolvedConfig.from_yaml("name: legacy\nexecution: {kind: ray_job}")


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, ["train", "--config", "CONFIG"]),
        (
            {"trainer_checkpoint": Path("/worker/check point;$(literal).ckpt")},
            [
                "train",
                "--config",
                "CONFIG",
                "--trainer-checkpoint",
                "/worker/check point;$(literal).ckpt",
            ],
        ),
        (
            {
                "operation": "evaluate",
                "model_package": "/worker/policy ' ;\n$(literal)",
                "device": "cuda:0",
                "overrides": ("name=quoted 'value';$(literal)",),
            },
            [
                "evaluate",
                "/worker/policy ' ;\n$(literal)",
                "--config",
                "CONFIG",
                "--device",
                "cuda:0",
                "-O",
                "name=quoted 'value';$(literal)",
            ],
        ),
        (
            {"operation": "evaluate", "model_package": "--config"},
            ["evaluate", "--config", "--config", "CONFIG"],
        ),
        (
            {
                "operation": "export",
                "checkpoint": "/worker/a ' ;.ckpt",
                "output": "/worker/out $(literal)",
            },
            [
                "export",
                "--config",
                "CONFIG",
                "--checkpoint",
                "/worker/a ' ;.ckpt",
                "--output",
                "/worker/out $(literal)",
            ],
        ),
    ],
)
def test_lifecycle_bootstrap_preserves_literal_argv_and_resets_execution(
    jobs_client, monkeypatch, kwargs, expected
):
    scenario = ScenarioSpec(
        name="resolved ' ; $(literal)\nscenario",
        execution=RayJobsSDKExecution(
            address="https://private-dashboard.test",
            verify="/private/ca.pem",
            runtime_env={"env_vars": {"TOKEN": "private-submission-token"}},
        ),
    )
    before = lower_model(scenario)
    received = []
    paths = []

    def cli(argv):
        config_index = 3 if argv[0] == "evaluate" else 2
        path = Path(argv[config_index])
        paths.append(path)
        remote = ResolvedConfig.load(path)
        assert isinstance(remote.execution, LocalExecution)
        assert remote.name == scenario.name
        assert lower_model(remote) == {
            **before,
            "execution": {"type": "local", "version": "1", "params": {}},
        }
        received.append([*argv[:config_index], "CONFIG", *argv[config_index + 1 :]])

    monkeypatch.setitem(sys.modules, "nexuml.cli.main", SimpleNamespace(app=cli))
    validations = []
    monkeypatch.setattr(
        "nexuml.execution.ray_job._validate_remote_resume",
        lambda config, checkpoint: validations.append((config, checkpoint)),
    )
    assert submit_ray_job(scenario, **kwargs) == "raysubmit_test"
    assert jobs_client.count == 1
    assert "submission_id" not in jobs_client.submission
    bootstrap = shlex.split(jobs_client.submission["entrypoint"])[-1]
    for secret in ("private-dashboard", "/private/ca.pem", "private-submission-token"):
        assert secret not in bootstrap
    exec(compile(bootstrap, "<mock-worker>", "exec"), {})
    assert received == [expected]
    assert len(validations) == int(kwargs.get("trainer_checkpoint") is not None)
    assert not paths[0].parent.exists()
    assert lower_model(scenario) == before


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"operation": "shell"}, "operation must"),
        ({"model_package": "worker/policy"}, "train does not accept model_package"),
        ({"checkpoint": "worker/model.ckpt"}, "train does not accept checkpoint"),
        ({"output": "worker/output"}, "train does not accept output"),
        ({"device": "cpu"}, "train does not accept device"),
        ({"overrides": ["name=x"]}, "train does not accept overrides"),
        ({"trainer_checkpoint": ""}, "trainer_checkpoint must"),
        ({"operation": "evaluate"}, "evaluate requires model_package"),
        (
            {"operation": "evaluate", "model_package": "worker/policy", "trainer_checkpoint": "x"},
            "evaluate does not accept trainer_checkpoint",
        ),
        (
            {"operation": "evaluate", "model_package": "worker/policy", "checkpoint": "x"},
            "evaluate does not accept checkpoint",
        ),
        (
            {"operation": "evaluate", "model_package": "worker/policy", "output": "x"},
            "evaluate does not accept output",
        ),
        (
            {"operation": "evaluate", "model_package": "worker/policy", "overrides": "name=x"},
            "overrides must",
        ),
        (
            {"operation": "evaluate", "model_package": "worker/policy", "overrides": ["\x00"]},
            "overrides must",
        ),
        ({"operation": "export", "checkpoint": "worker/model.ckpt"}, "explicit output"),
        ({"operation": "export", "output": "worker/output"}, "requires checkpoint"),
        (
            {"operation": "export", "checkpoint": "x", "output": "y", "device": "cpu"},
            "export does not accept device",
        ),
    ],
)
def test_shared_preflight_and_submission_reject_invalid_operations_before_sdk(
    jobs_client, kwargs, message
):
    from nexuml.execution.service import preflight

    scenario = ScenarioSpec(name="preflight", execution=RayJobsSDKExecution())
    with pytest.raises(ValueError, match=message):
        preflight(scenario, **kwargs)
    with pytest.raises(ValueError, match=message):
        submit_ray_job(scenario, **kwargs)
    assert jobs_client.connection is None
    assert jobs_client.count == 0


def test_preflight_and_settings_never_import_runtime_or_open_worker_paths(monkeypatch):
    from nexuml.execution import service

    # Finish registry discovery before blocking unrelated configuration-file reads.
    service.catalog()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ray", "isaaclab", "isaacsim", "mujoco"} or name in {
            "nexuml.training.lightning",
            "nexuml.core.compiler",
            "nexuml.core.export",
        }:
            pytest.fail(f"preflight imported runtime: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("opened worker path"))
    monkeypatch.setattr("torch.cuda._lazy_init", lambda: pytest.fail("initialized CUDA"))
    scenario = ScenarioSpec(name="inert", execution=RayJobsSDKExecution())
    for kwargs in (
        {"trainer_checkpoint": "/not/on/submitter.ckpt"},
        {"operation": "evaluate", "model_package": "/not/on/submitter/policy"},
        {"operation": "export", "checkpoint": "/remote/model.ckpt", "output": "/remote/out"},
        {
            "operation": "acceptance",
            "case": "preflight",
            "attempt_root": "/mnt/data/nexuml-qualification/campaign/attempt",
            "wall_seconds": 120,
        },
    ):
        assert service.preflight(scenario, **kwargs)["argv"]
    assert service.discover(scenario.execution).diagnostics
    assert any(row["type"] == "ray-jobs-sdk" for row in service.catalog())


def test_profile_uses_registered_schema_and_submission_cwd(tmp_path, monkeypatch):
    from nexuml.execution.profiles import load_ray_job_profile

    profile = tmp_path / "profile.yaml"
    profile.write_text(
        json.dumps(
            {
                "type": "ray-jobs-sdk",
                "version": "1",
                "params": {
                    "runtime_env": {"working_dir": "relative/project"},
                    "verify": "relative/ca.pem",
                },
            }
        )
    )
    monkeypatch.chdir(tmp_path.parent)
    execution = load_ray_job_profile(profile)
    assert execution.runtime_env["working_dir"] == "relative/project"
    assert execution.verify == "relative/ca.pem"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"trainer_checkpoint": "/worker/only/checkpoint.ckpt"},
        {"operation": "evaluate", "model_package": "/worker/only/policy"},
        {"operation": "export", "checkpoint": "/worker/model.ckpt", "output": "/worker/policy"},
        {
            "operation": "acceptance",
            "case": "preflight",
            "attempt_root": "/mnt/data/nexuml-qualification/campaign/attempt",
            "wall_seconds": 120,
        },
    ],
)
def test_submission_never_opens_artifacts_or_imports_local_runtime(
    jobs_client, monkeypatch, kwargs
):
    from nexuml.execution import service

    service.catalog()
    scenario = ScenarioSpec(name="submit-only", execution=RayJobsSDKExecution())
    original_import = builtins.__import__

    def guarded_import(name, *args, **kw):
        if name != "ray.job_submission" and (
            name.split(".")[0] in {"ray", "isaaclab", "isaacsim", "mujoco"}
            or name in {"nexuml.training.lightning", "nexuml.core.compiler", "nexuml.core.export"}
        ):
            pytest.fail(f"submission imported local runtime: {name}")
        return original_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kw: pytest.fail("opened worker path"))
    monkeypatch.setattr("torch.cuda._lazy_init", lambda: pytest.fail("initialized CUDA"))
    launch = service.run if kwargs.get("operation") == "acceptance" else submit_ray_job
    assert launch(scenario, **kwargs) == "raysubmit_test"
    assert jobs_client.count == 1


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {"kind": "ray_job"},
        {"type": "local", "version": "1", "params": {}},
        {"type": "ray-job", "version": "1", "params": {}},
        {"type": "ray-jobs-sdk", "version": "1", "params": {}, "kind": "ray_job"},
        {"type": "ray-jobs-sdk", "version": "1", "params": []},
        {"type": "ray-jobs-sdk", "version": "unknown", "params": {}},
    ],
)
def test_profile_rejects_legacy_and_other_backends_with_migration_diagnostic(tmp_path, data):
    from nexuml.execution.profiles import load_ray_job_profile

    profile = tmp_path / "invalid.yaml"
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="type: ray-jobs-sdk.*replace legacy kind"):
        load_ray_job_profile(profile)


@pytest.mark.parametrize(
    "case", ["preflight", "regression", "package", "recipe", "so101", "browser", "recovery"]
)
def test_acceptance_preflight_allowlists_fixed_cases(case):
    from nexuml.execution.service import preflight

    scenario = ScenarioSpec(name="acceptance", execution=RayJobsSDKExecution())
    assert preflight(
        scenario,
        operation="acceptance",
        case=case,
        attempt_root=Path("/mnt/data/nexuml-qualification/campaign/attempt"),
        wall_seconds=120,
    )["argv"] == [
        "acceptance",
        case,
        "/mnt/data/nexuml-qualification/campaign/attempt",
        "120",
        "{}",
    ]


@pytest.mark.parametrize(
    "invalid, message",
    [
        ({"case": None}, "fixed case"),
        ({"case": "shell"}, "fixed case"),
        ({"case": "recipe; echo injected"}, "fixed case"),
        ({"case": ["preflight"]}, "fixed case"),
        ({"attempt_root": None}, "attempt_root"),
        ({"attempt_root": ""}, "attempt_root"),
        ({"attempt_root": "relative/attempt"}, "attempt_root"),
        ({"attempt_root": "/worker/attempt"}, "attempt_root"),
        ({"attempt_root": "/mnt/data/nexuml-qualification"}, "attempt_root"),
        ({"attempt_root": "/mnt/data/nexuml-qualification/.."}, "attempt_root"),
        ({"attempt_root": "/mnt/data/nexuml-qualification/attempt/.."}, "attempt_root"),
        ({"attempt_root": "/mnt/data/nexuml-qualification-other/attempt"}, "attempt_root"),
        ({"attempt_root": "/worker/../other"}, "attempt_root"),
        ({"attempt_root": "/worker/attempt/.."}, "attempt_root"),
        ({"attempt_root": "/worker/bad\x00path"}, "attempt_root"),
        ({"wall_seconds": None}, "wall_seconds"),
        ({"wall_seconds": 0}, "wall_seconds"),
        ({"wall_seconds": -1}, "wall_seconds"),
        ({"wall_seconds": True}, "wall_seconds"),
        ({"wall_seconds": "120"}, "wall_seconds"),
        ({"wall_seconds": float("nan")}, "wall_seconds"),
        ({"wall_seconds": float("inf")}, "wall_seconds"),
        ({"wall_seconds": float("-inf")}, "wall_seconds"),
        ({"case_options": []}, "case_options"),
        ({"case_options": SimpleNamespace(value=1)}, "case_options"),
        ({"case_options": {1: "not a string key"}}, "case_options"),
        ({"case_options": {"nested": {1: "not a string key"}}}, "case_options"),
        ({"case_options": {"nested": ("not", "a list")}}, "case_options"),
        ({"case_options": {"path": Path("/worker/path")}}, "case_options"),
        ({"case_options": {"nested": [float("nan")]}}, "finite JSON"),
        ({"case_options": {"nested": {"value": float("inf")}}}, "finite JSON"),
        ({"trainer_checkpoint": "/worker/checkpoint"}, "does not accept trainer_checkpoint"),
        ({"model_package": "/worker/policy"}, "does not accept model_package"),
        ({"checkpoint": "/worker/checkpoint"}, "does not accept checkpoint"),
        ({"output": "/worker/output"}, "does not accept output"),
        ({"device": "cpu"}, "does not accept device"),
        ({"overrides": ["name=value"]}, "does not accept overrides"),
        ({"export_kind": "onnx"}, "train_package only"),
        ({"command": "echo injected"}, "only fixed case inputs"),
        ({"script": "/worker/script.py"}, "only fixed case inputs"),
        ({"module": "custom.module"}, "only fixed case inputs"),
        ({"fixture_path": "/worker/script.py"}, "only fixed case inputs"),
        ({"submission_id": "foreign-id"}, "submission_id"),
        ({"submission_id": 42}, "submission_id"),
        ({"submission_id": "a" * 32}, "submission_id"),
    ],
)
def test_acceptance_rejects_invalid_inputs_before_sdk(jobs_client, invalid, message):
    from nexuml.execution import service

    scenario = ScenarioSpec(name="acceptance", execution=RayJobsSDKExecution())
    kwargs = {
        "operation": "acceptance",
        "case": "preflight",
        "attempt_root": "/mnt/data/nexuml-qualification/campaign/attempt",
        "wall_seconds": 120,
        **invalid,
    }
    with pytest.raises(ValueError, match=message):
        service.preflight(scenario, **kwargs)
    with pytest.raises(ValueError, match=message):
        service.run(scenario, **kwargs)
    assert jobs_client.connection is None
    assert jobs_client.count == 0


def test_acceptance_rejects_circular_options_before_sdk(jobs_client):
    from nexuml.execution import service

    options = {}
    options["recursive"] = options
    scenario = ScenarioSpec(name="acceptance", execution=RayJobsSDKExecution())
    with pytest.raises(ValueError, match="case_options"):
        service.run(
            scenario,
            operation="acceptance",
            case="preflight",
            attempt_root="/mnt/data/nexuml-qualification/campaign/attempt",
            wall_seconds=120,
            case_options=options,
        )
    assert jobs_client.connection is None


@pytest.mark.parametrize(
    "operation, inputs",
    [
        ("train", {}),
        ("evaluate", {"model_package": "/worker/policy"}),
        ("export", {"checkpoint": "/worker/checkpoint", "output": "/worker/output"}),
    ],
)
@pytest.mark.parametrize(
    "field, value",
    [
        ("case", "recipe"),
        ("attempt_root", "/worker/attempt"),
        ("wall_seconds", 120),
        ("case_options", {}),
        ("submission_id", uuid4().hex),
    ],
)
def test_normal_lifecycle_rejects_acceptance_arguments(
    jobs_client, operation, inputs, field, value
):
    from nexuml.execution import service

    scenario = ScenarioSpec(name="normal", execution=RayJobsSDKExecution())
    with pytest.raises(ValueError, match=f"{operation} does not accept {field}"):
        service.run(scenario, operation=operation, **inputs, **{field: value})
    assert jobs_client.connection is None


def test_public_submission_convenience_does_not_expose_acceptance(jobs_client):
    scenario = ScenarioSpec(name="public", execution=RayJobsSDKExecution())
    with pytest.raises(ValueError, match="operation must be train, evaluate or export"):
        submit_ray_job(scenario, operation="acceptance")
    assert jobs_client.connection is None


def test_private_acceptance_passes_owned_uuid_without_reusing_environment_uuid(jobs_client):
    from nexuml.execution import service

    scenario = ScenarioSpec(name="private", execution=RayJobsSDKExecution())
    environments = []
    for _ in range(2):
        identity = uuid4().hex
        assert (
            service.run(
                scenario,
                operation="acceptance",
                case="preflight",
                attempt_root="/mnt/data/nexuml-qualification/campaign/attempt",
                wall_seconds=120,
                submission_id=identity,
            )
            == identity
        )
        assert jobs_client.submission["submission_id"] == identity
        environment = jobs_client.submission["runtime_env"]["env_vars"]["UV_PROJECT_ENVIRONMENT"]
        assert Path(environment).name != identity
        environments.append(environment)
    assert environments[0] != environments[1]


@pytest.mark.parametrize("with_setup", [False, True])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_acceptance_uses_shared_guarded_transport_and_fixed_fixture(
    jobs_client, monkeypatch, with_setup, exit_code
):
    from nexuml.execution import service

    monkeypatch.setattr("nexuml.execution.ray_job.uuid4", lambda: SimpleNamespace(hex="a" * 32))
    setup = ["uv", "sync", "--locked", "--extra", "isaac", "literal ' ; $(setup)"]
    execution = RayJobsSDKExecution(
        address="https://private-dashboard.test",
        verify="/private/ca.pem",
        setup_command=setup if with_setup else None,
        runtime_env={"working_dir": ".", "env_vars": {"TOKEN": "submitter-only"}},
        entrypoint_num_cpus=2,
        entrypoint_num_gpus=1,
        entrypoint_memory=4096,
        entrypoint_resources={"pool_gpu_workers": 1},
    )
    scenario = ScenarioSpec(name="resolved ' ; $(literal)\nscenario", execution=execution)
    before = lower_model(scenario)
    attempt_root = "/mnt/data/nexuml-qualification/attempt ' ; $(touch INJECTED)\npath"
    options = {"z": [True, None, 3.5, {"value": "' ; $(literal)\n"}], "a": "recipe"}
    config_paths = []
    fixture_calls = []

    def run_case(case, root, config, case_options):
        fixture_calls.append((case, root, case_options))
        config_paths.append(config)
        assert lower_model(ResolvedConfig.load(config)) == {
            **before,
            "execution": {"type": "local", "version": "1", "params": {}},
        }
        return exit_code

    def run_path(path):
        assert path == "tests/fixtures/physical_ai_acceptance.py"
        return {"run_case": run_case}

    monkeypatch.setattr("runpy.run_path", run_path)
    assert (
        service.run(
            scenario,
            operation="acceptance",
            case="recipe",
            attempt_root=attempt_root,
            wall_seconds=120.5,
            case_options=options,
        )
        == "raysubmit_test"
    )
    assert jobs_client.count == 1
    assert jobs_client.connection == {"address": execution.address, "verify": execution.verify}
    submission = jobs_client.submission
    guard = shlex.split(submission["entrypoint"])
    assert guard[:-1] == [
        "python",
        "tests/fixtures/physical_ai_guard.py",
        "--attempt-root",
        attempt_root,
        "--receipt",
        attempt_root + "/guard.json",
        "--deadline-seconds",
        "120.5",
        "--disk-floor-bytes",
        str(4 * 1024**3),
        "--disk-path",
        "/tmp/ray",
        "--",
        "sh",
        "-c",
    ]
    inner = guard[-1]
    if with_setup:
        prefix = shlex.join(setup) + " && exec "
        assert inner.startswith(prefix)
        inner = inner.removeprefix(prefix)
    python = shlex.split(inner)
    assert python[:-2] == shlex.split(execution.py_executable.replace("{run_id}", "a" * 32))
    assert python[-2] == "-c"
    bootstrap = python[-1]
    for secret in ("private-dashboard", "/private/ca.pem", "submitter-only", "setup_command"):
        assert secret not in bootstrap
    assert "nexuml.cli.main" not in bootstrap
    assert "uv sync" not in bootstrap
    with pytest.raises(SystemExit) as error:
        exec(compile(bootstrap, "<mock-acceptance-worker>", "exec"), {})
    assert error.value.code == exit_code
    assert fixture_calls == [("recipe", Path(attempt_root), options)]
    assert config_paths and not config_paths[0].parent.exists()
    assert submission["entrypoint_num_cpus"] == 2
    assert submission["entrypoint_num_gpus"] == 1
    assert submission["entrypoint_memory"] == 4096
    assert submission["entrypoint_resources"] == execution.entrypoint_resources
    assert submission["entrypoint_resources"] is not execution.entrypoint_resources
    env = submission["runtime_env"]["env_vars"]
    assert env["TOKEN"] == "submitter-only"
    if with_setup:
        assert env["UV_PROJECT_ENVIRONMENT"] == "/tmp/ray/nexuml-envs/" + "a" * 32
        assert env["UV_CACHE_DIR"] == "/tmp/ray/uv-cache"
        assert env["UV_LINK_MODE"] == "hardlink"
    assert lower_model(scenario) == before
