from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nexuml.cli import main
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import RayJobExecutionSpec, ScenarioSpec
from nexuml.execution import load_ray_job_profile, submit_ray_job

PROFILE = Path("configs/execution/ray.yaml")
runner = CliRunner()


def test_project_profile_supplies_locked_setup_and_private_local_environments():
    first = load_ray_job_profile(PROFILE)
    second = load_ray_job_profile(PROFILE)

    assert first.setup_command == [
        "uv",
        "sync",
        "--locked",
        "--python",
        "3.12",
        "--extra",
        "library",
        "--extra",
        "ray",
    ]
    assert first.entrypoint_num_gpus == 1
    assert first.entrypoint_resources == {"pool_gpu_workers": 1}
    assert first.address is None and first.verify is True
    for profile in (first, second):
        env = profile.runtime_env["env_vars"]
        venv = Path(env["UV_PROJECT_ENVIRONMENT"])
        assert venv.parent == Path("/tmp/ray/nexuml-envs")
        assert venv.name == "{run_id}"
        assert shlex.split(profile.py_executable) == [
            str(venv / "bin/python"),
            "-u",
            "-X",
            "faulthandler",
        ]
        assert env["UV_CACHE_DIR"] == "/tmp/ray/uv-cache"
        assert env["UV_LINK_MODE"] == "hardlink"
        assert env["UV_HTTP_TIMEOUT"] == "600"
        assert env["UV_CONCURRENT_DOWNLOADS"] == "4"
    assert first == second
    assert first.runtime_env["working_dir"] == "."
    assert "UV_CACHE_DIR" not in PROFILE.read_text()  # Settings come from the Python class.


@pytest.mark.parametrize(
    "content",
    [
        "",
        "[]",
        "kind: local",
        "kind: ray",
        "kind: ray_job\nunknown: true",
        "kind: ray_job\nentrypoint_num_gpus: -1",
        "kind: ray_job\nsetup_command: []",
        "kind: ray_job\nkind: ray_job",  # Duplicate YAML keys must not silently win.
        "kind: ray_job\nsetup_command: [",
        "!!python/object/apply:builtins.print [must-not-execute]",
    ],
)
def test_invalid_profile_fails_before_submission(tmp_path, monkeypatch, content):
    import nexuml.execution as execution

    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: ScenarioSpec(name="test"))
    monkeypatch.setattr(
        execution, "submit_ray_job", lambda _: pytest.fail("invalid profile submitted")
    )
    result = runner.invoke(main.app, ["train", "test", "--execution-profile", str(path)])
    assert result.exit_code == 1
    assert "Execution profile error" in result.output
    assert "Ray job submitted" not in result.output


def test_missing_profile_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: ScenarioSpec(name="test"))
    result = runner.invoke(
        main.app, ["train", "test", "--execution-profile", str(tmp_path / "missing.yaml")]
    )
    assert result.exit_code == 1
    assert "Execution profile error" in result.output


@pytest.mark.parametrize("source", ["name", "config", "file"])
def test_cli_profile_replaces_execution_then_applies_overrides(tmp_path, monkeypatch, source):
    import nexuml.execution as execution
    import nexuml.core.compiler as compiler
    import nexuml.training.lightning as lightning

    scenario = ScenarioSpec(name="profile-test")
    args = ["train"]
    if source == "name":
        monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: scenario)
        args += [scenario.name]
    elif source == "config":
        path = tmp_path / "scenario.yaml"
        ResolvedConfig.from_scenario(scenario).save(path)
        args += ["--config", str(path)]
    else:
        path = tmp_path / "scenario.py"
        path.write_text(
            "from nexuml.core.types import ScenarioSpec\n"
            "def scenario():\n    return ScenarioSpec(name='profile-test')\n"
        )
        args += ["--scenario-file", str(path), "--artifact-dir", str(tmp_path / "receipt")]
    submitted = []
    monkeypatch.setattr(
        execution, "submit_ray_job", lambda spec: submitted.append(spec) or "ray_profile"
    )
    monkeypatch.setattr(compiler, "compile", lambda *_: pytest.fail("compiled locally"))
    monkeypatch.setattr(lightning, "NexuSession", lambda **_: pytest.fail("trained locally"))
    setup = ["uv", "sync", "--locked", "--extra", "isaac", "--extra", "ray"]
    result = runner.invoke(
        main.app,
        [
            *args,
            "--execution-profile",
            str(PROFILE),
            "--max-epochs",
            "5",
            "-O",
            "training.max_epochs=2",
            "-O",
            f"execution.setup_command={json.dumps(setup)}",
            "-O",
            "execution.verify=/certs/private-ca.pem",
            "-O",
            'execution.entrypoint_resources={"pool_dgx_workers":1}',
            "-O",
            "execution.runtime_env.env_vars.NEXUML_LOGS_ROOT=/mnt/data/unique-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(submitted) == 1
    remote = submitted[0]
    assert isinstance(remote.execution, RayJobExecutionSpec)
    assert remote.training.max_epochs == 2
    assert remote.execution.setup_command == setup
    assert remote.execution.verify == "/certs/private-ca.pem"
    assert remote.execution.entrypoint_resources == {"pool_dgx_workers": 1}
    assert remote.execution.runtime_env["env_vars"]["UV_CACHE_DIR"] == "/tmp/ray/uv-cache"
    assert remote.execution.runtime_env["env_vars"]["NEXUML_LOGS_ROOT"] == "/mnt/data/unique-run"
    assert "Ray job submitted: ray_profile" in result.output
    assert "Training complete" not in result.output
    if source == "file":
        saved = ResolvedConfig.load(tmp_path / "receipt/resolved_scenario.yaml")
        assert saved.execution == remote.execution


def test_profile_requires_explicit_scenario_before_checkpoint_loading():
    result = runner.invoke(
        main.app,
        ["train", "--trainer-checkpoint", "worker-only.ckpt", "--execution-profile", str(PROFILE)],
    )
    assert result.exit_code == 1
    assert "requires a scenario/config/scenario-file" in result.output


def test_profile_still_rejects_submitter_checkpoint(monkeypatch):
    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: ScenarioSpec(name="test"))
    result = runner.invoke(
        main.app,
        [
            "train",
            "test",
            "--execution-profile",
            str(PROFILE),
            "--trainer-checkpoint",
            "local.ckpt",
        ],
    )
    assert result.exit_code == 1
    assert "does not transport --trainer-checkpoint" in result.output


def test_profile_cpu_overrides_clear_gpu_and_pool(monkeypatch):
    import nexuml.execution as execution

    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: ScenarioSpec(name="test"))
    submitted = []
    monkeypatch.setattr(
        execution, "submit_ray_job", lambda spec: submitted.append(spec) or "cpu_job"
    )
    result = runner.invoke(
        main.app,
        [
            "train",
            "test",
            "--execution-profile",
            str(PROFILE),
            "-O",
            "execution.entrypoint_num_gpus=0",
            "-O",
            "execution.entrypoint_resources={}",
            "-O",
            "training.accelerator=cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert submitted[0].execution.entrypoint_num_gpus == 0
    assert submitted[0].execution.entrypoint_resources == {}
    assert submitted[0].training.accelerator == "cpu"
    assert submitted[0].execution.setup_command is not None


def test_profile_submission_failure_does_not_fall_back(monkeypatch):
    import nexuml.execution as execution
    import nexuml.training.lightning as lightning

    monkeypatch.setattr(main, "_get_scenario_fn", lambda _: lambda: ScenarioSpec(name="test"))

    def fail(_):
        raise RuntimeError("profile upload rejected")

    monkeypatch.setattr(execution, "submit_ray_job", fail)
    monkeypatch.setattr(lightning, "NexuSession", lambda **_: pytest.fail("fell back locally"))
    result = runner.invoke(main.app, ["train", "test", "--execution-profile", str(PROFILE)])
    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "Ray job submitted" not in result.output


def test_profile_reaches_sdk_without_reentering_profile_selection(monkeypatch):
    sdk = pytest.importorskip("ray.job_submission")
    calls = []

    class Client:
        def __init__(self, address, **kwargs):
            assert address is None and kwargs == {"verify": True}

        def submit_job(self, **kwargs):
            calls.append(kwargs)
            return "ray_profile"

    monkeypatch.setattr(sdk, "JobSubmissionClient", Client)

    def assert_remote_config(args):
        assert args[:2] == ["train", "--config"]
        remote = ResolvedConfig.load(Path(args[2]))
        assert remote.execution.kind == "local"

    monkeypatch.setattr(main, "app", assert_remote_config)
    scenario = ScenarioSpec(name="test", execution=load_ray_job_profile(PROFILE))
    for _ in range(2):
        assert submit_ray_job(scenario) == "ray_profile"
        submitted = calls[-1]
        env = submitted["runtime_env"]["env_vars"]
        assert "{run_id}" not in env["UV_PROJECT_ENVIRONMENT"]
        assert env["UV_PROJECT_ENVIRONMENT"] + "/bin/python" in submitted["entrypoint"]
        assert "{run_id}" in scenario.execution.py_executable
        assert submitted["entrypoint_num_gpus"] == 1
        assert submitted["entrypoint_resources"] == {"pool_gpu_workers": 1}
        assert submitted["entrypoint"].startswith("uv sync --locked ")
        assert " && exec /tmp/ray/nexuml-envs/" in submitted["entrypoint"]
        bootstrap = shlex.split(submitted["entrypoint"])[-1]
        exec(compile(bootstrap, "<profile-bootstrap>", "exec"), {})
    assert (
        calls[0]["runtime_env"]["env_vars"]["UV_PROJECT_ENVIRONMENT"]
        != (calls[1]["runtime_env"]["env_vars"]["UV_PROJECT_ENVIRONMENT"])
    )


def test_ray_job_defaults_work_without_a_profile_and_local_default_stays_local():
    execution = RayJobExecutionSpec()
    assert execution.setup_command == ["uv", "sync", "--locked"]
    assert shlex.split(execution.py_executable)[0] == "/tmp/ray/nexuml-envs/{run_id}/bin/python"
    assert execution.entrypoint_num_gpus == 1
    assert execution.entrypoint_resources == {}
    assert execution.runtime_env["env_vars"]["UV_CACHE_DIR"] == "/tmp/ray/uv-cache"
    assert ScenarioSpec(name="local-default").execution.kind == "local"


def test_prepared_environment_opt_out_has_no_implicit_setup_or_uv_settings():
    execution = RayJobExecutionSpec(setup_command=None, entrypoint_num_gpus=0)
    assert execution.setup_command is None
    assert execution.py_executable == "python"
    assert execution.runtime_env == {"working_dir": "."}


def test_class_defaults_preserve_explicit_environment_and_interpreter():
    execution = RayJobExecutionSpec(
        runtime_env={
            "working_dir": ".",
            "env_vars": {
                "UV_CACHE_DIR": "/local/custom-cache",
                "UV_HTTP_TIMEOUT": "900",
                "UV_PROJECT_ENVIRONMENT": "/local/private venv",
                "TOKEN": "secret",
            },
        },
    )
    assert shlex.split(execution.py_executable)[0] == "/local/private venv/bin/python"
    assert execution.runtime_env["env_vars"]["UV_HTTP_TIMEOUT"] == "900"
    assert execution.runtime_env["env_vars"]["UV_CACHE_DIR"] == "/local/custom-cache"
    assert execution.runtime_env["env_vars"]["UV_LINK_MODE"] == "hardlink"
    assert execution.runtime_env["env_vars"]["TOKEN"] == "secret"
    explicit = RayJobExecutionSpec(py_executable="/prepared/python")
    assert explicit.py_executable == "/prepared/python"


def test_class_defaults_roundtrip_without_allocating_a_shared_environment():
    original = ScenarioSpec(name="roundtrip", execution=RayJobExecutionSpec())
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(original).to_yaml())
    assert restored.execution == original.execution
    assert "{run_id}" in restored.execution.runtime_env["env_vars"]["UV_PROJECT_ENVIRONMENT"]


def test_documented_cartpole_submission_keeps_setup_and_outputs_explicit(monkeypatch):
    import nexuml.execution as execution
    import nexuml.core.compiler as compiler
    import nexuml.training.lightning as lightning

    guide = Path("docs/how-to/isaac-cartpole.md").read_text()
    remote = guide.split("## Run remotely with Ray Jobs\n", 1)[1].split(
        "## Evaluate the exported policy\n", 1
    )[0]
    blocks = re.findall(r"```bash\n(.*?)```", remote, re.S)
    for block in blocks:
        subprocess.run(["bash", "-n"], input=block, text=True, check=True)
    block = next(block for block in blocks if "nexuml train" in block)
    assert block.index("RUN_ROOT=") < block.index("nexuml train")
    command = block[block.index("nexuml train") :]
    command = command.replace("$RAY_CA_BUNDLE", "/certs/ca.pem").replace(
        "$RUN_ROOT", "/mnt/data/doc-test"
    )
    submitted = []
    monkeypatch.setattr(
        execution, "submit_ray_job", lambda spec: submitted.append(spec) or "doc_job"
    )
    monkeypatch.setattr(compiler, "compile", lambda *_: pytest.fail("compiled locally"))
    monkeypatch.setattr(lightning, "NexuSession", lambda **_: pytest.fail("trained locally"))
    result = runner.invoke(main.app, shlex.split(command.replace("\\\n", ""))[1:])
    assert result.exit_code == 0, result.output
    scenario = ResolvedConfig.from_scenario(submitted[0])
    assert scenario.execution.setup_command == [
        "uv",
        "sync",
        "--locked",
        "--python",
        "3.12",
        "--extra",
        "isaac",
        "--extra",
        "ray",
    ]
    assert scenario.execution.verify == "/certs/ca.pem"
    assert scenario.execution.entrypoint_num_gpus == 1
    assert scenario.execution.entrypoint_resources == {"pool_gpu_workers": 1}
    assert scenario.execution.runtime_env["env_vars"]["NEXUML_LOGS_ROOT"] == "/mnt/data/doc-test"
    assert scenario.exports[0].output == "/mnt/data/doc-test/policy"
    assert scenario.interaction.environment.viewer.port == 8081
