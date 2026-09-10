from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nexuml import strategy
from nexuml.core.types import (
    EvalAlgorithmSpec,
    EvaluationSpec,
    LayerSpec,
    PipelineSpec,
    RayClusterTarget,
    RayExecutionSpec,
    ScenarioSpec,
)
from nexuml.core.serialization import lower_model
from nexuml.execution import ray as ray_execution
from nexuml_library.evaluation.visualizers.class_histogram import ClassHistogramVisualizer
from nexuml_library.layers.head.decision_rule import DecisionRulePipelineLayer


def test_ray_execution_config_is_placement_only():
    scenario = ScenarioSpec.model_validate(
        {
            "name": "distributed",
            "training": {"strategy": "fsdp"},
            "execution": {
                "kind": "ray",
                "target": {"kind": "cluster", "address": "ray://cluster:10001"},
                "workers": [2, 4],
                "resources_per_worker": {"CPU": 4, "GPU": 1},
                "storage_path": "s3://runs/nexuml",
            },
        }
    )

    assert isinstance(scenario.execution, RayExecutionSpec)
    assert scenario.execution.workers == (2, 4)
    assert scenario.training.strategy == "fsdp"
    assert not hasattr(scenario.execution, "strategy")


def test_ray_target_runtime_env_round_trips_through_scenario_serialization():
    scenario = ScenarioSpec.model_validate(
        {
            "name": "runtime-env",
            "execution": {
                "kind": "ray",
                "target": {
                    "kind": "cluster",
                    "working_dir": "/checkout",
                    "runtime_env": {
                        "excludes": ["data", ".venv"],
                        "uv": {
                            "packages": ["-e ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}"],
                        },
                    },
                },
            },
        }
    )

    assert isinstance(scenario.execution, RayExecutionSpec)
    round_tripped = RayExecutionSpec.model_validate(scenario.execution.model_dump(mode="json"))

    assert round_tripped.target.runtime_env == {
        "excludes": ["data", ".venv"],
        "uv": {"packages": ["-e ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}"]},
    }


def test_ray_worker_reuses_nexusession_and_reports_final_metrics(monkeypatch):
    ray_train = pytest.importorskip("ray.train")
    calls: list[str] = []
    reported = {}

    class FakeSession:
        @classmethod
        def from_scenario(cls, scenario, **kwargs):
            calls.append(f"session:{scenario.name}:{kwargs['enable_loggers']}")
            return cls()

        def run(self):
            calls.append("run")
            return SimpleNamespace(
                validation_results=[{"val/loss": 0.4}],
                test_results=[{"test/accuracy": 0.9}],
                eval_algorithm_results={"test/f1": 0.8},
            )

    import nexuml.training.lightning as lightning_backend

    monkeypatch.setattr(lightning_backend, "NexuSession", FakeSession)
    monkeypatch.setattr(
        ray_execution,
        "_prepare_session_trainer",
        lambda session: calls.append("trainer") or SimpleNamespace(),
    )
    monkeypatch.setattr(ray_train, "report", lambda metrics: reported.update(metrics))
    monkeypatch.setattr(
        ray_train,
        "get_context",
        lambda: SimpleNamespace(get_world_rank=lambda: 1),
    )

    scenario = ScenarioSpec(name="worker")
    ray_execution.train_loop_per_worker({"scenario": lower_model(scenario)})

    assert calls == ["session:worker:False", "trainer", "run"]
    assert reported == {
        "val/loss": 0.4,
        "test/accuracy": 0.9,
        "test/f1": 0.8,
        "nexuml/completed": 1,
    }


def test_disabled_session_loggers_preserve_callbacks(monkeypatch):
    import nexuml.tracking.logger as logger_module
    import nexuml.training.callbacks as callbacks_module
    from nexuml.training.lightning import NexuSession

    monkeypatch.setattr(
        logger_module,
        "create_loggers",
        lambda *_args, **_kwargs: pytest.fail("nonzero worker created loggers"),
    )
    monkeypatch.setattr(callbacks_module, "build_callbacks", lambda _specs: ["callback"])

    session = NexuSession.from_scenario(ScenarioSpec(name="worker"), enable_loggers=False)

    loggers = session.trainer_loggers
    assert isinstance(loggers, list)
    assert [type(logger).__name__ for logger in loggers] == ["DummyLogger"]
    assert session.trainer_callbacks == ["callback"]


def test_ray_strategy_uses_official_ray_classes(monkeypatch):
    lightning = pytest.importorskip("ray.train.lightning")

    class DDP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FSDP(DDP):
        pass

    class DeepSpeed(DDP):
        pass

    monkeypatch.setattr(lightning, "RayDDPStrategy", DDP)
    monkeypatch.setattr(lightning, "RayFSDPStrategy", FSDP)
    monkeypatch.setattr(lightning, "RayDeepSpeedStrategy", DeepSpeed)

    assert isinstance(ray_execution._ray_strategy("ddp"), DDP)
    assert isinstance(ray_execution._ray_strategy("fsdp"), FSDP)
    assert isinstance(ray_execution._ray_strategy("deepspeed"), DeepSpeed)


def test_ray_strategy_accepts_typed_factory():
    lightning = pytest.importorskip("ray.train.lightning")
    configured = strategy(lightning.RayFSDPStrategy, state_dict_type="full")

    assert isinstance(ray_execution._ray_strategy(configured), lightning.RayFSDPStrategy)


def test_ray_rejects_post_train_fit_layers_until_global_finalization_exists():
    scenario = ScenarioSpec(
        name="post-train",
        pipeline=PipelineSpec(
            stages={
                "post": [
                    LayerSpec(
                        component=DecisionRulePipelineLayer(),
                        keys_in=["features"],
                        keys_out=["score"],
                    )
                ]
            }
        ),
    )

    with pytest.raises(ray_execution.RayExecutionError, match="PostTrainFitLayer"):
        ray_execution._ensure_distributed_semantics(scenario)


def test_ray_rejects_evaluation_algorithms_until_global_aggregation_exists():
    scenario = ScenarioSpec(
        name="distributed-evaluation",
        evaluation=EvaluationSpec(
            algorithms=[EvalAlgorithmSpec(algorithm=ClassHistogramVisualizer())],
        ),
    )

    with pytest.raises(ray_execution.RayExecutionError, match=r"evaluation\.algorithms"):
        ray_execution._ensure_distributed_semantics(scenario)


def test_connect_uses_working_dir_and_uv(monkeypatch):
    ray = pytest.importorskip("ray")
    captured = {}

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://10.43.114.49:8333")
    monkeypatch.setenv(
        "AWS_ENDPOINT_URL_S3",
        "http://seaweedfs-cluster-s3.seaweedfs.svc.cluster.local:8333",
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", "0")
    monkeypatch.setenv("DALI_S3_NO_VERIFY_SSL", "1")
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            runtime_env={**captured["runtime_env"], "working_dir": "gcs://project.zip"}
        ),
    )

    scenario = ScenarioSpec.model_validate(
        {
            "name": "cluster",
            "execution": {
                "kind": "ray",
                "storage_path": "s3://runs/nexuml",
                "target": {
                    "kind": "cluster",
                    "address": "ray://cluster:10001",
                    "working_dir": ".",
                    "py_executable": "uv run --locked python",
                },
            },
        }
    )
    assert isinstance(scenario.execution, RayExecutionSpec)

    runtime_env = ray_execution._connect(scenario.execution)

    assert captured["address"] == "ray://cluster:10001"
    assert captured["runtime_env"]["working_dir"] == "."
    assert runtime_env["working_dir"] == "gcs://project.zip"
    assert captured["runtime_env"]["py_executable"] == "uv run --locked python"
    assert captured["runtime_env"]["env_vars"] == {
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "RAY_TRAIN_V2_ENABLED": "1",
        "RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S": "600",
        "TIMEOUT_FOR_SPECIFIC_SERVER_S": "600",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_ENDPOINT_URL": "http://10.43.114.49:8333",
        "AWS_ENDPOINT_URL_S3": "http://seaweedfs-cluster-s3.seaweedfs.svc.cluster.local:8333",
        "NEXUML_S3_VERIFY_SSL": "0",
        "DALI_S3_NO_VERIFY_SSL": "1",
        "AWS_REQUEST_CHECKSUM_CALCULATION": "WHEN_REQUIRED",
        "AWS_RESPONSE_CHECKSUM_VALIDATION": "WHEN_REQUIRED",
    }
    assert runtime_env["env_vars"]["AWS_ENDPOINT_URL"] == "http://10.43.114.49:8333"
    assert (
        runtime_env["env_vars"]["AWS_ENDPOINT_URL_S3"]
        == "http://seaweedfs-cluster-s3.seaweedfs.svc.cluster.local:8333"
    )
    run_config = ray_execution._run_config(scenario, scenario.execution, runtime_env)
    assert run_config.worker_runtime_env == runtime_env
    assert run_config.storage_path == "runs/nexuml"
    assert run_config.storage_filesystem.type_name == "s3"

    named_run_config = ray_execution._run_config(
        scenario, scenario.execution, runtime_env, "cluster-20260902-abcd1234"
    )
    assert named_run_config.name == "cluster-20260902-abcd1234"


def test_connect_preserves_target_tls_flag_precedence(monkeypatch):
    ray = pytest.importorskip("ray")
    captured = {}

    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", "0")
    monkeypatch.setenv("DALI_S3_NO_VERIFY_SSL", "1")
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(runtime_env=captured["runtime_env"]),
    )

    ray_execution._connect(
        RayExecutionSpec(
            target=RayClusterTarget(
                address="ray://cluster:10001",
                runtime_env={
                    "env_vars": {
                        "NEXUML_S3_VERIFY_SSL": "1",
                        "DALI_S3_NO_VERIFY_SSL": "1",
                    }
                },
            )
        )
    )

    assert captured["runtime_env"]["env_vars"]["NEXUML_S3_VERIFY_SSL"] == "1"
    assert captured["runtime_env"]["env_vars"]["DALI_S3_NO_VERIFY_SSL"] == "0"


def test_connect_derives_dali_flag_after_worker_env_merge(monkeypatch):
    ray = pytest.importorskip("ray")
    captured = {}

    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", "1")
    monkeypatch.setenv("DALI_S3_NO_VERIFY_SSL", "0")
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(runtime_env=captured["runtime_env"]),
    )

    ray_execution._connect(
        RayExecutionSpec(
            target=RayClusterTarget(
                address="ray://cluster:10001",
                runtime_env={"env_vars": {"NEXUML_S3_VERIFY_SSL": "0"}},
            )
        )
    )

    assert captured["runtime_env"]["env_vars"]["NEXUML_S3_VERIFY_SSL"] == "0"
    assert captured["runtime_env"]["env_vars"]["DALI_S3_NO_VERIFY_SSL"] == "1"


def test_connect_omits_unspecified_target_runtime_env(monkeypatch):
    ray = pytest.importorskip("ray")
    captured = {}

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(runtime_env=captured["runtime_env"]),
    )

    runtime_env = ray_execution._connect(
        RayExecutionSpec(
            target=RayClusterTarget(address="ray://cluster:10001", working_dir=None),
        )
    )

    assert "working_dir" not in captured["runtime_env"]
    assert "py_executable" not in captured["runtime_env"]
    assert "working_dir" not in runtime_env
    assert "py_executable" not in runtime_env


def test_connect_merges_native_target_runtime_env(monkeypatch):
    ray = pytest.importorskip("ray")
    captured = {}

    monkeypatch.delenv("RAY_RUNTIME_ENV_IGNORE_GITIGNORE", raising=False)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(runtime_env=captured["runtime_env"]),
    )

    target = RayClusterTarget(
        address="ray://cluster:10001",
        working_dir="/checkout",
        py_executable="uv run --locked python",
        runtime_env={
            "working_dir": "/nested",
            "py_executable": "nested-python",
            "excludes": ["models", "secrets"],
            "uv": {
                "packages": ["-e ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}"],
                "uv_pip_install_options": ["--link-mode", "copy"],
            },
            "env_vars": {
                "RAY_RUNTIME_ENV_IGNORE_GITIGNORE": "1",
                "RAY_TRAIN_V2_ENABLED": "custom",
                "CUSTOM": "value",
            },
        },
    )

    runtime_env = ray_execution._connect(RayExecutionSpec(target=target))

    assert captured["runtime_env"]["working_dir"] == "/checkout"
    assert captured["runtime_env"]["py_executable"] == "uv run --locked python"
    assert captured["runtime_env"]["excludes"] == ["models", "secrets"]
    assert captured["runtime_env"]["uv"]["packages"] == ["-e ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}"]
    assert "RAY_RUNTIME_ENV_IGNORE_GITIGNORE" not in os.environ
    assert captured["runtime_env"]["env_vars"]["RAY_TRAIN_V2_ENABLED"] == "custom"
    assert captured["runtime_env"]["env_vars"]["CUSTOM"] == "value"
    assert runtime_env == captured["runtime_env"]


def test_connect_restores_gitignore_env_after_ray_init_error(monkeypatch):
    ray = pytest.importorskip("ray")
    monkeypatch.setenv("RAY_RUNTIME_ENV_IGNORE_GITIGNORE", "previous")
    monkeypatch.setattr(ray, "is_initialized", lambda: False)

    def fail_init(**kwargs):
        raise RuntimeError("init failed")

    monkeypatch.setattr(ray, "init", fail_init)

    with pytest.raises(RuntimeError, match="init failed"):
        ray_execution._connect(
            RayExecutionSpec(
                target=RayClusterTarget(
                    address="ray://cluster:10001",
                    working_dir=None,
                    runtime_env={"env_vars": {"RAY_RUNTIME_ENV_IGNORE_GITIGNORE": "1"}},
                )
            )
        )

    assert os.environ["RAY_RUNTIME_ENV_IGNORE_GITIGNORE"] == "previous"


def test_connect_rejects_non_mapping_runtime_env_vars(monkeypatch):
    ray = pytest.importorskip("ray")
    monkeypatch.setattr(ray, "init", lambda **kwargs: pytest.fail("ray.init called"))

    with pytest.raises(
        ray_execution.RayExecutionError,
        match="runtime_env.env_vars must be a mapping",
    ):
        ray_execution._connect(
            RayExecutionSpec(
                target=RayClusterTarget(
                    address="ray://cluster:10001",
                    runtime_env={"env_vars": ["invalid"]},
                )
            )
        )


def test_native_runtime_env_schema_accepts_uv_dictionary():
    pytest.importorskip("ray")
    from ray.runtime_env import RuntimeEnv

    runtime_env = RuntimeEnv(
        working_dir=str(Path.cwd()),
        excludes=["data", ".venv"],
        uv=cast(
            Any,
            {
                "packages": ["-e ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}"],
                "uv_pip_install_options": ["--link-mode", "copy"],
            },
        ),
    )

    assert runtime_env["uv"]["packages"] == ["-e ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}"]


def test_native_runtime_env_uses_ordered_index_fallbacks_for_locked_packages():
    helper_path = Path(__file__).parents[2] / "external/NexuLibrary/configs/ray_runtime.py"
    if not helper_path.exists() or not helper_path.with_name("ray-requirements.txt").exists():
        pytest.skip("development Ray configs are not present in this checkout")

    helper_spec = importlib.util.spec_from_file_location("ray_runtime_indexes", helper_path)
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)

    options = helper.ray_runtime_env()["uv"]["uv_pip_install_options"]

    assert options == [
        "--link-mode",
        "copy",
        "--index",
        "https://pypi.org/simple",
        "--index",
        "https://download.pytorch.org/whl/cu128",
        "--index",
        "https://pypi.nvidia.com",
        "--index-strategy",
        "unsafe-first-match",
    ]
    assert "unsafe-best-match" not in options


def test_native_runtime_env_package_is_bounded_to_source_and_metadata(tmp_path):
    helper_path = Path(__file__).parents[2] / "external/NexuLibrary/configs/ray_runtime.py"
    if not helper_path.exists() or not helper_path.with_name("ray-requirements.txt").exists():
        pytest.skip("development Ray configs are not present in this checkout")

    helper_spec = importlib.util.spec_from_file_location("ray_runtime", helper_path)
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)

    packaging = pytest.importorskip("ray._private.runtime_env.packaging")
    package_path = tmp_path / "runtime.zip"
    checkout = helper_path.parents[3]
    packaging.create_package(
        str(checkout),
        package_path,
        include_gitignore=False,
        excludes=helper.ray_runtime_env()["excludes"],
    )

    with zipfile.ZipFile(package_path) as package:
        members = set(package.namelist())

    assert {
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src/nexuml/__init__.py",
        "library/pyproject.toml",
        "library/src/nexuml_library/__init__.py",
        "external/NexuLibrary/pyproject.toml",
        "external/NexuLibrary/src/nexulibrary/__init__.py",
        "external/NexuLibrary/configs/ray-requirements.txt",
    } <= members
    assert not any(
        member.startswith((".venv/", "data/", "exported_model/", ".git/"))
        or member.endswith(".ckpt")
        for member in members
    )
    assert all(
        member in {"pyproject.toml", "README.md", "LICENSE"}
        or member.startswith(("src/", "library/", "external/"))
        for member in members
    )


def test_native_uv_uri_tracks_included_files_and_accepts_fingerprint_comment(tmp_path, monkeypatch):
    helper_path = Path(__file__).parents[2] / "external/NexuLibrary/configs/ray_runtime.py"
    helper_spec = importlib.util.spec_from_file_location("ray_runtime_fingerprint", helper_path)
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)

    checkout = tmp_path / "checkout"
    for directory in (
        "src/nexuml",
        "library/src/nexuml_library",
        "external/NexuLibrary/src/nexulibrary",
        "external/NexuLibrary/configs",
    ):
        (checkout / directory).mkdir(parents=True)
    for relative in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "library/pyproject.toml",
        "library/README.md",
        "library/LICENSE",
        "external/NexuLibrary/pyproject.toml",
        "external/NexuLibrary/README.md",
        "external/NexuLibrary/LICENSE",
        "src/nexuml/module.py",
        "library/src/nexuml_library/module.py",
        "external/NexuLibrary/src/nexulibrary/module.py",
        "external/NexuLibrary/configs/ray-requirements.txt",
    ):
        (checkout / relative).write_text(relative)
    helper_file = checkout / "external/NexuLibrary/configs/ray_runtime.py"
    helper_file.write_text("# excluded helper source")
    monkeypatch.setattr(helper, "__file__", str(helper_file))

    from ray._private.runtime_env.uv import get_uri

    def uri() -> str:
        value = get_uri({"uv": helper.ray_runtime_env()["uv"]})
        assert value is not None
        return value

    base_uri = uri()
    source = checkout / "src/nexuml/module.py"
    source.write_text("changed source")
    assert uri() != base_uri
    source.write_text("src/nexuml/module.py")
    assert uri() == base_uri

    metadata = checkout / "pyproject.toml"
    metadata.write_text("changed metadata")
    assert uri() != base_uri
    metadata.write_text("pyproject.toml")
    assert uri() == base_uri

    requirements = checkout / "external/NexuLibrary/configs/ray-requirements.txt"
    requirements.write_text("changed requirements")
    assert uri() != base_uri
    requirements.write_text("external/NexuLibrary/configs/ray-requirements.txt")
    assert uri() == base_uri

    for relative in (
        "data/artifact.bin",
        "core/artifact.bin",
        "model/artifact.bin",
        ".cache/artifact.bin",
    ):
        ignored = checkout / relative
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_bytes(b"excluded")
    assert uri() == base_uri

    comment_requirements = tmp_path / "comment-requirements.txt"
    comment_requirements.write_text(helper.ray_runtime_env()["uv"]["packages"][-1] + "\n")
    result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--dry-run",
            "--offline",
            "--no-index",
            "--python",
            sys.executable,
            "-r",
            str(comment_requirements),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_connect_disables_uv_hook_before_first_ray_import():
    script = """
import builtins
import json
import os
import sys
from types import SimpleNamespace

import nexuml.cli.main
from nexuml.core.types import RayClusterTarget, RayExecutionSpec
from nexuml.execution import ray as ray_execution

assert "ray" not in sys.modules
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "1"
captured = {}
original_import = builtins.__import__
importing_ray = False

def import_ray(name, *args, **kwargs):
    global importing_ray
    outermost_ray_import = name == "ray" and not importing_ray
    if outermost_ray_import:
        importing_ray = True
    try:
        module = original_import(name, *args, **kwargs)
    finally:
        if outermost_ray_import:
            importing_ray = False
    if outermost_ray_import:
        builtins.__import__ = original_import
        from ray._private import ray_constants
        captured["ray_flag"] = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
        captured["env_at_import"] = os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"]
        module.is_initialized = lambda: False
        module.init = lambda **values: captured.update(values)
        module.get_runtime_context = lambda: SimpleNamespace(
            runtime_env=captured["runtime_env"]
        )
    return module

builtins.__import__ = import_ray
ray_execution._connect(
    RayExecutionSpec(target=RayClusterTarget(address="ray://offline", working_dir=None))
)

from ray._private.runtime_env import uv_runtime_env_hook
from ray.util.client import _apply_uv_hook_for_client

uv_runtime_env_hook.hook = lambda _runtime: (_ for _ in ()).throw(
    AssertionError("Ray Client UV hook was called")
)
hook_result = _apply_uv_hook_for_client(captured["runtime_env"])
print(json.dumps({
    "ray_flag": captured["ray_flag"],
    "env_at_import": captured["env_at_import"],
    "runtime_env": captured["runtime_env"],
    "hook_result": hook_result,
}))
"""
    env = os.environ.copy()
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "1"
    result = subprocess.run(
        ["uv", "run", "--offline", "--no-sync", "python", "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["ray_flag"] is False
    assert observed["env_at_import"] == "0"
    assert "working_dir" not in observed["runtime_env"]
    assert "py_executable" not in observed["runtime_env"]
    assert observed["hook_result"] == observed["runtime_env"]
