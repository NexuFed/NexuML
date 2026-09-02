from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexuml.core.types import (
    EvalAlgorithmSpec,
    EvaluationSpec,
    LayerSpec,
    PipelineSpec,
    RayExecutionSpec,
    ScenarioSpec,
)
from nexuml.execution import ray as ray_execution


def test_ray_execution_config_is_placement_only():
    scenario = ScenarioSpec.model_validate(
        {
            "name": "distributed",
            "training": {"strategy": "fsdp", "strategy_params": {"cpu_offload": False}},
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


def test_ray_worker_reuses_nexusession_and_reports_final_metrics(monkeypatch):
    ray_train = pytest.importorskip("ray.train")
    calls: list[str] = []
    reported = {}

    class FakeSession:
        @classmethod
        def from_scenario(cls, scenario):
            calls.append(f"session:{scenario.name}")
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

    scenario = ScenarioSpec(name="worker")
    ray_execution.train_loop_per_worker({"scenario": scenario.model_dump(mode="json")})

    assert calls == ["session:worker", "trainer", "run"]
    assert reported == {
        "val/loss": 0.4,
        "test/accuracy": 0.9,
        "test/f1": 0.8,
        "nexuml/completed": 1,
    }


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

    assert isinstance(ray_execution._ray_strategy("ddp", {}), DDP)
    assert isinstance(ray_execution._ray_strategy("fsdp", {"state_dict_type": "full"}), FSDP)
    assert isinstance(ray_execution._ray_strategy("deepspeed", {"stage": 2}), DeepSpeed)


def test_ray_rejects_post_train_fit_layers_until_global_finalization_exists(monkeypatch):
    from nexuml.core.post_train_layer import PostTrainFitLayer
    import nexuml.core.registry as registry_module

    class FakePostTrain(PostTrainFitLayer):
        def collect_batch(self, x, y):
            pass

        def finalize_fit(self):
            pass

        def _transform_forward(self, x, y):
            return x, y

    monkeypatch.setattr(
        registry_module,
        "get_registry",
        lambda: SimpleNamespace(get=lambda _key: FakePostTrain),
    )
    scenario = ScenarioSpec(
        name="post-train",
        pipeline=PipelineSpec(
            stages={
                "post": [
                    LayerSpec(
                        type_key="post_train",
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
            algorithms=[EvalAlgorithmSpec(type="class_histogram")],
        ),
    )

    with pytest.raises(ray_execution.RayExecutionError, match=r"evaluation\.algorithms"):
        ray_execution._ensure_distributed_semantics(scenario)


def test_connect_uses_working_dir_and_uv(monkeypatch):
    ray = pytest.importorskip("ray")
    captured = {}

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://seaweed-s3:8333")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "anonymous")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "anonymous")
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
                    "py_executable": (
                        "uv run --python 3.12 --locked --extra ray --extra s3 --extra dali python"
                    ),
                },
            },
        }
    )
    assert isinstance(scenario.execution, RayExecutionSpec)

    runtime_env = ray_execution._connect(scenario.execution)

    assert captured["address"] == "ray://cluster:10001"
    assert captured["runtime_env"]["working_dir"] == "."
    assert runtime_env["working_dir"] == "gcs://project.zip"
    assert captured["runtime_env"]["py_executable"] == (
        "uv run --python 3.12 --locked --extra ray --extra s3 --extra dali python"
    )
    assert captured["runtime_env"]["env_vars"] == {
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "RAY_TRAIN_V2_ENABLED": "1",
        "RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S": "600",
        "TIMEOUT_FOR_SPECIFIC_SERVER_S": "600",
        "AWS_ACCESS_KEY_ID": "anonymous",
        "AWS_SECRET_ACCESS_KEY": "anonymous",
        "AWS_ENDPOINT_URL": "http://seaweed-s3:8333",
        "AWS_REQUEST_CHECKSUM_CALCULATION": "WHEN_REQUIRED",
        "AWS_RESPONSE_CHECKSUM_VALIDATION": "WHEN_REQUIRED",
    }
    run_config = ray_execution._run_config(scenario, scenario.execution, runtime_env)
    assert run_config.worker_runtime_env == runtime_env
    assert run_config.storage_path == "runs/nexuml"
    assert run_config.storage_filesystem.type_name == "s3"

    named_run_config = ray_execution._run_config(
        scenario, scenario.execution, runtime_env, "cluster-20260902-abcd1234"
    )
    assert named_run_config.name == "cluster-20260902-abcd1234"
