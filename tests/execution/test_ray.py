from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexuml.core.types import RayExecutionSpec, ScenarioSpec
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
                validation_results=[{"loss": 0.4}],
                test_results=[{"accuracy": 0.9}],
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


def test_connect_uses_working_dir_and_uv(monkeypatch):
    ray = pytest.importorskip("ray")
    captured = {}

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: captured.update(kwargs))

    scenario = ScenarioSpec.model_validate(
        {
            "name": "cluster",
            "execution": {
                "kind": "ray",
                "target": {
                    "kind": "cluster",
                    "address": "ray://cluster:10001",
                    "working_dir": ".",
                },
            },
        }
    )
    assert isinstance(scenario.execution, RayExecutionSpec)

    ray_execution._connect(scenario.execution)

    assert captured["address"] == "ray://cluster:10001"
    assert captured["runtime_env"]["working_dir"] == "."
    assert captured["runtime_env"]["py_executable"] == "uv run"
