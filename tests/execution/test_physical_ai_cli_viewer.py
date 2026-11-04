"""Hermetic private fixture contracts: no CLI execution, native app, renderer or CUDA."""

import ast
import copy
import importlib.abc
import json
from pathlib import Path
import runpy
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
sys.path.insert(0, str(FIXTURES))
cli = importlib.import_module("isaaclab_viser_cli")
hooks = importlib.import_module("isaaclab_viser_cli_hooks")


def forbidden(*args, **kwargs):
    raise AssertionError("Hermetic check attempted native/CUDA/process execution")


class NoNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"isaaclab", "isaacsim", "isaaclab_newton", "viser", "ray"}:
            forbidden(fullname)


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment
    from nexuml_library.environments import _isaaclab_runtime

    monkeypatch.setattr(IsaacLabEnvironment, "build", forbidden)
    monkeypatch.setattr(_isaaclab_runtime, "_create_environment", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    finalizers = []
    monkeypatch.setattr(hooks.atexit, "register", lambda callback: finalizers.append(callback))
    guard = NoNativeImports()
    sys.meta_path.insert(0, guard)
    yield finalizers
    sys.meta_path.remove(guard)


def set_destination(monkeypatch, output, mode="train"):
    monkeypatch.setenv("NEXUML_SPECTATOR_PROBE_OUTPUT", str(output))
    monkeypatch.setenv("NEXUML_SPECTATOR_PROBE_MODE", mode)


@pytest.mark.parametrize("mode", sorted(hooks.MODES))
def test_prepare_public_cli_config_and_fixed_budgets(tmp_path, monkeypatch, mode):
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.types import ReinforcementLearningSpec
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment
    from nexuml_library.reinforcement.algorithms.ppo import PPO
    from nexuml_library.scenarios.reinforcement.isaaclab_sensors import isaac_mixed_cartpole_ppo

    monkeypatch.setenv("PYTHONPATH", "existing-import-path")
    output = tmp_path / mode
    prepared = cli.prepare(mode, output, 8090)
    config = ResolvedConfig.load(output / "config.yaml")
    replay = mode.startswith("replay")
    expected = [sys.executable, "-m", "nexuml.cli.main"]
    if replay:
        expected += [
            "evaluate",
            str(output.parent / "train-viewer/policy"),
            "--config",
            str(output / "config.yaml"),
            "--device",
            "cuda:0",
        ]
    else:
        expected += ["train", "--config", str(output / "config.yaml")]
    assert prepared["argv"] == expected
    env = prepared["env"]
    assert env["NEXUML_SPECTATOR_PROBE_OUTPUT"] == str(output)
    assert env["NEXUML_SPECTATOR_PROBE_MODE"] == mode
    assert env["PYTHONPATH"] == f"{FIXTURES}:existing-import-path"
    assert env["NEXUML_LOGS_ROOT"] == str(output / "native-logs")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert isinstance(config.training, ReinforcementLearningSpec)
    assert isinstance(config.training.algorithm, PPO)
    assert config.training.total_frames == 256
    assert config.training.frames_per_batch == 64
    assert config.training.collector.num_envs_per_collector == 2
    assert config.training.seed == isaac_mixed_cartpole_ppo().training.seed == 0
    assert config.training.algorithm.update_epochs == 1
    assert config.training.algorithm.minibatch_size == 64
    assert config.interaction is not None
    definition = config.interaction.environment
    original = isaac_mixed_cartpole_ppo().interaction.environment
    assert isinstance(definition, IsaacLabEnvironment) and isinstance(original, IsaacLabEnvironment)
    assert definition.task_config is not None and original.task_config is not None
    assert definition.task_config.kwargs == {"base_config": original.task_config.model_dump()}
    assert definition.task_config.factory == "isaaclab_viser_cli_hooks:configure_probe"
    assert definition.task_config.resolve() is hooks.configure_probe
    assert definition.model_dump(exclude={"task_config", "viewer"}) == original.model_dump(
        exclude={"task_config", "viewer"}
    )
    if mode.endswith("-viewer"):
        assert definition.viewer is not None
        assert definition.viewer.purposes == (("eval",) if replay else ("train",))
        assert definition.viewer.port == 8090
        assert definition.viewer.live_plots_update_interval == 1
    else:
        assert definition.viewer is None
    assert config.logging is not None and config.logging.tensorboard is not None
    assert config.logging.diagram is None
    assert Path(config.logging.tensorboard.log_dir).is_relative_to(output)
    assert Path(config.logging.temp_artifact_dir).is_relative_to(output)
    if replay:
        spec = config.policy_evaluation
        assert spec is not None
        assert spec.deterministic and spec.seed == 11
        assert spec.episodes == 32 and spec.max_steps_per_episode == 4
        assert not spec.video.enabled
        assert config.callbacks == config.exports == []
        assert "nonlearning" in prepared["comparison"]
    else:
        assert config.policy_evaluation is None
        assert config.training.evaluation.episodes == 0
        assert config.callbacks[0].factory == "isaaclab_viser_cli_hooks:observer_callback"
        assert config.callbacks[0].resolve() is hooks.observer_callback
        assert config.exports[0].output == str(output / "policy")
    assert not list(output.glob("*.pt"))
    assert not (output / "policy").exists()
    json.dumps(prepared)


@pytest.mark.parametrize("mode,port", [("unknown", 8090), ("train", 0), ("train", 64536)])
def test_prepare_rejects_invalid_before_writing(tmp_path, mode, port):
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        cli.prepare(mode, output, port)
    assert not output.exists()


def test_prepare_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "config.yaml").write_text("original")
    with pytest.raises(FileExistsError):
        cli.prepare("train", output, 8090)
    assert (output / "config.yaml").read_text() == "original"


class Runtime:
    def __init__(self):
        self.calls = []
        self.native = SimpleNamespace(sim=SimpleNamespace(visualizers=[]))

    def step(self, action):
        self.calls.append(("step", action))
        return "transition"

    def close(self):
        self.calls.append(("close",))
        return "closed"


def test_config_delegates_immutable_base_once_and_wraps_only_owner(tmp_path, monkeypatch):
    from nexuml_library.environments import _isaaclab_runtime
    from nexuml_library.environments.isaaclab_config import IsaacTaskConfigSpec
    from nexuml_library.scenarios.reinforcement.isaaclab_sensors import isaac_mixed_cartpole_ppo

    base = isaac_mixed_cartpole_ppo().interaction.environment.task_config.model_dump()
    original_base = copy.deepcopy(base)
    owner, other = Runtime(), Runtime()
    class_step, class_close = Runtime.step, Runtime.close
    monkeypatch.setattr(_isaaclab_runtime, "_active_runtime", owner)
    set_destination(monkeypatch, tmp_path)
    cfg, replacement = object(), object()
    delegated = []

    def build(spec, actual):
        delegated.append((spec.model_dump(), actual))
        return replacement

    monkeypatch.setattr(IsaacTaskConfigSpec, "build", build)
    assert hooks.configure_probe(cfg, base_config=base) is replacement
    assert delegated == [(original_base, cfg)]
    assert base == original_base
    assert Runtime.step is class_step and Runtime.close is class_close
    assert set(vars(other)) == {"calls", "native"}
    assert other.step.__func__ is class_step
    assert owner.calls == []
    events = []
    monkeypatch.setattr(hooks, "observe", lambda runtime, viewing: events.append("observe") or [])
    monkeypatch.setattr(hooks, "record", lambda *args: events.append("record") or {"native": True})
    monkeypatch.setattr(hooks, "progress", lambda runtime, steps, viewing: {"steps": steps})
    traces = []
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: traces.append(args))
    assert owner.step("action1") == owner.step("action2") == "transition"
    assert events == ["observe", "record", "record"]
    assert owner.calls == [("step", "action1"), ("step", "action2")]
    assert json.loads((tmp_path / "progress.json").read_text()) == {"steps": 2}
    assert json.loads((tmp_path / "ready.json").read_text()) == {"port": None, "mode": "train"}
    assert owner.close() == "closed"
    assert owner.calls[-1] == ("close",)
    assert len(traces) == 1 and len(traces[0][1]) == 2


def test_config_requires_active_owner(tmp_path, monkeypatch):
    from nexuml_library.environments import _isaaclab_runtime

    monkeypatch.setattr(_isaaclab_runtime, "_active_runtime", None)
    set_destination(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="active Isaac runtime"):
        hooks.configure_probe(object(), base_config={})


def test_viewer_step_delay_and_progress_are_test_only(tmp_path, monkeypatch):
    owner = Runtime()
    owner.native.sim.visualizers = [
        SimpleNamespace(_viewer=SimpleNamespace(_server=SimpleNamespace(get_port=lambda: 8090)))
    ]
    events = []
    monkeypatch.setattr(hooks, "observe", lambda *args: [])
    monkeypatch.setattr(hooks, "record", lambda *args: {})
    monkeypatch.setattr(hooks, "progress", lambda runtime, steps, viewing: {"steps": steps})
    monkeypatch.setattr(hooks.time, "sleep", events.append)
    hooks._wrap_runtime(owner, tmp_path, "train-viewer", tmp_path)
    assert owner.step("action") == "transition"
    assert events == [0.3]
    assert owner.calls == [("step", "action")]


def test_viewer_live_progress_uses_separate_ipc_and_retains_final_snapshot(tmp_path, monkeypatch):
    output, ipc = tmp_path / "artifacts", tmp_path / "local-ipc"
    output.mkdir()
    ipc.mkdir()
    owner = Runtime()
    owner.native.sim.visualizers = [
        SimpleNamespace(
            _viewer=SimpleNamespace(
                _server=SimpleNamespace(
                    get_port=lambda: 8081,
                    _websock_server=SimpleNamespace(
                        _server_thread=SimpleNamespace(is_alive=lambda: False)
                    ),
                    _thread_executor=SimpleNamespace(_threads=[]),
                )
            )
        )
    ]
    monkeypatch.setattr(hooks, "observe", lambda *args: [])
    monkeypatch.setattr(hooks, "record", lambda *args: {})
    monkeypatch.setattr(hooks, "progress", lambda runtime, steps, viewing: {"steps": steps})
    monkeypatch.setattr(hooks.time, "sleep", lambda *_: None)
    traces = []
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: traces.append(args))
    hooks._wrap_runtime(owner, output, "train-viewer", ipc)
    owner.step("first")
    assert json.loads((ipc / "progress.json").read_text()) == {"steps": 1}
    assert json.loads((ipc / "ready.json").read_text()) == {"port": 8081, "mode": "train-viewer"}
    assert json.loads((output / "ready.json").read_text()) == {"port": 8081, "mode": "train-viewer"}
    assert not (output / "progress.json").exists()
    owner.step("second")
    assert json.loads((ipc / "progress.json").read_text()) == {"steps": 2}
    assert not (output / "progress.json").exists()
    assert owner.close() == "closed"
    assert json.loads((output / "progress.json").read_text()) == {"steps": 2}
    assert len(traces) == 1 and traces[0][0] == output and len(traces[0][1]) == 2


@pytest.mark.parametrize("primary_fails", [False, True])
def test_durable_snapshot_failure_preserves_original_job_error(
    tmp_path, monkeypatch, primary_fails
):
    output, ipc = tmp_path / "artifacts", tmp_path / "local-ipc"
    output.mkdir()
    ipc.mkdir()
    owner = Runtime()
    monkeypatch.setattr(hooks, "observe", lambda *args: [])
    monkeypatch.setattr(hooks, "record", lambda *args: {})
    monkeypatch.setattr(hooks, "progress", lambda runtime, steps, viewing: {"steps": steps})
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: None)
    hooks._wrap_runtime(owner, output, "train", ipc)
    owner.step("first")
    original_write = Path.write_text
    storage_error = OSError("final progress retention failed")

    def write(path, text, *args, **kwargs):
        if path == output / "progress.json":
            raise storage_error
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    primary = RuntimeError("original job failure")
    with pytest.raises(RuntimeError if primary_fails else OSError) as caught:
        if primary_fails:
            try:
                raise primary
            finally:
                owner.close()
        else:
            owner.close()
    assert caught.value is (primary if primary_fails else storage_error)
    if primary_fails:
        assert "final progress retention failed" in primary.__notes__[0]
    assert not (output / "native-cleanup.json").exists()


@pytest.mark.parametrize("mode", ["train", "train-viewer"])
def test_config_routes_only_viewer_live_ipc_through_explicit_child_environment(
    tmp_path, monkeypatch, mode
):
    from nexuml_library.environments import _isaaclab_runtime
    from nexuml_library.environments.isaaclab_config import IsaacTaskConfigSpec

    owner = Runtime()
    monkeypatch.setattr(_isaaclab_runtime, "_active_runtime", owner)
    monkeypatch.setattr(IsaacTaskConfigSpec, "build", lambda *args: "configured")
    set_destination(monkeypatch, tmp_path, mode)
    ipc = tmp_path / "local-ipc"
    monkeypatch.setenv("NEXUML_SPECTATOR_PROBE_IPC", str(ipc))
    wrapped = []
    monkeypatch.setattr(hooks, "_wrap_runtime", lambda *args: wrapped.append(args))
    base = {"factory": "builtins:dict"}
    assert hooks.configure_probe(object(), base_config=base) == "configured"
    assert wrapped == [(owner, tmp_path, mode, ipc if mode.endswith("-viewer") else tmp_path)]
    if mode.endswith("-viewer"):
        monkeypatch.delenv("NEXUML_SPECTATOR_PROBE_IPC")
        with pytest.raises(KeyError, match="NEXUML_SPECTATOR_PROBE_IPC"):
            hooks.configure_probe(object(), base_config=base)


@pytest.mark.parametrize("mode", ["train", "train-viewer"])
def test_ready_is_published_only_after_first_native_progress(tmp_path, monkeypatch, mode):
    owner = Runtime()
    if mode.endswith("-viewer"):
        owner.native.sim.visualizers = [
            SimpleNamespace(_viewer=SimpleNamespace(_server=SimpleNamespace(get_port=lambda: 8081)))
        ]
    ready, progress = tmp_path / "ready.json", tmp_path / "progress.json"
    events = []

    def step(action):
        assert not ready.exists() and not progress.exists()
        events.append("native-step")
        return "transition"

    original_replace = Path.replace

    def replace(source, target):
        if Path(target) == ready:
            assert json.loads(progress.read_text()) == {"steps": 1}
            assert json.loads(source.read_text()) == {
                "port": 8081 if mode.endswith("-viewer") else None,
                "mode": mode,
            }
            events.append("ready-after-progress")
        return original_replace(source, target)

    monkeypatch.setattr(owner, "step", step)
    monkeypatch.setattr(hooks, "observe", lambda *args: [])
    monkeypatch.setattr(hooks, "record", lambda *args: {})
    monkeypatch.setattr(hooks, "progress", lambda runtime, steps, viewing: {"steps": steps})
    monkeypatch.setattr(hooks.time, "sleep", lambda *_: None)
    monkeypatch.setattr(Path, "replace", replace)
    hooks._wrap_runtime(owner, tmp_path, mode, tmp_path)
    assert owner.step("action") == "transition"
    assert events == ["native-step", "ready-after-progress"]
    assert ready.is_file() and json.loads(progress.read_text()) == {"steps": 1}


def test_first_native_failure_does_not_publish_ready_or_hide_original_error(tmp_path, monkeypatch):
    owner = Runtime()
    original = RuntimeError("first native step failed")

    def step(_action):
        raise original

    monkeypatch.setattr(owner, "step", step)
    monkeypatch.setattr(hooks, "observe", lambda *args: [])
    hooks._wrap_runtime(owner, tmp_path, "train", tmp_path)
    with pytest.raises(RuntimeError) as caught:
        owner.step("action")
    assert caught.value is original
    assert not (tmp_path / "ready.json").exists()
    assert not (tmp_path / "ready.tmp").exists()
    assert not (tmp_path / "progress.json").exists()


@pytest.mark.parametrize("closing_fails", [False, True])
def test_trace_failure_does_not_mask_primary_exception(tmp_path, monkeypatch, closing_fails):
    owner = Runtime()
    primary = RuntimeError("original close" if closing_fails else "original step")
    calls = []

    def close():
        calls.append("close")
        if closing_fails:
            raise primary

    monkeypatch.setattr(owner, "close", close)
    monkeypatch.setattr(
        hooks, "_write_trace", lambda *args: (_ for _ in ()).throw(OSError("trace"))
    )
    hooks._wrap_runtime(owner, tmp_path, "train", tmp_path)
    with pytest.raises(RuntimeError) as caught:
        if closing_fails:
            owner.close()
        else:
            try:
                raise primary
            finally:
                owner.close()
    assert caught.value is primary
    assert "trace" in primary.__notes__[0]
    assert calls == ["close"]
    assert not (tmp_path / "native-cleanup.json").exists()


def test_trace_failure_is_not_hidden_on_success(tmp_path, monkeypatch):
    owner = Runtime()
    monkeypatch.setattr(
        hooks, "_write_trace", lambda *args: (_ for _ in ()).throw(OSError("trace"))
    )
    hooks._wrap_runtime(owner, tmp_path, "train", tmp_path)
    with pytest.raises(OSError, match="trace"):
        owner.close()
    assert owner.calls == [("close",)]
    assert not (tmp_path / "native-cleanup.json").exists()


def test_clean_close_has_explicit_receipt_and_sentinel(tmp_path, monkeypatch, capsys, hermetic):
    owner = Runtime()
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: None)
    hooks._wrap_runtime(owner, tmp_path, "replay", tmp_path)
    assert owner.close() == "closed"
    assert not (tmp_path / "native-cleanup.json").exists()
    hermetic[0]()
    assert json.loads((tmp_path / "native-cleanup.json").read_text()) == {
        "status": "PASS",
        "mode": "replay",
        "steps": 0,
    }
    assert "CLI_PROBE_NATIVE_CLOSED_AND_OWNED_THREADS_STOPPED replay" in capsys.readouterr().out


def test_collection_close_precedes_training_progress_and_logger_cleanup(
    tmp_path, monkeypatch, hermetic
):
    owner = Runtime()

    class ProgressThread:
        def is_alive(self):
            return True

    threads = []
    monkeypatch.setattr(hooks, "enumerate_threads", lambda: threads)
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: None)
    hooks._wrap_runtime(owner, tmp_path, "train", tmp_path)
    threads.append(ProgressThread())
    assert owner.close() == "closed"
    assert not (tmp_path / "native-cleanup.json").exists()
    (tmp_path / "logger-finalization.json").write_text(
        json.dumps(
            [{"status": "success", "job_scope_active": True, "completed": True, "error": None}]
        )
    )
    threads.clear()
    hermetic[0]()
    assert json.loads((tmp_path / "native-cleanup.json").read_text())["status"] == "PASS"


def test_post_job_thread_leak_cannot_publish_cleanup_success(tmp_path, monkeypatch, hermetic):
    owner = Runtime()

    class LeakedThread:
        def is_alive(self):
            return True

    threads = []
    monkeypatch.setattr(hooks, "enumerate_threads", lambda: threads)
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: None)
    hooks._wrap_runtime(owner, tmp_path, "replay", tmp_path)
    owner.close()
    threads.append(LeakedThread())
    with pytest.raises(AssertionError):
        hermetic[0]()
    assert not (tmp_path / "native-cleanup.json").exists()


def test_native_close_checks_only_explicit_viewer_thread_ownership(tmp_path, monkeypatch):
    owner = Runtime()
    server = SimpleNamespace(
        get_port=lambda: 8081,
        _websock_server=SimpleNamespace(_server_thread=SimpleNamespace(is_alive=lambda: True)),
        _thread_executor=SimpleNamespace(_threads=()),
    )
    owner.native.sim.visualizers = [SimpleNamespace(_viewer=SimpleNamespace(_server=server))]
    monkeypatch.setattr(hooks, "observe", lambda *args: [])
    monkeypatch.setattr(hooks, "record", lambda *args: {})
    monkeypatch.setattr(hooks, "progress", lambda *args: {})
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: None)
    monkeypatch.setattr(hooks.time, "sleep", lambda *_: None)
    hooks._wrap_runtime(owner, tmp_path, "train-viewer", tmp_path)
    owner.step("action")
    with pytest.raises(AssertionError):
        owner.close()
    assert owner.calls[-1] == ("close",)
    assert not (tmp_path / "native-cleanup.json").exists()


def test_failed_native_close_cannot_publish_cleanup_success(tmp_path, monkeypatch, capsys):
    owner = Runtime()
    original = RuntimeError("native close failed")
    monkeypatch.setattr(owner, "close", lambda: (_ for _ in ()).throw(original))
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: None)
    hooks._wrap_runtime(owner, tmp_path, "failure-viewer", tmp_path)
    with pytest.raises(RuntimeError) as caught:
        owner.close()
    assert caught.value is original
    assert not (tmp_path / "native-cleanup.json").exists()
    assert "CLI_PROBE_NATIVE_CLOSED_AND_OWNED_THREADS_STOPPED" not in capsys.readouterr().out


def test_trace_artifacts_are_bounded_without_dropping_any_observation(tmp_path):
    trajectory = [
        {
            "observation": torch.zeros(2, 16, 32, 32),
            "history": torch.zeros(2, 16, 32, 32),
            "rng": torch.zeros(5056, dtype=torch.uint8),
        }
        for _ in range(128)
    ]
    hooks._write_trace(tmp_path, trajectory, "train")
    assert all(path.stat().st_size <= 32 * 1024**2 for path in tmp_path.glob("trajectory*.pt"))
    manifest = json.loads((tmp_path / "trajectory-manifest.json").read_text())
    assert manifest["steps"] == 128
    restored = []
    for part in manifest["parts"]:
        restored.extend(torch.load(tmp_path / part["path"], weights_only=True))
    assert len(restored) == 128
    torch.testing.assert_close(restored, trajectory, rtol=0, atol=0)


def test_native_step_exception_keeps_original_and_saves_partial_trace(tmp_path, monkeypatch):
    owner = Runtime()
    original = RuntimeError("original native step")
    calls = []

    def step(action):
        calls.append(action)
        raise original

    monkeypatch.setattr(owner, "step", step)
    monkeypatch.setattr(hooks, "observe", lambda *args: [])
    monkeypatch.setattr(hooks, "record", forbidden)
    hooks._wrap_runtime(owner, tmp_path, "train", tmp_path)
    with pytest.raises(RuntimeError) as caught:
        try:
            owner.step("action")
        finally:
            owner.close()
    assert caught.value is original and calls == ["action"]
    assert owner.calls == [("close",)]
    assert torch.load(tmp_path / "trajectory.pt", weights_only=True) == []


def test_record_copies_native_history_actions_and_rng_without_advancing_them(monkeypatch):
    import numpy as np
    from tensordict import TensorDict

    action = TensorDict({"action": torch.tensor([[0.2]])}, batch_size=(1,))
    observation = TensorDict({"state": torch.tensor([[1.0]])}, batch_size=(1,))
    history = TensorDict({"history": torch.tensor([[2.0]])}, batch_size=(1,))
    joints, velocity = torch.tensor([[3.0]]), torch.tensor([[4.0]])
    runtime = SimpleNamespace(
        native=SimpleNamespace(
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        joint_pos=SimpleNamespace(torch=joints),
                        joint_vel=SimpleNamespace(torch=velocity),
                    )
                )
            },
            episode_length_buf=torch.tensor([2]),
        ),
        _processing_state=history,
    )
    transition = SimpleNamespace(
        observation=observation,
        reward=torch.tensor([1.0]),
        terminated=torch.tensor([False]),
        truncated=torch.tensor([False]),
    )
    cuda_rng = torch.tensor([12], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_rng.clone())
    torch_rng, python_rng, numpy_rng = (
        torch.get_rng_state(),
        hooks.random.getstate(),
        np.random.get_state(legacy=True),
    )
    assert isinstance(numpy_rng, tuple)
    result = hooks.record(runtime, transition, action)
    assert set(result) == {
        "action",
        "observation",
        "reward",
        "terminated",
        "truncated",
        "joints",
        "velocity",
        "history",
        "episode_length",
        "rng",
        "cuda_rng",
        "python_rng",
        "numpy_rng",
    }
    assert torch.equal(result["rng"], torch_rng) and torch.equal(torch.get_rng_state(), torch_rng)
    assert result["python_rng"] == python_rng == hooks.random.getstate()
    assert torch.equal(result["cuda_rng"], cuda_rng)
    assert np.array_equal(result["numpy_rng"].numpy(), numpy_rng[1])
    current_numpy_rng = np.random.get_state(legacy=True)
    assert isinstance(current_numpy_rng, tuple)
    assert np.array_equal(current_numpy_rng[1], numpy_rng[1])
    for copied, original in (
        (result["action"]["action"], action["action"]),
        (result["observation"]["state"], observation["state"]),
        (result["history"]["history"], history["history"]),
        (result["joints"], joints),
        (result["velocity"], velocity),
    ):
        assert isinstance(original, torch.Tensor)
        assert torch.equal(copied, original) and copied.data_ptr() != original.data_ptr()


def training_fakes(monkeypatch, tmp_path, mode="train"):
    from nexuml_library.environments import _isaaclab_runtime

    set_destination(monkeypatch, tmp_path, mode)
    monkeypatch.setattr(_isaaclab_runtime, "_process_active", True)
    calls = []
    logger = SimpleNamespace(finalize=lambda status: calls.append(status))
    trainer = SimpleNamespace(loggers=[logger], save_checkpoint=lambda path: calls.append(path))
    module = SimpleNamespace(
        policy=torch.nn.Linear(2, 1), frames=256, batches=4, state_dict=lambda: {"test": True}
    )
    callback = hooks.observer_callback()
    callback.on_train_start(trainer, module)
    return callback, trainer, module, calls


def test_callback_asserts_training_saves_checkpoint_and_records_logger(tmp_path, monkeypatch):
    callback, trainer, module, calls = training_fakes(monkeypatch, tmp_path)
    with torch.no_grad():
        module.policy.weight.add_(1)
    callback.on_train_end(trainer, module)
    assert torch.load(tmp_path / "training-state.pt", weights_only=True) == {"test": True}
    assert calls == [tmp_path / "complete.ckpt"]
    trainer.loggers[0].finalize("success")
    assert calls[-1] == "success"
    assert json.loads((tmp_path / "logger-finalization.json").read_text()) == [
        {"status": "success", "job_scope_active": True, "completed": True, "error": None}
    ]


@pytest.mark.parametrize("invalid", ["unchanged", "nan", "frames", "batches"])
def test_callback_rejects_invalid_training(tmp_path, monkeypatch, invalid):
    callback, trainer, module, calls = training_fakes(monkeypatch, tmp_path)
    with torch.no_grad():
        if invalid != "unchanged":
            module.policy.weight.add_(1)
        if invalid == "nan":
            module.policy.weight.fill_(float("nan"))
    if invalid == "frames":
        module.frames = 255
    if invalid == "batches":
        module.batches = 3
    with pytest.raises(AssertionError):
        callback.on_train_end(trainer, module)
    assert not calls and not (tmp_path / "training-state.pt").exists()


@pytest.mark.parametrize("mode", ["failure-viewer", "cancel-viewer", "train-viewer"])
def test_callback_injects_only_labelled_third_batch(tmp_path, monkeypatch, mode):
    callback, trainer, module, _ = training_fakes(monkeypatch, tmp_path, mode)
    killed = []
    monkeypatch.setattr(hooks.os, "kill", lambda *args: killed.append(args))
    for index in (0, 1, 3):
        callback.on_train_batch_end(trainer, module, None, None, index)
    assert not killed
    if mode == "failure-viewer":
        with pytest.raises(RuntimeError, match="injected spectator training failure"):
            callback.on_train_batch_end(trainer, module, None, None, 2)
    else:
        callback.on_train_batch_end(trainer, module, None, None, 2)
    assert killed == ([(hooks.os.getpid(), signal.SIGTERM)] if mode == "cancel-viewer" else [])
    receipt = tmp_path / "injection.json"
    if mode in {"failure-viewer", "cancel-viewer"}:
        assert json.loads(receipt.read_text()) == {
            "mode": mode,
            "batch_idx": 2,
            "identity": "SIGTERM"
            if mode == "cancel-viewer"
            else "injected spectator training failure",
        }
    else:
        assert not receipt.exists()


def test_logger_original_failure_survives_trace_failure(tmp_path, monkeypatch):
    from nexuml_library.environments import _isaaclab_runtime

    set_destination(monkeypatch, tmp_path)
    monkeypatch.setattr(_isaaclab_runtime, "_process_active", True)
    original = RuntimeError("original logger failure")
    calls = []

    def finalize(status):
        calls.append(status)
        raise original

    trainer = SimpleNamespace(loggers=[SimpleNamespace(finalize=finalize)])
    callback = hooks.observer_callback()
    callback.on_train_start(trainer, SimpleNamespace(policy=torch.nn.Linear(1, 1)))
    monkeypatch.setattr(Path, "write_text", lambda *args: (_ for _ in ()).throw(OSError("trace")))
    with pytest.raises(RuntimeError) as caught:
        trainer.loggers[0].finalize("failed")
    assert caught.value is original
    assert calls == ["failed"]
    assert "trace" in original.__notes__[0]


def test_failed_logger_finalizer_cannot_qualify_expected_injected_failure(
    tmp_path, monkeypatch, hermetic
):
    from nexuml_library.environments import _isaaclab_runtime

    set_destination(monkeypatch, tmp_path, "failure-viewer")
    monkeypatch.setattr(_isaaclab_runtime, "_process_active", True)
    error = RuntimeError("original finalization error")

    def finalize(_status):
        raise error

    trainer = SimpleNamespace(loggers=[SimpleNamespace(finalize=finalize)])
    hooks.observer_callback().on_train_start(trainer, SimpleNamespace(policy=torch.nn.Linear(1, 1)))
    with pytest.raises(RuntimeError) as caught:
        trainer.loggers[0].finalize("failed")
    assert caught.value is error
    assert json.loads((tmp_path / "logger-finalization.json").read_text()) == [
        {
            "status": "failed",
            "job_scope_active": True,
            "completed": False,
            "error": "logger finalization failed",
        }
    ]
    owner = Runtime()
    monkeypatch.setattr(hooks, "_write_trace", lambda *args: None)
    hooks._wrap_runtime(owner, tmp_path, "failure-viewer", tmp_path)
    owner.close()
    with pytest.raises(AssertionError):
        hermetic[0]()
    assert not (tmp_path / "native-cleanup.json").exists()


def test_sources_no_direct_session_or_manual_native_loop_and_lazy_hooks():
    source = (FIXTURES / "isaaclab_viser_cli_hooks.py").read_text()
    tree = ast.parse(source)
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in statement.names]
                if isinstance(statement, ast.Import)
                else [statement.module]
            )
            assert all(
                isinstance(module, str) and module.split(".")[0] in sys.stdlib_module_names
                for module in modules
            )
    for name in ("isaaclab_viser_cli.py", "isaaclab_viser_cli_hooks.py"):
        text = (FIXTURES / name).read_text()
        assert "NexuSession" not in text and "export_package" not in text
        assert ".reset(" not in text and "for _ in range(" not in text
        assert "_active_runtime =" not in text


def test_browser_uses_required_interpreter_sandbox_and_preserves_controls():
    import nexuml

    browser = (FIXTURES / "isaaclab_viser_cli_browser.cjs").read_text()
    original = (
        Path(nexuml.__file__).parents[2] / "tests/fixtures/isaaclab_viser_browser.cjs"
    ).read_text()
    assert "process.env.NEXUML_SPECTATOR_PYTHON" in browser
    assert '"prepare", mode, output, portText' in browser
    assert "spawn(launch.argv[0], launch.argv.slice(1)" in browser
    assert "chromiumSandbox: true" in browser
    assert "--no-sandbox" not in browser and "--disable-setuid-sandbox" not in browser
    assert ".venv" not in browser and "isaaclab_viser_job.py" not in browser
    assert "const executablePath = process.env.NEXUML_CHROMIUM_EXECUTABLE;" in browser
    # Exact copied controls, canvas/console/errors, status, listener and native cleanup checks.
    control_start = "    async function connectPage() {"
    control_end = "    await browser.close(); browser = null;"
    original_progress = (
        'const progress = () => JSON.parse(fs.readFileSync(path.join(output, "progress.json"), '
        '"utf8"));'
    )
    assert "const progress = () => readProgress(ipc.path);" in browser
    assert (
        browser[browser.index(control_start) : browser.index(control_end)].replace(
            "const progress = () => readProgress(ipc.path);", original_progress
        )
        == original[original.index(control_start) : original.index(control_end)]
    )


def test_native_cli_caller_passes_literal_argv_and_child_only_environment(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    functions = runpy.run_path(str(root / "tests/core/test_isaaclab_integration.py"))
    execute = functions["_run"]
    calls = []
    monkeypatch.setenv("LD_LIBRARY_PATH", "native-libraries")
    monkeypatch.setenv("HOME", "native-home")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="public CLI completed")

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "literal output; $(no shell)"
    result = execute(
        ["-m", "nexuml.cli.main", "train", "--config", "literal config; $.yaml"],
        output,
        "public-cli",
        process_scope=False,
        extra_env={"NEXUML_SPECTATOR_PROBE_MODE": "train", "PYTHONPATH": str(FIXTURES)},
    )
    assert result == "public CLI completed"
    argv, kwargs = calls[0]
    assert argv == [
        sys.executable,
        "-m",
        "nexuml.cli.main",
        "train",
        "--config",
        "literal config; $.yaml",
    ]
    assert kwargs["env"]["NEXUML_SPECTATOR_PROBE_MODE"] == "train"
    assert kwargs["env"]["LD_LIBRARY_PATH"] == "native-libraries"
    assert kwargs["env"]["HOME"] == "native-home"
    assert kwargs["env"]["PYTHONPATH"] == str(FIXTURES)
    assert "NEXUML_SPECTATOR_PROBE_MODE" not in functions["os"].environ
    assert not kwargs.get("shell", False)


def test_native_cli_caller_uses_separate_modes_and_kuberay_safe_port():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "tests/core/test_isaaclab_integration.py").read_text())
    (function,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_native_spectator_public_cli_browser_train_replay_and_lifecycle"
    ]
    text = ast.unparse(function)
    assert "isaaclab_viser_cli.py" in text and "isaaclab_viser_cli_browser.cjs" in text
    assert "isaaclab_viser_job.py" not in text and "NexuSession" not in text
    assert "port = 8081" in text and "NEXUML_SPECTATOR_PYTHON" in text
    assert "sys.executable" in text and "headless('train')" in text and "headless('replay')" in text
    assert all(
        mode in text
        for mode in ("train-viewer", "replay-viewer", "failure-viewer", "cancel-viewer")
    )
