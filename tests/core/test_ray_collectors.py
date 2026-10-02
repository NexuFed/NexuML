"""Opted-in real Ray proof: factories, copied weights, exhaustion and learner failure."""

import json
import os
import signal
import tempfile
from threading import Event
from unittest.mock import Mock
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.discovery import environment
from nexuml.core.types import CollectorSpec, ReinforcementLearningSpec, RLEvaluationSpec
from nexuml.reinforcement.collector import build_collector, synchronize_policy
from nexuml.training.lightning import NexuSession
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo


@pytest.fixture
def local_ray(request, monkeypatch):
    import ray

    if ray.is_initialized():
        pytest.fail("Ray proof needs its own runtime; do not run it inside another Ray owner")
    root = Path(__file__).resolve().parents[2]
    # A short path avoids Ray's Unix socket path limit on long worktree/test directories.
    with tempfile.TemporaryDirectory(prefix="ray-") as directory:
        init = ray.init

        def isolated_init(**kwargs):
            return init(
                **kwargs,
                object_store_memory=80 * 1024**2,
                _temp_dir=directory,
                log_to_driver=False,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": os.pathsep.join(
                            (str(root / "src"), str(root / "library/src"))
                        ),
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    }
                },
            )

        monkeypatch.setattr(ray, "init", isolated_init)
        if getattr(request, "param", True):
            ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False)
        try:
            yield ray
        finally:
            ray.shutdown()


@pytest.fixture
def traced_environment(tmp_path):
    @environment(f"RayLifecycleTest_{uuid4().hex}")
    class TracedGym(GymnasiumEnvironment):
        events: str

        def build(self, context: EnvironmentBuildContext):
            runtime = super().build(context)
            path = self.events

            def record(event):
                with open(path, "a") as output:
                    output.write(json.dumps([event, os.getpid(), context.worker_rank]) + "\n")

            record("build")
            close = runtime.close

            def traced_close():
                close()
                record("close")

            runtime.close = traced_close
            return runtime

    return TracedGym(events=str(tmp_path / "events.jsonl"))


def assert_remote_environments_closed(definition):
    events = [json.loads(line) for line in Path(definition.events).read_text().splitlines()]
    builds = {(pid, rank) for event, pid, rank in events if event == "build"}
    closes = {(pid, rank) for event, pid, rank in events if event == "close"}
    assert len(builds) == 2 and all(pid != os.getpid() for pid, _ in builds)
    assert {rank for _, rank in builds} == {0, 1}
    assert builds == closes


@pytest.mark.requires_system
@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("num_envs", [1, 2])
def test_ray_factories_copied_policy_sync_and_iterator_exhaustion(
    local_ray,
    traced_environment,
    num_envs,
):
    from tensordict.nn import TensorDictModule

    linear = torch.nn.Linear(3, 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
    policy = TensorDictModule(linear, in_keys=["observation"], out_keys=["action"])
    training = ReinforcementLearningSpec(
        algorithm=PPO(),
        total_frames=24,
        frames_per_batch=8,
        collector=CollectorSpec(backend="ray", num_collectors=2, num_envs_per_collector=num_envs),
    )
    collector = build_collector(traced_environment, training, policy, "cpu")
    workers = list(collector.remote_collectors)
    try:
        iterator = iter(collector)
        first = next(iterator)
        assert first.numel() == 8 and torch.all(first["action"] == 0)
        with torch.no_grad():
            linear.bias.fill_(1)
        assert torch.all(next(iterator)["action"] == 0)
        synchronize_policy(collector, policy)
        assert torch.all(next(iterator)["action"] == 1)
        # Exhaustion invokes TorchRL's automatic shutdown, not a Lightning teardown hook.
        with pytest.raises(StopIteration):
            next(iterator)
        assert_remote_environments_closed(traced_environment)
        assert local_ray.is_initialized() and not collector.remote_collectors
        for worker in workers:
            with pytest.raises(local_ray.exceptions.RayActorError):
                local_ray.get(worker.__ray_ready__.remote(), timeout=5)
    finally:
        collector.shutdown()


@pytest.mark.requires_system
@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize(
    "local_ray", [True, False], indirect=True, ids=["caller-owned", "collector-owned"]
)
@pytest.mark.parametrize(
    "failure", [None, RuntimeError, KeyboardInterrupt], ids=["fit", "failure", "cancellation"]
)
def test_ray_lightning_fit_and_failure_close_workers(
    local_ray,
    traced_environment,
    failure,
    tmp_path,
    monkeypatch,
):
    caller_owned = local_ray.is_initialized()
    spec = cartpole_ppo(total_frames=32, frames_per_batch=16)
    assert spec.interaction is not None and isinstance(spec.training, ReinforcementLearningSpec)
    definition = traced_environment.model_copy(update={"env_name": "CartPole-v1"})
    spec.interaction.environment = definition
    spec.training.collector = CollectorSpec(backend="ray", num_collectors=2)
    spec.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    spec.training.evaluation = RLEvaluationSpec(episodes=0)
    session = NexuSession(spec, log_dir=tmp_path, enable_loggers=False, enable_progress_bar=False)
    assert session.lightning_module._collector is None and not Path(definition.events).exists()
    synchronized = []

    def checked_sync(collector, policy):
        synchronize_policy(collector, policy)
        expected = policy.state_dict()
        states = local_ray.get(
            [worker.state_dict.remote() for worker in collector.remote_collectors]
        )
        for state in states:
            actual = state["policy_state_dict"]
            assert actual.keys() == expected.keys()
            assert all(torch.equal(actual[key], value) for key, value in expected.items())
        synchronized.append(int(session.lightning_module.get_buffer("batches")))

    monkeypatch.setattr("nexuml.training.reinforcement.synchronize_policy", checked_sync)
    if failure is not None:

        def fail(*args, **kwargs):
            raise failure("Ray learner update failure")

        monkeypatch.setattr(session.lightning_module.algorithm, "update", fail)
        handler = signal.getsignal(signal.SIGINT)
        try:
            with pytest.raises(SystemExit if failure is KeyboardInterrupt else failure):
                session.fit()
        finally:
            signal.signal(signal.SIGINT, handler)
    else:
        result = session.run()
        assert int(session.lightning_module.frames) == 32
        assert result.trainer.global_step > 0
        assert synchronized == [1, 2]
    assert session.lightning_module._collector is None
    assert_remote_environments_closed(definition)
    assert local_ray.is_initialized() == caller_owned


@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
def test_ray_close_failure_kills_owned_actors_releases_lease_and_is_repeatable(monkeypatch):
    from nexuml.reinforcement.ray_collector import _GracefulRayCollector

    collector = object.__new__(_GracefulRayCollector)
    worker = Mock()
    lease = Mock()
    collector._remote_collectors = [worker]
    collector._runtime_lease = lease
    collector._stop_event = Event()
    collector._collection_thread = None
    collector._weight_sync_schemes = None
    get = Mock(side_effect=TimeoutError("close deadline"))
    kill = Mock()
    monkeypatch.setattr("nexuml.reinforcement.ray_collector.ray.get", get)
    monkeypatch.setattr("nexuml.reinforcement.ray_collector.ray.kill", kill)
    with pytest.raises(TimeoutError, match="close deadline"):
        collector.shutdown()
    kill.assert_called_once_with(worker)
    lease.release.assert_called_once()
    assert not collector.remote_collectors
    collector.shutdown()
    get.assert_called_once()
    kill.assert_called_once()


@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize(
    "collector, frames",
    [
        (CollectorSpec(backend="ray", sync=False), 8),
        (CollectorSpec(backend="ray", env_device="cuda"), 8),
        (CollectorSpec(backend="ray", num_collectors=2), 3),
    ],
)
def test_unvalidated_ray_topologies_fail_before_allocation(collector, frames):
    training = ReinforcementLearningSpec(
        algorithm=PPO(), collector=collector, frames_per_batch=frames
    )
    with pytest.raises(ValueError, match="Ray"):
        build_collector(GymnasiumEnvironment(), training, torch.nn.Linear(3, 1), "cpu")
