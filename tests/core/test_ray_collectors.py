"""Opted-in real Ray proof: factories, copied weights, exhaustion and learner failure."""

from contextlib import contextmanager
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
from pydantic import PrivateAttr

from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.discovery import environment
from nexuml.core.types import (
    CollectorSpec,
    RayCollectorSpec,
    RayEvaluationSpec,
    ReinforcementLearningSpec,
    RLEvaluationSpec,
)
from nexuml.reinforcement.collector import EnvironmentFactory, build_collector, synchronize_policy
from nexuml.training.lightning import NexuSession
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo, pendulum_ppo


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
        profile = getattr(request, "param", True)
        if profile:
            ray.init(
                address="local",
                num_cpus=2,
                num_gpus=int(profile == "gpu"),
                include_dashboard=False,
                resources={"snapshot-test": 1},
                labels={"nexuml-test-placement": "cpu-rollout"},
            )
        try:
            yield ray
        finally:
            ray.shutdown()


@pytest.fixture
def traced_environment(tmp_path):
    @environment(f"RayLifecycleTest_{uuid4().hex}")
    class TracedGym(GymnasiumEnvironment):
        events: str
        _scope_pid: int | None = PrivateAttr(default=None)

        def _record(self, event, rank=None):
            with open(self.events, "a") as output:
                output.write(json.dumps([event, os.getpid(), rank]) + "\n")

        @contextmanager
        def job_scope(self):
            assert self._scope_pid is None
            self._scope_pid = os.getpid()
            self._record("enter")
            try:
                yield
            finally:
                self._record("leave")
                self._scope_pid = None

        def build(self, context: EnvironmentBuildContext):
            assert self._scope_pid == os.getpid()
            runtime = super().build(context)
            self._record("build", context.worker_rank)
            close = runtime.close

            def traced_close():
                assert self._scope_pid == os.getpid()
                close()
                self._record("close", context.worker_rank)

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
    for pid, _ in builds:
        assert [event for event, owner, _ in events if owner == pid] == [
            "enter",
            "build",
            "close",
            "leave",
        ]


def test_worker_scope_unwinds_failed_build(traced_environment, monkeypatch):
    def fail(self, context):
        assert self._scope_pid == os.getpid()
        raise RuntimeError("worker build failure")

    monkeypatch.setattr(type(traced_environment), "build", fail)
    factory = EnvironmentFactory(
        traced_environment, EnvironmentBuildContext(purpose="train"), owns_job_scope=True
    )
    with pytest.raises(RuntimeError, match="worker build failure"):
        factory()
    events = [json.loads(line) for line in Path(traced_environment.events).read_text().splitlines()]
    assert [event for event, _, _ in events] == ["enter", "leave"]


@pytest.mark.requires_system
@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_snapshot_evaluation_uses_fresh_ray_workers_and_preserves_runtime(
    local_ray, tmp_path, monkeypatch
):
    pytest.importorskip("tensorboard")
    monkeypatch.setenv("NEXUML_LOGS_ROOT", str(tmp_path / "logs"))
    scenario = pendulum_ppo(total_frames=64, frames_per_batch=32)
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=16)
    scenario.training.evaluation = RLEvaluationSpec(
        at_start=True,
        every_n_frames=32,
        episodes=2,
        max_steps_per_episode=4,
        seed=23,
        ray=RayEvaluationSpec(resources={"snapshot-test": 1}, scheduling="skip_if_busy"),
    )
    result = NexuSession(scenario, log_dir=tmp_path, enable_progress_bar=False).run()
    assert int(result.lightning_module.frames) == 64
    assert int(result.lightning_module.batches) == 2
    reports = [
        json.loads(path.read_text()) for path in sorted(tmp_path.rglob("evaluation-frame-*.json"))
    ]
    frames = [r["source_training_frames"] for r in reports]
    assert frames in ([0, 64], [0, 32, 64])
    assert [r["source_training_batches"] for r in reports] == [frame // 32 for frame in frames]
    assert len({r["snapshot_sha256"] for r in reports}) == len(reports)
    assert len({r["evaluator_pid"] for r in reports}) == len(reports)
    for receipt in reports:
        assert receipt["evaluator_pid"] != os.getpid()
        assert receipt["evaluator_node_id"]
        assert receipt["evaluator_resources"] == {"CPU": 1.0, "snapshot-test": 1.0}
        assert [ep["seed"] for ep in receipt["episode_report"]["episodes"]] == [23, 24]
    assert local_ray.is_initialized()
    assert local_ray.get(local_ray.remote(lambda: "alive").remote(), timeout=10) == "alive"


@pytest.mark.requires_system
@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_ray_snapshot_cancels_running_evaluator_without_shutting_down_ray(
    local_ray, tmp_path, monkeypatch
):
    from nexuml.evaluation import _ray

    started = str(tmp_path / "started")
    closed = str(tmp_path / "closed")

    def blocking_worker(*args):
        try:
            Path(started).touch()
            gate = Event()
            for _ in range(600):
                gate.wait(0.1)
        finally:
            Path(closed).touch()

    monkeypatch.setattr(_ray, "_evaluate_on_worker", blocking_worker)
    spec = RayEvaluationSpec(timeout_s=30, resources={"snapshot-test": 1})
    scenario = pendulum_ppo()
    owner = NexuSession(scenario, log_dir=tmp_path).lightning_module
    pending = _ray.RayEvaluation(owner.policy, scenario, owner=owner, spec=spec)

    def cancel_when_running():
        if Path(started).exists():
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        try:
            pending.result(wait=True, check_cancelled=cancel_when_running)
        finally:
            pending.close()
    assert Path(closed).is_file()
    with pytest.raises(local_ray.exceptions.TaskCancelledError):
        local_ray.get(pending.reference, timeout=1)
    assert local_ray.is_initialized()


@pytest.mark.requires_system
@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_ray_worker_error_remains_actionable_and_other_batches_finish(
    local_ray, traced_environment
):
    from tensordict.nn import TensorDictModule

    policy = TensorDictModule(
        torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Tanh()),
        in_keys=["observation"],
        out_keys=["action"],
    )
    training = ReinforcementLearningSpec(
        algorithm=PPO(),
        total_frames=20,
        frames_per_batch=8,
        collector=CollectorSpec(backend="ray", num_collectors=2),
    )
    collector = build_collector(traced_environment, training, policy, "cpu")
    try:
        worker = collector.remote_collectors[0]
        with pytest.raises(local_ray.exceptions.RayTaskError, match="unavailable_for_test"):
            local_ray.get(worker.get_distant_attr.remote("unavailable_for_test"), timeout=10)
        assert [batch.numel() for batch in collector] == [8, 8, 8]
    finally:
        collector.shutdown()
    assert_remote_environments_closed(traced_environment)
    assert local_ray.is_initialized()


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
def test_ray_collector_placement_on_borrowed_cluster(local_ray, monkeypatch):
    from ray._private.state import state
    from ray.core.generated.gcs_pb2 import ActorTableData  # ty: ignore[unresolved-import]
    from tensordict.nn import TensorDictModule

    existing = local_ray.get_runtime_context().gcs_address
    another = "192.0.2.1:6379"  # A documentation address; mismatch must fail without contact.
    selector = {"nexuml-test-placement": "in(cpu-rollout,cpu-alternate)"}
    policy = TensorDictModule(
        torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Tanh()),
        in_keys=["observation"],
        out_keys=["action"],
    )
    definition = GymnasiumEnvironment(env_name="Pendulum-v1")
    training = ReinforcementLearningSpec(
        algorithm=PPO(),
        total_frames=8,
        frames_per_batch=8,
        collector=CollectorSpec(
            backend="ray",
            ray=RayCollectorSpec(resources={"snapshot-test": 1}, label_selector=selector),
        ),
    )
    assert training.collector.ray is not None
    init = Mock(side_effect=AssertionError("must borrow the fixture's Ray runtime"))
    monkeypatch.setattr(local_ray, "init", init)
    monkeypatch.setenv("RAY_ADDRESS", another)
    for address in (another, "auto"):
        training.collector.ray.address = address
        with pytest.raises(ValueError, match="conflicts with the initialized"):
            build_collector(definition, training, policy, "cpu")

    for address in (None, existing, "auto"):
        if address == "auto":
            monkeypatch.setenv("RAY_ADDRESS", existing)
        training.collector.ray.address = address
        collector = build_collector(definition, training, policy, "cpu")
        try:
            worker = collector.remote_collectors[0]
            row = state._connect_and_get_accessor().get_actor_info(worker._actor_id)
            actor = ActorTableData.FromString(row)
            assert dict(actor.required_resources) == {"CPU": 1.0, "snapshot-test": 1.0}
            assert set(actor.label_selector) == set(selector)
            expression = actor.label_selector["nexuml-test-placement"]
            assert expression.startswith("in(") and expression.endswith(")")
            assert set(expression[3:-1].split(",")) == {"cpu-rollout", "cpu-alternate"}
            assert next(iter(collector)).numel() == 8
        finally:
            collector.shutdown()
        assert local_ray.is_initialized()
        assert local_ray.get_runtime_context().gcs_address == existing
    init.assert_not_called()
    assert local_ray.get(local_ray.remote(lambda: "alive").remote(), timeout=10) == "alive"


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
        frames = session.lightning_module.frames
        assert isinstance(frames, torch.Tensor)
        assert int(frames) == 32
        assert result.trainer.global_step > 0
        assert synchronized == [1, 2]
    assert session.lightning_module._collector is None
    assert_remote_environments_closed(definition)
    assert local_ray.is_initialized() == caller_owned


@pytest.mark.requires_system
@pytest.mark.requires_gpu
@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
def test_ray_cuda_learner_synchronizes_cpu_workers(local_ray, tmp_path, monkeypatch):
    scenario = pendulum_ppo(total_frames=32, frames_per_batch=16)
    scenario.training.accelerator = "gpu"
    scenario.training.devices = 1
    scenario.training.collector = CollectorSpec(
        backend="ray", num_collectors=2, policy_device="cpu", ray=RayCollectorSpec()
    )
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    scenario.training.evaluation = RLEvaluationSpec(episodes=0)
    versions = []

    def checked_sync(collector, policy):
        expected = {key: value.cpu() for key, value in policy.state_dict().items()}
        assert all(value.is_cuda for value in policy.parameters())
        synchronize_policy(collector, policy)
        for state in local_ray.get(
            [worker.state_dict.remote() for worker in collector.remote_collectors], timeout=10
        ):
            torch.testing.assert_close(state["policy_state_dict"], expected, rtol=0, atol=0)
        versions.append(collector._published_version)

    monkeypatch.setattr("nexuml.training.reinforcement.synchronize_policy", checked_sync)
    result = NexuSession(
        scenario, log_dir=tmp_path, enable_loggers=False, enable_progress_bar=False
    ).run()
    assert result.lightning_module.fit_device == "cuda:0"
    assert int(result.lightning_module.get_buffer("frames")) == 32
    assert result.trainer.global_step == 8 and versions == [2, 3]
    assert result.lightning_module._collector is None and local_ray.is_initialized()


@pytest.mark.requires_system
@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize(
    "local_ray",
    [
        pytest.param(True, id="cpu"),
        pytest.param(
            "gpu",
            id="native-gpu-worker",
            marks=[
                pytest.mark.requires_gpu,
                pytest.mark.requires_simulator,
                pytest.mark.requires_optional("isaaclab"),
            ],
        ),
    ],
    indirect=True,
)
def test_ray_checkpoint_resumes_with_fresh_workers(local_ray, tmp_path):
    from lightning.pytorch.callbacks import Callback

    from nexuml.core.export import export_package, load_inference_package

    if local_ray.cluster_resources().get("GPU") == 1:
        from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo

        spec = isaac_cartpole_ppo(total_frames=32, frames_per_batch=16, num_envs=2)
        spec.training.accelerator = "cpu"
        spec.training.collector = CollectorSpec(
            backend="ray",
            num_envs_per_collector=2,
            env_device="cuda:0",
            policy_device="cuda:0",
            ray=RayCollectorSpec(num_gpus=1),
        )
    else:
        spec = pendulum_ppo(total_frames=32, frames_per_batch=16)
        spec.training.collector = CollectorSpec(backend="ray", num_collectors=2)
    assert spec.interaction is not None and isinstance(spec.training, ReinforcementLearningSpec)
    worker_count = spec.training.collector.num_collectors
    spec.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    spec.training.evaluation = RLEvaluationSpec(episodes=0)
    checkpoint = tmp_path / "partial.ckpt"
    previous_workers = set()

    class Interrupt(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            previous_workers.update(
                worker._actor_id for worker in pl_module._collector.remote_collectors
            )
            trainer.save_checkpoint(checkpoint)
            raise RuntimeError("checkpoint interruption")

    options = {"enable_loggers": False, "enable_progress_bar": False}
    first = NexuSession(spec, log_dir=tmp_path / "first", **options)
    first.trainer.callbacks.append(Interrupt())  # ty: ignore[missing-attribute]
    with pytest.raises(RuntimeError, match="checkpoint interruption"):
        first.fit()
    assert first.lightning_module._collector is None and len(previous_workers) == worker_count
    saved_policy = first.lightning_module.policy.state_dict()
    saved_optimizers = [optimizer.state_dict() for optimizer in first.trainer.optimizers]
    assert first.trainer.global_step == 4

    class Restored(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            assert int(pl_module.frames) == 16 and trainer.global_step == 4
            assert torch.all(batch["next", "policy_version"] == 1)
            assert batch["action"].device.type == "cpu"
            assert torch.isfinite(batch["action_log_prob"]).all()
            workers = {worker._actor_id for worker in pl_module._collector.remote_collectors}
            assert len(workers) == worker_count and workers.isdisjoint(previous_workers)
            torch.testing.assert_close(pl_module.policy.state_dict(), saved_policy, rtol=0, atol=0)
            for optimizer, expected in zip(trainer.optimizers, saved_optimizers, strict=True):
                torch.testing.assert_close(optimizer.state_dict(), expected, rtol=0, atol=0)

    resumed = NexuSession.from_trainer_checkpoint(
        checkpoint, log_dir=tmp_path / "resumed", **options
    )
    resumed.trainer.callbacks.append(Restored())  # ty: ignore[missing-attribute]
    result = resumed.run()
    assert int(result.lightning_module.get_buffer("frames")) == 32
    assert int(result.lightning_module.get_buffer("batches")) == 2
    assert result.trainer.global_step == 8 and result.policy is not None
    assert resumed.lightning_module._collector is None
    assert local_ray.is_initialized()
    package = export_package(result.policy, tmp_path / "policy", trainer=result.trainer)
    reloaded, _, _ = load_inference_package(package)
    torch.testing.assert_close(reloaded.state_dict(), result.policy.state_dict(), rtol=0, atol=0)


@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize("timeout", [False, True])
def test_ray_close_failure_kills_owned_actors_releases_lease_and_is_repeatable(
    monkeypatch, timeout
):
    from nexuml.reinforcement.ray_collector import _GracefulRayCollector

    collector = object.__new__(_GracefulRayCollector)
    workers = [Mock(), Mock()]
    lease = Mock()
    scheme = Mock()
    collector._remote_collectors = workers.copy()
    collector._runtime_lease = lease
    collector._stop_event = Event()
    collector._collection_thread = None
    collector._weight_sync_schemes = {"policy": scheme}
    closing = [worker.shutdown.remote.return_value for worker in workers]
    wait = Mock(return_value=(closing[:1], closing[1:]) if timeout else (closing, []))
    get = Mock(side_effect=RuntimeError("failed shutdown"))
    kill = Mock()
    monkeypatch.setattr("nexuml.reinforcement.ray_collector.ray.wait", wait)
    monkeypatch.setattr("nexuml.reinforcement.ray_collector.ray.get", get)
    monkeypatch.setattr("nexuml.reinforcement.ray_collector.ray.kill", kill)
    error, message = (
        (TimeoutError, "close deadline") if timeout else (RuntimeError, "failed shutdown")
    )
    with pytest.raises(error, match=message):
        collector.shutdown()
    wait.assert_called_once_with(closing, num_returns=2, timeout=30)
    assert [call.args for call in kill.call_args_list] == [(workers[1],), (workers[0],)]
    scheme.shutdown.assert_called_once()
    assert lease.release.called
    assert not collector.remote_collectors
    collector.shutdown()
    assert get.call_count == int(not timeout)
    assert kill.call_count == 2
    scheme.shutdown.assert_called_once()


@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize(
    "collector, frames, error",
    [
        (CollectorSpec(backend="ray", sync=False), 8, "synchronous collectors"),
        (
            CollectorSpec(backend="ray", policy_sync_interval_batches=2),
            8,
            "policy_sync_interval_batches=1",
        ),
        (
            CollectorSpec(backend="ray", env_device="cuda"),
            8,
            "without a GPU reservation requires CPU collectors",
        ),
        (CollectorSpec(backend="ray", num_collectors=2), 3, "frames_per_batch must be divisible"),
    ],
)
def test_unvalidated_ray_topologies_fail_before_allocation(collector, frames, error):
    training = ReinforcementLearningSpec(
        algorithm=PPO(), collector=collector, frames_per_batch=frames
    )
    with pytest.raises(ValueError, match=error):
        build_collector(GymnasiumEnvironment(), training, torch.nn.Linear(3, 1), "cpu")


def test_ray_cuda_learner_needs_explicit_placement():
    collector = CollectorSpec(backend="ray", policy_device="cpu")
    training = ReinforcementLearningSpec(algorithm=PPO(), collector=collector)
    with pytest.raises(ValueError, match="requires explicit Ray placement"):
        build_collector(GymnasiumEnvironment(), training, None, "cuda:0")


def test_ray_placement_config_round_trips_without_changing_default_dump():
    assert "ray" not in CollectorSpec(backend="ray").model_dump()
    config = {
        "address": "head.example:6379",
        "resources": {"rollout": 0.5},
        "label_selector": {"role": "cpu-rollout"},
    }
    spec = CollectorSpec.model_validate({"backend": "ray", "ray": config})
    assert isinstance(spec.ray, RayCollectorSpec)
    assert spec.model_dump()["ray"] == config
    for backend in ("direct", "process"):
        with pytest.raises(ValueError, match="placement requires backend='ray'"):
            CollectorSpec(backend=backend, ray=spec.ray)


@pytest.mark.parametrize(
    "placement",
    [
        {"address": address}
        for address in (
            "local",
            "http://head.example:8265",
            "ray://head.example:10001",
            "user:password@head.example:6379",
            "head.example",
            "head.example:6379/path",
            "head.example:65536",
        )
    ]
    + [
        {"resources": {name: amount}}
        for name, amount in (
            ("CPU", 1),
            ("GPU", 1),
            ("memory", 1),
            ("", 1),
            ("rollout", 0),
            ("rollout", float("inf")),
        )
    ]
    + [
        {"label_selector": {"role": 1}},
        {"num_gpus": 0.5},
        {"num_gpus": 2},
        {"num_gpus": True},
        {"runtime_env": {}},
    ],
)
def test_invalid_ray_placement_config_rejected(placement):
    with pytest.raises(ValueError):
        CollectorSpec.model_validate({"backend": "ray", "ray": placement})


@pytest.mark.parametrize(
    "devices,valid",
    [
        ({"env_device": "cpu"}, False),
        ({"env_device": "cuda:1"}, False),
        ({"policy_device": "auto"}, False),
        ({"storing_device": "cuda:0"}, False),
        ({}, True),
    ],
)
def test_ray_gpu_placement_requires_worker_local_mapping(devices, valid):
    config = {
        "backend": "ray",
        "ray": {"address": "head.example:6379", "num_gpus": 1},
        "env_device": "cuda:0",
        "policy_device": "cuda:0",
        **devices,
    }
    if valid:
        assert CollectorSpec.model_validate(config).model_dump()["ray"]["num_gpus"] == 1
    else:
        with pytest.raises(ValueError, match="Ray GPU collection requires"):
            CollectorSpec.model_validate(config)


@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize(
    "address, env_address, attempted, error, message",
    [
        (None, None, None, ValueError, "requires an initialized Ray runtime or address"),
        ("auto", None, None, ConnectionError, "Could not find any running Ray instance"),
        ("auto", "local", None, ValueError, "not start Local Ray"),
        ("auto", "http://head.example:8265", None, ValueError, "Ray Core host:port"),
        ("auto", "head.example:6379", "head.example:6379", ConnectionError, "unreachable target"),
        ("head.example:6379", "local", "head.example:6379", ConnectionError, "unreachable target"),
    ],
)
def test_ray_placement_never_falls_back_to_local(
    monkeypatch, address, env_address, attempted, error, message
):
    import ray
    from ray._private import services

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    if env_address is not None:
        monkeypatch.setenv("RAY_ADDRESS", env_address)
    monkeypatch.setattr(services, "find_gcs_addresses", lambda: set())
    monkeypatch.setattr(services, "find_bootstrap_address", lambda temp_dir: None)
    assert not ray.is_initialized()
    init = Mock(side_effect=ConnectionError("unreachable target"))
    monkeypatch.setattr(ray, "init", init)
    training = ReinforcementLearningSpec(
        algorithm=PPO(),
        collector=CollectorSpec(backend="ray", ray=RayCollectorSpec(address=address)),
    )
    with pytest.raises(error, match=message):
        build_collector(GymnasiumEnvironment(), training, torch.nn.Linear(3, 1), "cpu")
    if attempted is None:
        init.assert_not_called()
    else:
        init.assert_called_once_with(address=attempted)
    assert not ray.is_initialized()


@pytest.mark.requires_optional("ray")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize("release", ["0.13.0", "0.14.1", "0.15.0"])
def test_ray_private_lifecycle_is_gated_before_worker_allocation(monkeypatch, release):
    from nexuml.reinforcement import ray_collector

    monkeypatch.setattr(ray_collector, "version", lambda name: release)
    monkeypatch.setattr(
        ray_collector.RayCollector, "__init__", lambda *a, **k: pytest.fail("worker allocation")
    )
    with pytest.raises(RuntimeError, match="qualified for TorchRL 0.14.0"):
        ray_collector._GracefulRayCollector()
