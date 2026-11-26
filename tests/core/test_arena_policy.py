"""Hermetic Arena policy protocol checks; no native startup or score claims."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast
import json
import signal
import sys

import pytest
import torch
from tensordict import TensorDict
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.compiler import compile
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import export_package
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import LoggingSpec, MLflowSpec
from nexuml_library.evaluation import arena
from nexuml_library.evaluation.arena import (
    ArenaBenchmarkEnvironment,
    ArenaPolicy,
    run_arena_condition,
)
from nexuml_library.environments._isaaclab_so101 import GOALS
from nexuml_library.evaluation.arena_so101 import CONDITIONS, _TARGETS_RAD, so101_arena_evaluation
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_so101_joint_reach_ppo
from test_policy_evaluation_cli import fail_training, native_logging as native_logging


@pytest.fixture
def so101():
    scenario = isaac_so101_joint_reach_ppo()
    definition = scenario.interaction.environment
    contract = definition.describe()
    policy = CompiledPolicy.from_pipeline(compile(scenario), contract).eval()
    return policy, contract, definition.native_bindings()


def native(num_envs=1, period=1 / 30):
    return SimpleNamespace(
        unwrapped=SimpleNamespace(
            num_envs=num_envs,
            step_dt=period,
            single_action_space=SimpleNamespace(shape=(5,), dtype="float32"),
            device="cpu",
        )
    )


def test_arena_bridge_decodes_the_stock_so101_policy_without_remapping(so101):
    policy, contract, bindings = so101
    observations = torch.linspace(-0.35, 0.35, 30).reshape(2, 15)
    action_field = contract.actions["action"].model_copy(update={"description": "SO-101 arm"})
    batch_contract = contract.model_copy(
        update={"num_envs": 2, "actions": {"action": action_field}}
    )
    adapter = ArenaPolicy(policy, batch_contract, bindings, deterministic=True)
    expected = policy(TensorDict({"observation": observations}, batch_size=[2]))["action"]
    action = adapter.get_action(native(2), {"policy": observations})
    assert action.shape == (2, 5)
    assert torch.equal(action, expected)


@pytest.mark.parametrize("mismatch", ["joints", "mode", "period", "processing", "observation"])
def test_arena_bridge_rejects_equal_width_but_different_semantics(so101, mismatch):
    policy, contract, bindings = so101
    action = contract.actions["action"]
    observation = contract.observations["observation"]
    if mismatch == "joints":
        assert action.physical is not None
        action = action.model_copy(
            update={
                "physical": action.physical.model_copy(
                    update={"joints": action.physical.joints[::-1]}
                )
            }
        )
    if mismatch == "mode":
        assert action.physical is not None
        action = action.model_copy(
            update={
                "physical": action.physical.model_copy(update={"control_mode": "relative_position"})
            }
        )
    if mismatch == "period":
        contract = contract.model_copy(update={"control_period_s": 1 / 60})
        action = action.model_copy(update={"rate_hz": 60.0})
    if mismatch == "observation":
        observation = observation.model_copy(update={"description": "reordered arm state"})
    if mismatch == "processing":
        bindings = bindings.model_copy(update={"processing": "raw"})
    contract = contract.model_copy(
        update={"actions": {"action": action}, "observations": {"observation": observation}}
    )
    with pytest.raises(ValueError, match="Evaluation|Arena"):
        ArenaPolicy(policy, contract, bindings, deterministic=True)


def test_arena_bridge_rejects_wrong_native_clock_and_missing_tensors(so101):
    policy, contract, bindings = so101
    adapter = ArenaPolicy(policy, contract, bindings, deterministic=True)
    with pytest.raises(ValueError, match="control period"):
        adapter.get_action(native(period=1 / 60), {"policy": torch.zeros(1, 15)})
    with pytest.raises(ValueError, match="missing binding"):
        adapter.get_action(native(), {})
    with pytest.raises(ValueError, match="declared tensor"):
        adapter.get_action(native(), {"policy": torch.zeros(15)})


def test_arena_bridge_resets_only_the_streams_requested_by_arena(so101):
    _, contract, bindings = so101

    class Stateful:
        processor = object()

        def __init__(self):
            self.contract = contract

        def initialize_state(self, batch, *, device):
            return TensorDict({"counter": torch.zeros(batch, device=device)}, batch_size=batch)

        def step(self, observation, state, *, deterministic):
            assert not deterministic
            counter = state["counter"] + 1
            return (
                TensorDict({"action": counter[:, None].expand(-1, 5)}, batch_size=[2]),
                TensorDict({"counter": counter}, batch_size=[2], device=state.device),
            )

        def reset_state(self, state, mask):
            assert torch.equal(mask, torch.tensor([False, True]))
            return state.masked_fill(mask, 0)

    adapter = ArenaPolicy(
        cast(CompiledPolicy, Stateful()),
        contract.model_copy(update={"num_envs": 2}),
        bindings.model_copy(update={"processing": "raw"}),
        deterministic=False,
    )
    obs = {"policy": torch.zeros(2, 15)}
    assert torch.equal(adapter.get_action(native(2), obs)[:, 0], torch.tensor([1.0, 1.0]))
    adapter.reset(env_ids=torch.tensor([1]))
    assert torch.equal(adapter.get_action(native(2), obs)[:, 0], torch.tensor([2.0, 1.0]))
    adapter.reset()
    assert torch.equal(adapter.get_action(native(2), obs)[:, 0], torch.tensor([1.0, 1.0]))


@pytest.mark.parametrize("records", ["valid", "duplicate", "unknown_success", "missing"])
def test_arena_condition_uses_native_runner_and_rejects_incomplete_results(
    monkeypatch, so101, tmp_path, records
):
    policy, contract, bindings = so101
    policy.train()
    bridge = ArenaPolicy(policy, contract, bindings, deterministic=True)
    env = native()
    env.unwrapped.cfg = SimpleNamespace(seed=13)
    recorder = env.unwrapped.episode_recorder = SimpleNamespace()
    recorder.set_job_name = lambda name: setattr(recorder, "name", name)
    recorder.set_output_path = lambda path: setattr(recorder, "path", path)
    calls = []

    def rollout(environment, adapter, *, num_steps, num_episodes):
        assert environment is env and adapter is bridge
        assert num_steps is None and num_episodes == 2
        assert not policy.training and torch.is_inference_mode_enabled()
        adapter.reset()
        assert adapter.get_action(environment, {"policy": torch.zeros(1, 15)}).shape == (1, 5)
        row = dict(job_name=recorder.name, env_id=0, seed=13, episode_length=120)
        rows = [
            {**row, "episode_in_env": 0, "success": True},
            {**row, "episode_in_env": 1, "success": False, "variations": {"target": "left"}},
        ]
        if records == "missing":
            rows.pop()
        elif records == "duplicate":
            rows[1]["episode_in_env"] = 0
        elif records == "unknown_success":
            rows[1]["success"] = None
        recorder.path.write_text("\n".join(json.dumps(value) for value in rows))
        calls.append(rows)

    monkeypatch.setitem(
        sys.modules,
        "isaaclab_arena.evaluation.policy_runner",
        SimpleNamespace(rollout_policy=rollout),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_arena.recording.episode_results",
        SimpleNamespace(
            read_episode_records=lambda p: [json.loads(s) for s in p.read_text().splitlines()]
        ),
    )

    def run():
        return run_arena_condition(
            env, bridge, condition="held-out", seed=13, episodes=2, output=tmp_path / "attempt"
        )

    if records == "valid":
        assert run() == calls[0]
    else:
        with pytest.raises(ValueError, match="Arena (returned|completed)"):
            run()
    assert policy.training and bridge.state is None
    with pytest.raises(FileExistsError):
        run()


@pytest.mark.requires_optional("torchrl")
def test_public_evaluate_matches_package_and_checkpoint_at_the_suite_boundary(
    so101, native_logging, monkeypatch, tmp_path
):
    from hashlib import sha256

    import lightning as L

    from nexuml.training.reinforcement import NexuRLLightningModule

    _, contract, bindings = so101
    module = NexuRLLightningModule(isaac_so101_joint_reach_ppo())
    module.frames.fill_(1024)
    checkpoint = tmp_path / "trainer.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": L.__version__,
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
        },
        checkpoint,
    )
    package = export_package(module.policy, tmp_path / "policy", checkpoint_path=checkpoint)
    config = tmp_path / "evaluation.yaml"
    scenario = so101_arena_evaluation()
    scenario.logging = LoggingSpec(mlflow=MLflowSpec())
    ResolvedConfig.from_scenario(scenario).save(config)
    actions = []

    def run(suite, loaded, spec):
        assert suite.describe() == loaded.contract == contract
        assert suite.conditions == CONDITIONS
        assert spec.episodes == 3 and spec.seed is None
        bridge = ArenaPolicy(loaded, contract, bindings, deterministic=spec.deterministic)
        observations = torch.linspace(-0.25, 0.25, 15).reshape(1, 15)
        actions.append(bridge.get_action(native(), {"policy": observations}))
        metrics = {"eval/rl/episodes": 3.0, "eval/rl/task_successes": 1.0}
        return metrics, {"suite": suite.suite, "completed_episodes": spec.episodes}

    monkeypatch.setattr(ArenaBenchmarkEnvironment, "run_benchmark", run)
    monkeypatch.setattr(L.Trainer, "__init__", fail_training)
    monkeypatch.setattr("nexuml.training.reinforcement.build_collector", fail_training)
    for args in ([str(package)], ["--trainer-checkpoint", str(checkpoint)]):
        result = CliRunner().invoke(app, ["evaluate", *args, "--config", str(config)])
        assert result.exit_code == 0, (result.output, result.exception)
    reports = [
        json.loads(data)
        for _, data, name in native_logging.artifacts
        if name == "evaluation-episodes.json"
    ]
    assert len(reports) == 2
    for report in reports:
        assert report["suite"] == "so101-joint-reach" and report["completed_episodes"] == 3
        assert report["source_training_frames"] == 1024
    assert reports[0]["source_policy_files_sha256"] == reports[1]["source_policy_files_sha256"]
    assert reports[1]["source_checkpoint_sha256"] == sha256(checkpoint.read_bytes()).hexdigest()
    assert torch.equal(actions[0], actions[1])
    assert native_logging.loggers[0].statuses == ["success", "success"]


def test_arena_targets_are_disjoint_from_stock_training_targets():
    assert CONDITIONS.keys() == _TARGETS_RAD.keys()
    assert len(set(_TARGETS_RAD.values())) == len(CONDITIONS)
    assert not set(GOALS).intersection(_TARGETS_RAD.values())


def test_runtime_admission_allows_compatible_minor_series_without_exact_commit_pins(monkeypatch):
    versions = {
        "isaaclab-arena": "0.4.1",
        "isaaclab": "3.0.2",
        "nvidia-curobo": "0.9.0",
        "isaacsim": "6.2.1.0",
    }
    monkeypatch.setattr(arena, "version", versions.__getitem__)
    monkeypatch.setattr(arena, "_source_revision", lambda p: "git-id" if p == "isaaclab" else None)
    monkeypatch.setattr(arena, "find_spec", lambda _: object())
    assert arena._runtime_identity() == {
        "arena": {"version": "0.4.1", "revision": None},
        "isaac_lab": {"version": "3.0.2", "revision": "git-id"},
        "curobo": {"version": "0.9.0", "revision": None},
        "isaac_sim": {"version": "6.2.1.0", "revision": None},
    }
    monkeypatch.setattr(arena, "find_spec", lambda _: None)
    with pytest.raises(RuntimeError, match="no curobo import package"):
        arena._runtime_identity()
    versions["isaaclab"] = "2.3.0"
    with pytest.raises(RuntimeError, match="requires Isaac Lab 3.0"):
        arena._runtime_identity()


@pytest.mark.parametrize("cancelled", [False, True])
def test_named_suite_preserves_denominators_and_actual_failure(so101, monkeypatch, cancelled):
    policy, _, _ = so101
    scenario = so101_arena_evaluation()
    assert scenario.interaction is not None and scenario.policy_evaluation is not None
    suite = scenario.interaction.evaluation_environment
    assert isinstance(suite, ArenaBenchmarkEnvironment)
    events = []

    @contextmanager
    def launch(_, args):
        assert args["headless"] and args["visualizer"] == ["none"] and args["livestream"] == 0
        events.append("launch")
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            yield
        finally:
            events.append("release")

    def build(_, *, condition, seed, device):
        assert device == "cuda:0" and seed == CONDITIONS[condition]
        events.append(f"build-{condition}")
        return condition, suite.contract

    def condition(env, _, *, condition, seed, episodes, output):
        assert env == condition and episodes == 1
        if cancelled:
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
        return [
            {
                "seed": seed,
                "episode_length": 50,
                "success": condition == "near_home",
                "failure_trace": [{"step": 50, "joint_error_rad": 0.1}],
            }
        ]

    monkeypatch.delenv("LIVESTREAM", raising=False)
    monkeypatch.setattr(
        policy, "parameters", lambda: iter([SimpleNamespace(device=torch.device("cuda:0"))])
    )
    monkeypatch.setattr(arena, "_runtime_identity", lambda: {"arena": "verified-fixture"})
    monkeypatch.setattr(type(suite.factory), "build", build)
    monkeypatch.setattr(arena, "run_arena_condition", condition)
    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(launch_simulation=launch))
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_arena.evaluation.resource_cleanup",
        SimpleNamespace(close_environment=lambda env: events.append(f"close-{env}")),
    )
    previous = signal.getsignal(signal.SIGTERM)
    if cancelled:
        with pytest.raises(SystemExit) as exit_status, suite.job_scope():
            suite.run_benchmark(policy, scenario.policy_evaluation)
        assert exit_status.value.code == 143
        assert events == ["launch", "build-near_home", "close-near_home", "release"]
        assert signal.getsignal(signal.SIGTERM) is previous
        return

    with suite.job_scope():
        metrics, report = suite.run_benchmark(policy, scenario.policy_evaluation)
        assert events == [
            "launch",
            "build-near_home",
            "close-near_home",
            "build-left",
            "close-left",
            "build-right",
            "close-right",
        ]
    assert events[-1] == "release"
    assert signal.getsignal(signal.SIGTERM) is previous
    assert metrics["eval/rl/task_success_rate"] == 1 / 3
    assert [row["seed"] for row in report["episodes"]] == list(CONDITIONS.values())
    assert report["selected_failure"]["condition"] == "left"
    assert report["selected_failure"]["episode_id"] == 1
    assert report["selected_failure"]["trace"] == [{"step": 50, "joint_error_rad": 0.1}]
    assert all("failure_trace" not in row for row in report["episodes"])
