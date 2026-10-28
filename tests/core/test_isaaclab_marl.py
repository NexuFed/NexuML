"""CPU fake-native proofs, not Isaac application/asset or learning acceptance."""

import builtins
from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.types import AgentGroupContract, InteractionContract, TensorFieldContract
from nexuml.reinforcement.torchrl_env import _TorchRLEnvironmentAdapter
from nexuml_library.environments import _isaaclab_runtime
from nexuml_library.environments.isaaclab import IsaacBindings, IsaacLabEnvironment


def _definition(*, team_reward=False, nested=False):
    groups = {
        "workers": AgentGroupContract(
            agents=("a", "b"),
            observations={"observation": TensorFieldContract(shape=(2,), dtype="float32")},
            actions={
                "action": TensorFieldContract(
                    shape=(1,), dtype="float32", low=-1, high=1, rate_hz=60
                )
            },
            action_space="continuous",
        ),
        "camera": AgentGroupContract(
            agents=("c",),
            observations={"image_features": TensorFieldContract(shape=(3,), dtype="float32")},
            actions={"choice": TensorFieldContract(shape=(), dtype="int64", num_values=3)},
            action_space="discrete",
        ),
    }
    return IsaacLabEnvironment(
        task="CPU-Fixed-MARL",
        contract=InteractionContract(
            groups=groups,
            action_space="structured",
            control_period_s=1 / 60,
            team_reward=team_reward,
        ),
        bindings=IsaacBindings(
            groups={
                name: IsaacBindings(
                    observations={key: ("policy",) if nested else () for key in group.observations},
                    action=next(iter(group.actions)),
                )
                for name, group in groups.items()
            }
        ),
    )


class _Native:
    def __init__(self, count=3, *, nested=False):
        self.num_envs = count
        self.device = torch.device("cpu")
        self.possible_agents = ["a", "b", "c"]
        self.agents = list(self.possible_agents)
        self.cfg = SimpleNamespace(compute_final_obs=True)
        self.step_dt = 1 / 60
        self.action_spaces = {
            "a": Box(-1, 1, shape=(1,), dtype=np.float32),
            "b": Box(-1, 1, shape=(1,), dtype=np.float32),
            "c": Discrete(3),
        }
        self.observation_spaces: dict[str, Any] = {
            a: Box(-np.inf, np.inf, shape=(3 if a == "c" else 2,), dtype=np.float32)
            for a in self.agents
        }
        self.nested = nested
        if nested:
            self.observation_spaces = {
                a: {"policy": space} for a, space in self.observation_spaces.items()
            }
        self.reset_buf = torch.zeros(count, dtype=torch.bool)
        self.next_reset = torch.zeros_like(self.reset_buf)
        self.terminated = {a: torch.zeros_like(self.reset_buf) for a in self.agents}
        self.truncated = {a: torch.zeros_like(self.reset_buf) for a in self.agents}
        self.rewards = {a: torch.full((count,), float(i + 1)) for i, a in enumerate(self.agents)}
        self.reset_calls = 0
        self.step_calls = []
        self.close_calls = 0
        self.include_final = True
        self.finals = {}
        self.observations = {}

    def _wrap(self, values):
        return {a: {"policy": v} if self.nested else v for a, v in values.items()}

    def reset(self, *, seed=None):
        self.reset_calls += 1
        self.observations = {
            a: torch.full((self.num_envs, 3 if a == "c" else 2), 1000.0)
            for a in self.possible_agents
        }
        return self._wrap(self.observations), {}

    def step(self, actions):
        assert set(actions) == set(self.possible_agents)
        assert actions["a"].shape == actions["b"].shape == (self.num_envs, 1)
        assert actions["c"].shape == (self.num_envs,)
        assert actions["c"].dtype == torch.int64
        self.step_calls.append({a: v.clone() for a, v in actions.items()})
        self.reset_buf.copy_(self.next_reset)
        index = len(self.step_calls)
        self.finals = {
            a: torch.full((self.num_envs, 3 if a == "c" else 2), float(index * 10 + i))
            for i, a in enumerate(self.possible_agents)
        }
        self.observations = {a: v.clone() for a, v in self.finals.items()}
        for value in self.observations.values():
            value[self.reset_buf] *= -1
        finals = self._wrap(self.finals)
        extras = {a: {"final_obs": v} if self.include_final else {} for a, v in finals.items()}
        return (
            self._wrap(self.observations),
            self.rewards,
            self.terminated,
            self.truncated,
            extras,
        )

    def close(self):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _no_native_or_cuda(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0].startswith(("isaaclab", "isaacsim")):
            raise AssertionError(f"CPU boundary tests cannot import {name}")
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU boundary tests cannot initialize/probe CUDA")

    monkeypatch.setattr(builtins, "__import__", guarded)
    for name in ("init", "_lazy_init", "is_available"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def make_runtime(monkeypatch):
    runtimes = []

    def build(count=3, *, definition=None, native=None, nested=False):
        definition = definition or _definition(nested=nested)
        native = native or _Native(count, nested=nested)

        def create(owner, _definition, _context):
            owner.native = native
            owner._marl = owner._direct = True

        monkeypatch.setattr(_isaaclab_runtime, "_create_environment", create)
        runtime = _isaaclab_runtime._IsaacLabRuntime(
            definition,
            EnvironmentBuildContext(purpose="train", device="cpu", num_envs=count),
        )
        runtimes.append(runtime)
        return runtime

    yield build
    for runtime in reversed(runtimes):
        runtime.close()


def _actions(count=3):
    return TensorDict(
        {
            "workers": TensorDict(
                {"action": torch.tensor([0.2, -0.4]).reshape(1, 2, 1).expand(count, 2, 1)},
                batch_size=(count, 2),
            ),
            "camera": TensorDict(
                {"choice": torch.full((count, 1), 2, dtype=torch.int64)}, batch_size=(count, 1)
            ),
        },
        batch_size=(count,),
    )


def test_group_bindings_are_relative_to_each_ordered_agent(make_runtime):
    runtime = make_runtime(nested=True)
    initial = runtime.reset()
    assert initial.batch_size == (3,)
    assert initial["workers"].batch_size == (3, 2)
    assert initial["camera", "image_features"].shape == (3, 1, 3)
    step = runtime.step(_actions())
    assert step.reward is None
    assert torch.equal(runtime.native.step_calls[0]["a"], torch.full((3, 1), 0.2))
    assert torch.equal(runtime.native.step_calls[0]["b"], torch.full((3, 1), -0.4))
    assert step.groups["workers", "reward"][0, :, 0].tolist() == [1, 2]
    assert step.groups["camera", "reward"][0, 0, 0] == 3


def test_partial_agent_ends_keep_raw_facts_and_invalid_suffix_until_actual_reset(make_runtime):
    runtime = make_runtime()
    native = runtime.native
    runtime.reset()
    native.terminated["a"][0] = True
    native.truncated["b"][1] = True
    first = runtime.step(_actions())
    assert not (first.terminated | first.truncated).any()
    assert first.groups["workers", "valid"].all()  # Terminal samples still contribute.
    assert not first.observation["workers", "valid"][0, 0]
    assert first.observation["workers", "valid"][0, 1]
    native.terminated["a"].zero_()
    native.truncated["b"].zero_()
    assert first.groups["workers", "raw_terminated"][0, 0]
    second = runtime.step(_actions())
    assert not second.groups["workers", "raw_terminated"].any()
    assert second.groups["workers", "terminated"][0, 0]
    assert second.groups["workers", "truncated"][1, 1]
    assert not second.groups["workers", "valid"][0, 0]
    assert not second.groups["workers", "valid"][1, 1]
    assert (second.observation["workers", "observation"][0, 0] == 20).all()
    # The native boundary, not an invented any/all reduction, ends every role in row zero.
    native.next_reset[0] = True
    third = runtime.step(_actions())
    assert third.terminated[:, 0].tolist() == [True, False, False]
    assert third.groups["camera", "done"][:, 0, 0].tolist() == [True, False, False]
    assert (third.observation["workers", "observation"][0, 0] == 30).all()
    reset = runtime.reset(mask=torch.tensor([True, False, False]))
    assert native.reset_calls == 1
    assert (reset["workers", "observation"][0, 0] == -30).all()
    assert reset["workers", "valid"][0].all()
    assert not reset["workers", "valid"][1, 1]
    assert (third.observation["workers", "observation"][0, 0] == 30).all()
    with pytest.raises(ValueError, match="full reset or cached"):
        runtime.reset(mask=torch.tensor([True, False, False]))
    native.next_reset.zero_()
    fourth = runtime.step(_actions())
    assert fourth.groups["workers", "valid"][0].all()
    assert not fourth.groups["workers", "valid"][1, 1]
    assert not fourth.groups["workers", "terminated"][0].any()


def test_unanimous_timeout_classifies_actual_boundary_but_does_not_create_one(make_runtime):
    runtime = make_runtime()
    runtime.reset()
    for value in runtime.native.truncated.values():
        value[0:2] = True
    runtime.native.next_reset[1] = True
    transition = runtime.step(_actions())
    assert not transition.terminated.any()
    assert transition.truncated[:, 0].tolist() == [False, True, False]
    assert transition.groups["workers", "raw_truncated"][0].all()


def test_simultaneous_native_failure_and_timeout_preserve_true_termination(make_runtime):
    runtime = make_runtime()
    runtime.reset()
    runtime.native.next_reset[0] = True
    for value in (*runtime.native.terminated.values(), *runtime.native.truncated.values()):
        value[0] = True
    transition = runtime.step(_actions())
    assert transition.terminated[0] and transition.truncated[0]
    assert transition.groups["workers", "terminated"][0].all()
    assert transition.groups["workers", "truncated"][0].all()


def test_missing_pre_reset_finals_rejects_instead_of_using_reset_state(make_runtime):
    runtime = make_runtime()
    runtime.reset()
    runtime.native.next_reset[0] = True
    runtime.native.include_final = False
    with pytest.raises(ValueError, match="pre-reset final_obs"):
        runtime.step(_actions())


@pytest.mark.parametrize("bad", ["shape", "dtype", "nonfinite"])
def test_final_observations_validate_each_agent_before_stacking(make_runtime, bad):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_reset[0] = True

    def wrap(values):
        result = dict(values)
        if values is native.finals:
            if bad == "shape":
                result["a"] = torch.zeros(3, 3)
            elif bad == "dtype":
                result["a"] = result["a"].to(torch.float16)
            else:
                result["a"] = result["a"].clone()
                result["a"][0, 0] = float("nan")
        return result

    native._wrap = wrap
    with pytest.raises(ValueError, match="shape/dtype/device|nonfinite"):
        runtime.step(_actions())


def test_unavailable_privileged_290_state_binding_rejects():
    definition = _definition()
    contract = definition.contract.model_copy(
        update={
            "critic_observations": {"state": TensorFieldContract(shape=(290,), dtype="float32")}
        }
    )
    with pytest.raises(ValueError, match="pre-reset final-state"):
        IsaacLabEnvironment(task="CPU-Handover", contract=contract, bindings=definition.bindings)
    bindings = definition.bindings.model_copy(update={"critic_observations": {"state": ("state",)}})
    with pytest.raises(ValueError, match="pre-reset final-state"):
        IsaacLabEnvironment(task="CPU-Handover", contract=definition.contract, bindings=bindings)


def test_grouped_declaration_cannot_bypass_the_retained_so101_physics_gate():
    definition = _definition()
    with pytest.raises(ValueError, match="validated PhysX"):
        IsaacLabEnvironment(
            task="NexuML-SO101-Joint-Reach",
            physics="newton_mjwarp",
            contract=definition.contract,
            bindings=definition.bindings,
        )


@pytest.mark.parametrize("change", ["agents", "possible_agents", "observations"])
def test_dynamic_membership_rejects(make_runtime, change):
    runtime = make_runtime()
    runtime.reset()
    if change == "observations":
        runtime.native._wrap = lambda values: {a: v for a, v in values.items() if a != "b"}
    else:
        setattr(runtime.native, change, ["b", "a", "c"])
    with pytest.raises(ValueError, match="membership"):
        runtime.step(_actions())


@pytest.mark.parametrize("count", [1, 3])
def test_declared_team_reward_requires_equal_native_signal_not_mean(make_runtime, count):
    runtime = make_runtime(count, definition=_definition(team_reward=True))
    runtime.reset()
    with pytest.raises(ValueError, match="genuinely equal"):
        runtime.step(_actions(count))
    for value in runtime.native.rewards.values():
        value.fill_(7)
    transition = runtime.step(_actions(count))
    assert transition.reward.shape == (count, 1)
    assert (transition.reward == 7).all()
    assert (transition.groups["workers", "reward"] == 7).all()


@pytest.mark.parametrize(
    "change", ["observation_dtype", "action_dtype", "action_shape", "limits", "dt"]
)
def test_native_spaces_and_rates_are_validated_before_reset(make_runtime, change):
    native = _Native()
    if change == "observation_dtype":
        native.observation_spaces["a"] = Box(-1, 1, (2,), dtype=np.float64)
    elif change == "action_dtype":
        native.action_spaces["a"] = Box(-1, 1, (1,), dtype=np.float64)
    elif change == "action_shape":
        native.action_spaces["a"] = Box(-1, 1, (2,), dtype=np.float32)
    elif change == "limits":
        native.action_spaces["a"] = Box(-0.5, 0.5, (1,), dtype=np.float32)
    else:
        native.step_dt *= 2
    with pytest.raises(ValueError, match="mismatch|limits"):
        make_runtime(native=native)
    assert native.reset_calls == 0
    assert native.close_calls == 1


def test_declared_field_rate_and_actual_actions_are_validated(make_runtime):
    definition = _definition()
    group = definition.contract.groups["workers"]
    field = group.actions["action"].model_copy(update={"rate_hz": 30})
    bad_group = group.model_copy(update={"actions": {"action": field}})
    bad_contract = definition.contract.model_copy(
        update={"groups": {**definition.contract.groups, "workers": bad_group}}
    )
    with pytest.raises(ValueError, match="field rate"):
        make_runtime(definition=definition.model_copy(update={"contract": bad_contract}))
    runtime = make_runtime()
    runtime.reset()
    actions = _actions()
    actions["workers", "action"] = torch.full((3, 2, 1), 1.1)
    with pytest.raises(ValueError, match="upper bound"):
        runtime.step(actions)
    assert not runtime.native.step_calls


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("team_reward", [False, True])
def test_real_torchrl_specs_reset_step_rollout_axes_and_physical_frames(
    make_runtime, count, team_reward
):
    from torchrl.envs.utils import check_env_specs

    runtime = make_runtime(count, definition=_definition(team_reward=team_reward))
    if team_reward:
        for value in runtime.native.rewards.values():
            value.fill_(7)
    env = _TorchRLEnvironmentAdapter(runtime)
    try:
        assert env.batch_size == (count,)
        assert env.observation_spec["workers"].shape == (count, 2)
        assert env.full_action_spec["camera", "choice"].shape == (count, 1)
        assert ("reward" in env.full_reward_spec.keys()) is team_reward
        check_env_specs(env)
        initial = env.reset()
        runtime.native.terminated["a"][0] = True
        first = env.step(initial.update(_actions(count)))
        assert first["workers", "valid"].all()
        assert not first["next", "workers", "valid"][0, 0]
        assert not first["next", "done"].any()
        runtime.native.terminated["a"].zero_()
        rollout = env.rollout(4, break_when_any_done=False)
        assert rollout.batch_size == (count, 4)
        assert rollout["workers"].batch_size == (count, 4, 2)
        assert rollout["camera"].batch_size == (count, 4, 1)
        assert rollout.numel() == count * 4
        assert rollout["next", "workers", "reward"].shape == (count, 4, 2, 1)
        assert ("reward" in rollout["next"].keys()) is team_reward
    finally:
        env.close()


def test_torchrl_agent_only_ends_never_reset_native_or_reactivate_inactive_agents(make_runtime):
    runtime = make_runtime()
    runtime.native.terminated["a"][0] = True
    env = _TorchRLEnvironmentAdapter(runtime)
    try:
        rollout = env.rollout(4, break_when_any_done=False)
        assert runtime.native.reset_calls == 1
        assert rollout["workers", "valid"][0, :, 0, 0].tolist() == [True, False, False, False]
        assert rollout["workers", "valid"][0, :, 1, 0].all()
        assert not rollout["next", "done"].any()
        assert rollout["next", "workers", "raw_terminated"][0, :, 0, 0].all()
    finally:
        env.close()


def test_actual_collector_trajectory_ids_follow_native_environment_not_agent_endings(make_runtime):
    from torchrl.collectors import Collector

    runtime = make_runtime()
    native = runtime.native
    native.terminated["a"][0] = True
    env = _TorchRLEnvironmentAdapter(runtime)
    collector = Collector(
        env, frames_per_batch=12, total_frames=12, auto_register_policy_transforms=False
    )
    try:
        batch = next(iter(collector))
        assert batch.batch_size == (3, 4)
        assert not batch["next", "done"].any()
        assert native.reset_calls == 1
        assert batch["workers", "valid"][0, :, 0, 0].tolist() == [True, False, False, False]
        assert batch["workers", "valid"][0, :, 1, 0].all()
        ids = batch["collector", "traj_ids"]
        assert all(row.unique().numel() == 1 for row in ids)
        assert env.done_keys == ["done", "terminated", "truncated"]
    finally:
        collector.shutdown()


@pytest.mark.parametrize("count", [1, 3])
def test_torchrl_native_reset_returns_finals_then_reuses_cached_reset_once(make_runtime, count):
    runtime = make_runtime(count)
    env = _TorchRLEnvironmentAdapter(runtime)
    try:
        initial = env.reset()
        runtime.native.next_reset[0] = True
        transition, following = env.step_and_maybe_reset(initial.update(_actions(count)))
        assert transition["next", "done"][0]
        assert not transition["next", "workers", "raw_terminated"].any()
        assert transition["next", "workers", "done"][0].all()
        assert (transition["next", "workers", "observation"][0, 0] == 10).all()
        assert (following["workers", "observation"][0, 0] == -10).all()
        assert following["workers", "valid"].all()
        assert runtime.native.reset_calls == 1
    finally:
        env.close()


@pytest.mark.parametrize(
    ("marl_config", "marl_env", "grouped", "error"),
    [
        (True, True, True, None),
        (True, False, True, "DirectMARLEnv"),
        (False, True, True, "grouped contract"),
        (True, True, False, "grouped contract"),
    ],
)
def test_native_admission_uses_config_and_environment_types_without_launching_native(
    monkeypatch, tmp_path, marl_config, marl_env, grouped, error
):
    import gymnasium as gym
    from nexuml_library.environments import _isaaclab_tasks

    class ManagerConfig(SimpleNamespace):
        pass

    class DirectConfig(SimpleNamespace):
        pass

    class MARLConfig(SimpleNamespace):
        pass

    class ManagerEnv:
        pass

    class DirectEnv:
        pass

    class MARLEnv:
        pass

    definition = _definition() if grouped else IsaacLabEnvironment()
    cfg = (MARLConfig if marl_config else DirectConfig)(
        sim=SimpleNamespace(device="cuda:0", physics=object()),
        scene=SimpleNamespace(num_envs=3),
    )
    events = []

    @contextmanager
    def launch(*args):
        events.append("launch")
        yield
        events.append("close")

    source = tmp_path / "isaaclab/__init__.py"
    modules = {
        "isaaclab": SimpleNamespace(__file__=str(source)),
        "isaaclab_tasks": SimpleNamespace(),
        "isaaclab.app": SimpleNamespace(launch_simulation=launch),
        "isaaclab.utils": SimpleNamespace(has_kit=lambda: False, to_dict=lambda cfg: vars(cfg)),
        "isaaclab_tasks.utils": SimpleNamespace(parse_env_cfg=lambda *args, **kwargs: cfg),
        "isaaclab_tasks.utils.hydra": SimpleNamespace(),
        "isaaclab.sim": SimpleNamespace(
            SimulationContext=SimpleNamespace(instance=lambda: None, clear_instance=lambda: None)
        ),
        "isaaclab.envs": SimpleNamespace(
            ManagerBasedRLEnvCfg=ManagerConfig,
            DirectRLEnvCfg=DirectConfig,
            DirectMARLEnvCfg=MARLConfig,
            ManagerBasedRLEnv=ManagerEnv,
            DirectRLEnv=DirectEnv,
            DirectMARLEnv=MARLEnv,
        ),
    }
    modules["isaaclab_tasks.utils.hydra"].resolve_presets = lambda *args, **kwargs: cfg
    monkeypatch.setattr(
        _isaaclab_runtime, "sys", SimpleNamespace(version_info=(3, 12), modules=sys.modules)
    )
    monkeypatch.setattr(_isaaclab_runtime, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        _isaaclab_runtime,
        "distribution",
        lambda _: SimpleNamespace(
            read_text=lambda _: json.dumps({"vcs_info": {"commit_id": definition.revision}}),
            locate_file=lambda _: source,
        ),
    )
    monkeypatch.setattr(_isaaclab_tasks, "register_reference_tasks", lambda: None)
    monkeypatch.setattr(gym, "spec", lambda _: SimpleNamespace(id=definition.task))
    monkeypatch.setattr(
        gym,
        "make",
        lambda *args, **kwargs: SimpleNamespace(unwrapped=MARLEnv() if marl_env else DirectEnv()),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    owner = SimpleNamespace(
        device=torch.device("cuda:0"),
        _rendering=False,
        _launcher=None,
        contract=definition.describe(),
    )

    def close():
        if owner._launcher is not None:
            owner._launcher.__exit__(None, None, None)
        _isaaclab_runtime._active_runtime = None

    owner.close = close
    context = EnvironmentBuildContext(purpose="train", device="cuda:0", num_envs=3)
    with definition.job_scope():
        if error is None:
            _isaaclab_runtime._create_environment(owner, definition, context)
            assert owner._marl and owner._direct
        else:
            with pytest.raises(ValueError, match=error):
                _isaaclab_runtime._create_environment(owner, definition, context)
    assert events == (["launch", "close"] if marl_config and grouped else [])
