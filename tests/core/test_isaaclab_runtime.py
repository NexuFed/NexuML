"""CPU-only proofs of Isaac transition normalization and owned lifetime."""

import asyncio
import builtins
import traceback
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext
from nexuml_library.environments import _isaaclab_runtime
from nexuml_library.environments.isaaclab import IsaacLabEnvironment


class _FakeLauncher:
    def __init__(self, events, *, close_error=None):
        self.events = events
        self.close_error = close_error
        self.enter_calls = 0
        self.exit_calls = []

    def __enter__(self):
        self.enter_calls += 1
        self.events.append("launcher.enter")
        return self

    def __exit__(self, *args):
        self.exit_calls.append(args)
        self.events.append("launcher.exit")
        if self.close_error is not None:
            raise self.close_error


class _FakeNative:
    def __init__(self, num_envs=3, *, close_error=None):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.single_observation_space = {"policy": SimpleNamespace(shape=(4,))}
        self.single_action_space = SimpleNamespace(shape=(1,))
        self.step_dt = 1 / 60
        self.cfg = SimpleNamespace(compute_final_obs=True)
        self.term = SimpleNamespace(
            cfg=SimpleNamespace(scale=100.0, clip=None),
            IO_descriptor=SimpleNamespace(joint_names=["slider_to_cart"], offset=0.0),
        )
        self.action_manager = SimpleNamespace(active_terms=["joint_effort"], get_term=self.get_term)
        self.observation = torch.zeros(num_envs, 4)
        self.final_observation = torch.zeros(num_envs, 4)
        self.reward = torch.zeros(num_envs)
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.truncated = torch.zeros(num_envs, dtype=torch.bool)
        self.next_terminated = [False] * num_envs
        self.next_truncated = [False] * num_envs
        self.include_final_obs = True
        self.reset_calls = []
        self.step_calls = []
        self.events = []
        self.close_error = close_error
        self.close_calls = 0

    def get_term(self, name):
        assert name == "joint_effort"
        self.events.append("action.validate")
        return self.term

    def reset(self, *, seed=None, env_ids=None):
        assert env_ids is not None
        self.reset_calls.append((seed, env_ids.clone()))
        self.events.append("native.reset")
        values = torch.arange(self.num_envs * 4, dtype=torch.float32).reshape(self.num_envs, 4)
        self.observation[env_ids] = values[env_ids] + 1000 * len(self.reset_calls)
        self.final_observation.fill_(-999)
        self.reward.fill_(-777)
        self.terminated.zero_()
        self.truncated.zero_()
        return {"policy": self.observation}, {}

    def step(self, action):
        assert action.shape == (self.num_envs, 1)
        assert action.dtype == torch.float32 and action.device == self.device
        self.step_calls.append(action.clone())
        self.events.append("native.step")
        index = len(self.step_calls)
        values = torch.arange(self.num_envs * 4, dtype=torch.float32).reshape(self.num_envs, 4)
        self.final_observation.copy_(values + 100 * index)
        self.observation.copy_(self.final_observation)
        self.reward.copy_(torch.arange(self.num_envs, dtype=torch.float32) + index + 0.5)
        self.terminated.copy_(torch.tensor(self.next_terminated))
        self.truncated.copy_(torch.tensor(self.next_truncated))
        done = self.terminated | self.truncated
        self.observation[done] = values[done] - 100 * index
        info = {"final_obs": {"policy": self.final_observation}} if self.include_final_obs else {}
        return {"policy": self.observation}, self.reward, self.terminated, self.truncated, info

    def close(self):
        self.close_calls += 1
        self.events.append("native.close")
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture(autouse=True)
def forbid_native_and_cuda(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root.startswith(("isaaclab", "isaacsim")) or root in {"omni", "pxr"}:
            raise AssertionError(f"CPU tests must not import {name}")
        return original_import(name, *args, **kwargs)

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("CPU tests must not probe or initialize CUDA")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for name in ("is_available", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden_cuda)


@pytest.fixture
def make_runtime(monkeypatch):
    runtimes = []

    def build(
        native=None,
        *,
        num_envs=3,
        launcher=None,
        creation_error=None,
        attach_native=True,
        device="cpu",
    ):
        if native is None:
            native = _FakeNative(num_envs)
        if launcher is None:
            launcher = _FakeLauncher(native.events)
        definition = IsaacLabEnvironment()
        context = EnvironmentBuildContext(
            purpose="eval" if num_envs == 1 else "train",
            device=device,
            num_envs=num_envs,
            seed=17,
        )

        def create_environment(owner, received_definition, received_context):
            assert received_definition is definition
            assert received_context is context
            owner._launcher = launcher
            launcher.__enter__()
            if attach_native:
                owner.native = native
            if creation_error is not None:
                raise creation_error

        monkeypatch.setattr(_isaaclab_runtime, "_create_environment", create_environment)
        runtime = _isaaclab_runtime._IsaacLabRuntime(definition, context)
        runtimes.append(runtime)
        return runtime

    yield build
    for runtime in reversed(runtimes):
        runtime.close()


def _actions(num_envs=3):
    batch = () if num_envs == 1 else (num_envs,)
    return TensorDict({"action": torch.full((*batch, 1), 0.25)}, batch_size=batch)


def test_constructor_owns_one_validated_native_batch(make_runtime):
    runtime = make_runtime()
    assert runtime.native.events == ["launcher.enter", "action.validate"]
    assert runtime._launcher.enter_calls == 1
    assert runtime.native.reset_calls == []
    assert runtime.contract.num_envs == 3
    assert runtime.contract.observations["observation"].shape == (4,)
    assert runtime.contract.actions["action"].shape == (1,)
    assert runtime.device == torch.device("cpu")
    assert runtime.render_fps == pytest.approx(60)
    reset = runtime.reset(seed=29)
    assert reset.batch_size == (3,)
    assert reset["observation"].shape == (3, 4)
    assert runtime.native.reset_calls[0][0] == 29
    assert torch.equal(runtime.native.reset_calls[0][1], torch.arange(3))


def test_unindexed_cuda_alias_matches_native_cuda_zero_without_initializing_cuda(make_runtime):
    native = _FakeNative()
    native.device = torch.device("cuda:0")
    runtime = make_runtime(native, device="cuda")
    assert runtime.device == torch.device("cuda:0")
    assert native.events == ["launcher.enter", "action.validate"]


@pytest.mark.parametrize(
    "terminated,truncated",
    [
        pytest.param([False] * 3, [False] * 3, id="none-done"),
        pytest.param([True, False, False], [False] * 3, id="partial-termination"),
        pytest.param([False] * 3, [False, True, False], id="partial-truncation"),
        pytest.param([True, False, False], [True, False, False], id="partial-both"),
        pytest.param([True] * 3, [False] * 3, id="all-termination"),
        pytest.param([False] * 3, [True] * 3, id="all-truncation"),
        pytest.param([True] * 3, [True] * 3, id="all-both"),
        pytest.param([True, False, True], [False, True, True], id="all-mixed"),
    ],
)
def test_same_step_truth_table_keeps_final_and_reset_states_separate(
    make_runtime, terminated, truncated
):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_terminated, native.next_truncated = terminated, truncated
    transition = runtime.step(_actions())
    expected_final = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100
    done = torch.tensor(terminated) | torch.tensor(truncated)
    expected_next = expected_final.clone()
    expected_next[done] -= 200
    assert transition.observation.batch_size == (3,)
    assert torch.equal(transition.observation["observation"], expected_final)
    assert (
        transition.reward.shape
        == transition.terminated.shape
        == transition.truncated.shape
        == (3, 1)
    )
    assert torch.equal(transition.reward, torch.tensor([[1.5], [2.5], [3.5]]))
    assert torch.equal(transition.terminated[:, 0], torch.tensor(terminated))
    assert torch.equal(transition.truncated[:, 0], torch.tensor(truncated))
    assert transition.reward.dtype == torch.float32
    assert transition.terminated.dtype == transition.truncated.dtype == torch.bool
    assert transition.observation.device == transition.reward.device == torch.device("cpu")
    next_observation = runtime.reset(mask=done)["observation"]
    assert torch.equal(next_observation, expected_next)
    assert torch.equal(next_observation[~done], expected_final[~done])
    assert len(native.reset_calls) == 1, "acknowledging auto-reset must not reset physics twice"
    assert torch.equal(native.step_calls[0], _actions()["action"])


def test_pending_rows_can_be_acknowledged_separately_without_disturbing_continuing_rows(
    make_runtime,
):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_terminated = [True, True, False]
    transition = runtime.step(_actions())
    cached = native.observation.clone()
    for mask in ([True, False, False], [False, True, False]):
        observation = runtime.reset(mask=torch.tensor(mask).reshape(3, 1))
        assert torch.equal(observation["observation"], cached)
    assert len(native.reset_calls) == 1
    assert torch.equal(transition.observation["observation"][2], cached[2])


def test_explicit_masked_reset_only_resets_requested_nonpending_rows(make_runtime):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_terminated = [True, False, False]
    runtime.step(_actions())
    before = native.observation.clone()
    observation = runtime.reset(seed=91, mask=torch.tensor([True, True, False]))["observation"]
    assert len(native.reset_calls) == 2
    assert native.reset_calls[-1][0] == 91
    assert torch.equal(native.reset_calls[-1][1], torch.tensor([1]))
    assert torch.equal(observation[[0, 2]], before[[0, 2]])
    assert torch.equal(observation[1], torch.tensor([2004.0, 2005.0, 2006.0, 2007.0]))
    # Once acknowledged, another explicit request starts a new episode.
    runtime.reset(mask=torch.tensor([True, False, False]))
    assert torch.equal(native.reset_calls[-1][1], torch.tensor([0]))


@pytest.mark.parametrize("pending", [False, True])
def test_explicit_full_reset_always_calls_native_even_after_auto_reset(make_runtime, pending):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_truncated = [pending] * 3
    runtime.step(_actions())
    for seed in (101, 102):
        reset = runtime.reset(seed=seed)
        assert native.reset_calls[-1][0] == seed
        assert torch.equal(native.reset_calls[-1][1], torch.arange(3))
        assert torch.equal(reset["observation"], native.observation)
    assert len(native.reset_calls) == 3


@pytest.mark.parametrize(
    "terminated,truncated", [(False, False), (True, False), (False, True), (True, True)]
)
def test_singleton_evaluation_unbatches_outputs_and_rebatches_actions(
    make_runtime, terminated, truncated
):
    runtime = make_runtime(num_envs=1)
    initial = runtime.reset()
    assert initial.batch_size == ()
    assert initial["observation"].shape == (4,)
    native = runtime.native
    native.next_terminated, native.next_truncated = [terminated], [truncated]
    transition = runtime.step(_actions(1))
    assert transition.observation.batch_size == ()
    assert transition.observation["observation"].shape == (4,)
    assert torch.equal(transition.observation["observation"], torch.arange(4) + 100)
    assert (
        transition.reward.shape == transition.terminated.shape == transition.truncated.shape == (1,)
    )
    assert transition.terminated.item() is terminated
    assert transition.truncated.item() is truncated
    assert torch.equal(native.step_calls[0], torch.tensor([[0.25]]))
    next_observation = runtime.reset(mask=torch.tensor(terminated or truncated))
    assert next_observation.batch_size == ()
    assert torch.equal(next_observation["observation"], native.observation[0])
    assert len(native.reset_calls) == 1


def test_returned_buffers_survive_later_native_steps_and_resets(make_runtime):
    runtime = make_runtime()
    initial = runtime.reset()
    expected_initial = initial["observation"].clone()
    native = runtime.native
    native.next_terminated = [True, False, False]
    native.next_truncated = [False, True, False]
    first = runtime.step(_actions())
    expected_observation = first.observation["observation"].clone()
    expected_reward = first.reward.clone()
    expected_terminated, expected_truncated = first.terminated.clone(), first.truncated.clone()
    cached = runtime.reset(mask=torch.tensor([True, True, False]))
    expected_cached = cached["observation"].clone()
    native.next_terminated = native.next_truncated = [False] * 3
    runtime.step(_actions())
    runtime.reset()
    assert torch.equal(initial["observation"], expected_initial)
    assert torch.equal(cached["observation"], expected_cached)
    assert torch.equal(first.observation["observation"], expected_observation)
    assert torch.equal(first.reward, expected_reward)
    assert torch.equal(first.terminated, expected_terminated)
    assert torch.equal(first.truncated, expected_truncated)


def test_mutating_returned_reset_state_cannot_corrupt_runtime_cache(make_runtime):
    runtime = make_runtime()
    initial = runtime.reset()
    expected = initial["observation"].clone()
    initial["observation"].fill_(-123)
    cached = runtime.reset(mask=torch.zeros(3, dtype=torch.bool))
    assert torch.equal(cached["observation"], expected)
    cached["observation"].fill_(-456)
    assert torch.equal(
        runtime.reset(mask=torch.zeros(3, dtype=torch.bool))["observation"], expected
    )
    assert len(runtime.native.reset_calls) == 1


@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize(
    "terminated,truncated",
    [
        ([False] * 3, [False] * 3),
        ([True, False, False], [False, True, False]),
        ([True] * 3, [False] * 3),
        ([False] * 3, [True] * 3),
        ([True] * 3, [True] * 3),
    ],
)
def test_generic_adapter_handoff_and_exact_two_step_gae(make_runtime, terminated, truncated):
    from torchrl.objectives.value.functional import generalized_advantage_estimate

    from nexuml.reinforcement.torchrl_env import _TorchRLEnvironmentAdapter

    runtime = make_runtime()
    native = runtime.native
    env = _TorchRLEnvironmentAdapter(runtime)
    try:
        root = env.reset()
        native.next_terminated, native.next_truncated = terminated, truncated
        root["action"] = _actions()["action"]
        first, next_root = env.step_and_maybe_reset(root)
        current_values = [root["observation"][:, :1].clone()]
        current_values.append(next_root["observation"][:, :1].clone())
        assert len(native.reset_calls) == 1, "all-done and partial masks must consume the cache"
        torch.testing.assert_close(next_root["observation"], native.observation)
        final = first["next", "observation"].clone()
        native.next_terminated = native.next_truncated = [False] * 3
        next_root["action"] = _actions()["action"]
        second, _ = env.step_and_maybe_reset(next_root)
        assert torch.equal(first["next", "observation"], final)

        rollout = torch.stack([first, second], dim=1)
        assert isinstance(rollout, TensorDict)
        assert rollout.batch_size == (3, 2) and rollout.numel() == 6
        value = torch.stack(current_values, dim=1)
        next_value = rollout["next", "observation"][..., :1]
        reward = rollout["next", "reward"]
        done, term = rollout["next", "done"], rollout["next", "terminated"]
        assert isinstance(next_value, torch.Tensor)
        assert isinstance(reward, torch.Tensor)
        assert isinstance(done, torch.Tensor)
        assert isinstance(term, torch.Tensor)
        advantage, target = generalized_advantage_estimate(
            gamma=0.9,
            lmbda=0.95,
            state_value=value,
            next_state_value=next_value,
            reward=reward,
            done=done,
            terminated=term,
        )
        delta = reward + 0.9 * (~term) * next_value - value
        expected = delta.clone()
        expected[:, 0] += 0.9 * 0.95 * (~done[:, 0]) * delta[:, 1]
        torch.testing.assert_close(advantage, expected)
        torch.testing.assert_close(target, expected + value)
        timeout = torch.tensor(truncated) & ~torch.tensor(terminated)
        if timeout.any():
            torch.testing.assert_close(
                target[timeout, 0], reward[timeout, 0] + 0.9 * final[timeout, :1]
            )
            wrong = reward[timeout, 0] + 0.9 * current_values[1][timeout]
            assert not torch.allclose(target[timeout, 0], wrong)
    finally:
        env.close()


@pytest.mark.parametrize("done", [False, True])
def test_final_observations_are_required_only_when_a_row_finishes(make_runtime, done):
    runtime = make_runtime()
    runtime.reset()
    runtime.native.include_final_obs = False
    runtime.native.next_truncated = [False, done, False]
    if done:
        with pytest.raises(ValueError, match="pre-reset final_obs"):
            runtime.step(_actions())
    else:
        assert torch.equal(
            runtime.step(_actions()).observation["observation"], runtime.native.observation
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("num_envs", 4, "dimensions/device/control period"),
        ("device", "meta", "dimensions/device/control period"),
        ("single_observation_space", {"policy": SimpleNamespace(shape=(5,))}, "dimensions/device"),
        ("single_action_space", SimpleNamespace(shape=(2,)), "dimensions/device"),
        ("step_dt", 1 / 30, "control period"),
        ("cfg", SimpleNamespace(compute_final_obs=False), "final-state settings"),
        ("active_terms", ["other_term"], "action terms"),
        ("joint_names", ["wrong_joint"], "joint order/scale/offset/clipping"),
        ("offset", 0.1, "joint order/scale/offset/clipping"),
        ("scale", 1.0, "joint order/scale/offset/clipping"),
        ("clip", (-1.0, 1.0), "joint order/scale/offset/clipping"),
    ],
)
def test_constructor_rejects_native_contract_mismatches_and_closes_owners(
    make_runtime, field, value, match
):
    native = _FakeNative()
    target = native
    if field == "active_terms":
        target = native.action_manager
    elif field in {"joint_names", "offset"}:
        target = native.term.IO_descriptor
    elif field in {"scale", "clip"}:
        target = native.term.cfg
    setattr(target, field, value)
    launcher = _FakeLauncher(native.events)
    with pytest.raises(ValueError, match=match):
        make_runtime(native, launcher=launcher)
    assert native.close_calls == 1 and len(launcher.exit_calls) == 1
    assert native.reset_calls == native.step_calls == []
    assert native.events[-2:] == ["native.close", "launcher.exit"]


@pytest.mark.parametrize(
    "value,match",
    [
        (torch.zeros(3, 5), "shape/dtype/device"),
        (torch.zeros(3, 4, dtype=torch.float64), "shape/dtype/device"),
        (torch.empty(3, 4, device="meta"), "shape/dtype/device"),
        (torch.full((3, 4), float("nan")), "finite"),
        (torch.full((3, 4), float("inf")), "finite"),
    ],
)
@pytest.mark.parametrize("source", ["reset", "step", "final_obs"])
def test_native_observation_shape_dtype_device_and_finiteness_are_validated(
    make_runtime, monkeypatch, value, match, source
):
    runtime = make_runtime()
    native = runtime.native
    if source == "reset":
        monkeypatch.setattr(native, "reset", lambda **kwargs: ({"policy": value}, {}))
        with pytest.raises(ValueError, match=match):
            runtime.reset()
    else:
        runtime.reset()
        original_step = native.step

        def step(action):
            observations, reward, terminated, truncated, info = original_step(action)
            if source == "step":
                observations = {"policy": value}
            else:
                terminated[0] = True
                info["final_obs"] = {"policy": value}
            return observations, reward, terminated, truncated, info

        monkeypatch.setattr(native, "step", step)
        with pytest.raises(ValueError, match=match):
            runtime.step(_actions())


@pytest.mark.parametrize("field", ["reward", "terminated", "truncated"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_native_transition_signal_shape_dtype_and_device_are_validated(
    make_runtime, monkeypatch, field, invalid
):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    original_step = native.step

    def step(action):
        result = list(original_step(action))
        index = {"reward": 1, "terminated": 2, "truncated": 3}[field]
        value = result[index]
        if invalid == "shape":
            value = value.reshape(3, 1)
        elif invalid == "dtype":
            value = value.to(torch.float64 if field == "reward" else torch.int64)
        else:
            value = value.to("meta")
        result[index] = value
        return tuple(result)

    monkeypatch.setattr(native, "step", step)
    with pytest.raises(ValueError, match=rf"{field} shape/dtype/device"):
        runtime.step(_actions())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_native_reward_must_be_finite(make_runtime, monkeypatch, value):
    runtime = make_runtime()
    runtime.reset()
    original_step = runtime.native.step

    def step(action):
        result = original_step(action)
        result[1][0] = value
        return result

    monkeypatch.setattr(runtime.native, "step", step)
    with pytest.raises(ValueError, match="rewards must be finite"):
        runtime.step(_actions())


@pytest.mark.parametrize(
    "value,batch,match",
    [
        (torch.zeros(3, 2), (3,), "must have shape"),
        (torch.zeros(3, 1, dtype=torch.float64), (3,), "must have dtype"),
        (torch.zeros(1), (), "native batch and device"),
        (torch.full((3, 1), float("nan")), (3,), "nonfinite"),
        (torch.full((3, 1), float("inf")), (3,), "upper bound"),
        (torch.full((3, 1), -float("inf")), (3,), "lower bound"),
        (torch.full((3, 1), 1.01), (3,), "upper bound"),
        (torch.full((3, 1), -1.01), (3,), "lower bound"),
        (None, (3,), "missing required field"),
    ],
)
def test_invalid_actions_are_rejected_before_native_step(make_runtime, value, batch, match):
    runtime = make_runtime()
    runtime.reset()
    actions = TensorDict({"action": value} if value is not None else {}, batch_size=batch)
    with pytest.raises(ValueError, match=match):
        runtime.step(actions)
    assert runtime.native.step_calls == []


def test_actions_on_wrong_device_fail_before_tensor_computation(make_runtime):
    runtime = make_runtime()
    runtime.reset()
    actions = TensorDict({"action": torch.empty(3, 1, device="meta")}, batch_size=(3,))
    with pytest.raises(ValueError, match="native batch and device"):
        runtime.step(actions)
    assert runtime.native.step_calls == []


@pytest.mark.parametrize("mask", [torch.ones(3), torch.ones(2, dtype=torch.bool)])
def test_invalid_reset_masks_do_not_call_native_reset(make_runtime, mask):
    runtime = make_runtime()
    runtime.reset()
    with pytest.raises(ValueError, match="one boolean per environment"):
        runtime.reset(mask=mask)
    assert len(runtime.native.reset_calls) == 1


def test_closed_or_uninitialized_runtime_rejects_operations(make_runtime):
    runtime = make_runtime()
    native = runtime.native
    with pytest.raises(RuntimeError, match="Reset an open Isaac runtime"):
        runtime.step(_actions())
    runtime.reset()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.reset()
    with pytest.raises(RuntimeError, match="Reset an open Isaac runtime"):
        runtime.step(_actions())
    assert len(native.reset_calls) == 1
    assert native.step_calls == []


def test_owned_close_is_ordered_and_idempotent(make_runtime):
    runtime = make_runtime()
    native, launcher = runtime.native, runtime._launcher
    runtime.reset()
    runtime.step(_actions())
    runtime.close()
    runtime.close()
    assert native.close_calls == 1
    assert launcher.exit_calls == [(None, None, None)]
    assert native.events[-2:] == ["native.close", "launcher.exit"]
    assert runtime.native is None and runtime._launcher is None
    assert runtime._observation is None and runtime._pending_reset is None


def test_task_outcomes_are_cloned_before_native_cache_reuse(make_runtime):
    runtime = make_runtime()
    runtime._so101 = True
    native = runtime.native
    native.joint_reach_success = torch.tensor([False, True, False])
    native.joint_reach_error = torch.tensor([0.3, 0.01, 0.2])
    runtime.reset()
    native.next_terminated = [False, True, False]
    transition = runtime.step(_actions())
    native.joint_reach_success.zero_()
    native.joint_reach_error.fill_(99)
    assert torch.equal(transition.info["task_success"], torch.tensor([[False], [True], [False]]))
    torch.testing.assert_close(
        transition.info["joint_error_rad"], torch.tensor([[0.3], [0.01], [0.2]])
    )


@pytest.mark.parametrize("fail_native,fail_launcher", [(True, False), (False, True), (True, True)])
def test_close_attempts_all_owners_and_preserves_first_cleanup_error(
    make_runtime, fail_native, fail_launcher
):
    native_error, launcher_error = (
        RuntimeError("native close failed"),
        RuntimeError("launcher close failed"),
    )
    native = _FakeNative(close_error=native_error if fail_native else None)
    launcher = _FakeLauncher(native.events, close_error=launcher_error if fail_launcher else None)
    runtime = make_runtime(native, launcher=launcher)
    with pytest.raises(RuntimeError) as caught:
        runtime.close()
    assert caught.value is (native_error if fail_native else launcher_error)
    if fail_native and fail_launcher:
        assert "launcher close failed" in "\n".join(caught.value.__notes__)
    runtime.close()
    assert native.close_calls == len(launcher.exit_calls) == 1
    assert native.events[-2:] == ["native.close", "launcher.exit"]


@pytest.mark.parametrize("attach_native", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError, KeyboardInterrupt])
def test_initialization_failure_or_cancellation_preserves_primary_and_releases_owners(
    make_runtime, attach_native, error_type
):
    primary = error_type("initialization interrupted")
    native = _FakeNative()
    launcher = _FakeLauncher(native.events)
    with pytest.raises(error_type) as caught:
        make_runtime(native, launcher=launcher, creation_error=primary, attach_native=attach_native)
    assert caught.value is primary
    assert launcher.enter_calls == len(launcher.exit_calls) == 1
    assert native.close_calls == int(attach_native)
    assert native.reset_calls == native.step_calls == []


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_native,fail_launcher", [(True, False), (False, True), (True, True)])
def test_initialization_primary_and_all_cleanup_failures_remain_visible(
    make_runtime, error_type, fail_native, fail_launcher
):
    primary = error_type("primary initialization failure")
    native = _FakeNative(close_error=RuntimeError("native close failed") if fail_native else None)
    launcher = _FakeLauncher(
        native.events, close_error=RuntimeError("launcher close failed") if fail_launcher else None
    )
    with pytest.raises(error_type) as caught:
        make_runtime(native, launcher=launcher, creation_error=primary)
    assert caught.value is primary
    assert native.close_calls == len(launcher.exit_calls) == 1
    rendered = "".join(traceback.format_exception(caught.value))
    if fail_native:
        assert "native close failed" in rendered
    if fail_launcher:
        assert "launcher close failed" in rendered
