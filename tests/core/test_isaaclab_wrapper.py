"""CPU probes of the installed public wrapper, not native Isaac compatibility proof.

The native-shaped Gym fake cannot enter TorchRL's installation/class-gated Isaac
autoreset path without Isaac installed. No wrapper internals or availability flags
are patched; explicit public reader/transform composition is probed separately.
"""

from contextlib import closing

import pytest
import torch
from tensordict import TensorDict

gym = pytest.importorskip("gymnasium")
batch_space = pytest.importorskip("gymnasium.vector.utils").batch_space
TransformedEnv = pytest.importorskip("torchrl.envs").TransformedEnv
IsaacLabWrapper = pytest.importorskip("torchrl.envs.libs.isaac_lab").IsaacLabWrapper
VecGymEnvTransform = pytest.importorskip("torchrl.envs.transforms").VecGymEnvTransform
generalized_advantage_estimate = pytest.importorskip(
    "torchrl.objectives.value.functional"
).generalized_advantage_estimate


_CASES = [pytest.param((False, False), (False, False), id="none")]
for _mask_name, _mask in (("partial", (True, False)), ("all", (True, True))):
    for _flag_name, _terminated, _truncated in (
        ("termination", _mask, (False, False)),
        ("timeout", (False, False), _mask),
        ("both", _mask, _mask),
    ):
        _CASES.append(pytest.param(_terminated, _truncated, id=f"{_mask_name}-{_flag_name}"))


class _NativeShapedEnv(gym.Env):
    # This name alone does not activate the installed wrapper's Isaac class gate.
    __module__ = "isaaclab.envs.manager_based_rl_env"
    num_envs = 2
    device = "cpu"

    def __init__(self, terminated, truncated):
        self.single_action_space = gym.spaces.Box(-1, 1, shape=(1,))
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(-float("inf"), float("inf"), shape=(1,))}
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.initial_terminated = torch.tensor(terminated)
        self.initial_truncated = torch.tensor(truncated)
        self.observation_buf = torch.zeros(2, 1)
        self.reward_buf = torch.zeros(2)
        self.terminated_buf = torch.zeros(2, dtype=torch.bool)
        self.truncated_buf = torch.zeros(2, dtype=torch.bool)
        self.reset_count = torch.zeros(2, dtype=torch.long)
        self.reset_calls = []
        self.extras = {}
        self.steps = 0
        self.last_action = None
        self.closed = False

    def reset(self, *, seed=None, env_ids=None, options=None):
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        self.reset_calls.append(ids.clone())
        self.reset_count[ids] += 1
        self.observation_buf[ids, 0] = -100 * self.reset_count[ids].float() - ids.float()
        return {"policy": self.observation_buf}, {}

    def step(self, action):
        assert isinstance(action, torch.Tensor) and action.device.type == "cpu"
        assert action.shape == (2, 1) and (action.abs() <= 1).all()
        self.last_action = action
        self.steps += 1
        self.observation_buf.copy_(torch.tensor([[11.0], [22.0]]) * self.steps)
        self.reward_buf.copy_(torch.tensor([1.0, 2.0]) * self.steps)
        self.terminated_buf.copy_(self.initial_terminated if self.steps == 1 else False)
        self.truncated_buf.copy_(self.initial_truncated if self.steps == 1 else False)
        done = self.terminated_buf | self.truncated_buf
        if done.any():
            # Pinned manager env: full-batch policy group before same-step reset,
            # with no Gymnasium _final_obs mask (extras may persist afterward).
            self.extras["final_obs"] = {"policy": self.observation_buf.clone()}
            ids = done.nonzero(as_tuple=True)[0]
            self.reset_count[ids] += 1
            self.observation_buf[ids, 0] = -100 * self.reset_count[ids].float() - ids.float()
        return (
            {"policy": self.observation_buf},
            self.reward_buf,
            self.terminated_buf,
            self.truncated_buf,
            self.extras,
        )

    def close(self):
        self.closed = True


def _input(env):
    td = env.reset().clone()
    td["action"] = torch.tensor([[0.25], [-0.5]])
    return td


def _final_reader(info, td):
    final = info.get("final_obs")
    if final is not None:
        td["final"] = TensorDict({"policy": final["policy"].clone()}, batch_size=[2])
    return td


def _value_target(next_value, reward, done, terminated):
    _, target = generalized_advantage_estimate(
        gamma=0.9,
        lmbda=0.95,
        state_value=torch.zeros(2, 1, 1),
        next_state_value=next_value.unsqueeze(1),
        reward=reward.unsqueeze(1),
        done=done.unsqueeze(1),
        terminated=terminated.unsqueeze(1),
    )
    return target.squeeze(1)


@pytest.mark.parametrize("native_autoreset", [False, True])
@pytest.mark.parametrize("terminated,truncated", _CASES)
def test_public_wrapper_ignores_final_obs_on_native_shaped_cpu_fake(
    native_autoreset, terminated, truncated
):
    fake = _NativeShapedEnv(terminated, truncated)
    with closing(IsaacLabWrapper(fake, device="cpu", native_autoreset=native_autoreset)) as env:
        # No fake module injection, private availability flags, or wrapper subclass.
        assert type(env) is IsaacLabWrapper
        assert env.batch_size == torch.Size([2])
        td = _input(env)
        transition = env.step(td)
        assert fake.last_action is td["action"]
        final = torch.tensor([[11.0], [22.0]])
        done = torch.tensor(terminated) | torch.tensor(truncated)
        expected_returned = final.clone()
        expected_returned[done, 0] = torch.tensor([-200.0, -201.0])[done]
        torch.testing.assert_close(transition["next", "policy"], expected_returned)
        torch.testing.assert_close(
            transition["next", "terminated"].squeeze(-1), torch.tensor(terminated)
        )
        torch.testing.assert_close(
            transition["next", "truncated"].squeeze(-1), torch.tensor(truncated)
        )
        torch.testing.assert_close(transition["next", "done"].squeeze(-1), done)
        if done.any():
            assert "_final_obs" not in fake.extras
            torch.testing.assert_close(fake.extras["final_obs"]["policy"], final)
            assert not torch.equal(transition["next", "policy"][done], final[done])
        else:
            torch.testing.assert_close(transition["next", "policy"], final)

        actual = _value_target(
            transition["next", "policy"],
            transition["next", "reward"],
            transition["next", "done"],
            transition["next", "terminated"],
        )
        expected = torch.tensor([[1.0], [2.0]]) + 0.9 * final * (
            ~torch.tensor(terminated).unsqueeze(-1)
        )
        torch.testing.assert_close(
            _value_target(
                final,
                transition["next", "reward"],
                transition["next", "done"],
                transition["next", "terminated"],
            ),
            expected,
        )
        timeouts_only = torch.tensor(truncated) & ~torch.tensor(terminated)
        if timeouts_only.any():
            assert not torch.allclose(actual[timeouts_only], expected[timeouts_only])
        else:
            torch.testing.assert_close(actual, expected)
    assert fake.closed


@pytest.mark.parametrize("native_autoreset", [False, True])
@pytest.mark.parametrize("terminated,truncated", _CASES)
def test_public_reader_and_vec_transform_do_not_fix_reset_lifecycle(
    native_autoreset, terminated, truncated
):
    fake = _NativeShapedEnv(terminated, truncated)
    wrapper = IsaacLabWrapper(fake, device="cpu", native_autoreset=native_autoreset)
    wrapper.set_info_dict_reader(_final_reader)
    with closing(
        TransformedEnv(wrapper, VecGymEnvTransform(native_autoreset=native_autoreset))
    ) as env:
        transition = env.step(_input(env))
        done = torch.tensor(terminated) | torch.tensor(truncated)
        final = torch.tensor([[11.0], [22.0]])
        cached_reset = torch.tensor([[-200.0], [-201.0]])
        if done.any() and native_autoreset:
            # The public transform alone is a pass-through, not native wiring.
            torch.testing.assert_close(transition["next", "policy"][done], cached_reset[done])
            torch.testing.assert_close(transition["next", "final", "policy"], final)
        else:
            # A reader can supply real terminal observations to the default transform.
            torch.testing.assert_close(transition["next", "policy"], final)

        root = env.step_mdp(transition)
        if done.any() and not done.all():
            with pytest.raises(RuntimeError, match="Partial resets are not handled at this level"):
                env.maybe_reset(root)
            assert len(fake.reset_calls) == 1
        else:
            root = env.maybe_reset(root)
            if done.all():
                # Full reset reaches native reset again, ignoring its already-reset cache.
                assert len(fake.reset_calls) == 2
                torch.testing.assert_close(fake.reset_count, torch.tensor([3, 3]))
                torch.testing.assert_close(root["policy"], torch.tensor([[-300.0], [-301.0]]))
                assert not torch.equal(root["policy"], cached_reset)
            else:
                assert len(fake.reset_calls) == 1
                torch.testing.assert_close(root["policy"], final)


@pytest.mark.parametrize("native_autoreset", [False, True])
@pytest.mark.parametrize("mask", [(False, False), (True, False), (True, True)])
def test_public_step_and_maybe_reset_handoff(native_autoreset, mask):
    fake = _NativeShapedEnv((False, False), mask)
    with closing(IsaacLabWrapper(fake, device="cpu", native_autoreset=native_autoreset)) as env:
        td = _input(env)
        if any(mask) and not all(mask):
            with pytest.raises(RuntimeError, match="Partial resets are not handled at this level"):
                env.step_and_maybe_reset(td)
            assert len(fake.reset_calls) == 1
        else:
            _, root = env.step_and_maybe_reset(td)
            if all(mask):
                assert len(fake.reset_calls) == 2
                torch.testing.assert_close(root["policy"], torch.tensor([[-300.0], [-301.0]]))
                torch.testing.assert_close(
                    fake.extras["final_obs"]["policy"], torch.tensor([[11.0], [22.0]])
                )
            else:
                assert len(fake.reset_calls) == 1
                torch.testing.assert_close(root["policy"], torch.tensor([[11.0], [22.0]]))


@pytest.mark.parametrize("native_autoreset", [False, True])
def test_reused_native_buffers_preserve_reward_flags_but_not_observations(native_autoreset):
    fake = _NativeShapedEnv((False, False), (True, False))
    with closing(IsaacLabWrapper(fake, device="cpu", native_autoreset=native_autoreset)) as env:
        first = env.step(_input(env))
        snapshot = first["next"].clone()
        second_input = env.step_mdp(first).clone()
        second_input["action"] = torch.zeros(2, 1)
        env.step(second_input)
        for key in ("reward", "terminated", "truncated", "done"):
            torch.testing.assert_close(first["next", key], snapshot[key])
        assert not torch.equal(first["next", "policy"], snapshot["policy"])
        torch.testing.assert_close(first["next", "policy"], torch.tensor([[22.0], [44.0]]))
