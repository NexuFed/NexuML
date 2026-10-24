"""Opt-in native transition oracle; no learner, model downloads or quality claim."""

import sys

import torch
from tensordict import TensorDict
from torchrl.objectives.value.functional import generalized_advantage_estimate

from nexuml.core.components import EnvironmentBuildContext
from nexuml.reinforcement.torchrl_env import _TorchRLEnvironmentAdapter
from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime
from nexuml_library.environments.isaaclab import IsaacLabEnvironment


def main():
    """Compare real native end/reset states to the generic TorchRL handoff and GAE."""
    task, physics = sys.argv[1:]
    definition = IsaacLabEnvironment.model_validate({"task": task, "physics": physics})
    runtime = definition.build(
        EnvironmentBuildContext(
            purpose="train",
            device="cuda:0",
            num_envs=4,
            seed=17,
        )
    )
    assert isinstance(runtime, _IsaacLabRuntime)
    env = _TorchRLEnvironmentAdapter(runtime, device="cuda:0", seed=17)
    native = runtime.native
    reset_calls = []
    original_reset = native.reset

    def reset(**kwargs):
        reset_calls.append(kwargs)
        return original_reset(**kwargs)

    native.reset = reset
    try:
        robot = native.scene["cartpole" if task == "Isaac-Cartpole-Direct" else "robot"]
        joint_ids, names = robot.find_joints("slider_to_cart")
        assert names == ["slider_to_cart"]
        cases = [(0, False, False)] + [
            (count, term, trunc)
            for count in (1, 4)
            for term, trunc in ((True, False), (False, True), (True, True))
        ]
        for count, terminate, truncate in cases:
            env.set_seed(17)
            root = env.reset().clone()
            positions = robot.data.joint_pos.torch.clone()
            positions[:, joint_ids] = 0
            if terminate:
                positions[:count, joint_ids] = 3.5
            robot.write_joint_position_to_sim_index(position=positions)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(positions))
            if truncate:
                native.episode_length_buf[:count] = native.max_episode_length - 1
            before_resets = len(reset_calls)
            root["action"] = torch.zeros((4, 1), device="cuda:0")
            first, next_root = env.step_and_maybe_reset(root)
            expected_done = torch.arange(4, device="cuda:0") < count
            assert torch.equal(first["next", "terminated"][:, 0], expected_done & terminate)
            assert torch.equal(first["next", "truncated"][:, 0], expected_done & truncate)
            assert len(reset_calls) == before_resets, "auto-reset was acknowledged twice"
            final = first["next", "observation"].clone()
            next_values = next_root["observation"].clone()
            if count:
                assert not torch.equal(final[:count], next_values[:count])
                assert (native.episode_length_buf[:count] == 0).all()
            next_root["action"] = torch.zeros((4, 1), device="cuda:0")
            second, _ = env.step_and_maybe_reset(next_root)
            assert torch.equal(first["next", "observation"], final)
            rollout = torch.stack([first, second], dim=1)
            assert isinstance(rollout, TensorDict) and rollout.batch_size == (4, 2)
            value = torch.stack([root["observation"][:, :1], next_values[:, :1]], dim=1) + 100
            next_value = rollout["next", "observation"][..., :1] + 100
            reward = rollout["next", "reward"]
            done, term = rollout["next", "done"], rollout["next", "terminated"]
            assert isinstance(next_value, torch.Tensor) and isinstance(reward, torch.Tensor)
            assert isinstance(done, torch.Tensor) and isinstance(term, torch.Tensor)
            advantage, target = generalized_advantage_estimate(
                0.9,
                0.95,
                value,
                next_value,
                reward,
                done,
                term,
            )
            delta = reward + 0.9 * (~term) * next_value - value
            expected = delta.clone()
            expected[:, 0] += 0.9 * 0.95 * (~done[:, 0]) * delta[:, 1]
            torch.testing.assert_close(advantage, expected)
            torch.testing.assert_close(target, expected + value)
            print("NATIVE_TRANSITION_GAE", task, physics, count, terminate, truncate, flush=True)

        root = env.reset().clone()
        root["action"] = torch.zeros((4, 1), device="cuda:0")
        _, root = env.step_and_maybe_reset(root)
        previous = root["observation"].clone()
        lengths = native.episode_length_buf.clone()
        before_resets = len(reset_calls)
        mask = torch.tensor([True, False, False, False], device="cuda:0")
        if task == "Isaac-Cartpole-Direct":
            try:
                runtime.reset(mask=mask)
            except ValueError as error:
                assert "explicit partial native reset" in str(error)
            else:
                raise AssertionError("Direct task silently accepted an unsupported partial reset")
            assert len(reset_calls) == before_resets
            assert torch.equal(native.episode_length_buf, lengths)
        else:
            reset_observation = runtime.reset(mask=mask)
            assert len(reset_calls) == before_resets + 1
            reset_value = reset_observation["observation"]
            assert isinstance(reset_value, torch.Tensor)
            assert torch.equal(reset_value[1:], previous[1:])
            assert torch.equal(native.episode_length_buf[1:], lengths[1:])
            assert native.episode_length_buf[0] == 0
        print("NATIVE_EXPLICIT_MASK_RESET_VERIFIED", flush=True)
    finally:
        env.close()
    print("NATIVE_ADAPTER_TRANSITIONS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
