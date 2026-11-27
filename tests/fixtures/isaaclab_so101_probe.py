"""Native five-joint reaching controls and outcomes; not vial placement or learned success."""

import math

import torch
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext
from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime
from nexuml_library.environments._isaaclab_so101 import (
    GRIPPER_POSITION,
    HOLD_SECONDS,
    HOME,
    JOINT_NAMES,
    JOINT_LIMITS,
    TARGET_SCALE,
)
from nexuml_library.environments.isaaclab import IsaacLabEnvironment


def main():
    """Check genuine signed physical motion, fixed gripper, limits and held outcomes."""
    runtime = IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach").build(
        EnvironmentBuildContext(purpose="train", device="cuda:0", num_envs=10, seed=17)
    )
    assert isinstance(runtime, _IsaacLabRuntime)
    env = runtime.native
    try:
        robot = env.scene["robot"]
        ids, names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        assert names == list(JOINT_NAMES)
        gripper, _ = robot.find_joints("gripper")
        initial = runtime.reset()["observation"]
        assert isinstance(initial, torch.Tensor)
        home = torch.tensor(HOME, device="cuda:0")
        command = home.repeat(10, 1)
        for joint in range(5):
            command[2 * joint, joint] += 0.15
            command[2 * joint + 1, joint] -= 0.15
        # Choose a genuine distant task goal so the sign probe does not auto-reset midway.
        env.joint_reach_target[:] = torch.tensor((-0.2, -0.3, 0.3, 0.15, -0.15), device="cuda:0")
        for _ in range(20):
            transition = runtime.step(TensorDict({"action": command / TARGET_SCALE}, [10]))
            assert not transition.terminated.any() and not transition.truncated.any()
            assert transition.reward is not None and torch.isfinite(transition.reward).all()
        positions = robot.data.joint_pos.torch
        for joint in range(5):
            displacement = positions[2 * joint : 2 * joint + 2, ids[joint]] - home[joint]
            assert displacement[0] > 0.02 and displacement[1] < -0.02, (joint, displacement)
        limits = torch.tensor(JOINT_LIMITS, device="cuda:0")
        assert ((positions >= limits[:, 0]) & (positions <= limits[:, 1])).all()
        assert (positions[:, gripper] - GRIPPER_POSITION).abs().max() < 0.02
        print("SO101_SIGN_LIMIT_FIXED_GRIPPER_VERIFIED", flush=True)

        runtime.reset(seed=17)
        goal = torch.tensor(HOME, device="cuda:0").repeat(10, 1)
        env.joint_reach_target[:] = goal
        steps = math.ceil(HOLD_SECONDS / env.step_dt)
        for index in range(steps):
            transition = runtime.step(TensorDict({"action": goal / TARGET_SCALE}, [10]))
            assert transition.info is not None
            assert torch.equal(transition.terminated[:, 0], env.joint_reach_success)
            if index < steps - 1:
                assert not transition.info["task_success"].any(), "target was not held long enough"
        assert transition.info is not None
        assert transition.terminated.all() and transition.info["task_success"].all()
        assert not transition.truncated.any()
        assert transition.info["joint_error_rad"].max() <= 0.05
        assert (env.episode_length_buf == 0).all()
        print("SO101_NATIVE_HELD_SUCCESS_VERIFIED", flush=True)

        runtime.reset(seed=17)
        env.joint_reach_target[:] = torch.full_like(goal, 0.5)
        env.episode_length_buf[0] = env.max_episode_length - 1
        transition = runtime.step(TensorDict({"action": goal / TARGET_SCALE}, [10]))
        assert transition.info is not None
        assert transition.truncated[:, 0].tolist() == [True] + [False] * 9
        assert not transition.terminated.any() and not transition.info["task_success"].any()
        assert transition.info["joint_error_rad"][0] > 0.05
        print("SO101_NATIVE_TIMEOUT_NOT_SUCCESS_VERIFIED", flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
