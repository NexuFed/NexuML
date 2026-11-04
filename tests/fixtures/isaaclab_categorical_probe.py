"""Real native class-to-force and signed-motion oracle; no convergence claim."""

import shutil

import torch
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext
from nexuml_library.scenarios.reinforcement.isaaclab_categorical import (
    isaac_categorical_cartpole_ppo,
)


def main():
    assert shutil.disk_usage(".").free > 4 * 2**30
    scenario = isaac_categorical_cartpole_ppo()
    assert scenario.interaction is not None
    definition = scenario.interaction.environment
    with definition.job_scope():
        runtime = definition.build(
            EnvironmentBuildContext(purpose="train", device="cuda:0", num_envs=3, seed=0)
        )
        try:
            runtime.reset(seed=0)
            native = runtime.native
            assert native.single_action_space.n == 3
            robot = native.cartpole
            joints, _ = robot.find_joints(["slider_to_cart", "cart_to_pole"], preserve_order=True)
            ids = torch.arange(3, dtype=torch.int32, device="cuda:0")
            robot.write_joint_position_to_sim_index(
                position=torch.zeros(3, 2, device="cuda:0"), joint_ids=joints, env_ids=ids
            )
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros(3, 2, device="cuda:0"), joint_ids=joints, env_ids=ids
            )
            transition = runtime.step(
                TensorDict({"action": torch.arange(3, device="cuda:0")}, batch_size=(3,))
            )
            assert torch.equal(
                native.actions, torch.tensor([[-100.0], [0.0], [100.0]], device="cuda:0")
            )
            slider, _ = robot.find_joints(["slider_to_cart"], preserve_order=True)
            velocity = robot.data.joint_vel.torch[:, slider].flatten()
            assert velocity[0] < 0 and velocity[2] > 0 and velocity[1].abs() < 1e-5
            assert transition.observation["observation"].shape == (3, 4)
            print(
                "NATIVE_CATEGORICAL_CLASS_FORCE_AND_SIGNED_MOTION_VERIFIED",
                velocity.cpu().tolist(),
                flush=True,
            )
        finally:
            runtime.close()
    assert shutil.disk_usage(".").free > 4 * 2**30


if __name__ == "__main__":
    main()
