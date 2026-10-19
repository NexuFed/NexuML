"""Bounded native state/control/reset proof; not NexuML training or convergence."""

import argparse
from importlib import import_module
import json
import math
from typing import Any, cast

import torch


def main():
    """Check the pinned upstream manager task without constructing a NexuML learner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", choices=["Isaac-Cartpole", "Isaac-Reach-Franka", "NexuML-SO101-Joint-Reach"]
    )
    args = parser.parse_args()
    import gymnasium as gym

    import_module("isaaclab_tasks")
    launch_simulation = import_module("isaaclab.app").launch_simulation
    parse_env_cfg = import_module("isaaclab_tasks.utils").parse_env_cfg

    from nexuml_library.environments.isaaclab import IsaacLabEnvironment

    definition = IsaacLabEnvironment(task=args.task)
    contract = definition.describe()
    assert contract.control_period_s is not None
    so101 = args.task == "NexuML-SO101-Joint-Reach"
    if so101:
        from nexuml_library.environments._isaaclab_so101 import (
            JOINT_NAMES,
            make_so101_config,
            validate_asset,
        )

        cfg = make_so101_config(num_envs=8, device="cuda:0", seed=0)
    else:
        cfg = parse_env_cfg(
            args.task, device="cuda:0", num_envs=8, overrides=["physics=isaacsim_physx"]
        )
    cfg.seed = 0
    cfg.compute_final_obs = True
    cfg.observations.policy.enable_corruption = False
    cfg.episode_length_s = 4 * cfg.sim.dt * cfg.decimation
    with launch_simulation(
        cfg,
        {"headless": True, "device": "cuda:0", "visualizer": ["none"], "visualizer_explicit": True},
    ):
        env = cast(
            Any,
            import_module("isaaclab.envs").ManagerBasedRLEnv(cfg=cfg)
            if so101
            else gym.make(args.task, cfg=cfg).unwrapped,
        )
        try:
            if so101:
                validate_asset(env)
            observations, _ = env.reset(seed=0)
            assert observations["policy"].shape == (8, *contract.observations["observation"].shape)
            assert observations["policy"].device == torch.device("cuda:0")
            assert math.isclose(env.step_dt, contract.control_period_s)
            name = "joint_effort" if args.task == "Isaac-Cartpole" else "arm_action"
            expected_joints = (
                ["slider_to_cart"]
                if args.task == "Isaac-Cartpole"
                else list(JOINT_NAMES)
                if so101
                else [f"panda_joint{i}" for i in range(1, 8)]
            )
            scale = 100.0 if args.task == "Isaac-Cartpole" else 0.5
            term = env.action_manager.get_term(name)
            descriptor = term.IO_descriptor
            assert descriptor.joint_names == expected_joints
            assert term.cfg.scale == scale and term.cfg.clip is None
            if args.task == "Isaac-Reach-Franka":
                assert term.cfg.use_default_offset
            offset = torch.as_tensor(descriptor.offset, device="cuda:0")
            action = torch.full((8, len(expected_joints)), 0.1, device="cuda:0")
            # First force a partial timeout, then let all rows reach the bounded horizon.
            env.episode_length_buf[0] = env.max_episode_length - 1
            counts = []
            for step in range(8):
                obs, reward, terminated, truncated, info = env.step(action)
                assert obs["policy"].device == reward.device == action.device
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert reward.shape == terminated.shape == truncated.shape == (8,)
                assert terminated.dtype == truncated.dtype == torch.bool
                done = terminated | truncated
                if step == 0:
                    assert truncated[0] and not truncated[1:].any()
                if done.any():
                    final = info["final_obs"]["policy"]
                    assert final.shape == obs["policy"].shape and torch.isfinite(final).all()
                    assert not torch.equal(final[done], obs["policy"][done])
                # Native reset clears raw actions on done rows; continuing rows retain mapping.
                if (~done).any():
                    torch.testing.assert_close(
                        term.processed_actions[~done], (scale * action + offset)[~done]
                    )
                counts.append(int(done.sum()))
            robot = env.scene["robot"]
            print(
                json.dumps(
                    {
                        "task": args.task,
                        "revision": definition.revision,
                        "physics": definition.physics,
                        "seed": 0,
                        "num_envs": 8,
                        "frames": 64,
                        "control_period_s": env.step_dt,
                        "action_joints": expected_joints,
                        "scale": scale,
                        "offset": offset.tolist(),
                        "asset": cfg.scene.robot.spawn.usd_path,
                        "joint_names": robot.joint_names,
                        "joint_limits": robot.data.joint_pos_limits.torch[0].tolist(),
                        "default_joint_positions": robot.data.default_joint_pos.torch[0].tolist(),
                        "done_rows_per_step": counts,
                    },
                    indent=2,
                ),
                flush=True,
            )
            print("NATIVE_STATE_CONTROL_RESET_OK", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
