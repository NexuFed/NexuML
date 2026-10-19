"""Minimal, lazily constructed SO-101 joint-reaching task for the pinned Isaac profile."""

from importlib import import_module
import math
from typing import Any

import torch


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_POSITION = 0.2555162024919699
TARGET_SCALE = 0.5
HOME = (0.0, -0.25, 0.25, 0.0, 0.0)
GOALS = (HOME, (0.2, -0.2, 0.2, -0.15, 0.15), (-0.2, -0.3, 0.3, 0.15, -0.15))
TOLERANCE = 0.05
HOLD_SECONDS = 0.25
ASSET_SOURCE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac"
    "/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/so101_new_calib.usda"
)
JOINT_LIMITS = (
    (-1.919859886, 1.919859886),
    (-1.745329857, 1.745329857),
    (-1.689999938, 1.689999938),
    (-1.658059835, 1.658059835),
    (-2.743849754, 2.841209650),
    (-0.174532995, 1.745329857),
)
STIFFNESS = (48.419899, 47.782635, 14.715496, 43.607166, 54.873917, 68.250793)
DAMPING = (3.334595, 2.563165, 0.343250, 4.139243, 2.968392, 2.818730)


def validate_asset(env) -> None:
    """Reject a different articulation or authored PhysX control profile before acting.

    Raises:
        ValueError: If the maintained asset, joints, limits or actuator gains differ.
    """
    robot = env.scene["robot"]
    cfg = env.cfg.scene.robot
    expected_names = [*JOINT_NAMES, "gripper"]
    if (
        cfg.spawn.usd_path != ASSET_SOURCE
        or cfg.spawn.variants.get("Physics") != "physx"
        or robot.joint_names != expected_names
    ):
        raise ValueError("SO-101 asset source/PhysX variant/joints differ from the pinned profile")
    for value, expected in (
        (robot.data.joint_pos_limits.torch, JOINT_LIMITS),
        (robot.data.joint_stiffness.torch, STIFFNESS),
        (robot.data.joint_damping.torch, DAMPING),
        (robot.data.joint_effort_limits.torch, [3.35] * 6),
        (robot.data.joint_vel_limits.torch, [30.0] * 6),
        (robot.data.default_joint_pos.torch, (*HOME, GRIPPER_POSITION)),
    ):
        reference = torch.tensor(expected, device=env.device, dtype=value.dtype)
        if not torch.allclose(value, reference.expand_as(value), rtol=1e-5, atol=1e-6):
            raise ValueError(
                "SO-101 native joint limits/actuator gains differ from the pinned asset"
            )
    if any(
        term.stiffness is not None or term.damping is not None for term in cfg.actuators.values()
    ):
        raise ValueError("SO-101 must preserve the maintained USD actuator gains")


def normalized_joint_error(joint_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean squared joint error normalized by the declared one-radian target range."""
    return ((joint_pos - target) / (2 * TARGET_SCALE)).square().mean(dim=-1)


def _target_positions(env) -> torch.Tensor:
    """Return goals in radians without sampling during the manager's initial shape probe."""
    if not hasattr(env, "joint_reach_target"):
        return torch.tensor(HOME, device=env.device).repeat(env.num_envs, 1)
    return env.joint_reach_target


def _joint_error(env, asset_cfg) -> torch.Tensor:
    """Return an unweighted error; Isaac's reward manager applies the negative weight and dt."""
    positions = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    # Keep pre-reset outcomes: native same-step reset replaces the goal and hold counter.
    env.joint_reach_error = (positions - env.joint_reach_target).abs().mean(dim=-1)
    return normalized_joint_error(positions, env.joint_reach_target)


def _reach_success(env, asset_cfg) -> torch.Tensor:
    """Return success after all arm joints stay within tolerance for the required control steps."""
    positions = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    within = (positions - env.joint_reach_target).abs().amax(dim=-1) <= TOLERANCE
    env.joint_reach_hold = torch.where(within, env.joint_reach_hold + 1, 0)
    env.joint_reach_success = env.joint_reach_hold >= math.ceil(HOLD_SECONDS / env.step_dt)
    return env.joint_reach_success


def _reset_joint_reach(env, env_ids) -> None:
    """Reset named joints, fix the gripper target, and sample goals on Isaac's seeded device RNG."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints([*JOINT_NAMES, "gripper"], preserve_order=True)
    positions = torch.tensor((*HOME, GRIPPER_POSITION), device=env.device).repeat(len(env_ids), 1)
    robot.write_joint_position_to_sim_index(
        position=positions, joint_ids=joint_ids, env_ids=env_ids
    )
    robot.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(positions), joint_ids=joint_ids, env_ids=env_ids
    )
    # Unspecified targets default to zero, which would close the separately held gripper.
    robot.set_joint_position_target(positions, joint_ids=joint_ids, env_ids=env_ids)
    if not hasattr(env, "joint_reach_target"):
        env.joint_reach_target = torch.tensor(HOME, device=env.device).repeat(env.num_envs, 1)
        env.joint_reach_hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    goals = torch.tensor(GOALS, device=env.device)
    choices = torch.randint(len(GOALS), (len(env_ids),), device=env.device)
    env.joint_reach_target[env_ids] = goals[choices]
    env.joint_reach_hold[env_ids] = 0


def make_so101_config(*, num_envs: int, device: str, seed: int) -> Any:
    """Return a minimal manager config only inside the prepared Isaac execution process."""
    assets = import_module("isaaclab.assets")
    envs = import_module("isaaclab.envs")
    mdp = import_module("isaaclab.envs.mdp")
    managers = import_module("isaaclab.managers")
    scene = import_module("isaaclab.scene")
    sim = import_module("isaaclab.sim")
    configclass = import_module("isaaclab.utils").configclass
    physx = import_module("isaaclab_physx.physics")
    robot_cfg = import_module("isaaclab_assets.robots.so101").SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    robot_cfg.spawn.variants = {"Robot": "robot", "Sensor": "sensors", "Physics": "physx"}
    robot_cfg.init_state.joint_pos = dict(zip(JOINT_NAMES, HOME, strict=True)) | {
        "gripper": GRIPPER_POSITION
    }
    robot_cfg.init_state.joint_vel = {".*": 0.0}
    arm_cfg = managers.SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)

    @configclass
    class SceneCfg(scene.InteractiveSceneCfg):
        """Ground, maintained SO-101 articulation, and light only."""

        ground = assets.AssetBaseCfg(prim_path="/World/ground", spawn=sim.GroundPlaneCfg())
        robot = robot_cfg
        light = assets.AssetBaseCfg(
            prim_path="/World/Light", spawn=sim.DomeLightCfg(intensity=2000.0)
        )

    @configclass
    class ActionsCfg:
        """Five absolute position targets; the gripper is not a policy action."""

        arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(JOINT_NAMES),
            preserve_order=True,
            scale=TARGET_SCALE,
            offset=0.0,
            use_default_offset=False,
            clip=None,
        )

    @configclass
    class PolicyCfg(managers.ObservationGroupCfg):
        """Ordered positions [rad], velocities [rad/s], and goals [rad]."""

        joint_pos = managers.ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": arm_cfg})
        joint_vel = managers.ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": arm_cfg})
        target = managers.ObservationTermCfg(func=_target_positions)
        enable_corruption = False
        concatenate_terms = True
        history_length = 0

    @configclass
    class ObservationsCfg:
        """State-only policy observations, without a separate critic group."""

        policy = PolicyCfg()

    @configclass
    class RewardsCfg:
        """Negative normalized mean squared error, integrated by the native reward manager."""

        joint_error = managers.RewardTermCfg(
            func=_joint_error, weight=-1.0, params={"asset_cfg": arm_cfg}
        )

    @configclass
    class TerminationsCfg:
        """Held target success is terminal; the four-second horizon is a distinct timeout."""

        success = managers.TerminationTermCfg(func=_reach_success, params={"asset_cfg": arm_cfg})
        time_out = managers.TerminationTermCfg(func=mdp.time_out, time_out=True)

    @configclass
    class EventsCfg:
        """Per-environment home, goal, and success-hold reset."""

        reset_joint_reach = managers.EventTermCfg(func=_reset_joint_reach, mode="reset")

    return envs.ManagerBasedRLEnvCfg(
        scene=SceneCfg(num_envs=num_envs, env_spacing=1.5),
        sim=sim.SimulationCfg(
            device=device, dt=1 / 120, render_interval=4, physics=physx.PhysxCfg()
        ),
        seed=seed,
        decimation=4,
        episode_length_s=4.0,
        compute_final_obs=True,
        observations=ObservationsCfg(),
        actions=ActionsCfg(),
        rewards=RewardsCfg(),
        terminations=TerminationsCfg(),
        events=EventsCfg(),
    )
