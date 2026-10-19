"""Minimal, lazily constructed SO-101 joint-reaching task for the pinned Isaac profile."""

from importlib import import_module
from hashlib import sha256
import math
from typing import Any, cast

import torch


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_POSITION = 0.2555162024919699
TARGET_SCALE = 0.5
HOME = (0.0, -0.25, 0.25, 0.0, 0.0)
GOALS = (HOME, (0.2, -0.2, 0.2, -0.15, 0.15), (-0.2, -0.3, 0.3, 0.15, -0.15))
TOLERANCE = 0.05
HOLD_SECONDS = 0.25
ASSET_SOURCE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac"
    "/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/so101_new_calib.usda"
)
# Raw authored layers, not USD copies rewritten into a process-local resolver cache.
ASSET_LAYER_SHA256 = {
    "so101_new_calib.usda": "d4067639489ecc99f7da2637de72098fd6155a2d54f32a8a49afe412a291612c",
    "payloads/base.usda": "f7125f84967821eb2540a3b6d5674f2681be722dec2f5343ba5718d474412205",
    "payloads/robot.usda": "3fcc5e3444b6ef1e2c11a36cf5d2cdc26dd577246aaf11e327cefae02399192b",
    "payloads/Robot/robot.usda": "a3fd163e24ad42fa534085b089f7f579b29254c709a043e0f68c8bf8aea704d4",
    "payloads/Physics/physics.usda": (
        "859dfddc29fd5e4c0ec753f6842d87d26d3fcd381f5efa9722d68a4789a3ae27"
    ),
    "payloads/Physics/physx.usda": (
        "478fc4568b13906897027c683a11c382fd0b1d81f9d79229fbd35956b00cd1af"
    ),
    "payloads/geometries.usd": "2fb01028b0866a3f3f9c38fa3d56eb9ba174a93810ab460621a4e639aa7cafec",
    "payloads/instances.usda": "bd026f2b4cb445c2eeee8acbb489886484e0928aaca006b1520140c84b3e159c",
    "payloads/materials.usda": "92a576efc9e6f642b23c3fb44c1dc71fe578f1141bb8a5980de91bc6d34f7f09",
    "payloads/Sensor/sensors.usda": (
        "be447310348d89308a09706dfa6b61540e441e85095d65c0b39f8cc4f3e03f93"
    ),
    "payloads/so101-camera_mount/SO-101_camera_mount.usda": (
        "e5f5acbf93fce4ffad87ceefcc982a0b9552060ff46c6e07ca67ca27b65617a2"
    ),
    "payloads/so101-camera_mount/payloads/base.usda": (
        "678c619c6125b587747bf570741bfcb8968dcfebda15a0e1405162ecd85fe73b"
    ),
    "payloads/so101-camera_mount/payloads/geometries.usd": (
        "ac913f1be2c235ae27214a5d6f6008f206491d31c6e844f937da5ec27fe3b653"
    ),
    "payloads/so101-camera_mount/payloads/instances.usda": (
        "f625094f7a97e2d971ddef19ca4dbea0d0eb86ef2ef9648ec50cd3bb203c34a3"
    ),
    "payloads/so101-camera_mount/payloads/materials.usda": (
        "19b6e7dc1331d76ccec313ae6f17d53acb64f74e9bcd7bde2180c5e03f615ac8"
    ),
}
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


def _validate_asset_content() -> None:
    read_file = import_module("isaaclab.utils.assets").read_file
    root = ASSET_SOURCE.rsplit("/", 1)[0]
    for name, expected in ASSET_LAYER_SHA256.items():
        if sha256(read_file(f"{root}/{name}").getvalue()).hexdigest() != expected:
            raise ValueError(f"SO-101 authored asset content differs from pinned layer: {name}")


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
    _validate_asset_content()


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
    robot_cfg = cast(Any, import_module("isaaclab_assets.robots.so101").SO101_CFG).replace(
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
