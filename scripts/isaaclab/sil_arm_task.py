"""Native Isaac Lab task for the simulated SIL arm, without ROS or Kit UI."""

import gymnasium as gym
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SpawnerCfg
from isaaclab.sim.utils.prims import clone
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from ros2_sil import _HORIZON, _JOINT_FIELDS, _JOINTS, _PERIOD, _TARGET, _TOLERANCE, _arm


@clone
def _spawn(prim_path, cfg, translation=None, orientation=None):
    return _arm(get_current_stage(), prim_path)


@configclass
class SilArmSceneCfg(InteractiveSceneCfg):
    """Declare the procedural arm through Isaac Lab's ordinary scene cloner."""

    arm: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Arm",
        spawn=SpawnerCfg(func=_spawn),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0}, joint_vel={".*": 0}),
        actuators={
            "position": ImplicitActuatorCfg(
                joint_names_expr=list(_JOINTS),
                stiffness=34.906585,
                damping=3.4906585,
                effort_limit_sim=20.0,
            )
        },
    )


@configclass
class SilArmCfg(DirectRLEnvCfg):
    """Apply one absolute position command for 24 native physics steps."""

    decimation = 24
    episode_length_s = _HORIZON * _PERIOD
    action_space = gym.spaces.Box(-1.0, 1.0, (2,))
    observation_space = dict.fromkeys(_JOINT_FIELDS, 2)
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=_PERIOD / decimation, render_interval=decimation, gravity=(0, 0, 0), physics=PhysxCfg()
    )
    scene: SilArmSceneCfg = SilArmSceneCfg(num_envs=8, env_spacing=2.0)


class SilArm(DirectRLEnv):
    """Reuse the bridge asset and score measured joint tracking."""

    def _setup_scene(self):
        self.arm = self.scene["arm"]
        self.goal = torch.tensor(_TARGET, device=self.device).expand(self.num_envs, -1)

    def _pre_physics_step(self, actions):
        self.command = actions

    def _apply_action(self):
        self.arm.actuators.target_command.set_position_index(value=self.command)

    def _get_observations(self):
        return {
            "policy": {
                "joint_position": self.arm.data.joint_pos.torch.clone(),
                "joint_velocity": self.arm.data.joint_vel.torch.clone(),
                "target_position": self.goal,
            }
        }

    def _get_rewards(self):
        return -(self.arm.data.joint_pos.torch - self.goal).norm(dim=-1)

    def _get_dones(self):
        success = -self._get_rewards() <= _TOLERANCE
        return success, (self.episode_length_buf >= self.max_episode_length) & ~success

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        arm = self.arm
        pos = torch.empty((len(env_ids), 2), device=self.device).uniform_(-0.04, 0.04)
        arm.write_joint_position_to_sim_index(position=pos, env_ids=env_ids)
        arm.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos), env_ids=env_ids)
        arm.actuators.target_command.set_position_index(value=pos, env_ids=env_ids)


def controls(native):
    """Verify the registered native joint ordering and limits.

    Returns:
        The native absolute-position mapping in radians.
    """
    from nexuml.core.types import PhysicalTensorContract

    assert type(native) is SilArm and tuple(native.arm.joint_names) == _JOINTS
    torch.testing.assert_close(
        native.arm.data.soft_joint_pos_limits.torch[0],
        torch.tensor([[-1.0, 1.0]] * 2, device=native.device),
    )
    return PhysicalTensorContract(
        units=("rad", "rad"),
        frame="joint",
        joints=tuple(native.arm.joint_names),
        control_mode="absolute_position",
        scale=(1.0, 1.0),
        offset=(0.0, 0.0),
    )


gym.register(
    id="NexuML-SIL-Arm-v0",
    entry_point=f"{__name__}:SilArm",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:SilArmCfg"},
)
