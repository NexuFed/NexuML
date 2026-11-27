"""Native DirectRLEnv project example: balance the pole while centering the cart."""

import gymnasium as gym

from isaaclab.utils import configclass
from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg


@configclass
class CenteringCartpoleCfg(CartpoleEnvCfg):
    """Keep native Cartpole assets and controls; own the reward and track bounds."""

    episode_length_s = 1.0
    compute_final_obs = True
    max_cart_pos = 0.5
    initial_cart_position_range = (-0.04, 0.04)
    initial_cart_velocity_range = (0.0, 0.0)
    initial_pole_angle_range = (0.0, 0.0)
    initial_pole_velocity_range = (0.0, 0.0)


class CenteringCartpole(CartpoleEnv):
    """Reward an upright pole and centered cart, with a cost for leaving the track."""

    def _get_rewards(self):
        cart = self.joint_pos[:, self._cart_dof_idx].squeeze(-1)
        pole = self.joint_pos[:, self._pole_dof_idx].squeeze(-1)
        score = pole.cos() - (cart / self.cfg.max_cart_pos).square()
        return (score - self.reset_terminated.float()) * self.step_dt


gym.register(
    id="NexuML-Cartpole-Direct-Centering",
    entry_point=f"{__name__}:CenteringCartpole",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:CenteringCartpoleCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg:CartpoleDirectPPORunnerCfg"
        ),
    },
)
