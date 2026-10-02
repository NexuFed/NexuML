"""Continuous/discrete CPU reference scenarios with overrideable frame budgets."""

from nexuml.core.discovery import scenario
from nexuml.core.types import InteractionSpec, PolicySpec, ReinforcementLearningSpec, ScenarioSpec
from nexuml_library.action_adapters.adapters import GaussianActionAdapter, CategoricalActionAdapter
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.reinforcement.policies import continuous_policy_mlp, discrete_policy_mlp


@scenario("pendulum-ppo")
def pendulum_ppo(total_frames=50_000, frames_per_batch=1_024) -> ScenarioSpec:
    """Reference bounded continuous control.

    Returns:
        A finite-frame Pendulum PPO scenario.
    """
    return ScenarioSpec(
        name="pendulum-ppo",
        pipeline=continuous_policy_mlp(),
        interaction=InteractionSpec(environment=GymnasiumEnvironment(env_name="Pendulum-v1")),
        policy=PolicySpec(action_adapter=GaussianActionAdapter()),
        training=ReinforcementLearningSpec(
            algorithm=PPO(),
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            accelerator="cpu",
            devices=1,
            seed=0,
        ),
    )


@scenario("cartpole-ppo")
def cartpole_ppo(total_frames=50_000, frames_per_batch=1_024) -> ScenarioSpec:
    """Reference categorical control.

    Returns:
        A finite-frame CartPole PPO scenario.
    """
    return ScenarioSpec(
        name="cartpole-ppo",
        pipeline=discrete_policy_mlp(),
        interaction=InteractionSpec(environment=GymnasiumEnvironment(env_name="CartPole-v1")),
        policy=PolicySpec(action_adapter=CategoricalActionAdapter()),
        training=ReinforcementLearningSpec(
            algorithm=PPO(),
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            accelerator="cpu",
            devices=1,
            seed=0,
        ),
    )
