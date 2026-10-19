"""State-only Isaac PPO references with one direct native GPU batch."""

from nexuml.core.discovery import scenario
from nexuml.core.types import (
    CollectorSpec,
    InteractionSpec,
    LoggingSpec,
    PolicySpec,
    RLEvaluationSpec,
    ReinforcementLearningSpec,
    ScenarioSpec,
    TensorBoardSpec,
)
from nexuml_library.action_adapters.adapters import GaussianActionAdapter
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.reinforcement.policies import continuous_policy_mlp


def _isaac_ppo(name, task, total_frames, frames_per_batch, num_envs) -> ScenarioSpec:
    definition = IsaacLabEnvironment(task=task)
    return ScenarioSpec(
        name=name,
        pipeline=continuous_policy_mlp(action_dim=definition.describe().actions["action"].shape[0]),
        interaction=InteractionSpec(environment=definition),
        policy=PolicySpec(action_adapter=GaussianActionAdapter()),
        logging=LoggingSpec(tensorboard=TensorBoardSpec()),
        training=ReinforcementLearningSpec(
            algorithm=PPO(),
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            accelerator="gpu",
            devices=1,
            seed=0,
            collector=CollectorSpec(
                backend="direct",
                num_collectors=1,
                num_envs_per_collector=num_envs,
                sync=True,
                env_device="cuda:0",
                storing_device="cuda:0",
            ),
            evaluation=RLEvaluationSpec(episodes=0, at_start=False, every_n_frames=None),
        ),
    )


@scenario("isaac-cartpole-ppo")
def isaac_cartpole_ppo(total_frames=50_000, frames_per_batch=1_024, num_envs=32) -> ScenarioSpec:
    """Configure continuous cart effort, not categorical Gymnasium CartPole.

    Returns:
        A finite-frame Isaac Cartpole PPO scenario.
    """
    return _isaac_ppo(
        "isaac-cartpole-ppo", "Isaac-Cartpole", total_frames, frames_per_batch, num_envs
    )


@scenario("isaac-franka-reach-ppo")
def isaac_franka_reach_ppo(
    total_frames=50_000, frames_per_batch=1_024, num_envs=32
) -> ScenarioSpec:
    """Configure normalized seven-joint position offsets around the default pose.

    Returns:
        A finite-frame Isaac Franka Reach PPO scenario.
    """
    return _isaac_ppo(
        "isaac-franka-reach-ppo", "Isaac-Reach-Franka", total_frames, frames_per_batch, num_envs
    )


@scenario("isaac-so101-joint-reach-ppo")
def isaac_so101_joint_reach_ppo(
    total_frames=50_000, frames_per_batch=1_024, num_envs=32
) -> ScenarioSpec:
    """Configure five-arm-joint reaching with a fixed, non-policy gripper.

    Returns:
        A finite-frame simulation-only SO-101 Joint Reach PPO scenario.
    """
    return _isaac_ppo(
        "isaac-so101-joint-reach-ppo",
        "NexuML-SO101-Joint-Reach",
        total_frames,
        frames_per_batch,
        num_envs,
    )
