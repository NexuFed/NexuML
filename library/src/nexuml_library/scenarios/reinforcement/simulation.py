"""Static-contract visual and MuJoCo reference scenarios; no simulator on discovery."""

from nexuml.core.discovery import scenario
from nexuml.core.types import InteractionContract, ScenarioSpec, TensorFieldContract
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.policies import continuous_policy_mlp, discrete_policy_mlp
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo, pendulum_ppo


@scenario("visual-cartpole-ppo")
def visual_cartpole_ppo(total_frames=50_000, frames_per_batch=1_024) -> ScenarioSpec:
    """Consume rendered RGB tensors directly.

    Returns:
        A small image-to-categorical-action reference scenario.
    """
    spec = cartpole_ppo(total_frames, frames_per_batch)
    spec.name = "visual-cartpole-ppo"
    spec.interaction.environment = GymnasiumEnvironment(
        env_name="CartPole-v1",
        image_shape=(3, 32, 32),
        contract=InteractionContract(
            observations={
                "image": TensorFieldContract(
                    shape=(3, 32, 32),
                    dtype="uint8",
                    modality="image",
                    low=0,
                    high=255,
                )
            },
            actions={"action": TensorFieldContract(shape=(), dtype="int64", num_values=2)},
            action_space="discrete",
        ),
    )
    spec.pipeline = discrete_policy_mlp(observation_keys=("image",))
    return spec


@scenario("mujoco-humanoid-ppo")
def mujoco_humanoid_ppo(total_frames=100_000, frames_per_batch=2_048) -> ScenarioSpec:
    """High-dimensional humanoid joint control with explicit v5 proprioception.

    Returns:
        An optional MuJoCo humanoid scenario.
    """
    spec = pendulum_ppo(total_frames, frames_per_batch)
    spec.name = "mujoco-humanoid-ppo"
    spec.interaction.environment = GymnasiumEnvironment(
        env_name="Humanoid-v5",
        kwargs={
            "include_cinert_in_observation": False,
            "include_cvel_in_observation": False,
            "include_qfrc_actuator_in_observation": False,
            "include_cfrc_ext_in_observation": False,
        },
        contract=InteractionContract(
            observations={
                "observation": TensorFieldContract(
                    shape=(45,),
                    dtype="float64",
                    modality="proprioception",
                )
            },
            actions={
                "action": TensorFieldContract(
                    shape=(17,),
                    dtype="float32",
                    modality="joint_command",
                    low=-0.4,
                    high=0.4,
                )
            },
            action_space="continuous",
        ),
    )
    spec.pipeline = continuous_policy_mlp(action_dim=17)
    return spec


@scenario("mujoco-reacher-ppo")
def mujoco_reacher_ppo(total_frames=50_000, frames_per_batch=1_024) -> ScenarioSpec:
    """Two-joint articulated control distinct from humanoid locomotion.

    Returns:
        An optional MuJoCo Reacher scenario.
    """
    spec = pendulum_ppo(total_frames, frames_per_batch)
    spec.name = "mujoco-reacher-ppo"
    spec.interaction.environment = GymnasiumEnvironment(
        env_name="Reacher-v5",
        contract=InteractionContract(
            observations={
                "observation": TensorFieldContract(
                    shape=(10,),
                    dtype="float64",
                    modality="proprioception",
                )
            },
            actions={
                "action": TensorFieldContract(
                    shape=(2,),
                    dtype="float32",
                    modality="joint_command",
                    low=-1,
                    high=1,
                )
            },
            action_space="continuous",
        ),
    )
    spec.pipeline = continuous_policy_mlp(action_dim=2)
    return spec
