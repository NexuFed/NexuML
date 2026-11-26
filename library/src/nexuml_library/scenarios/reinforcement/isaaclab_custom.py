"""Static example of a genuinely registered custom native task and typed configuration."""

from typing import Any

from nexuml.core.discovery import scenario
from nexuml.core.types import (
    FactorySpec,
    InteractionContract,
    PhysicalSystemContract,
    PhysicalTensorContract,
    ScenarioSpec,
    TensorFieldContract,
)
from nexuml_library.environments.isaaclab import (
    IsaacBindings,
    IsaacLabEnvironment,
    isaac_task_config,
)
from nexuml_library.environments._isaaclab_tasks import configure_custom_cartpole
from .isaaclab import isaac_cartpole_ppo


@scenario("isaac-custom-cartpole-ppo")
def isaac_custom_cartpole_ppo(
    total_frames=50_000, frames_per_batch=1_024, num_envs=32
) -> ScenarioSpec:
    """Configure the installed custom task without loading its native configuration.

    Returns:
        Ordinary PPO with explicit native I/O and replacement-config customization.
    """
    spec = isaac_cartpole_ppo(total_frames, frames_per_batch, num_envs)
    contract = IsaacLabEnvironment(physics="newton_mjwarp").describe()
    spec.name = "isaac-custom-cartpole-ppo"
    spec.interaction.environment = IsaacLabEnvironment(
        task="NexuML-Custom-Cartpole",
        physics="newton_mjwarp",
        contract=contract,
        bindings=IsaacBindings(observations={"observation": ("policy",)}),
        task_config=isaac_task_config(
            configure_custom_cartpole, episode_length_s=2.0, env_spacing=3.0
        ),
    )
    return spec


def describe_centering_controls(native: Any) -> PhysicalTensorContract:
    """Inspect the exact inherited native effort mechanism on the live task.

    Returns:
        Physical description read from the instantiated native controls.

    Raises:
        ValueError: If the registered task's verified control mechanism changed.
    """
    from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

    from nexuml_isaac_tasks.centering_cartpole import CenteringCartpole

    if (
        type(native) is not CenteringCartpole
        or type(native)._pre_physics_step is not CartpoleEnv._pre_physics_step
        or type(native)._apply_action is not CartpoleEnv._apply_action
        or native.cfg.action_noise_model is not None
    ):
        raise ValueError("Expected the native centering task's unmodified effort controls")
    return PhysicalTensorContract(
        units=("N",),
        frame="joint",
        joints=tuple(native.cartpole.joint_names[index] for index in native._cart_dof_idx.tolist()),
        control_mode="effort",
        scale=(native.action_scale,),
        offset=(0.0,),
    )


@scenario("isaac-custom-direct-cartpole-ppo")
def isaac_custom_direct_cartpole_ppo(
    total_frames=50_000, frames_per_batch=1_024, num_envs=32
) -> ScenarioSpec:
    """Use the registered native centering task without modifying its configuration.

    Returns:
        A direct PPO scenario with explicit physical controls and observation bindings.
    """
    spec = isaac_cartpole_ppo(total_frames, frames_per_batch, num_envs)
    assert spec.interaction is not None
    spec.name = "isaac-custom-direct-cartpole-ppo"
    spec.interaction.environment = IsaacLabEnvironment(
        task="nexuml_isaac_tasks.centering_cartpole:NexuML-Cartpole-Direct-Centering",
        physics=None,
        bindings=IsaacBindings(
            observations={"observation": ("policy",)},
            direct_controls=FactorySpec.from_factory(describe_centering_controls),
        ),
        contract=InteractionContract(
            action_space="continuous",
            control_period_s=1 / 60,
            observations={
                "observation": TensorFieldContract(
                    shape=(4,),
                    dtype="float32",
                    modality="vector",
                    physical=PhysicalTensorContract(
                        units=("m", "rad", "m/s", "rad/s"),
                        frame="joint",
                        joints=("slider_to_cart", "cart_to_pole") * 2,
                        sensor_id="relative-joint-position-and-velocity",
                    ),
                )
            },
            actions={
                "action": TensorFieldContract(
                    shape=(1,),
                    dtype="float32",
                    low=-1,
                    high=1,
                    rate_hz=60,
                    physical=PhysicalTensorContract(
                        units=("N",),
                        frame="joint",
                        joints=("slider_to_cart",),
                        control_mode="effort",
                        scale=(100.0,),
                        offset=(0.0,),
                    ),
                )
            },
            physical=PhysicalSystemContract(
                embodiment_id="prismatic-cart-unactuated-hinge-pole",
                calibration_id="upright-zero-positive-tilt-opposes-positive-cart-axis-v1",
                asset_source=(
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
                    "Isaac/6.2/Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd"
                ),
                asset_id="sha256:c98ce5dbb174876998052d486036fb79e07f52320851526a4ff61c60ed2db043",
            ),
        ),
    )
    return spec
