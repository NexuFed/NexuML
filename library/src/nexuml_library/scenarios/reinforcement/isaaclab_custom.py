"""Static example of a genuinely registered custom native task and typed configuration."""

from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec
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
