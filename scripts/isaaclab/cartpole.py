"""Declarative state-only Cartpole: nexuml train --scenario-file this_file.py."""

from time import time_ns

from nexuml.core.types import ExportSpec, ScenarioSpec
from nexuml_library.environments.isaaclab_config import ViserSpec
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_learning_ppo


def scenario() -> ScenarioSpec:
    """Declare training, a private spectator and a unique complete-policy export.

    Returns:
        The native state-only Cartpole learning reference.
    """
    spec = isaac_cartpole_learning_ppo()
    spec.interaction.environment = spec.interaction.environment.model_copy(
        update={"viewer": ViserSpec(port=8080, max_visible_envs=4, lookat=(0.0, 0.0, 2.0))}
    )
    spec.exports = [ExportSpec(output=f"rl-runs/cartpole-{time_ns()}/policy")]
    return spec
