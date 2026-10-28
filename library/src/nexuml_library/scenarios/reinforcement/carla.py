"""Camera plus vehicle-state reference control, not a production driving stack."""

from nexuml.core.discovery import scenario
from nexuml.core.types import InteractionSpec, RLEvaluationSpec, ScenarioSpec
from nexuml_library.environments.carla import CarlaLaneFollowingEnvironment
from nexuml_library.reinforcement.policies import continuous_policy_mlp
from nexuml_library.scenarios.reinforcement.gymnasium import pendulum_ppo


@scenario("carla-lane-following-ppo")
def carla_lane_following_ppo(total_frames=10_000, frames_per_batch=128) -> ScenarioSpec:
    """Use a dedicated, already-running CARLA server and its matching Python client.

    Returns:
        A bounded camera/ego-state PPO integration scenario.
    """
    spec = pendulum_ppo(total_frames, frames_per_batch)
    spec.name = "carla-lane-following-ppo"
    definition = CarlaLaneFollowingEnvironment()
    spec.interaction = InteractionSpec(environment=definition)
    spec.pipeline = continuous_policy_mlp(
        observation_keys=tuple(definition.describe().observations),
        action_dim=3,
    )
    spec.training.evaluation = RLEvaluationSpec(episodes=3, max_steps_per_episode=300)
    return spec
