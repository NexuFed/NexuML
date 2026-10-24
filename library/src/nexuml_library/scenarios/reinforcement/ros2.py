"""The maintained ROS joint-reach task with algorithm-independent policy semantics."""

from nexuml.core.discovery import scenario
from nexuml.core.types import InteractionSpec, PolicySpec, RLEvaluationSpec, ScenarioSpec
from nexuml_library.action_adapters.adapters import GaussianActionAdapter
from nexuml_library.environments.ros2.joint_reach import Ros2JointReachEnvironment
from nexuml_library.reinforcement.policies import continuous_policy_mlp
from nexuml_library.scenarios.reinforcement.gymnasium import pendulum_ppo


@scenario("ros2-gazebo-joint-reach-ppo")
def ros2_joint_reach_ppo(total_frames=10_000, frames_per_batch=128) -> ScenarioSpec:
    """Connect only when explicitly executed against an operator-provided ROS graph.

    Returns:
        A bounded two-joint simulation reference scenario.
    """
    spec = pendulum_ppo(total_frames, frames_per_batch)
    spec.name = "ros2-gazebo-joint-reach-ppo"
    definition = Ros2JointReachEnvironment()
    spec.interaction = InteractionSpec(environment=definition)
    spec.pipeline = continuous_policy_mlp(
        observation_keys=tuple(definition.describe().observations),
        action_dim=2,
    )
    spec.policy = PolicySpec(action_adapter=GaussianActionAdapter(action_key="joint_command"))
    spec.training.evaluation = RLEvaluationSpec(episodes=3, max_steps_per_episode=200)
    return spec
