"""Distinct scalar categorical effort reference using the ordinary PPO path."""

from nexuml.core.discovery import scenario
from nexuml.core.types import PhysicalTensorContract, PolicySpec, ScenarioSpec, TensorFieldContract
from nexuml_library.action_adapters.adapters import CategoricalActionAdapter
from nexuml_library.environments._isaaclab_categorical import ACTION_FORCES
from nexuml_library.environments.isaaclab import IsaacBindings, IsaacLabEnvironment
from nexuml_library.reinforcement.policies import discrete_policy_mlp
from .isaaclab import isaac_cartpole_ppo


@scenario("isaac-categorical-cartpole-ppo")
def isaac_categorical_cartpole_ppo(
    total_frames=50_000, frames_per_batch=1_024, num_envs=32
) -> ScenarioSpec:
    """Configure three explicit force commands without relabeling continuous Cartpole.

    Returns:
        Ordinary categorical PPO over the distinctly registered direct native task.
    """
    spec = isaac_cartpole_ppo(total_frames, frames_per_batch, num_envs)
    reference = IsaacLabEnvironment(
        task="Isaac-Cartpole-Direct", physics="newton_mjwarp"
    ).describe()
    action = TensorFieldContract(
        shape=(),
        dtype="int64",
        num_values=3,
        low=0,
        high=2,
        rate_hz=60,
        description="Classes 0, 1, 2 command -100, 0, 100 N on slider_to_cart.",
        physical=PhysicalTensorContract(
            units=("N",),
            frame="joint",
            joints=("slider_to_cart",),
            control_mode="effort",
            scale=(1.0,),
            offset=(0.0,),
            class_controls=tuple((force,) for force in ACTION_FORCES),
        ),
    )
    contract = reference.model_copy(
        update={"action_space": "discrete", "actions": {"action": action}}
    )
    assert spec.interaction is not None
    spec.name = "isaac-categorical-cartpole-ppo"
    spec.interaction.environment = IsaacLabEnvironment(
        task="NexuML-Cartpole-Categorical",
        physics="newton_mjwarp",
        contract=contract,
        bindings=IsaacBindings(observations={"observation": ("policy",)}),
    )
    spec.pipeline = discrete_policy_mlp(num_actions=3)
    spec.policy = PolicySpec(action_adapter=CategoricalActionAdapter())
    return spec
