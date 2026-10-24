"""The learning reference is explicit and does not alter the sensor smoke recipe."""

from nexuml.core.config import ResolvedConfig
from nexuml.core.scenario_loader import load_scenario_file
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.scenarios.reinforcement.isaaclab import (
    isaac_cartpole_learning_ppo,
    isaac_cartpole_ppo,
)
from nexuml_library.scenarios.reinforcement.isaaclab_sensors import isaac_mixed_cartpole_ppo


def test_learning_recipe_keeps_native_task_and_declares_learning_budget():
    spec = isaac_cartpole_learning_ppo()
    environment = spec.interaction.environment
    training = spec.training
    assert environment.task == "Isaac-Cartpole"
    assert environment.physics == "newton_mjwarp"
    assert environment.task_config == isaac_cartpole_ppo().interaction.environment.task_config
    assert environment.describe().observations["observation"].shape == (4,)
    assert training.collector.num_envs_per_collector == 512
    assert training.frames_per_batch == 8192
    assert training.total_frames == 2097152
    assert training.seed == 0
    assert training.evaluation.episodes == 0
    algorithm = training.algorithm
    assert algorithm.actor_lr == algorithm.critic_lr == 0.001
    assert algorithm.update_epochs == 5
    assert algorithm.minibatch_size == 2048
    assert algorithm.entropy_coef == 0.005
    assert spec.pipeline.stages["policy"][0].component.hidden_dims == (32, 32)
    assert algorithm.critic.stages["value"][0].component.hidden_dims == (32, 32)
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(spec).to_yaml()).to_scenario()
    assert restored == spec


def test_learning_configuration_does_not_change_existing_recipes():
    state = isaac_cartpole_ppo()
    sensor = isaac_mixed_cartpole_ppo()
    isaac_cartpole_learning_ppo()
    assert state.training.total_frames == 50000
    assert state.training.algorithm.actor_lr == 0.0003
    assert sensor.training.total_frames == 256
    assert sensor.training.frames_per_batch == 64
    assert sensor.interaction.environment.task == "NexuML-Cartpole-Sensors"


def test_declarative_example_has_private_bounded_viewer_and_unique_export():
    first = load_scenario_file("scripts/isaaclab/cartpole.py").scenario
    second = load_scenario_file("scripts/isaaclab/cartpole.py").scenario
    assert first.interaction is not None
    environment = first.interaction.environment
    assert isinstance(environment, IsaacLabEnvironment)
    viewer = environment.viewer
    assert viewer is not None
    assert viewer.bind_address == "127.0.0.1" and viewer.port == 8080
    assert viewer.max_visible_envs == 4 and viewer.purposes == ("train",)
    output = first.exports[0].output
    assert output is not None
    assert output.startswith("rl-runs/cartpole-")
    assert output.endswith("/policy")
    assert first.exports[0].output != second.exports[0].output
    assert first.training == second.training
