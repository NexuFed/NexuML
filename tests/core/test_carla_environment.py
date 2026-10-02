"""CARLA configuration, fake-server wiring, and a separately opted-in live smoke."""

import copy
import builtins
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import export_package, load_inference_package, load_package
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import ReinforcementLearningSpec, RLEvaluationSpec
from nexuml.training.lightning import NexuSession
from nexuml_library.environments.carla import CarlaLaneFollowingEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.carla import carla_lane_following_ppo


def test_contract_round_trip_and_export_without_carla(tmp_path, monkeypatch):
    spec = carla_lane_following_ppo(total_frames=16, frames_per_batch=16)
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(spec).to_yaml()).to_scenario()
    assert restored.interaction is not None and restored.policy is not None
    contract = restored.interaction.environment.describe()
    policy = CompiledPolicy(
        compile(restored), restored.policy.action_adapter.build(contract), contract
    )
    observations = TensorDict(
        {
            key: torch.zeros(field.shape, dtype=getattr(torch, field.dtype))
            for key, field in contract.observations.items()
        },
        batch_size=[],
    )
    path = tmp_path / "carla-policy.nexu"
    export_package(policy, path)
    loaded, _, _ = load_package(path)
    torch.testing.assert_close(loaded(observations)["action"], policy(observations)["action"])
    monkeypatch.setitem(sys.modules, "carla", None)
    original_import = builtins.__import__

    def without_interaction_dependencies(name, *args, **kwargs):
        if name.split(".")[0] in {"carla", "rclpy", "torchrl", "gymnasium"}:
            raise ImportError(f"{name} must not be imported for packaged inference")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_interaction_dependencies)
    packaged, _, metadata = load_inference_package(path)
    torch.testing.assert_close(packaged(observations)["action"], policy(observations)["action"])
    assert not any(
        dependency["module"] in {"carla", "rclpy"}
        for dependency in metadata["external_dependencies"]
    )
    assert contract.observations["front_rgb"].modality == "image"
    assert contract.observations["ego_state"].shape == (3,)


def test_missing_client_and_unsupported_topology_fail_without_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "carla", None)
    definition = CarlaLaneFollowingEnvironment()
    with pytest.raises(ImportError, match="matching simulator server"):
        definition.build(EnvironmentBuildContext(purpose="train"))
    with pytest.raises(ValueError, match="one direct collector"):
        definition.build(EnvironmentBuildContext(purpose="train", num_envs=2))


def test_fake_server_exercises_real_runtime_frames_controls_and_cleanup(monkeypatch):
    events, actors = [], []

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

        def __sub__(self, other):
            return Location(self.x - other.x, self.y - other.y, self.z - other.z)

    transform = SimpleNamespace(location=Location(), rotation=SimpleNamespace(yaw=0.0))
    waypoint = SimpleNamespace(
        transform=SimpleNamespace(
            location=Location(),
            rotation=SimpleNamespace(yaw=0.0),
            get_right_vector=lambda: Location(y=1.0),
        ),
        lane_width=3.0,
    )
    settings = SimpleNamespace(
        synchronous_mode=False, no_rendering_mode=False, fixed_delta_seconds=None
    )

    class Actor:
        def __init__(self, kind):
            self.kind, self.callback = kind, None
            actors.append(self)

        def listen(self, callback):
            self.callback = callback

        def stop(self):
            events.append((self.kind, "stop"))

        def destroy(self):
            events.append((self.kind, "destroy"))
            return True

        def get_transform(self):
            return transform

        def get_velocity(self):
            return Location(x=2.0)

        def apply_control(self, control):
            events.append(control)

    class World:
        frame = 0
        send_camera = True

        def get_settings(self):
            return copy.copy(settings)

        def apply_settings(self, new):
            events.append(new.synchronous_mode)

        def get_map(self):
            return SimpleNamespace(
                get_spawn_points=lambda: [transform], get_waypoint=lambda location: waypoint
            )

        def get_blueprint_library(self):
            return SimpleNamespace(
                find=lambda name: SimpleNamespace(id=name, set_attribute=lambda key, value: None)
            )

        def try_spawn_actor(self, blueprint, spawn):
            return Actor(blueprint.id)

        def spawn_actor(self, blueprint, spawn, attach_to):
            return Actor(blueprint.id)

        def tick(self, timeout):
            self.frame += 1
            if self.send_camera:
                camera = next(
                    actor for actor in reversed(actors) if actor.kind == "sensor.camera.rgb"
                )
                camera.callback(
                    SimpleNamespace(
                        frame=self.frame,
                        height=2,
                        width=2,
                        raw_data=bytes([10, 20, 30, 255] * 4),
                    )
                )
            return self.frame

    world = World()
    carla = SimpleNamespace(
        Client=lambda host, port: SimpleNamespace(
            set_timeout=lambda timeout: None, get_world=lambda: world
        ),
        Location=Location,
        Transform=lambda location=None: location,
        VehicleControl=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, "carla", carla)
    definition = CarlaLaneFollowingEnvironment(
        image_width=2, image_height=2, max_episode_steps=2, timeout_s=0.01
    )
    runtime = definition.build(EnvironmentBuildContext(purpose="train"))
    try:
        observation = runtime.reset(seed=0)
        assert observation["front_rgb"][:, 0, 0].tolist() == [30, 20, 10]
        assert observation["ego_state"].tolist() == [2.0, 0.0, 0.0]
        action = TensorDict({"action": torch.tensor([0.25, 0.5, 0.0])}, batch_size=[])
        transition = runtime.step(action)
        assert not transition.terminated.item() and not transition.truncated.item()
        assert any(getattr(event, "steer", None) == 0.25 for event in events)
        assert runtime.step(action).truncated.item()
        runtime.reset(seed=0)
        runtime._on_collision(object())
        assert runtime.step(action).terminated.item()
        runtime.reset(seed=0)
        transform.location.y = 3.0
        assert runtime.step(action).terminated.item()
        transform.location.y = 0.0
        runtime.reset(seed=0)
        world.send_camera = False
        with pytest.raises(TimeoutError, match="camera timed out"):
            runtime.step(action)
        with pytest.raises(RuntimeError, match="Reset"):
            runtime.step(action)
    finally:
        runtime.close()
        runtime.close()
    assert events[-1] is False
    assert len(
        [event for event in events if isinstance(event, tuple) and event[1] == "destroy"]
    ) == len(actors)


@pytest.mark.requires_system
@pytest.mark.requires_optional("carla")
@pytest.mark.requires_optional("torchrl")
def test_live_carla_collection_evaluation_and_policy_export(tmp_path):
    import os

    spec = carla_lane_following_ppo(total_frames=16, frames_per_batch=16)
    assert spec.interaction is not None and isinstance(spec.training, ReinforcementLearningSpec)
    spec.interaction.environment = CarlaLaneFollowingEnvironment(
        host=os.environ.get("NEXUML_CARLA_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEXUML_CARLA_PORT", "2000")),
    )
    spec.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    spec.training.evaluation = RLEvaluationSpec(episodes=1, max_steps_per_episode=2)
    result = NexuSession(
        spec, log_dir=tmp_path, enable_loggers=False, enable_progress_bar=False
    ).run()
    assert result.policy is not None and result.interaction_results
    assert result.trainer.global_step > 0
    path = tmp_path / "live-carla.nexu"
    export_package(result.policy, path)
    loaded, _, _ = load_package(path)
    assert isinstance(loaded, CompiledPolicy)
