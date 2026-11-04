"""Camera/ego-state lane control on a dedicated CARLA server; no import-time client."""

import math
import random
from queue import Empty, Queue
from time import monotonic
from typing import Any

import numpy as np
import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.discovery import environment
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import InteractionContract, TensorFieldContract
from nexuml.interaction.runtime import InteractionStep


@environment("CarlaLaneFollowingEnvironment")
class CarlaLaneFollowingEnvironment(EnvironmentDefinition):
    """Simulation-only, single tick owner; runtime restores settings and owns its actors."""

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=2000, ge=1, le=65535)
    timeout_s: float = Field(default=10.0, gt=0)
    image_width: int = Field(default=32, gt=0)
    image_height: int = Field(default=32, gt=0)
    vehicle_blueprint: str = "vehicle.tesla.model3"
    control_period_s: float = Field(default=0.05, gt=0, le=0.1)
    max_episode_steps: int = Field(default=300, gt=0)
    target_speed_m_s: float = Field(default=8.0, gt=0)

    def describe(self) -> InteractionContract:
        return InteractionContract(
            observations={
                "front_rgb": TensorFieldContract(
                    shape=(3, self.image_height, self.image_width),
                    dtype="uint8",
                    modality="image",
                    low=0,
                    high=255,
                ),
                "ego_state": TensorFieldContract(
                    shape=(3,),
                    dtype="float32",
                    modality="vehicle_state",
                    description="speed (m/s), signed lane-center offset (m), heading error (rad)",
                ),
            },
            actions={
                "action": TensorFieldContract(
                    shape=(3,),
                    dtype="float32",
                    modality="vehicle_control",
                    low=[-1.0, 0.0, 0.0],
                    high=[1.0, 1.0, 1.0],
                    description="steering, throttle, brake",
                    rate_hz=1 / self.control_period_s,
                )
            },
            action_space="continuous",
            control_period_s=self.control_period_s,
        )

    def build(self, context: EnvironmentBuildContext):
        if context.num_envs != 1 or context.worker_rank != 0:
            raise ValueError("CARLA requires one direct collector and a dedicated server")
        try:
            import carla  # ty: ignore[unresolved-import]
        except ImportError as error:
            raise ImportError(
                "CARLA requires a matching simulator server and CARLA Python client built for "
                "your Python version; neither is installed by nexuml[reinforcement]"
            ) from error
        return _CarlaRuntime(self, context, carla)


class _CarlaRuntime:
    def __init__(self, definition, context, carla):
        self.definition, self.context, self.carla = definition, context, carla
        self.contract = definition.describe()
        self.actors = []
        self.world: Any = None
        self.original_settings = None
        self.closed, self.ready = False, False
        self.frames = Queue()
        try:
            client = carla.Client(definition.host, definition.port)
            client.set_timeout(definition.timeout_s)
            self.world = client.get_world()
            settings = self.world.get_settings()
            if settings.synchronous_mode or settings.no_rendering_mode:
                raise ValueError("CARLA needs an idle asynchronous server with rendering enabled")
            self.original_settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = definition.control_period_s
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            self.world.apply_settings(settings)
            self.map = self.world.get_map()
        except BaseException:
            self.close()
            raise

    def _destroy_actors(self):
        error = None
        for actor in reversed(self.actors):
            try:
                try:
                    if hasattr(actor, "stop"):
                        actor.stop()
                finally:
                    if not actor.destroy():
                        raise RuntimeError("CARLA actor destruction failed")
            except Exception as failure:
                error = error or failure
        self.actors.clear()
        if error is not None:
            raise error

    def reset(self, *, seed=None, mask=None):
        if self.closed:
            raise RuntimeError("CARLA environment is closed")
        self.ready = False
        self._destroy_actors()
        self.frames = Queue()
        self.collided = False
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA map has no vehicle spawn points")
        spawn = random.Random(seed).choice(spawn_points)
        self.vehicle = self.world.try_spawn_actor(
            blueprint_library.find(self.definition.vehicle_blueprint),
            spawn,
        )
        if self.vehicle is None:
            raise RuntimeError("CARLA vehicle spawn is occupied; use an empty dedicated server")
        self.actors.append(self.vehicle)
        camera_blueprint = blueprint_library.find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", str(self.definition.image_width))
        camera_blueprint.set_attribute("image_size_y", str(self.definition.image_height))
        camera_blueprint.set_attribute("sensor_tick", "0.0")
        self.camera = self.world.spawn_actor(
            camera_blueprint,
            self.carla.Transform(self.carla.Location(x=1.5, z=2.4)),
            attach_to=self.vehicle,
        )
        self.actors.append(self.camera)
        self.camera.listen(self.frames.put)
        collision = self.world.spawn_actor(
            blueprint_library.find("sensor.other.collision"),
            self.carla.Transform(),
            attach_to=self.vehicle,
        )
        self.actors.append(collision)
        collision.listen(self._on_collision)
        self.steps = 0
        observation, _ = self._observe()
        self.ready = True
        return observation

    def _on_collision(self, event):
        self.collided = True

    def _observe(self):
        frame = self.world.tick(self.definition.timeout_s)
        deadline = monotonic() + self.definition.timeout_s
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CARLA camera timed out waiting for frame {frame}")
            try:
                image = self.frames.get(timeout=remaining)
            except Empty as error:
                raise TimeoutError(f"CARLA camera timed out waiting for frame {frame}") from error
            if image.frame == frame:
                break
            if image.frame > frame:
                raise RuntimeError(
                    "CARLA camera/world frames diverged; another client may be ticking"
                )
        rgb = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        rgb = rgb[:, :, 2::-1].transpose(2, 0, 1).copy()
        transform, velocity = self.vehicle.get_transform(), self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        waypoint = self.map.get_waypoint(transform.location)
        offset, heading, off_lane = 0.0, 0.0, waypoint is None
        if waypoint is not None:
            delta = transform.location - waypoint.transform.location
            right = waypoint.transform.get_right_vector()
            offset = delta.x * right.x + delta.y * right.y
            angle = math.radians(transform.rotation.yaw - waypoint.transform.rotation.yaw)
            heading = math.atan2(math.sin(angle), math.cos(angle))
            off_lane = abs(offset) > waypoint.lane_width / 2
        observation = TensorDict(
            {
                "front_rgb": torch.as_tensor(rgb, device=self.context.device),
                "ego_state": torch.tensor(
                    [speed, offset, heading], dtype=torch.float32, device=self.context.device
                ),
            },
            batch_size=[],
        )
        CompiledPolicy._validate_fields(observation, self.contract.observations, "Observation")
        return observation, off_lane

    def step(self, actions):
        if self.closed or not self.ready:
            raise RuntimeError("Reset the CARLA environment before stepping")
        CompiledPolicy._validate_fields(actions, self.contract.actions, "Action")
        if actions.batch_size:
            raise ValueError("CARLA requires unbatched actions")
        steer, throttle, brake = actions["action"].detach().cpu().tolist()
        self.ready = False
        self.vehicle.apply_control(
            self.carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        )
        observation, off_lane = self._observe()
        self.steps += 1
        speed, offset, heading = observation["ego_state"].tolist()
        # ponytail: local lane-center reward; add route/traffic objectives for driving benchmarks.
        reward = speed * math.cos(heading) - abs(offset) - abs(heading)
        reward -= 0.5 * abs(speed - self.definition.target_speed_m_s)
        terminated = self.collided or off_lane
        truncated = not terminated and self.steps >= self.definition.max_episode_steps
        self.ready = not (terminated or truncated)
        device = self.context.device
        return InteractionStep(
            observation=observation,
            reward=torch.tensor([reward], dtype=torch.float32, device=device),
            terminated=torch.tensor([terminated], device=device),
            truncated=torch.tensor([truncated], device=device),
        )

    def close(self):
        if self.closed:
            return
        self.closed, self.ready = True, False
        try:
            self._destroy_actors()
        finally:
            if self.original_settings is not None:
                self.world.apply_settings(self.original_settings)
