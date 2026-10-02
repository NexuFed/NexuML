"""Bounded joint-reach task, independent of ROS message types and reward transport."""

import torch
from pydantic import Field, model_validator
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.discovery import environment
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import InteractionContract, TensorFieldContract
from nexuml.interaction.runtime import InteractionStep
from nexuml_library.environments.ros2.transport import Ros2Transport


@environment("Ros2JointReachEnvironment")
class Ros2JointReachEnvironment(EnvironmentDefinition):
    """Explicit simulation contract; no ROS connection during description."""

    joint_names: tuple[str, ...] = ("joint1", "joint2")
    initial_positions: tuple[float, ...] = (0.0, 0.0)
    target_positions: tuple[float, ...] = (0.5, -0.5)
    joint_low: tuple[float, ...] = (-1.0, -1.0)
    joint_high: tuple[float, ...] = (1.0, 1.0)
    joint_state_topic: str = "/joint_states"
    command_topic: str = "/joint_trajectory_controller/joint_trajectory"
    reset_service: str = "/reset_simulation"
    namespace: str = ""
    use_sim_time: bool = True
    control_period_s: float = Field(default=0.05, gt=0)
    timeout_s: float = Field(default=2.0, gt=0)
    max_episode_steps: int = Field(default=200, gt=0)
    success_tolerance: float = Field(default=0.05, gt=0)

    @model_validator(mode="after")
    def validate_joint_configuration(self):
        count = len(self.joint_names)
        if (
            count == 0
            or len(set(self.joint_names)) != count
            or any(not name.strip() for name in self.joint_names)
        ):
            raise ValueError("joint_names must be nonempty and unique")
        values = (self.initial_positions, self.target_positions, self.joint_low, self.joint_high)
        if any(len(value) != count for value in values):
            raise ValueError("joint positions and bounds must match joint_names")
        for initial, target, low, high in zip(*values):
            if not low <= initial <= high or not low <= target <= high or low >= high:
                raise ValueError("initial/target positions must lie within ordered joint limits")
        if not self.joint_state_topic or not self.command_topic or not self.reset_service:
            raise ValueError("ROS topics and the simulation reset service must be explicit")
        return self

    def describe(self) -> InteractionContract:
        shape = (len(self.joint_names),)
        return InteractionContract(
            observations={
                "joint_position": TensorFieldContract(
                    shape=shape, dtype="float32", modality="proprioception"
                ),
                "joint_velocity": TensorFieldContract(
                    shape=shape, dtype="float32", modality="proprioception"
                ),
                "target_position": TensorFieldContract(
                    shape=shape, dtype="float32", modality="joint_target"
                ),
            },
            actions={
                "joint_command": TensorFieldContract(
                    shape=shape,
                    dtype="float32",
                    modality="joint_command",
                    low=list(self.joint_low),
                    high=list(self.joint_high),
                    rate_hz=1 / self.control_period_s,
                )
            },
            action_space="continuous",
            control_period_s=self.control_period_s,
        )

    def build(self, context: EnvironmentBuildContext):
        if context.num_envs != 1:
            raise ValueError(
                "ROS joint control supports one robot per worker; use isolated ROS namespaces"
            )
        from nexuml_library.environments.ros2.rclpy_transport import RclpyTransport

        return _JointReachRuntime(self, context, RclpyTransport(self, context))


class _JointReachRuntime:
    def __init__(self, definition, context, transport: Ros2Transport):
        if context.num_envs != 1:
            raise ValueError("ROS joint control supports one robot per worker")
        self.definition, self.context, self.transport = definition, context, transport
        self.contract = definition.describe()
        self.steps = 0
        self.closed = False
        self._ready = False

    def _observation(self):
        position, velocity = self.transport.receive_joint_state(self.definition.timeout_s)
        observations = TensorDict(
            {
                "joint_position": torch.tensor(
                    position, dtype=torch.float32, device=self.context.device
                ),
                "joint_velocity": torch.tensor(
                    velocity, dtype=torch.float32, device=self.context.device
                ),
                "target_position": torch.tensor(
                    self.definition.target_positions,
                    dtype=torch.float32,
                    device=self.context.device,
                ),
            },
            batch_size=[],
        )
        CompiledPolicy._validate_fields(observations, self.contract.observations, "Observation")
        return observations

    def reset(self, *, seed=None, mask=None):
        if self.closed:
            raise RuntimeError("ROS environment is closed")
        self._ready = False
        self.transport.reset(self.definition.timeout_s)
        self.transport.send_joint_command(
            list(self.definition.initial_positions), self.definition.control_period_s
        )
        observation = self._observation()
        self.steps = 0
        self._ready = True
        return observation

    def step(self, actions):
        if self.closed or not self._ready:
            raise RuntimeError("Reset the ROS environment before stepping")
        CompiledPolicy._validate_fields(actions, self.contract.actions, "Action")
        if actions.batch_size:
            raise ValueError("One ROS robot requires unbatched actions")
        self._ready = False
        self.transport.send_joint_command(
            actions["joint_command"].detach().cpu().tolist(), self.definition.control_period_s
        )
        observation = self._observation()
        self.steps += 1
        error = (
            (observation["joint_position"] - observation["target_position"]).square().sum().sqrt()
        )
        success = bool(error <= self.definition.success_tolerance)
        done = success or self.steps >= self.definition.max_episode_steps
        self._ready = not done
        return InteractionStep(
            observation=observation,
            reward=-error.reshape(1),
            terminated=torch.tensor([success], device=self.context.device),
            truncated=torch.tensor([not success and done], device=self.context.device),
        )

    def close(self):
        if not self.closed:
            self.closed = True
            self._ready = False
            self.transport.close()
