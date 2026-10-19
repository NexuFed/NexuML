"""Pinned state-only Isaac contracts; native resources exist only after build."""

from typing import TYPE_CHECKING, Literal

import torch
from pydantic import model_validator

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.discovery import environment
from nexuml.core.types import (
    InteractionContract,
    PhysicalSystemContract,
    PhysicalTensorContract,
    ReinforcementLearningSpec,
    TensorFieldContract,
)

if TYPE_CHECKING:
    from nexuml.interaction.runtime import EnvironmentRuntime


_TASK_PROFILES: dict[str, tuple[int, int, float, str, str]] = {
    "Isaac-Cartpole": (
        4,
        1,
        1 / 60,
        "Pinned Cartpole state-only policy observations.",
        "Joint order: slider_to_cart; continuous joint effort = 100 * action N.",
    ),
    "Isaac-Cartpole-Direct": (
        4,
        1,
        1 / 60,
        "Relative cart/pole positions followed by relative cart/pole velocities.",
        "Joint order: slider_to_cart; continuous joint effort = 100 * action N.",
    ),
    "Isaac-Reach-Franka": (
        32,
        7,
        1 / 30,
        "Pinned Franka Reach state-only policy observations.",
        "Joint order: panda_joint1, panda_joint2, panda_joint3, panda_joint4, "
        "panda_joint5, panda_joint6, panda_joint7; joint-position targets = "
        "pinned default joint pose + 0.5 * action rad (default-pose offsets).",
    ),
    "NexuML-SO101-Joint-Reach": (
        15,
        5,
        1 / 30,
        "Five arm positions (rad), velocities (rad/s), and target positions (rad), "
        "each ordered shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.",
        "Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll; "
        "absolute joint-position targets = 0.5 * action rad in [-0.5, 0.5], "
        "without default-pose offsets, using the pinned SO101_CFG PhysX asset; "
        "gripper is fixed at 0.2555162024919699 rad, not a policy control.",
    ),
}


@environment("IsaacLabEnvironment")
class IsaacLabEnvironment(EnvironmentDefinition):
    """Immutable candidate profile; description is not native compatibility proof."""

    task: Literal[
        "Isaac-Cartpole", "Isaac-Cartpole-Direct", "Isaac-Reach-Franka", "NexuML-SO101-Joint-Reach"
    ] = "Isaac-Cartpole"
    revision: Literal[
        "ae37b028ea415c91ea2bc32609efcd759ed2b974",
        "70ab24fec3eabf51222817db0cdb5a4e7133bd7e",
    ] = "70ab24fec3eabf51222817db0cdb5a4e7133bd7e"
    physics: Literal["isaacsim_physx", "newton_mjwarp", "ovphysx"] = "isaacsim_physx"

    @model_validator(mode="after")
    def _validate_profile(self):
        if self.revision == "ae37b028ea415c91ea2bc32609efcd759ed2b974" and (
            self.physics != "isaacsim_physx" or self.task == "Isaac-Cartpole-Direct"
        ):
            raise ValueError("Legacy Isaac artifacts retain their original task/PhysX profile")
        if self.task == "NexuML-SO101-Joint-Reach" and self.physics != "isaacsim_physx":
            raise ValueError(
                "SO-101 joint reach requires its validated PhysX asset/control profile"
            )
        return self

    def describe(self) -> InteractionContract:
        obs_size, action_size, period, observation_description, mapping = _TASK_PROFILES[self.task]
        observation_physical = action_physical = system_physical = None
        if (
            self.task == "NexuML-SO101-Joint-Reach"
            and self.revision == "70ab24fec3eabf51222817db0cdb5a4e7133bd7e"
        ):
            from ._isaaclab_so101 import (
                ASSET_LAYER_SHA256,
                ASSET_MANIFEST_ID,
                ASSET_SOURCE,
                JOINT_NAMES,
            )

            mapping += (
                f" Asset={ASSET_SOURCE}; sha256={ASSET_LAYER_SHA256['so101_new_calib.usda']}."
            )
            observation_physical = PhysicalTensorContract(
                units=("rad",) * 5 + ("rad/s",) * 5 + ("rad",) * 5,
                frame="robot_joint",
                joints=JOINT_NAMES * 3,
                sensor_id="simulation:joint-state-and-task-goal",
            )
            action_physical = PhysicalTensorContract(
                units=("rad",),
                frame="robot_joint",
                joints=JOINT_NAMES,
                control_mode="absolute_position",
                scale=(0.5,),
                offset=(0.0,),
            )
            system_physical = PhysicalSystemContract(
                embodiment_id="simulation:so101-calibrated-fixed-gripper",
                asset_source=ASSET_SOURCE,
                asset_id=ASSET_MANIFEST_ID,
                calibration_id="simulation:usd-authored-joint-zero",
            )
        return InteractionContract(
            observations={
                "observation": TensorFieldContract(
                    shape=(obs_size,),
                    dtype="float32",
                    modality="vector",
                    description=observation_description,
                    physical=observation_physical,
                )
            },
            actions={
                "action": TensorFieldContract(
                    shape=(action_size,),
                    dtype="float32",
                    low=-1,
                    high=1,
                    rate_hz=1 / period,
                    description=(
                        f"task={self.task}; revision={self.revision}; physics={self.physics}; "
                        f"normalized dimensionless [-1, 1] controls; period={period} s. {mapping}"
                    ),
                    physical=action_physical,
                )
            },
            action_space="continuous",
            num_envs=1,
            control_period_s=period,
            physical=system_physical,
        )

    def validate_training(
        self, training: ReinforcementLearningSpec, *, device: str | None = None
    ) -> None:
        collector = training.collector
        if collector.backend != "direct" or not collector.sync or collector.num_collectors != 1:
            raise ValueError(
                "Isaac Lab requires one synchronous direct collector: backend='direct', "
                "sync=True, num_collectors=1"
            )
        if training.devices != 1 or training.strategy != "auto":
            raise ValueError("Isaac Lab requires training.devices=1 and training.strategy='auto'")
        configured_devices = {
            "collector.env_device": collector.env_device,
            "collector.storing_device": collector.storing_device,
        }
        if collector.policy_device != "auto":
            configured_devices["collector.policy_device"] = collector.policy_device
        if device is not None:
            configured_devices["learner device"] = device
        indices = set()
        for name, value in configured_devices.items():
            try:
                parsed = torch.device(value)
            except (RuntimeError, TypeError) as error:
                raise ValueError(
                    f"Isaac Lab requires a CUDA device for {name}, got {value!r}"
                ) from error
            if parsed.type != "cuda":
                raise ValueError(f"Isaac Lab requires a CUDA device for {name}, got {value!r}")
            indices.add(parsed.index if parsed.index is not None else 0)
        if len(indices) != 1:
            raise ValueError(
                "Isaac Lab environment, storage, policy, and learner must use the same CUDA "
                "device index (cuda means cuda:0)"
            )
        if training.frames_per_batch % collector.num_envs_per_collector:
            raise ValueError(
                "Isaac Lab frames_per_batch must be divisible by collector.num_envs_per_collector"
            )
        evaluation = training.evaluation
        if (
            evaluation.episodes > 0
            or evaluation.at_start
            or evaluation.every_n_frames is not None
            or evaluation.video.enabled
        ):
            raise ValueError(
                "Isaac Lab does not support automatic fit evaluation or video. Disable episodes, "
                "at_start, every_n_frames, and video; use fresh-process replay instead: "
                "nexuml evaluate <policy-package> --device cuda:0 "
                "--override training.evaluation.episodes=3"
            )

    def build(self, context: EnvironmentBuildContext) -> "EnvironmentRuntime":
        from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime

        return _IsaacLabRuntime(self, context)
