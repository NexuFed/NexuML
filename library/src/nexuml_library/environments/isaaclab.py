"""Portable native task declarations; simulation exists only inside an owned job."""

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import Field, model_validator

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.discovery import environment
from nexuml.core.types import (
    InteractionContract,
    PhysicalSystemContract,
    PhysicalTensorContract,
    ReinforcementLearningSpec,
    TensorFieldContract,
)
from .isaaclab_config import (
    IsaacBindings,
    IsaacTaskConfigSpec,
    ViserSpec,
    isaac_task_config,
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


@environment("IsaacLabEnvironment", version="2")
class IsaacLabEnvironment(EnvironmentDefinition):
    """Trusted registered task plus explicit portable configuration and I/O meaning."""

    task: str = Field(default="Isaac-Cartpole", min_length=1, pattern=r"\S")
    revision: Literal["70ab24fec3eabf51222817db0cdb5a4e7133bd7e",] = (
        "70ab24fec3eabf51222817db0cdb5a4e7133bd7e"
    )
    physics: str | None = "isaacsim_physx"
    presets: tuple[str, ...] = ()
    task_config: IsaacTaskConfigSpec | None = None
    contract: InteractionContract | None = None
    bindings: IsaacBindings | None = None
    viewer: ViserSpec | None = None

    @model_validator(mode="after")
    def _validate_profile(self):
        selections = (*self.presets, *((self.physics,) if self.physics is not None else ()))
        if any(
            not item or not item.replace("_", "").replace("-", "").isalnum() for item in selections
        ):
            raise ValueError(
                "Native physics/presets must be named selections, not override strings"
            )
        if len(set(self.presets)) != len(self.presets):
            raise ValueError("Native presets must be unique and ordered")
        if self.task == "NexuML-SO101-Joint-Reach" and self.physics != "isaacsim_physx":
            raise ValueError(
                "SO-101 joint reach requires its validated PhysX asset/control profile"
            )
        if self.task not in _TASK_PROFILES and (self.contract is None or self.bindings is None):
            raise ValueError("Custom native tasks require an explicit contract and bindings")
        if (
            self.bindings is not None
            and not self.bindings.groups
            and any(not path for path in self.bindings.observations.values())
        ):
            raise ValueError("Single-agent native observation paths must be nonempty")
        if self.contract is not None:
            if self.contract.num_envs != 1:
                raise ValueError(
                    "Environment count belongs to the collector, not the task contract"
                )
            if self.contract.control_period_s is None:
                raise ValueError("Native tasks require an explicit control period")
            bindings = self.native_bindings()
            if self.contract.groups:
                if tuple(bindings.groups) != tuple(self.contract.groups):
                    raise ValueError("Native group bindings must match ordered contract groups")
                if self.contract.critic_observations or bindings.critic_observations:
                    raise ValueError(
                        "Privileged critic state has no qualified pre-reset final-state source"
                    )
                for name, group in self.contract.groups.items():
                    binding = bindings.groups[name]
                    if set(binding.observations) != set(group.observations):
                        raise ValueError(
                            "Native group observation bindings must match the contract"
                        )
                    if set(group.actions) != {binding.action}:
                        raise ValueError("Native groups require one explicitly bound action")
                    if binding.critic_observations or binding.processing != "processed":
                        raise ValueError("Native grouped raw/privileged processing is unsupported")
                if bindings.processing != "processed":
                    raise ValueError("Native grouped raw processing is unsupported")
                return self
            if bindings.groups:
                raise ValueError("Grouped native bindings require a grouped contract")
            if set(bindings.observations) != set(self.contract.observations):
                raise ValueError("Native observation bindings must exactly match the contract")
            if set(self.contract.actions) != {bindings.action}:
                raise ValueError("Native single-agent tasks require one explicitly bound action")
        return self

    def native_bindings(self) -> IsaacBindings:
        """Return declared native paths, with the maintained reference's explicit mapping."""
        return self.bindings or IsaacBindings(observations={"observation": ("policy",)})

    def checkpoint_identity(self) -> dict:
        """Retain native task meaning while permitting display-only viewer changes.

        Returns:
            Task-defining declaration with spectator settings excluded.
        """
        identity = super().checkpoint_identity()
        identity["params"].pop("viewer", None)
        return identity

    def describe(self) -> InteractionContract:
        if self.contract is not None:
            return self.contract
        obs_size, action_size, period, observation_description, mapping = _TASK_PROFILES[self.task]
        observation_physical = action_physical = system_physical = None
        if self.task == "NexuML-SO101-Joint-Reach":
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
                        f"task={self.task}; physics={self.physics}; "
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

    def job_scope(self) -> AbstractContextManager[None]:
        """Own native lifetime through checkpoint/export and logger finalization.

        Returns:
            Lazy whole-job scope; entering it alone allocates no simulator.
        """
        from nexuml_library.environments._isaaclab_runtime import isaaclab_process

        return isaaclab_process(self)


__all__ = [
    "IsaacLabEnvironment",
    "IsaacTaskConfigSpec",
    "IsaacBindings",
    "ViserSpec",
    "isaac_task_config",
]
