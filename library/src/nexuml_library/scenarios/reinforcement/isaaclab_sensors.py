"""Bounded raw-camera Cartpole PPO recipes with explicit processing ownership."""

from typing import Literal

from nexuml.core.discovery import scenario
from nexuml.core.types import (
    InteractionContract,
    PhysicalTensorContract,
    PolicySpec,
    ScenarioSpec,
    SensorProcessingContract,
    TensorFieldContract,
)
from nexuml_library.action_adapters.adapters import GaussianActionAdapter
from nexuml_library.environments._isaaclab_sensors import (
    camera_geometry_identity,
    configure_sensor_cartpole,
)
from nexuml_library.environments.isaaclab import (
    IsaacBindings,
    IsaacLabEnvironment,
    isaac_task_config,
)
from nexuml_library.layers.processing.sensor_history import SensorHistoryProcessor
from nexuml_library.reinforcement.policies import continuous_policy_mlp
from .isaaclab import isaac_cartpole_ppo


def _sensor_cartpole_ppo(
    name: str,
    modalities: list[str],
    total_frames: int,
    frames_per_batch: int,
    num_envs: int,
    processing_owner: Literal["package", "external"],
) -> ScenarioSpec:
    assets = {"camera_geometry": camera_geometry_identity(32, 32)}
    fields = {
        "rgb": TensorFieldContract(
            shape=(6, 32, 32),
            dtype="float32",
            modality="rgb",
            low=0,
            high=1,
            rate_hz=60,
            physical=PhysicalTensorContract(
                units=("1",), frame="camera_optical", sensor_id="tiled_camera"
            ),
            sensor=SensorProcessingContract(
                owner=processing_owner,
                assets=assets,
                raw_shape=(32, 32, 3),
                raw_dtype="uint8",
                raw_layout="HWC",
                units="uint8_sRGB",
                scale=1 / 255,
                history_length=2,
                initialization="repeat_first",
                history_order="oldest_first",
                timing="post_control",
                latency_steps=0,
            ),
        ),
        "depth": TensorFieldContract(
            shape=(4, 32, 32),
            dtype="float32",
            modality="depth",
            low=0,
            high=100,
            rate_hz=60,
            description="Oldest-first z-depth metres and 0/1 validity channel for each frame.",
            physical=PhysicalTensorContract(
                units=("m",), frame="camera_optical", sensor_id="tiled_camera"
            ),
            sensor=SensorProcessingContract(
                owner=processing_owner,
                assets=assets,
                raw_shape=(32, 32, 1),
                raw_dtype="float32",
                raw_layout="HWC",
                units="m",
                scale=1.0,
                offset=0.0,
                depth_representation="z_depth",
                depth_min=0.1,
                depth_max=100.0,
                invalid_depth="mask_zero",
                history_length=2,
                initialization="repeat_first",
                history_order="oldest_first",
                timing="post_control",
                latency_steps=0,
            ),
        ),
        "segmentation": TensorFieldContract(
            shape=(6, 32, 32),
            dtype="float32",
            modality="segmentation",
            low=0,
            high=1,
            rate_hz=60,
            physical=PhysicalTensorContract(
                units=("1",), frame="camera_optical", sensor_id="tiled_camera"
            ),
            sensor=SensorProcessingContract(
                owner=processing_owner,
                assets=assets,
                raw_shape=(32, 32, 1),
                raw_dtype="int32",
                raw_layout="HWC",
                units="class_id",
                labels=((0, "BACKGROUND"), (1, "UNLABELLED"), (2, "cartpole")),
                label_set_id="cartpole-class-v1",
                history_length=2,
                initialization="repeat_first",
                history_order="oldest_first",
                timing="post_control",
                latency_steps=0,
            ),
        ),
    }
    observations = {key: fields[key] for key in modalities}
    paths: dict[str, tuple[str, ...]] = {key: ("images", key) for key in modalities}
    if len(modalities) == 3:
        observations["state"] = (
            IsaacLabEnvironment(task="Isaac-Cartpole").describe().observations["observation"]
        )
        paths["state"] = ("policy",)
    contract = InteractionContract(
        observations=observations,
        actions={
            "action": TensorFieldContract(
                shape=(1,),
                dtype="float32",
                low=-1,
                high=1,
                rate_hz=60,
                physical=PhysicalTensorContract(
                    units=("N",),
                    frame="joint",
                    joints=("slider_to_cart",),
                    control_mode="effort",
                    scale=(100.0,),
                    offset=(0.0,),
                ),
            )
        },
        action_space="continuous",
        control_period_s=1 / 60,
    )
    spec = isaac_cartpole_ppo(total_frames, frames_per_batch, num_envs)
    assert spec.interaction is not None
    spec.name = name
    spec.interaction.environment = IsaacLabEnvironment(
        task="NexuML-Cartpole-Sensors",
        physics="newton_mjwarp",
        presets=("rgb", "newton_renderer"),
        task_config=isaac_task_config(
            configure_sensor_cartpole, modalities=modalities, width=32, height=32
        ),
        contract=contract,
        bindings=IsaacBindings(observations=paths, processing="raw"),
    )
    spec.pipeline = continuous_policy_mlp(tuple(observations), action_dim=1, hidden_dims=(32,))
    spec.policy = PolicySpec(
        action_adapter=GaussianActionAdapter(),
        processor=SensorHistoryProcessor() if processing_owner == "package" else None,
    )
    return spec


@scenario("isaac-rgb-cartpole-ppo")
def isaac_rgb_cartpole_ppo(
    total_frames: int = 256,
    frames_per_batch: int = 64,
    num_envs: int = 4,
    processing_owner: Literal["package", "external"] = "package",
) -> ScenarioSpec:
    """Configure normalized RGB history without upstream native image preprocessing.

    Returns:
        Bounded RGB Cartpole PPO with explicitly owned two-frame history.
    """
    return _sensor_cartpole_ppo(
        "isaac-rgb-cartpole-ppo",
        ["rgb"],
        total_frames,
        frames_per_batch,
        num_envs,
        processing_owner,
    )


@scenario("isaac-depth-cartpole-ppo")
def isaac_depth_cartpole_ppo(
    total_frames: int = 256,
    frames_per_batch: int = 64,
    num_envs: int = 4,
    processing_owner: Literal["package", "external"] = "package",
) -> ScenarioSpec:
    """Configure metre z-depth history with explicit zero filling and validity masks.

    Returns:
        Bounded depth Cartpole PPO with explicitly owned two-frame history.
    """
    return _sensor_cartpole_ppo(
        "isaac-depth-cartpole-ppo",
        ["depth"],
        total_frames,
        frames_per_batch,
        num_envs,
        processing_owner,
    )


@scenario("isaac-segmentation-cartpole-ppo")
def isaac_segmentation_cartpole_ppo(
    total_frames: int = 256,
    frames_per_batch: int = 64,
    num_envs: int = 4,
    processing_owner: Literal["package", "external"] = "package",
) -> ScenarioSpec:
    """Configure the measured native class vocabulary as ordered one-hot history.

    Returns:
        Bounded segmentation Cartpole PPO with explicitly owned two-frame history.
    """
    return _sensor_cartpole_ppo(
        "isaac-segmentation-cartpole-ppo",
        ["segmentation"],
        total_frames,
        frames_per_batch,
        num_envs,
        processing_owner,
    )


@scenario("isaac-mixed-cartpole-ppo")
def isaac_mixed_cartpole_ppo(
    total_frames: int = 256,
    frames_per_batch: int = 64,
    num_envs: int = 4,
    processing_owner: Literal["package", "external"] = "package",
) -> ScenarioSpec:
    """Configure RGB, depth and segmentation history alongside the declared native state.

    Returns:
        Bounded mixed Cartpole PPO without undeclared privileged actor observations.
    """
    return _sensor_cartpole_ppo(
        "isaac-mixed-cartpole-ppo",
        ["rgb", "depth", "segmentation"],
        total_frames,
        frames_per_batch,
        num_envs,
        processing_owner,
    )
