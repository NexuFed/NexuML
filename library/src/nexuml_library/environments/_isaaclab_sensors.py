"""Raw Cartpole camera configuration; native imports belong to the owned job."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from nexuml.core.types import InteractionContract
    from nexuml_library.environments.isaaclab_config import IsaacBindings


CAMERA_POSITION = (-5.0, 0.0, 2.0)
CAMERA_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
CAMERA_CONVENTION = "world"
CAMERA_FOCAL_LENGTH = 24.0
CAMERA_HORIZONTAL_APERTURE = 20.955
CAMERA_CLIPPING_RANGE = (0.1, 100.0)


def _validate_resolution(width: int, height: int) -> None:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive integers, not booleans")


def _geometry_digest(
    *,
    width: int,
    height: int,
    position: tuple[float, ...],
    rotation_xyzw: tuple[float, ...],
    convention: str,
    focal_length: float,
    horizontal_aperture: float,
    clipping_range: tuple[float, ...],
) -> str:
    _validate_resolution(width, height)
    geometry = {
        "width": width,
        "height": height,
        "position": [float(value) for value in position],
        "rotation_xyzw": [float(value) for value in rotation_xyzw],
        "convention": convention,
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
        "clipping_range": [float(value) for value in clipping_range],
    }
    return sha256(
        json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def camera_geometry_identity(width: int, height: int) -> str:
    """Identify the fixed camera projection and pose at the selected resolution.

    Returns:
        Canonical SHA-256 of the declared primitive geometry values.

    """
    return _geometry_digest(
        width=width,
        height=height,
        position=CAMERA_POSITION,
        rotation_xyzw=CAMERA_ROTATION_XYZW,
        convention=CAMERA_CONVENTION,
        focal_length=CAMERA_FOCAL_LENGTH,
        horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
        clipping_range=CAMERA_CLIPPING_RANGE,
    )


def validate_camera_geometry(cfg: Any, expected: str) -> None:
    """Compare an actual resolved CameraCfg against the declared geometry identity.

    Raises:
        ValueError: If source geometry is missing, invalid or different.
    """
    try:
        if (
            cfg.spawn.projection_type != "pinhole"
            or cfg.spawn.distortion is not None
            or (
                cfg.spawn.vertical_aperture is not None
                and not math.isclose(
                    cfg.spawn.vertical_aperture,
                    cfg.spawn.horizontal_aperture * cfg.height / cfg.width,
                    rel_tol=1e-7,
                )
            )
            or cfg.spawn.horizontal_aperture_offset != 0.0
            or cfg.spawn.vertical_aperture_offset != 0.0
            or cfg.spawn.f_stop != 0.0
        ):
            raise ValueError("Camera geometry requires the declared centered, undistorted pinhole")
        actual = _geometry_digest(
            width=cfg.width,
            height=cfg.height,
            position=cfg.offset.pos,
            rotation_xyzw=cfg.offset.rot,
            convention=cfg.offset.convention,
            focal_length=cfg.spawn.focal_length,
            horizontal_aperture=cfg.spawn.horizontal_aperture,
            clipping_range=cfg.spawn.clipping_range,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Camera geometry is missing or invalid") from error
    if actual != expected:
        raise ValueError("Camera geometry differs from the declared camera_geometry identity")


def camera_frame(env: Any, *, data_type: str) -> torch.Tensor:
    """Snapshot a raw camera buffer before native same-step reset can reuse it.

    Returns:
        An owned tensor clone in the native camera's raw layout and dtype.
    """
    return env.scene["tiled_camera"].data.output[data_type].torch.clone()


def validate_native_sensors(
    env: Any, contract: InteractionContract, bindings: IsaacBindings
) -> None:
    """Admit only the qualified raw Newton camera snapshot and its actual metadata.

    Tensor values, dtype and device are checked separately by the runtime's state path.
    This check neither reads pixels as semantic proof nor imports native modules.

    Raises:
        ValueError: If a declared image sensor cannot be verified against the native config.
    """
    for key, field in contract.observations.items():
        sensor = field.sensor
        if sensor is None or field.modality not in {"rgb", "depth", "segmentation"}:
            continue
        try:
            if bindings.processing != "raw":
                raise ValueError("processed image bindings have no qualified native source")
            physical = field.physical
            if (
                physical is None
                or physical.sensor_id != "tiled_camera"
                or physical.frame != "camera_optical"
            ):
                raise ValueError(
                    "requires physical sensor_id='tiled_camera', frame='camera_optical'"
                )
            path = bindings.observations[key]
            if len(path) != 2:
                raise ValueError("raw source path must name exactly one observation group and term")
            group = getattr(env.cfg.observations, path[0])
            term = getattr(group, path[1])
            if term.func is not camera_frame:
                raise ValueError(
                    "observation term must use the exact camera_frame snapshot function"
                )
            if (
                term.modifiers not in (None, [])
                or term.noise is not None
                or term.clip is not None
                or term.scale is not None
                or any(
                    type(value) is not int or value != 0
                    for value in (term.history_length, term.delay_min_lag, term.delay_max_lag)
                )
                or group.enable_corruption is not False
                or type(group.history_length) is not int
                or group.history_length != 0
                or group.concatenate_terms is not False
            ):
                raise ValueError("raw source cannot apply native processing, history or delay")

            camera = env.scene["tiled_camera"]
            cfg = camera.cfg
            if cfg.prim_path != env.scene.env_regex_ns + "/Camera":
                raise ValueError("camera mounting must remain fixed at the environment root")
            renderer = cfg.renderer_cfg
            if renderer.renderer_type != "newton_warp":
                raise ValueError("requires explicit renderer_type='newton_warp'")
            validate_camera_geometry(cfg, sensor.assets["camera_geometry"])
            data = camera.data
            origins = env.scene.env_origins
            positions, rotations, intrinsics = (
                data.pos_w.torch,
                data.quat_w_world.torch,
                data.intrinsic_matrices.torch,
            )
            focal = cfg.width * cfg.spawn.focal_length / cfg.spawn.horizontal_aperture
            expected_intrinsics = origins.new_tensor(
                [
                    [focal, 0.0, cfg.width * 0.5],
                    [0.0, focal, cfg.height * 0.5],
                    [0.0, 0.0, 1.0],
                ]
            ).expand(origins.shape[0], 3, 3)
            if (
                positions.shape != origins.shape
                or rotations.shape != (origins.shape[0], 4)
                or intrinsics.shape != expected_intrinsics.shape
                or not torch.allclose(
                    positions - origins,
                    origins.new_tensor(cfg.offset.pos).expand_as(origins),
                    rtol=0,
                    atol=1e-4,
                )
                or not torch.allclose(
                    rotations.abs(),
                    origins.new_tensor(CAMERA_ROTATION_XYZW).expand_as(rotations),
                    rtol=0,
                    atol=1e-5,
                )
                or not torch.allclose(intrinsics, expected_intrinsics, rtol=1e-6, atol=1e-5)
            ):
                raise ValueError("actual camera pose/intrinsics differ from declared calibration")
            if (
                field.rate_hz is None
                or not math.isfinite(field.rate_hz)
                or field.rate_hz <= 0
                or not math.isclose(cfg.update_period, 1 / field.rate_hz, rel_tol=1e-9)
                or contract.control_period_s is None
                or not math.isclose(cfg.update_period, contract.control_period_s, rel_tol=1e-9)
                or sensor.timing != "post_control"
                or sensor.latency_steps != 0
            ):
                raise ValueError(
                    "sensor rate/control period must match post_control, zero-latency updates"
                )

            data_type = term.params["data_type"]
            if set(term.params) != {"data_type"} or data_type not in cfg.data_types:
                raise ValueError("raw data_type must be explicitly enabled on the actual camera")
            if field.modality == "rgb":
                data_types, dtype, units, physical_units = ("rgb",), "uint8", "uint8_sRGB", ("1",)
            elif field.modality == "depth":
                data_types = (
                    ("depth", "distance_to_image_plane")
                    if sensor.depth_representation == "z_depth"
                    else ("distance_to_camera",)
                )
                dtype, units, physical_units = "float32", "m", ("m",)
                if (
                    sensor.depth_representation not in {"z_depth", "range_depth"}
                    or sensor.depth_min is None
                    or sensor.depth_max is None
                    or sensor.depth_min <= 0
                    or sensor.depth_max <= sensor.depth_min
                    or tuple(cfg.spawn.clipping_range) != (sensor.depth_min, sensor.depth_max)
                    or renderer.depth_clipping_behavior != "zero"
                    or sensor.invalid_depth != "mask_zero"
                ):
                    raise ValueError(
                        "depth requires matching metre bounds and zero-sentinel masking"
                    )
            else:
                data_types, dtype, units, physical_units = (
                    ("semantic_segmentation",),
                    "int32",
                    "class_id",
                    ("1",),
                )
                expected_labels = {0: "BACKGROUND", 1: "UNLABELLED", 2: "cartpole"}
                if (
                    renderer.colorize_semantic_segmentation is not False
                    or sensor.label_set_id != "cartpole-class-v1"
                    or dict(sensor.labels) != expected_labels
                ):
                    raise ValueError("segmentation requires uncolorized cartpole-class-v1 labels")
                actual_labels = camera.data.info["semantic_segmentation"]["idToLabels"]
                if not isinstance(actual_labels, dict):
                    raise ValueError("segmentation idToLabels must be a dictionary")
                normalized = {}
                for identifier, label in actual_labels.items():
                    if type(identifier) is int:
                        native_id = identifier
                    elif (
                        type(identifier) is str
                        and identifier.isascii()
                        and identifier.isdecimal()
                        and str(int(identifier)) == identifier
                    ):
                        native_id = int(identifier)
                    else:
                        raise ValueError("segmentation idToLabels contains an invalid integer ID")
                    if native_id < 0 or native_id in normalized:
                        raise ValueError(
                            "segmentation idToLabels contains a negative or duplicate ID"
                        )
                    normalized[native_id] = label
                if normalized != {
                    identifier: {"class": label} for identifier, label in expected_labels.items()
                }:
                    raise ValueError("actual segmentation idToLabels differs from declared labels")

            if data_type not in data_types:
                raise ValueError(
                    "raw data_type does not match the declared modality/representation"
                )
            channels = 3 if field.modality == "rgb" else 1
            if (
                sensor.raw_layout != "HWC"
                or sensor.raw_shape != (cfg.height, cfg.width, channels)
                or sensor.raw_dtype != dtype
                or sensor.units != units
                or physical.units != physical_units
            ):
                raise ValueError(
                    "raw layout/shape/dtype/units differ from the qualified camera source"
                )
        except Exception as error:
            raise ValueError(f"Native sensor {key!r}: {error}") from error


def configure_sensor_cartpole(cfg: Any, *, modalities: list[str], width: int, height: int) -> None:
    """Configure a resolved upstream camera task, without creating native resources.

    The four-control-step horizon deliberately exercises bounded reset qualification.

    Raises:
        ValueError: If modalities, dimensions or the explicitly selected renderer differ.
    """
    _validate_resolution(width, height)
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(name not in ("rgb", "depth", "segmentation") for name in modalities)
        or len(set(modalities)) != len(modalities)
    ):
        raise ValueError("Camera modalities must be unique rgb, depth or segmentation names")
    camera = cfg.scene.tiled_camera
    if camera.renderer_cfg.renderer_type != "newton_warp":
        raise ValueError("Sensor Cartpole requires explicit renderer_type='newton_warp'")

    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.utils import configclass

    camera.width, camera.height = width, height
    camera.data_types = [
        "semantic_segmentation" if name == "segmentation" else name for name in modalities
    ]
    camera.update_period = cfg.sim.dt * cfg.decimation
    camera.renderer_cfg.colorize_semantic_segmentation = False
    camera.renderer_cfg.depth_clipping_behavior = "zero"
    camera.spawn.focal_length = CAMERA_FOCAL_LENGTH
    camera.spawn.horizontal_aperture = CAMERA_HORIZONTAL_APERTURE
    camera.spawn.clipping_range = CAMERA_CLIPPING_RANGE
    camera.offset.pos = CAMERA_POSITION
    camera.offset.rot = CAMERA_ROTATION_XYZW
    camera.offset.convention = CAMERA_CONVENTION
    cfg.scene.robot.spawn.semantic_tags = [("class", "cartpole")]

    state_group = cfg.observations.critic
    state_group.enable_corruption = False
    state_group.history_length = 0

    @configclass
    class RawFrames(ObservationGroupCfg):
        """Selected raw terms only; portable processing owns all frame history."""

        rgb = (
            ObservationTermCfg(func=camera_frame, params={"data_type": "rgb"})
            if "rgb" in modalities
            else None
        )
        depth = (
            ObservationTermCfg(func=camera_frame, params={"data_type": "depth"})
            if "depth" in modalities
            else None
        )
        segmentation = (
            ObservationTermCfg(func=camera_frame, params={"data_type": "semantic_segmentation"})
            if "segmentation" in modalities
            else None
        )
        concatenate_terms = False
        enable_corruption = False
        history_length = 0

    @configclass
    class NewObservations:
        """Expose explicit reference state separately from raw camera terms."""

        policy = state_group
        images = RawFrames()

    cfg.observations = NewObservations()
    cfg.episode_length_s = 4 * cfg.sim.dt * cfg.decimation
