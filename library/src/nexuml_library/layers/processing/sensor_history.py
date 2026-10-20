"""Declared sensor transforms and caller-owned, oldest-first frame history."""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict
from torch import nn

from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import InteractionContract, SensorProcessingContract, TensorFieldContract


@layer("SensorHistoryProcessor")
class SensorHistoryProcessor(LayerDefinition):
    """Build processing solely from the interaction contract's sensor descriptors."""

    def build(self, context: LayerBuildContext) -> _SensorHistoryRuntime:
        if "interaction_contract" not in context.metadata:
            raise ValueError("SensorHistoryProcessor requires interaction_contract metadata")
        owner = context.metadata.get("execution_owner", "package")
        if owner not in {"package", "runtime"}:
            raise ValueError("Sensor processing execution_owner must be package or runtime")
        return _SensorHistoryRuntime(
            InteractionContract.model_validate(context.metadata["interaction_contract"]),
            process_external=owner == "runtime",
        )


class _SensorHistoryRuntime(nn.Module):
    def __init__(self, contract: InteractionContract, *, process_external: bool = False):
        super().__init__()
        self.fields = contract.observations
        self.raw_fields: dict[str, TensorFieldContract] = {}
        self.frame_shapes: dict[str, tuple[int, ...]] = {}
        for key, field in self.fields.items():
            sensor = field.sensor
            if sensor is None or (sensor.owner == "external" and not process_external):
                self.raw_fields[key] = field
                continue
            self.frame_shapes[key] = self._frame_shape(key, field, sensor)
            self.raw_fields[key] = TensorFieldContract(
                shape=sensor.raw_shape,
                dtype=sensor.raw_dtype,
                modality=field.modality,
                rate_hz=field.rate_hz,
                sensor=sensor,
            )

    @staticmethod
    def _frame_shape(
        key: str, field: TensorFieldContract, sensor: SensorProcessingContract
    ) -> tuple[int, ...]:
        if (sensor.processor_id, sensor.processor_version) != ("SensorHistoryProcessor", "1"):
            raise ValueError(f"Sensor {key!r} has an unsupported processor identity/version")
        if sensor.timing != "post_control" or sensor.latency_steps != 0:
            raise ValueError(f"Sensor {key!r} requires post_control timing and zero latency_steps")
        if sensor.raw_dtype not in {
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "bfloat16",
            "float32",
            "float64",
        }:
            raise ValueError(f"Sensor {key!r} has unsupported raw dtype {sensor.raw_dtype!r}")
        shape = sensor.raw_shape
        if field.modality in {"state", "vector", "proprioception", "generic"}:
            if sensor.raw_layout != "vector" or len(shape) != 1:
                raise ValueError(f"State sensor {key!r} requires a vector layout")
            frame_shape = shape
        elif field.modality in {"rgb", "depth", "segmentation"}:
            if sensor.raw_layout == "HWC" and len(shape) == 3:
                channels, height, width = shape[2], shape[0], shape[1]
            elif sensor.raw_layout == "CHW" and len(shape) == 3:
                channels, height, width = shape
            elif sensor.raw_layout == "HW" and len(shape) == 2:
                channels, (height, width) = 1, shape
            else:
                raise ValueError(f"Image sensor {key!r} has incompatible raw layout/shape")
            if field.modality == "rgb":
                if channels != 3 or sensor.raw_layout == "HW":
                    raise ValueError(f"RGB sensor {key!r} requires three declared channels")
                output_channels = 3
            elif field.modality == "depth":
                if channels != 1:
                    raise ValueError(f"Depth sensor {key!r} requires one raw channel")
                if (
                    sensor.depth_representation is None
                    or sensor.units != "m"
                    or sensor.depth_min is None
                    or sensor.depth_max is None
                ):
                    raise ValueError(
                        f"Depth sensor {key!r} requires representation/bounds in metres"
                    )
                output_channels = 2 if sensor.invalid_depth == "mask_zero" else 1
            else:
                if channels != 1 or not sensor.labels:
                    raise ValueError(f"Segmentation sensor {key!r} requires one channel and labels")
                if sensor.raw_dtype not in {"uint8", "int8", "int16", "int32", "int64"}:
                    raise ValueError(f"Segmentation sensor {key!r} requires an integer raw dtype")
                if sensor.scale != 1 or sensor.offset != 0:
                    raise ValueError(f"Segmentation sensor {key!r} uses one-hot values, not affine")
                output_channels = len(sensor.labels)
            frame_shape = (output_channels, height, width)
        else:
            raise ValueError(f"Sensor {key!r} has unsupported modality {field.modality!r}")
        expected = (frame_shape[0] * sensor.history_length, *frame_shape[1:])
        if field.dtype != "float32" or field.shape != expected:
            raise ValueError(f"Processed sensor {key!r} must declare float32 shape {expected}")
        return frame_shape

    def initialize_state(self, batch_size=(), *, device=None) -> TensorDict:
        if type(batch_size) is int:
            batch_size = (batch_size,)
        if not isinstance(batch_size, (tuple, list, torch.Size)) or any(
            type(size) is not int or size < 0 for size in batch_size
        ):
            raise ValueError("State batch_size must contain nonnegative integers")
        batch = torch.Size(batch_size)
        device = torch.device("cpu" if device is None else device)
        state = TensorDict({}, batch_size=batch, device=device)
        for key, shape in self.frame_shapes.items():
            sensor = self.fields[key].sensor
            assert sensor is not None
            state[key] = TensorDict(
                {
                    "frames": torch.zeros(
                        *batch, sensor.history_length, *shape, dtype=torch.float32, device=device
                    ),
                    "initialized": torch.zeros(batch, dtype=torch.bool, device=device),
                },
                batch_size=batch,
                device=device,
            )
        return state

    def _validate_state(self, state: TensorDict) -> torch.device:
        if not isinstance(state, TensorDict) or state.device is None:
            raise ValueError("Processing state must be a TensorDict with an explicit device")
        if set(state.keys()) != set(self.frame_shapes):
            raise ValueError("Processing state fields do not match the package-owned sensors")
        for key, shape in self.frame_shapes.items():
            sensor = self.fields[key].sensor
            assert sensor is not None
            child = state[key]
            if (
                not isinstance(child, TensorDict)
                or child.batch_size != state.batch_size
                or child.device != state.device
                or set(child.keys()) != {"frames", "initialized"}
            ):
                raise ValueError(f"Invalid processing state structure for {key!r}")
            for name, expected_shape, dtype in (
                ("frames", (*state.batch_size, sensor.history_length, *shape), torch.float32),
                ("initialized", tuple(state.batch_size), torch.bool),
            ):
                tensor = child[name]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tuple(tensor.shape) != expected_shape
                    or tensor.dtype != dtype
                    or tensor.device != state.device
                ):
                    raise ValueError(f"Invalid processing state {key!r}/{name}: shape/dtype/device")
            frames = cast(torch.Tensor, child["frames"])
            initialized = cast(torch.Tensor, child["initialized"])
            if not torch.isfinite(frames).all():
                raise ValueError(f"Processing state {key!r} contains nonfinite history")
            if torch.any(frames[~initialized] != 0):
                raise ValueError(
                    f"Uninitialized processing state {key!r} must contain zero history"
                )
        return state.device

    def _frame(self, key: str, raw: torch.Tensor, batch_dims: int) -> torch.Tensor:
        field = self.fields[key]
        sensor = field.sensor
        assert sensor is not None
        if sensor.raw_layout == "HWC":
            raw = raw.movedim(-1, -3)
        elif sensor.raw_layout == "HW":
            raw = raw.unsqueeze(batch_dims)
        if field.modality == "segmentation":
            values = raw.squeeze(batch_dims)
            masks = torch.stack(
                [values == identifier for identifier, _ in sensor.labels], batch_dims
            )
            if not masks.any(dim=batch_dims).all():
                raise ValueError(f"Segmentation sensor {key!r} contains unknown label IDs")
            return masks.to(torch.float32)
        values = raw.to(torch.float32)
        if field.modality == "depth":
            assert sensor.depth_min is not None and sensor.depth_max is not None
            valid = torch.isfinite(raw) & (raw >= sensor.depth_min) & (raw <= sensor.depth_max)
            if sensor.invalid_depth == "reject":
                if not valid.all():
                    raise ValueError(f"Depth sensor {key!r} contains invalid depth")
            else:
                depth = torch.where(valid, values * sensor.scale + sensor.offset, 0.0)
                return torch.cat((depth, valid.to(torch.float32)), dim=batch_dims)
        return values * sensor.scale + sensor.offset

    def step(self, observation: TensorDict, state: TensorDict) -> tuple[TensorDict, TensorDict]:
        device = self._validate_state(state)
        if not isinstance(observation, TensorDict) or observation.batch_size != state.batch_size:
            raise ValueError("Observation and processing state batch axes must match")
        for key, field in self.raw_fields.items():
            tensor = observation.get(key)
            if isinstance(tensor, torch.Tensor) and tensor.device != device:
                raise ValueError(f"Raw observation {key!r} and state must use the same device")
            sensor = self.fields[key].sensor
            mask_depth = (
                key in self.frame_shapes
                and self.fields[key].modality == "depth"
                and sensor is not None
                and sensor.invalid_depth == "mask_zero"
            )
            CompiledPolicy._validate_fields(
                observation, {key: field}, "Raw observation", require_finite=not mask_depth
            )
        processed, next_state = observation.clone(), state.clone()
        batch_dims = len(state.batch_size)
        for key, shape in self.frame_shapes.items():
            sensor = self.fields[key].sensor
            assert sensor is not None
            frame = self._frame(key, cast(torch.Tensor, observation[key]), batch_dims).unsqueeze(
                batch_dims
            )
            previous = cast(torch.Tensor, state[key, "frames"])
            advanced = torch.cat(
                (previous.narrow(batch_dims, 1, sensor.history_length - 1), frame), dim=batch_dims
            )
            if sensor.initialization == "repeat_first":
                initial = frame.expand(*state.batch_size, sensor.history_length, *shape)
            else:
                initial = torch.cat(
                    (
                        torch.zeros_like(previous.narrow(batch_dims, 0, sensor.history_length - 1)),
                        frame,
                    ),
                    dim=batch_dims,
                )
            flags = cast(torch.Tensor, state[key, "initialized"])
            initialized = flags.reshape(*state.batch_size, *([1] * (len(shape) + 1)))
            frames = torch.where(initialized, advanced, initial)
            next_state[key, "frames"] = frames
            next_state[key, "initialized"] = torch.ones_like(flags)
            processed[key] = frames.reshape(
                *state.batch_size, sensor.history_length * shape[0], *shape[1:]
            ).clone()
        CompiledPolicy._validate_fields(processed, self.fields, "Processed observation")
        return processed, next_state

    def reset_state(self, state: TensorDict, mask: torch.Tensor) -> TensorDict:
        device = self._validate_state(state)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != state.batch_size
            or mask.device != device
        ):
            raise ValueError(
                "Reset mask must have the exact state batch shape/device and bool dtype"
            )
        result = state.clone()
        for key, shape in self.frame_shapes.items():
            selected = mask.reshape(*state.batch_size, *([1] * (len(shape) + 1)))
            result[key, "frames"] = torch.where(
                selected, 0.0, cast(torch.Tensor, state[key, "frames"])
            )
            result[key, "initialized"] = cast(torch.Tensor, state[key, "initialized"]) & ~mask
        return result
