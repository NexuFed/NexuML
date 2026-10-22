"""Core type definitions for NexuML pipeline specifications."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from keyword import iskeyword
from math import isclose, prod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from nexuml.core.components import (
    ActionAdapterDefinition,
    DataSourceDefinition,
    EnvironmentDefinition,
    EvalAlgorithmDefinition,
    ExecutionBackendDefinition,
    LayerDefinition,
    LoaderBackendDefinition,
    RLAlgorithmDefinition,
)
from nexuml.core.factory import factory_values, normalize_json_value, resolve_factory
from nexuml.execution.definitions import LocalExecution

if TYPE_CHECKING:
    from nexuml.evaluation.utils import FeatureStore


class AxisKeySpec(BaseModel):
    """Typed axis-key declaration with source provenance.

    ``source`` indicates where the axis value is resolved from at runtime:

    - ``"x"``: from the transformed pipeline output TensorDict (e.g. a
      reconstructed feature or anomaly score produced by a pipeline layer).
    - ``"y"``: from the label TensorDict — most common for grouping axes such
      as ``machine``, ``section``, or ``target``.
    - ``"metadata"``: from the dataset metadata DataFrame attached to the batch
      by ``_attach_eval_metadata``.  Use this source for non-tensor columns that
      are not part of the label TensorDict but are needed for grouped evaluation
      (e.g. ``basename``, ``domain``).  Only available during eval/test phases.
    """

    key: str
    source: Literal["x", "y", "metadata"] = "y"


class SpecModel(BaseModel):
    """Shared base model for scenario specifications."""

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpecModel":
        return cls.model_validate(data)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.model_dump(mode="json") == other
        return super().__eq__(other)


class FactorySpec(SpecModel):
    """Portable constructor configuration authored through a typed factory helper."""

    factory: str
    args: list[object] = Field(default_factory=list)
    kwargs: dict[str, object] = Field(default_factory=dict)

    @field_validator("args", mode="before")
    @classmethod
    def _validate_args(cls, value: Any) -> list[object]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("args must be a list or tuple")
        return [normalize_json_value(item, f"args[{index}]") for index, item in enumerate(value)]

    @field_validator("kwargs", mode="before")
    @classmethod
    def _validate_kwargs(cls, value: Any) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError("kwargs must be a mapping")
        normalized = normalize_json_value(value, "kwargs")
        assert isinstance(normalized, dict)
        return cast(dict[str, object], normalized)

    @classmethod
    def from_factory(cls, factory: object, *args: object, **kwargs: object) -> Self:
        """Capture an importable factory and portable constructor arguments.

        Returns:
            Validated factory specification.
        """
        return cls(**cast(Any, factory_values(factory, args, kwargs, label=cls.__name__)))

    def resolve(self) -> Callable[..., object]:
        """Resolve the configured import target.

        Returns:
            Imported callable.
        """
        return resolve_factory(self.factory, label=type(self).__name__)

    def build(self, *runtime_args: object, **runtime_kwargs: object) -> object:
        """Construct the configured object with any runtime-supplied arguments.

        Returns:
            Constructed runtime object.

        Raises:
            TypeError: If construction fails.
        """
        kwargs = {**self.kwargs, **runtime_kwargs}
        try:
            return self.resolve()(*runtime_args, *self.args, **kwargs)
        except Exception as exc:
            raise TypeError(
                f"Could not construct {type(self).__name__} from {self.factory!r}: {exc}"
            ) from exc


class LayerSpec(SpecModel):
    """Specification for a single layer in the pipeline."""

    component: LayerDefinition
    keys_in: list[str] | dict[str, str]
    keys_out: list[str]
    label_key: str | list[str] | None = None
    label_in_x: bool = False
    meta_in: dict[str, str] | None = None
    meta_out: dict[str, str] | None = None
    delay_epochs: int = 0
    update_every_n_epochs: int = 1


class PipelineSpec(SpecModel):
    """Ordered specification of pipeline stages, each containing layer specs."""

    stages: dict[str, list[LayerSpec]] = Field(default_factory=dict)


class PhysicalTensorContract(SpecModel):
    """Versioned physical interpretation; tensor values retain their numeric bounds."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    units: tuple[Annotated[str, Field(min_length=1, pattern=r"\S")], ...] = Field(min_length=1)
    frame: str = Field(min_length=1, pattern=r"\S")
    joints: tuple[Annotated[str, Field(min_length=1, pattern=r"\S")], ...] = ()
    control_mode: Literal["effort", "absolute_position", "relative_position"] | None = None
    scale: tuple[float, ...] | None = None
    offset: tuple[float, ...] | None = None
    sensor_id: str | None = Field(default=None, min_length=1, pattern=r"\S")
    class_controls: tuple[tuple[float, ...], ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_control(self) -> Self:
        if self.control_mode is None and (self.scale is not None or self.offset is not None):
            raise ValueError("Physical action scale/offset requires an explicit control mode")
        if self.control_mode is not None and (not self.joints or self.scale is None):
            raise ValueError("Physical control requires ordered joints and an explicit scale")
        if self.control_mode is not None and len(self.joints) != len(set(self.joints)):
            raise ValueError("Physical action joints must be unique and ordered")
        if self.scale is not None and any(value == 0 for value in self.scale):
            raise ValueError("Physical action scale must be nonzero")
        if self.class_controls is not None and (
            self.control_mode is None
            or not self.class_controls
            or any(len(row) != len(self.joints) for row in self.class_controls)
        ):
            raise ValueError("Categorical controls require one physical command per ordered joint")
        return self


class PhysicalSystemContract(SpecModel):
    """Declared identity, not proof of real hardware calibration or safety."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    embodiment_id: str = Field(min_length=1, pattern=r"\S")
    asset_source: str = Field(min_length=1, pattern=r"\S")
    asset_id: str = Field(min_length=1, pattern=r"\S")
    calibration_id: str = Field(min_length=1, pattern=r"\S")


class SensorProcessingContract(SpecModel):
    """Explicit raw sensor meaning and ownership of portable processing/history."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    owner: Literal["package", "external"]
    processor_id: str = Field(default="SensorHistoryProcessor", min_length=1, pattern=r"\S")
    processor_version: str = Field(default="1", min_length=1, pattern=r"\S")
    assets: dict[str, Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]] = Field(
        default_factory=dict
    )
    raw_shape: tuple[Annotated[StrictInt, Field(gt=0)], ...]
    raw_dtype: str = Field(min_length=1, pattern=r"\S")
    raw_layout: Literal["vector", "HWC", "CHW", "HW"]
    units: str = Field(min_length=1, pattern=r"\S")
    scale: float = Field(default=1.0, strict=True)
    offset: float = Field(default=0.0, strict=True)
    depth_representation: Literal["z_depth", "range_depth"] | None = None
    invalid_depth: Literal["reject", "mask_zero"] = "reject"
    depth_min: float | None = Field(default=None, strict=True)
    depth_max: float | None = Field(default=None, strict=True)
    labels: tuple[tuple[Annotated[StrictInt, Field(ge=0)], str], ...] = ()
    label_set_id: str | None = Field(default=None, min_length=1, pattern=r"\S")
    history_length: StrictInt = Field(default=1, ge=1)
    history_order: Literal["oldest_first"] = "oldest_first"
    initialization: Literal["repeat_first", "zeros"] = "repeat_first"
    timing: Literal["post_control", "pre_action"] = "post_control"
    latency_steps: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_sensor(self) -> Self:
        ids = [identifier for identifier, _ in self.labels]
        names = [name for _, name in self.labels]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("Segmentation label IDs and values must be unique and ordered")
        if any(not name.strip() for name in names):
            raise ValueError("Segmentation label values must be nonempty")
        if self.depth_min is not None and self.depth_min < 0:
            raise ValueError("depth_min must be nonnegative")
        if self.depth_max is not None and (
            self.depth_min is None or self.depth_max <= self.depth_min
        ):
            raise ValueError("Depth bounds must satisfy 0 <= depth_min < depth_max")
        if self.owner == "package" and self.depth_representation is not None:
            if self.units != "m" or self.depth_min is None or self.depth_max is None:
                raise ValueError(
                    "Package-owned depth requires explicit bounds in metres (units='m')"
                )
        return self


class TensorFieldContract(SpecModel):
    """Portable tensor leaf; shape excludes environment and inference batch axes."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    shape: tuple[Annotated[int, Field(strict=True, gt=0)], ...]
    dtype: str = Field(min_length=1)
    modality: str = "generic"
    low: float | list[float] | None = None
    high: float | list[float] | None = None
    num_values: int | None = Field(default=None, gt=0)
    rate_hz: float | None = Field(default=None, gt=0)
    description: str | None = None
    physical: PhysicalTensorContract | None = None
    axes: tuple[Annotated[str, Field(min_length=1, pattern=r"\S")], ...] | None = None
    sensor: SensorProcessingContract | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.axes is not None:
            if len(self.axes) != len(self.shape) or len(set(self.axes)) != len(self.axes):
                raise ValueError("Named feature axes must be unique and match the tensor rank")
            if {"batch", "environment"}.intersection(self.axes):
                raise ValueError("Batch/environment axes belong outside the tensor feature shape")
        size = prod(self.shape)
        for name in ("low", "high"):
            bound = getattr(self, name)
            if isinstance(bound, list) and len(bound) != size:
                raise ValueError(f"{name} must contain {size} values for shape {self.shape}")
        if self.low is not None and self.high is not None:
            low = self.low if isinstance(self.low, list) else [self.low] * size
            high = self.high if isinstance(self.high, list) else [self.high] * size
            if any(lo >= hi for lo, hi in zip(low, high, strict=True)):
                raise ValueError("low must be less than high for every tensor element")
        if self.physical is not None:
            width = self.shape[-1] if self.shape else 1
            for name in ("units", "scale", "offset"):
                values = getattr(self.physical, name)
                if values is not None and len(values) not in (1, width):
                    raise ValueError(f"Physical {name} must have one or {width} values")
            if self.physical.joints and len(self.physical.joints) != width:
                raise ValueError(f"Physical joint order must contain {width} names")
            if self.physical.class_controls is not None and (
                self.shape != ()
                or self.dtype != "int64"
                or self.num_values != len(self.physical.class_controls)
            ):
                raise ValueError(
                    "Categorical physical controls require matching scalar int64 classes"
                )
        return self


def _validate_module_identifier(value: str) -> None:
    # Match public Module/ModuleDict names without importing torch during discovery.
    reserved = {
        "training",
        "T_destination",
        "forward",
        "keys",
        "values",
        "items",
        "update",
        "clear",
        "pop",
        "add_module",
        "register_module",
        "register_parameter",
        "register_buffer",
        "register_backward_hook",
        "register_full_backward_hook",
        "register_full_backward_pre_hook",
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_load_state_dict_pre_hook",
        "register_load_state_dict_post_hook",
        "register_state_dict_pre_hook",
        "register_state_dict_post_hook",
        "call_super_init",
        "dump_patches",
        "get_buffer",
        "get_extra_state",
        "get_parameter",
        "get_submodule",
        "set_extra_state",
        "set_submodule",
        "state_dict",
        "load_state_dict",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "children",
        "named_children",
        "modules",
        "named_modules",
        "apply",
        "train",
        "eval",
        "requires_grad_",
        "zero_grad",
        "share_memory",
        "to",
        "to_empty",
        "type",
        "float",
        "double",
        "half",
        "bfloat16",
        "cpu",
        "cuda",
        "xpu",
        "ipu",
        "mtia",
        "compile",
        "extra_repr",
    }
    if not value.isidentifier() or value.startswith("_") or iskeyword(value) or value in reserved:
        raise ValueError(f"Invalid module identifier {value!r}")


class AgentGroupContract(SpecModel):
    """Ordered compatible agents sharing one local observation/action interface."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    agents: tuple[Annotated[str, Field(min_length=1, pattern=r"\S")], ...] = Field(min_length=1)
    observations: dict[str, TensorFieldContract] = Field(min_length=1)
    actions: dict[str, TensorFieldContract] = Field(min_length=1)
    action_space: Literal["continuous", "discrete"]

    @model_validator(mode="after")
    def validate_agents(self) -> Self:
        if len(self.agents) != len(set(self.agents)):
            raise ValueError("Group agent identities must be unique and ordered")
        reserved = {
            "reward",
            "done",
            "terminated",
            "truncated",
            "valid",
            "raw_terminated",
            "raw_truncated",
            "state_value",
            "advantage",
            "value_target",
        }
        if reserved.intersection(self.observations.keys() | self.actions.keys()):
            raise ValueError("Group fields must not overwrite learning or transition signals")
        InteractionContract(
            observations=self.observations, actions=self.actions, action_space=self.action_space
        )
        return self


class InteractionContract(SpecModel):
    """Portable observation/action truth, independent of simulator frameworks."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    observations: dict[str, TensorFieldContract] = Field(default_factory=dict)
    actions: dict[str, TensorFieldContract] = Field(default_factory=dict)
    action_space: Literal["continuous", "discrete", "multi_discrete", "structured"]
    num_envs: int = Field(default=1, gt=0)
    control_period_s: float | None = Field(default=None, gt=0)
    physical: PhysicalSystemContract | None = None
    groups: dict[str, AgentGroupContract] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    critic_observations: dict[str, TensorFieldContract] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    team_reward: bool = Field(default=False, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def validate_keys(self) -> Self:
        if self.groups:
            if self.observations or self.actions:
                raise ValueError("Grouped contracts require empty root observations/actions")
            agents = []
            for name, group in self.groups.items():
                _validate_module_identifier(name)
                if name in {"next", "collector", "reward", "done", "terminated", "truncated"}:
                    raise ValueError("Group names must not overwrite root transition metadata")
                agents.extend(group.agents)
                self.group_contract(name)
            if len(agents) != len(set(agents)):
                raise ValueError("Agent identities must be globally unique across groups")
        elif not self.observations or not self.actions:
            raise ValueError("Ungrouped contracts require nonempty observations and actions")
        for fields in (self.observations, self.actions, self.critic_observations):
            if any(not key.strip() for key in fields):
                raise ValueError("interaction field keys must be nonempty")
        if self.observations.keys() & self.actions.keys():
            raise ValueError("observation and action keys must be distinct")
        if self.critic_observations.keys() & (
            self.observations.keys() | self.actions.keys() | self.groups.keys()
        ):
            raise ValueError("Critic-only keys must be distinct from actor fields and groups")
        for field in self.actions.values():
            if (
                field.physical is not None
                and field.rate_hz is not None
                and self.control_period_s is not None
                and not isclose(field.rate_hz * self.control_period_s, 1.0)
            ):
                raise ValueError("Physical action rate must agree with the control period")
        return self

    def group_contract(self, name: str) -> InteractionContract:
        """Return only this group's actor-visible local contract.

        Returns:
            Ordinary contract retaining physical identity and control timing.
        """
        group = self.groups[name]
        return InteractionContract(
            observations=group.observations,
            actions=group.actions,
            action_space=group.action_space,
            num_envs=self.num_envs,
            control_period_s=self.control_period_s,
            physical=self.physical,
        )


class InteractionSpec(SpecModel):
    """Interaction source, separate from datasets and learning algorithms."""

    environment: EnvironmentDefinition
    evaluation_environment: EnvironmentDefinition | None = None


class GroupPolicySpec(SpecModel):
    """Existing local neural graph and explicit actor identity for each agent."""

    pipeline: PipelineSpec
    action_adapter: ActionAdapterDefinition
    actor_ids: tuple[str, ...] = Field(min_length=1)
    processor: LayerDefinition | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("actor_ids")
    @classmethod
    def validate_actor_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _validate_module_identifier(value)
        return values


class PolicySpec(SpecModel):
    """Deployable action semantics attached to the scenario's neural pipeline."""

    action_adapter: ActionAdapterDefinition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    batch_mode: Literal["independent", "preserve"] = "independent"
    processor: LayerDefinition | None = Field(default=None, exclude_if=lambda value: value is None)
    groups: dict[str, GroupPolicySpec] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_grouped_policy(self) -> Self:
        if self.groups:
            if self.action_adapter is not None or self.processor is not None:
                raise ValueError("Grouped policies declare adapters/processors on each group only")
            for name in self.groups:
                _validate_module_identifier(name)
        elif self.action_adapter is None:
            raise ValueError("Ungrouped policy requires an action_adapter")
        return self


_DEFAULT_OPTIMIZER_KWARGS: dict[str, object] = {"lr": 1e-3}


class OptimizerSpec(FactorySpec):
    """Specification for an optimizer."""

    factory: str = "torch.optim.adam:Adam"
    kwargs: dict[str, object] = Field(default_factory=lambda: dict(_DEFAULT_OPTIMIZER_KWARGS))


_DEFAULT_SCHEDULER_KWARGS: dict[str, object] = {"factor": 1.0, "total_iters": 0}


class SchedulerSpec(FactorySpec):
    """Specification for a learning rate scheduler."""

    factory: str = "torch.optim.lr_scheduler:ConstantLR"
    kwargs: dict[str, object] = Field(default_factory=lambda: dict(_DEFAULT_SCHEDULER_KWARGS))
    warmup: str | int | None = None

    def resolve_warmup(self, max_epochs: int) -> int:
        """Resolve the warmup spec to an integer number of epochs.

        Accepts an integer (used directly), a percentage string like ``"5%"``
        (computed as a fraction of *max_epochs*), or ``None`` (returns 0).
        The result is always at least 1 when a non-zero warmup is specified.

        Returns:
            Resolved warmup epoch count.
        """
        if self.warmup is None:
            return 0
        if isinstance(self.warmup, int):
            return max(1, self.warmup)
        w = str(self.warmup).strip()
        if w.endswith("%"):
            pct = float(w[:-1]) / 100.0
            return max(1, round(pct * max_epochs))
        return max(1, int(w))


class StrategySpec(FactorySpec):
    """Specification for an importable Lightning strategy."""


class AutoBatchSizeSpec(SpecModel):
    """Structured config for runtime automatic training batch-size resolution."""

    mode: Literal["auto"] = "auto"
    min: int = 1
    max: int = 128
    candidates: Literal["power_of_two"] = "power_of_two"
    safety: Literal["largest", "previous_power_of_two", "margin"] = "previous_power_of_two"
    margin: float = 0.8

    @model_validator(mode="after")
    def validate_bounds(self) -> "AutoBatchSizeSpec":
        if self.min <= 0:
            raise ValueError("auto batch size min must be positive")
        if self.max <= 0:
            raise ValueError("auto batch size max must be positive")
        if self.min > self.max:
            raise ValueError("auto batch size min must be <= max")
        if self.safety == "margin" and not (0 < self.margin <= 1):
            raise ValueError("auto batch size margin must be in (0, 1]")
        return self


BatchSizeSpec = int | AutoBatchSizeSpec


class TrainingSpec(SpecModel):
    """Specification for training configuration."""

    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)
    loss_keys: dict[str, float] = Field(default_factory=dict)
    metric_keys: list[str] = Field(default_factory=list)
    progress_bar_keys: list[str] | Literal["all", "none"] = Field(default_factory=lambda: ["loss"])
    max_epochs: int = 10
    max_steps: int | None = Field(default=None, gt=0)
    seed: int | None = None
    deterministic: bool = False
    batch_size: BatchSizeSpec = 64
    lr: float = 1e-3
    accelerator: str = "auto"
    devices: str | int = "auto"
    strategy: str | StrategySpec = "auto"
    precision: str | int = "32-true"

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: BatchSizeSpec) -> BatchSizeSpec:
        if isinstance(value, int) and value <= 0:
            raise ValueError("training.batch_size must be positive")
        return value


class CollectorSpec(SpecModel):
    """Rollout topology and synchronization, independent of learner placement."""

    backend: Literal["direct", "process", "ray"] = "direct"
    num_collectors: int = Field(default=1, gt=0)
    num_envs_per_collector: int = Field(default=1, gt=0)
    sync: bool = True
    env_device: str = "cpu"
    storing_device: str = "cpu"
    policy_device: str = "auto"
    policy_sync_interval_batches: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        if self.backend == "direct" and self.num_collectors != 1:
            raise ValueError("direct collection requires num_collectors=1")
        return self


class RLVideoSpec(SpecModel):
    """Bounded first-episode recording; formats select downloadable artifacts."""

    enabled: bool = False
    format: Literal["gif", "mp4", "both"] = "gif"
    max_frames: int = Field(default=500, gt=0)
    max_size: int = Field(default=256, gt=0)
    fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class RLEvaluationSpec(SpecModel):
    """Fresh episodic policy evaluation, not dataset evaluation phases."""

    episodes: int = Field(default=5, ge=0)
    deterministic: bool = True
    seed: int | None = None
    max_steps_per_episode: int | None = Field(default=None, gt=0)
    every_n_frames: int | None = Field(default=None, gt=0, strict=True)
    at_start: bool = False
    video: RLVideoSpec = Field(default_factory=RLVideoSpec)

    @model_validator(mode="after")
    def validate_recording(self) -> Self:
        if self.video.enabled and self.episodes == 0:
            raise ValueError("evaluation.video requires evaluation.episodes > 0")
        return self


# Keep class identities and archived config names compatible; these options are not RL-specific.
PolicyEvaluationSpec = RLEvaluationSpec
PolicyVideoSpec = RLVideoSpec


class ReinforcementLearningSpec(SpecModel):
    """Finite-frame Lightning learning; interaction owns the environment."""

    mode: Literal["reinforcement"] = "reinforcement"
    algorithm: RLAlgorithmDefinition
    collector: CollectorSpec = Field(default_factory=CollectorSpec)
    evaluation: RLEvaluationSpec = Field(default_factory=RLEvaluationSpec)
    total_frames: int = Field(default=100_000, gt=0)
    frames_per_batch: int = Field(default=2_048, gt=0)
    seed: int | None = None
    accelerator: str = "auto"
    devices: str | int = "auto"
    strategy: str | StrategySpec = "auto"
    precision: str | int = "32-true"


class RayJobExecutionSpec(SpecModel):
    """Submit the normal local CLI asynchronously through Ray Jobs."""

    kind: Literal["ray_job"] = "ray_job"
    address: str | None = None
    verify: Annotated[str, Field(min_length=1)] | bool = True
    setup_command: list[str] | None = Field(default=None, min_length=1)
    py_executable: str = Field(default="python", min_length=1)
    runtime_env: dict[str, Any] = Field(default_factory=lambda: {"working_dir": "."})
    entrypoint_num_cpus: float = Field(default=0, ge=0, allow_inf_nan=False)
    entrypoint_num_gpus: float = Field(default=0, ge=0, allow_inf_nan=False)
    entrypoint_memory: int | None = Field(default=None, ge=0)
    entrypoint_resources: dict[str, Annotated[float, Field(gt=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )

    @field_validator("setup_command")
    @classmethod
    def validate_setup_command(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (
            not value[0].strip() or any("\x00" in argument for argument in value)
        ):
            raise ValueError("setup_command requires an executable and arguments without NUL bytes")
        return value

    @field_validator("entrypoint_resources")
    @classmethod
    def validate_resource_names(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not name.strip() for name in value):
            raise ValueError("Ray Job resource names must not be empty")
        return value


class TargetSpec(SpecModel):
    """Specification for a synthetic target."""

    type: str  # "multiclass", "multilabel", "regression"
    key: str  # TensorDict key for this target
    num_classes: int | None = None
    num_outputs: int | None = None
    positive_fraction: float | None = None


class DatasetSpec(SpecModel):
    """Specification for a single dataset in the pipeline."""

    source: DataSourceDefinition
    modality: str = "audio"
    max_samples: int | None = None
    split_type: str = "fit"  # "fit", "all", "keep"


class LoaderSpec(SpecModel):
    """Specification for the data loader."""

    backend: LoaderBackendDefinition = Field(default_factory=lambda: _default_loader_backend())
    # None defers to TrainingSpec.batch_size; explicit loader values take precedence.
    batch_size: int | None = None
    num_workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    weighted_sampling: bool = False
    shuffle_train: bool = True

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("data.loader.batch_size must be positive")
        return value


def _default_loader_backend() -> LoaderBackendDefinition:
    from nexuml.data.loaders.definitions import TorchLoader

    return TorchLoader()


class WriterSpec(FactorySpec):
    """Specification for an importable dataset export backend."""

    factory: str = "nexuml.data.export.numpy_files:NumpyBackend"

    def backend_name(self) -> str:
        """Return the stable registry name for the configured backend class.

        Returns:
            Registered export backend name.

        Raises:
            TypeError: If the configured factory is not an export backend class.
        """
        from nexuml.data.export.backend import ExportBackend, get_export_backend_name

        backend_type = self.resolve()
        if not isinstance(backend_type, type) or not issubclass(backend_type, ExportBackend):
            raise TypeError(f"Writer factory must be an ExportBackend class, got {backend_type!r}")
        return get_export_backend_name(backend_type)


class PreprocessingSpec(SpecModel):
    """Optional preprocessing stage contract."""

    enabled: bool = False
    source_view: str = "raw"
    target_view: str = "prepared"
    path: str | None = None
    until_x_keys: list[str] = Field(default_factory=list)
    until_y_keys: list[str] = Field(default_factory=list)
    x_keys: list[str] | None = None
    y_keys: list[str] | None = None
    include_labels: bool = True
    label_prefix: str = "label__"
    writer: WriterSpec = Field(default_factory=lambda: WriterSpec())
    overwrite: bool = False


class DataSpec(SpecModel):
    """Specification for data configuration."""

    source: DataSourceDefinition | None = None
    targets: list[TargetSpec] = Field(default_factory=list)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    datasets: list[DatasetSpec] = Field(default_factory=list)
    loader: LoaderSpec = Field(default_factory=LoaderSpec)
    input_shapes: dict[str, list[int]] = Field(default_factory=dict)
    input_contracts: dict[str, TensorFieldContract] = Field(default_factory=dict)
    num_classes: int | None = None
    merge_labels: dict[str, dict] | None = None
    feature_key: str = "features"
    skip_pipeline_stages: list[str] = Field(default_factory=list)
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)

    @model_validator(mode="after")
    def validate_splits(self) -> "DataSpec":
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must sum to 1.0")
        for key, contract in self.input_contracts.items():
            if not key.strip():
                raise ValueError("Input contract keys must be nonempty")
            if key in self.input_shapes and tuple(self.input_shapes[key]) != contract.shape:
                raise ValueError(f"Input shape and contract disagree for {key!r}")
        return self


class DistanceEstimatorSpec(SpecModel):
    """Specification for a distance estimator used in anomaly detection."""

    group_label_keys: list[str] = Field(default_factory=list)
    missing_label_policy: Literal["error", "skip", "unknown"] = "error"
    fallback_policy: Literal["error", "parent", "global", "parent_or_global"] = "error"
    storage_backend: Literal["ram", "memmap"] = "memmap"
    storage_path: str | None = None
    max_samples: int | None = None
    retain_storage: bool = False

    def create_feature_store(self) -> FeatureStore:
        """Create the feature store configured for estimator fitting.

        Returns:
            Configured feature store runtime.
        """
        from nexuml.evaluation.utils import create_feature_store

        return create_feature_store(
            self.storage_backend,
            max_samples=self.max_samples,
            storage_path=self.storage_path,
            retain_storage=self.retain_storage,
        )


class EvalAlgorithmSpec(SpecModel):
    """Specification for a post-training evaluation algorithm."""

    algorithm: EvalAlgorithmDefinition
    name: str | None = None
    enabled: bool = True
    axis_keys: list[AxisKeySpec | str] = Field(default_factory=list)
    feature_key: str | None = None
    label_key: str | None = None

    def contract_summary(self) -> dict[str, Any]:
        """Return a summary of the key contract for diagnostics."""
        return {
            "type": self.algorithm.component_name,
            "name": self.name,
            "feature_key": self.feature_key,
            "label_key": self.label_key,
            "axis_keys": [
                a if isinstance(a, str) else {"key": a.key, "source": a.source}
                for a in self.axis_keys
            ],
        }


class EvaluationSpec(SpecModel):
    """Specification for evaluation configuration."""

    metrics: list[str] = Field(default_factory=lambda: ["mse", "mae"])
    algorithms: list[EvalAlgorithmSpec] = Field(default_factory=list)
    test_result_metrics: list[str] | Literal["all", "none"] = "none"


class TensorBoardSpec(SpecModel):
    """TensorBoard logger configuration."""

    log_dir: str = ".experiments/tensorboard"


class DVCLiveSpec(SpecModel):
    """DVCLive logger configuration."""

    dir: str = ".experiments/dvclive"


class MLflowSpec(SpecModel):
    """MLflow logger configuration."""

    tracking_uri: str = "file:./.experiments/mlflow"
    artifact_location: str | None = None
    experiment_name: str | None = None
    log_model: bool = False
    run_id: str | None = None
    parent_run_id: str | None = None
    auto_nested_runs: bool = True
    tags: dict[str, Any] = Field(default_factory=dict)
    synchronous: bool | None = None


class DiagramSpec(SpecModel):
    """Mermaid diagram export configuration."""

    enabled: bool = True
    depth: int = 2
    direction: str = "TB"
    show_params: bool = True
    show_shapes: bool = True
    show_metrics: bool = True
    output_dir: str = ".experiments/diagrams"


class LoggingSpec(SpecModel):
    """Experiment logging configuration."""

    tensorboard: TensorBoardSpec | None = None
    dvclive: DVCLiveSpec | None = None
    mlflow: MLflowSpec | None = None
    diagram: DiagramSpec | None = Field(default_factory=DiagramSpec)
    temp_artifact_dir: str = "/tmp/mlruns"
    experiment_name: str = "NexuML"
    run_name: str | None = None
    log_system_metrics: bool = False


class CallbackSpec(FactorySpec):
    """Specification for a training callback."""


class TuningSpec(SpecModel):
    """Specification for hyperparameter tuning with Optuna."""

    n_trials: int = 50
    directions: list[str] = Field(default_factory=lambda: ["minimize"])
    metric_key: str = "val/loss"
    storage: str = ".experiments/optuna/optuna.log"
    prune: bool = False


class CheckpointLoadSpec(SpecModel):
    """Selective weight loading policy for resume/fine-tune flows."""

    source: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    allow_missing: bool = True
    allow_shape_mismatch: bool = True
    freeze_loaded: bool = False


class ExportSpec(SpecModel):
    """Requested export artifact."""

    kind: Literal["train_package", "onnx", "safetensors"] = "train_package"
    output: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class ScenarioSpec(SpecModel):
    """Complete scenario specification combining all components."""

    name: str
    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)
    training: TrainingSpec | ReinforcementLearningSpec = Field(default_factory=TrainingSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    interaction: InteractionSpec | None = None
    policy: PolicySpec | None = None
    policy_evaluation: PolicyEvaluationSpec | None = None
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    logging: LoggingSpec | None = None
    callbacks: list[CallbackSpec] = Field(default_factory=list)
    tuning: TuningSpec | None = None
    checkpoint: CheckpointLoadSpec | None = None
    exports: list[ExportSpec] = Field(default_factory=list)
    execution: ExecutionBackendDefinition = Field(default_factory=LocalExecution)

    @model_validator(mode="after")
    def validate_reinforcement(self) -> Self:
        if self.policy_evaluation is not None and (self.interaction is None or self.policy is None):
            raise ValueError("policy_evaluation requires scenario.interaction and scenario.policy")
        if isinstance(self.training, ReinforcementLearningSpec):
            if self.interaction is None or self.policy is None:
                raise ValueError("reinforcement training requires scenario.interaction and policy")
            if self.execution.component_name not in {"local", "ray-jobs-sdk"}:
                # ponytail: single learner; enable Ray placement after distributed lifecycle tests.
                raise ValueError(
                    "Ray learner placement for RL is not validated; use local execution"
                )
            if self.training.devices not in (1, "auto") or self.training.strategy != "auto":
                raise ValueError("RL currently supports one learner with training.strategy='auto'")
        return self

    @field_validator("execution", mode="before")
    @classmethod
    def restore_execution(cls, value: Any) -> ExecutionBackendDefinition:
        if isinstance(value, ExecutionBackendDefinition):
            return value
        from nexuml.core.serialization import restore_component

        if not isinstance(value, Mapping) or "kind" in value:
            raise ValueError(
                "execution requires type/version/params; replace legacy execution.kind."
            )
        return cast(
            ExecutionBackendDefinition, restore_component(kind="execution_backend", value=value)
        )
