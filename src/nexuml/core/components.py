"""Typed component definitions and explicit runtime build contexts."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from torch import nn

    from nexuml.core.policy import CompiledPolicy
    from nexuml.core.types import InteractionContract
    from nexuml.interaction.runtime import EnvironmentRuntime


class ComponentDefinition(BaseModel, ABC):
    """Immutable semantic configuration for a registered component."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=False,
    )

    kind: ClassVar[str]
    component_name: ClassVar[str]
    component_version: ClassVar[str] = "1"


@dataclass(frozen=True, slots=True)
class LayerBuildContext:
    """Runtime values supplied by the pipeline compiler to a layer definition."""

    input_sizes: Mapping[str, tuple[int, ...]]
    keys_in: list[str]
    keys_out: list[str]
    label_key: str | list[str] | None = None
    label_in_x: bool = False
    num_classes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    shared_storage: Any = None
    shared_outputs: list[str] | str | None = None
    shared_inputs: list[str] | str | None = None
    delay_epochs: int = 0
    update_every_n_epochs: int = 1

    def runtime_kwargs(self, **overrides: Any) -> dict[str, Any]:
        values = {
            "input_sizes": dict(self.input_sizes),
            "keys_in": self.keys_in,
            "keys_out": self.keys_out,
            "label_key": self.label_key,
            "label_in_x": self.label_in_x,
            "num_classes": self.num_classes,
            "shared_memory": self.shared_storage,
            "shared_outputs": self.shared_outputs,
            "shared_inputs": self.shared_inputs,
            "delay_epochs": self.delay_epochs,
            "update_every_n_epochs": self.update_every_n_epochs,
        }
        values.update(overrides)
        return values


@dataclass(frozen=True, slots=True)
class EvalBuildContext:
    """Runtime routing values supplied to an evaluation definition."""

    feature_key: str | None = None
    label_key: str | None = None


class LayerDefinition(ComponentDefinition):
    """Semantic configuration that materializes a pipeline layer."""

    kind = "layer"
    requires_post_train_fit: ClassVar[bool] = False

    @abstractmethod
    def build(self, context: LayerBuildContext) -> Any:
        """Build the mutable pipeline layer runtime."""


class DataSourceDefinition(ComponentDefinition):
    """Semantic configuration that materializes a dataset."""

    kind = "data_source"

    @abstractmethod
    def build(self) -> Any:
        """Build the mutable dataset runtime."""


class EvalAlgorithmDefinition(ComponentDefinition):
    """Semantic configuration that materializes an evaluation algorithm."""

    kind = "eval_algorithm"

    @abstractmethod
    def build(self, context: EvalBuildContext) -> Any:
        """Build the mutable evaluation runtime."""


class LoaderBackendDefinition(ComponentDefinition):
    """Semantic configuration that materializes a loader backend."""

    kind = "loader_backend"

    @abstractmethod
    def build(self) -> Any:
        """Build the loader backend runtime."""


@dataclass(frozen=True, slots=True)
class EnvironmentBuildContext:
    """Execution-local settings for a training or evaluation environment."""

    purpose: Literal["train", "eval"]
    seed: int | None = None
    device: str = "cpu"
    worker_rank: int = 0
    num_envs: int = 1

    def __post_init__(self) -> None:
        if self.purpose not in ("train", "eval"):
            raise ValueError("environment purpose must be train or eval")
        if self.num_envs < 1 or self.worker_rank < 0:
            raise ValueError("num_envs must be positive and worker_rank nonnegative")


class EnvironmentDefinition(ComponentDefinition):
    """Portable environment configuration, never a live simulator or transport."""

    kind = "environment"

    @abstractmethod
    def describe(self) -> "InteractionContract":
        """Describe tensor I/O without hardware connections or retained resources."""

    @abstractmethod
    def build(self, context: EnvironmentBuildContext) -> "EnvironmentRuntime":
        """Materialize reset/step/close resources at the execution site."""


class ActionAdapterDefinition(ComponentDefinition):
    """Portable configuration for converting neural outputs into valid actions."""

    kind = "action_adapter"

    @abstractmethod
    def build(self, contract: "InteractionContract") -> "nn.Module":
        """Build the deployable action adapter, independent of any learner."""


class RLAlgorithmBuildContext(NamedTuple):
    """Learner inputs, deliberately excluding live interaction resources."""

    policy: "CompiledPolicy"
    contract: "InteractionContract"


class RLAlgorithmDefinition(ComponentDefinition):
    """Portable configuration for a checkpoint-visible reinforcement learner."""

    kind = "rl_algorithm"

    @abstractmethod
    def build(self, context: RLAlgorithmBuildContext) -> "nn.Module":
        """Build algorithm-owned checkpointable state, not an environment."""
