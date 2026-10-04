"""Typed component definitions and explicit runtime build contexts."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


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


class ExecutionBackendDefinition(ComponentDefinition):
    """Installation-owned placement settings and execution behavior."""

    kind = "execution_backend"
    label: ClassVar[str] = ""
    dependencies: ClassVar[tuple[str, ...]] = ()
    capabilities: ClassVar[dict[str, bool]] = {}
    presentation: ClassVar[dict[str, list[str]]] = {}

    def preflight(self, scenario: Any, **kwargs: Any) -> dict[str, Any]:
        """Validate a request without allocating workers.

        Returns:
            Backend-specific review information.

        Raises:
            ValueError: If requested resume or exports are unsupported.
        """
        if kwargs.get("trainer_checkpoint") and not self.capabilities.get("resume", False):
            raise ValueError("Trainer checkpoint resume is local-only; the backend owns recovery.")
        if scenario.exports and not self.capabilities.get("artifacts", False):
            raise ValueError("Configured model exports are unsupported by the selected backend.")
        return {}

    def discover(self, **kwargs: Any) -> Any:
        """Return an explicit unknown snapshot when a library has no capacity probe."""
        from nexuml.execution.schemas import ResourceSnapshot

        return ResourceSnapshot(
            backend=self.component_name,
            source=self.component_name,
            diagnostics=["This backend does not expose resource discovery."],
        )

    def inspect(self, reference: Any) -> dict[str, Any]:
        """Refuse native observation that the backend has not implemented.

        Raises:
            NotImplementedError: When no native inspection is available.
        """
        raise NotImplementedError("Native reference inspection is unsupported by this backend.")

    def cancel(self, reference: Any) -> dict[str, Any]:
        """Refuse to confuse submitter termination with native cancellation.

        Raises:
            NotImplementedError: When native cancellation cannot be confirmed.
        """
        raise NotImplementedError("Native cancellation is unsupported by this backend.")

    @abstractmethod
    def run(self, scenario: Any, **kwargs: Any) -> Any:
        """Execute the canonical lifecycle and return its native result.

        Shared dispatch supplies ``review`` from preflight and caller-specific keyword
        options (including an optional observer). Definitions accept these via kwargs.
        """
