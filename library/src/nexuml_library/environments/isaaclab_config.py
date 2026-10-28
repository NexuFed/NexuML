"""Portable native-task customization and private spectator declarations."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator

from nexuml.core.factory import factory_target
from nexuml.core.types import FactorySpec, SpecModel


def _validate_signature(fn: object, kwargs: Mapping[str, object]) -> None:
    if (
        not inspect.isfunction(fn)
        or inspect.iscoroutinefunction(fn)
        or inspect.isgeneratorfunction(fn)
        or inspect.isasyncgenfunction(fn)
    ):
        raise TypeError("Isaac task configuration requires a synchronous function")
    signature = inspect.signature(fn, eval_str=False)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("Isaac task configuration requires a first positional config argument")
    if signature.return_annotation is inspect.Signature.empty:
        raise TypeError("Isaac task configuration requires a return annotation")
    for index, parameter in enumerate(parameters):
        if index and parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError("Isaac task configuration requires declared named arguments")
        if parameter.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"Isaac task configuration argument {parameter.name!r} needs an annotation"
            )
    signature.bind(object(), **kwargs)


class IsaacTaskConfigSpec(FactorySpec):
    """Trusted importable function applied to an execution-local native config.

    Restoration validates portable data only. Function resolution and execution
    belong to the native job, not YAML loading or discovery.
    """

    model_config = ConfigDict(frozen=True, validate_default=True, allow_inf_nan=False)

    @field_validator("args")
    @classmethod
    def _require_empty_args(cls, value: list[object]) -> list[object]:
        if value:
            raise ValueError(
                "Isaac task configuration accepts only named kwargs; args must be empty"
            )
        return value

    @classmethod
    def from_factory(cls, factory: object, *args: object, **kwargs: object) -> Self:
        """Capture a typed top-level function and JSON-safe named arguments.

        Returns:
            Portable customization specification.
        """
        spec = super().from_factory(factory, *args, **kwargs)
        _validate_signature(factory, spec.kwargs)
        return spec

    def build(self, *runtime_args: object, **runtime_kwargs: object) -> object:
        """Apply customization, retaining the input when the function returns None.

        Returns:
            Mutated input config or the function's replacement config.

        Raises:
            TypeError: If execution does not supply exactly one configuration.
        """
        if len(runtime_args) == 1 and not runtime_kwargs:
            cfg = runtime_args[0]
        elif not runtime_args and set(runtime_kwargs) == {"cfg"}:
            cfg = runtime_kwargs["cfg"]
        else:
            raise TypeError("Isaac task customization requires exactly one config")
        fn = self.resolve()
        factory_target(fn, label=type(self).__name__)
        _validate_signature(fn, self.kwargs)
        result = fn(cfg, **self.kwargs)
        return cfg if result is None else result


def isaac_task_config(fn: Callable[..., object], **kwargs: object) -> IsaacTaskConfigSpec:
    """Capture trusted native-config customization without serializing native objects.

    Returns:
        Portable customization specification.
    """
    return IsaacTaskConfigSpec.from_factory(fn, **kwargs)


class IsaacBindings(SpecModel):
    """Map declared observation fields to literal native observation mapping keys.

    Source paths index mappings only; an empty group path selects the whole agent
    tensor. Components are never evaluated as configuration overrides. The action
    names the declared local action field.
    """

    model_config = ConfigDict(frozen=True, validate_default=True, allow_inf_nan=False)

    observations: dict[
        Annotated[str, Field(min_length=1)],
        tuple[Annotated[str, Field(min_length=1)], ...],
    ] = Field(default_factory=dict)
    groups: dict[str, IsaacBindings] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    critic_observations: dict[str, tuple[Annotated[str, Field(min_length=1)], ...]] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    action: str = Field(default="action", min_length=1)
    processing: Literal["raw", "processed"] = Field(
        default="processed", exclude_if=lambda value: value == "processed"
    )

    @model_validator(mode="after")
    def _validate_groups(self):
        if self.groups:
            if self.observations or any(group.groups for group in self.groups.values()):
                raise ValueError("Grouped native bindings require one flat binding per group")
        elif not self.observations:
            raise ValueError("Native bindings require observations or groups")
        return self


class ViserSpec(SpecModel):
    """Inert loopback-only spectator settings, never a simulator control surface."""

    model_config = ConfigDict(frozen=True, validate_default=True, allow_inf_nan=False)

    purposes: tuple[Literal["train", "eval"], ...] = Field(default=("train",), min_length=1)
    bind_address: Literal["127.0.0.1", "localhost"] = "127.0.0.1"
    # Native Viser can retry up to 1000 consecutive ports after a bind failure.
    port: Annotated[StrictInt, Field(ge=1, le=64535)] = 8080
    eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible_env_indices: tuple[Annotated[StrictInt, Field(ge=0)], ...] | None = None
    max_visible_envs: Annotated[StrictInt, Field(gt=0)] | None = None
    enable_markers: bool = True
    enable_live_plots: bool = True
    live_plots_update_interval: Annotated[StrictInt, Field(gt=0)] = 5

    @field_validator("purposes", "visible_env_indices")
    @classmethod
    def _require_unique(cls, value: tuple | None) -> tuple | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("purposes and visible_env_indices must contain distinct values")
        return value
