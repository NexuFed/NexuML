"""Adapter wrapping plain torch.nn.Modules for TensorDict pipeline compatibility."""

from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, cast

import torch
import torch.nn as nn
from pydantic import Field, field_validator

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


def _normalize_json_value(value: Any, path: str) -> JsonValue:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} mappings must use string keys")
        return {key: _normalize_json_value(value[key], f"{path}.{key}") for key in sorted(value)}
    raise ValueError(
        f"{path} contains unsupported {type(value).__name__}; "
        "use only null, booleans, integers, finite floats, strings, lists/tuples, "
        "and string-key mappings"
    )


def _resolve_factory(target: str) -> Callable[..., object]:
    module_name, separator, name = target.partition(":")
    if (
        not separator
        or not module_name
        or not name
        or module_name == "__main__"
        or "." in name
        or "<" in name
    ):
        raise ValueError(
            f"Direct module factory target {target!r} must name a top-level importable callable "
            "as 'module:name'"
        )
    try:
        factory = getattr(importlib.import_module(module_name), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not import direct module factory {target!r}: {exc}") from exc
    if not callable(factory):
        raise TypeError(f"Direct module factory {target!r} is not callable")
    return factory


def _factory_target(factory: object) -> str:
    if isinstance(factory, nn.Module):
        raise TypeError(
            "nn_module() requires an importable factory, not a live torch.nn.Module instance"
        )
    if inspect.ismethod(factory) and factory.__self__ is not None:
        raise ValueError("nn_module() does not support bound methods")
    if not callable(factory):
        raise TypeError("nn_module() factory must be callable")

    module_name = getattr(factory, "__module__", None)
    name = getattr(factory, "__qualname__", None)
    if (
        not isinstance(module_name, str)
        or not isinstance(name, str)
        or module_name == "__main__"
        or "." in name
        or "<" in name
    ):
        raise ValueError(
            "nn_module() factory must be a top-level importable class or function; "
            "lambdas, closures, local/nested definitions, and __main__ targets are unsupported"
        )

    target = f"{module_name}:{name}"
    resolved = _resolve_factory(target)
    if resolved is not factory:
        raise ValueError(f"nn_module() factory {target!r} does not re-import to the same object")
    return target


@layer("NnModule")
class NnModuleLayer(LayerDefinition):
    """Portable definition for an importable one-input/one-output PyTorch module."""

    factory: str
    args: list[JsonValue] = Field(default_factory=list)
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("args", mode="before")
    @classmethod
    def _validate_args(cls, value: Any) -> list[JsonValue]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("args must be a list or tuple")
        return [_normalize_json_value(item, f"args[{index}]") for index, item in enumerate(value)]

    @field_validator("kwargs", mode="before")
    @classmethod
    def _validate_kwargs(cls, value: Any) -> dict[str, JsonValue]:
        if not isinstance(value, Mapping):
            raise ValueError("kwargs must be a mapping")
        normalized = _normalize_json_value(value, "kwargs")
        assert isinstance(normalized, dict)
        return normalized

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        if len(context.keys_in) != 1 or len(context.keys_out) != 1:
            raise ValueError(
                "NnModule requires exactly one input key and one output key; "
                "use a registered LayerDefinition for richer routing"
            )
        if context.label_key is not None or context.label_in_x:
            raise ValueError(
                "NnModule cannot consume labels; use a registered LayerDefinition for label routing"
            )

        factory = _resolve_factory(self.factory)
        try:
            module = factory(*self.args, **self.kwargs)
        except Exception as exc:
            raise TypeError(
                f"Could not construct direct module from {self.factory!r}: {exc}"
            ) from exc
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Direct module factory {self.factory!r} must return torch.nn.Module, "
                f"got {type(module).__name__}"
            )
        return TorchModuleAdapter(module=module, **context.runtime_kwargs())


P = ParamSpec("P")


def nn_module(factory: Callable[P, nn.Module], *args: P.args, **kwargs: P.kwargs) -> NnModuleLayer:
    """Create a portable direct-module definition from an importable factory.

    Returns:
        Validated universal layer definition.
    """
    return NnModuleLayer(
        factory=_factory_target(factory),
        args=cast(Any, args),
        kwargs=cast(Any, kwargs),
    )


class TorchModuleAdapter(PipelineLayer):
    """Wraps any torch.nn.Module to conform to PipelineLayer interface."""

    def __init__(
        self,
        module: nn.Module,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        self.module = module

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        result = self.module(x)
        if not isinstance(result, torch.Tensor):
            target = f"{type(self.module).__module__}.{type(self.module).__qualname__}"
            raise TypeError(
                f"Direct module {target} must return one torch.Tensor, got {type(result).__name__}"
            )
        return result
