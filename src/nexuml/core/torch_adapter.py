"""Adapter wrapping plain torch.nn.Modules for TensorDict pipeline compatibility."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, cast

import torch
import torch.nn as nn
from pydantic import Field, field_validator

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml.core.factory import (
    factory_target,
    normalize_json_value,
    resolve_factory,
)


@layer("NnModule")
class NnModuleLayer(LayerDefinition):
    """Portable definition for an importable one-input/one-output PyTorch module."""

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

        factory = resolve_factory(self.factory, label="Direct module factory")
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

    Raises:
        TypeError: If *factory* is a live module instance.
    """
    if isinstance(factory, nn.Module):
        raise TypeError(
            "nn_module() requires an importable factory, not a live torch.nn.Module instance"
        )
    return NnModuleLayer(
        factory=factory_target(factory, label="nn_module() factory"),
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
