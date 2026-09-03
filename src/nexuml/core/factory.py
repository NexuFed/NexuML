"""Portable importable-factory helpers for typed Python configuration."""

from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, cast

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.strategies import Strategy
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from nexuml.core.types import (
        CallbackSpec,
        OptimizerSpec,
        SchedulerSpec,
        StrategySpec,
        WriterSpec,
    )
    from nexuml.data.export.backend import ExportBackend

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


def normalize_json_value(value: Any, path: str) -> JsonValue:
    """Normalize one constructor value to portable JSON data.

    Returns:
        Normalized JSON value.

    Raises:
        ValueError: If the value cannot be represented portably.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} mappings must use string keys")
        return {key: normalize_json_value(value[key], f"{path}.{key}") for key in sorted(value)}
    raise ValueError(
        f"{path} contains unsupported {type(value).__name__}; "
        "use only null, booleans, integers, finite floats, strings, lists/tuples, "
        "and string-key mappings"
    )


def resolve_factory(target: str, *, label: str = "factory") -> Callable[..., object]:
    """Resolve a portable ``module:name`` target.

    Returns:
        Imported callable.

    Raises:
        TypeError: If the imported target is not callable.
        ValueError: If the target is malformed or cannot be imported.
    """
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
            f"{label} target {target!r} must name a top-level importable callable as 'module:name'"
        )
    try:
        factory = getattr(importlib.import_module(module_name), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not import {label} {target!r}: {exc}") from exc
    if not callable(factory):
        raise TypeError(f"{label} {target!r} is not callable")
    return factory


def factory_target(factory: object, *, label: str = "factory") -> str:
    """Return and verify the portable target for an importable callable.

    Returns:
        Portable ``module:name`` target.

    Raises:
        TypeError: If *factory* is not callable.
        ValueError: If *factory* cannot be re-imported safely.
    """
    if inspect.ismethod(factory) and factory.__self__ is not None:
        raise ValueError(f"{label} does not support bound methods")
    if not callable(factory):
        raise TypeError(f"{label} must be callable")

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
            f"{label} must be a top-level importable class or function; "
            "lambdas, closures, local/nested definitions, and __main__ targets are unsupported"
        )

    target = f"{module_name}:{name}"
    resolved = resolve_factory(target, label=label)
    if resolved is not factory:
        raise ValueError(f"{label} {target!r} does not re-import to the same object")
    return target


P = ParamSpec("P")


def callback(factory: Callable[P, Callback], *args: P.args, **kwargs: P.kwargs) -> CallbackSpec:
    """Configure an importable Lightning callback factory.

    Returns:
        Portable callback specification.
    """
    from nexuml.core.types import CallbackSpec

    return CallbackSpec.from_factory(factory, *args, **kwargs)


def optimizer(
    factory: Callable[Concatenate[Any, P], Optimizer], *args: P.args, **kwargs: P.kwargs
) -> OptimizerSpec:
    """Configure an optimizer factory; model parameters are supplied at runtime.

    Returns:
        Portable optimizer specification.
    """
    from nexuml.core.types import OptimizerSpec

    return OptimizerSpec.from_factory(factory, *args, **kwargs)


def scheduler(
    factory: Callable[Concatenate[Optimizer, P], LRScheduler],
    *args: P.args,
    **kwargs: P.kwargs,
) -> SchedulerSpec:
    """Configure a scheduler factory; its optimizer is supplied at runtime.

    Returns:
        Portable scheduler specification.
    """
    from nexuml.core.types import SchedulerSpec

    return SchedulerSpec.from_factory(factory, *args, **kwargs)


def strategy(factory: Callable[P, Strategy], *args: P.args, **kwargs: P.kwargs) -> StrategySpec:
    """Configure an importable Lightning strategy factory.

    Returns:
        Portable strategy specification.
    """
    from nexuml.core.types import StrategySpec

    return StrategySpec.from_factory(factory, *args, **kwargs)


def writer(factory: Callable[P, ExportBackend], *args: P.args, **kwargs: P.kwargs) -> WriterSpec:
    """Configure an importable dataset export backend factory.

    Returns:
        Portable writer specification.
    """
    from nexuml.core.types import WriterSpec

    return WriterSpec.from_factory(factory, *args, **kwargs)


def factory_values(
    factory: object,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Build validated Pydantic field values for a factory helper.

    Returns:
        Factory target and constructor arguments.
    """
    return {
        "factory": factory_target(factory, label=label),
        "args": cast(Any, args),
        "kwargs": cast(Any, kwargs),
    }
