"""Generic component lowering and restoration at config boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePath
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin

from pydantic import BaseModel

from nexuml.core.components import ComponentDefinition
from nexuml.core.registry import get_component_registry


def lower_component(component: ComponentDefinition) -> dict[str, Any]:
    """Lower a concrete definition to its stable serialized identity.

    Returns:
        Portable component identity and validated parameters.
    """
    entry = get_component_registry().entry_for_type(type(component))
    return {
        "type": entry.name,
        "version": entry.version,
        "params": component.model_dump(mode="json"),
    }


def restore_component(*, kind: str, value: Mapping[str, Any]) -> ComponentDefinition:
    """Restore a concrete definition from stable serialized data.

    Returns:
        Validated concrete component definition.

    Raises:
        TypeError: If serialized parameters are not a mapping.
        ValueError: If identity fields are missing or invalid.
    """
    name = value.get("type")
    version = value.get("version")
    params = value.get("params", {})
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError("serialized components require string type and version fields")
    if not isinstance(params, Mapping):
        raise TypeError("serialized component params must be a mapping")
    definition_type = get_component_registry().get_type(kind, name, version)
    return definition_type.model_validate(params)


def lower_model(model: BaseModel) -> dict[str, Any]:
    """Recursively lower a Pydantic model and all component definitions.

    Returns:
        Portable plain model data.
    """
    return {name: _lower_value(getattr(model, name)) for name in type(model).model_fields}


def _lower_value(value: Any) -> Any:
    if isinstance(value, ComponentDefinition):
        return lower_component(value)
    if isinstance(value, BaseModel):
        return lower_model(value)
    if isinstance(value, Mapping):
        return {str(key): _lower_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_lower_value(item) for item in value]
    if isinstance(value, PurePath):
        return str(value)
    return value


def restore_model_data(data: Mapping[str, Any], model_type: type[BaseModel]) -> dict[str, Any]:
    """Restore nested component definitions before model validation.

    Returns:
        Model data containing concrete component definition instances.
    """
    restored = dict(data)
    for name, field in model_type.model_fields.items():
        if name in restored:
            restored[name] = _restore_value(restored[name], field.annotation)
    return restored


def _restore_value(value: Any, annotation: Any) -> Any:
    component_type = _model_subclass(annotation, ComponentDefinition)
    if component_type is not None and isinstance(value, Mapping):
        definition_type = cast(type[ComponentDefinition], component_type)
        restored = restore_component(kind=definition_type.kind, value=value)
        if not isinstance(restored, component_type):
            raise TypeError(
                f"Expected {component_type.__name__}, restored {type(restored).__name__}"
            )
        return restored

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list and args and isinstance(value, list):
        return [_restore_value(item, args[0]) for item in value]
    if origin is dict and len(args) == 2 and isinstance(value, Mapping):
        return {key: _restore_value(item, args[1]) for key, item in value.items()}

    nested_model = _model_subclass(annotation, BaseModel)
    if nested_model is not None and isinstance(value, Mapping):
        return restore_model_data(value, nested_model)
    return value


def _model_subclass(annotation: Any, base: type[BaseModel]) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, base):
        return annotation
    if get_origin(annotation) in (Union, UnionType):
        for option in get_args(annotation):
            match = _model_subclass(option, base)
            if match is not None:
                return match
    return None
