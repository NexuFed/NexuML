"""Static catalog descriptions from native definitions, never runtime construction."""

from types import UnionType
from typing import Annotated, Literal, TypedDict, Union, get_args, get_origin

from nexuml.core.components import (
    ComponentDefinition,
    DataSourceDefinition,
    EvalAlgorithmDefinition,
    _KeyRole,
)


class InputBinding(TypedDict):
    """A definition-local key field or surrounding evaluation context slot."""

    source: Literal["parameter", "context"]
    field: str
    domain: Literal["x", "y"]
    collection: bool
    required: bool
    default: str | list[str] | None


class ComponentIO(TypedDict):
    """Known input bindings and labels; unknown is distinct from known-empty."""

    inputs: list[InputBinding]
    inputs_known: bool
    labels: list[str] | None


def _key_role(annotation: object) -> tuple[Literal["x", "y"], bool] | None:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        for metadata in args[1:]:
            if isinstance(metadata, _KeyRole):
                return metadata.domain, False
        return _key_role(args[0])
    elif origin is list:
        if role := _key_role(args[0]):
            return role[0], True
    elif origin in (Union, UnionType):
        for member in args:
            if role := _key_role(member):
                return role
    return None


def component_io(definition: type[ComponentDefinition]) -> ComponentIO:
    """Describe public field annotations and shared context rules without instances.

    Returns:
        Static bindings, contract completeness and fixed source labels.
    """
    result: ComponentIO = {"inputs": [], "inputs_known": False, "labels": None}
    if issubclass(definition, DataSourceDefinition):
        if definition.fixed_label_keys is not None:
            result["labels"] = list(definition.fixed_label_keys)
    if not issubclass(definition, EvalAlgorithmDefinition):
        return result
    if definition.context_inputs is not None:
        result["inputs_known"] = True
        for name, domain in (("feature_key", "x"), ("label_key", "y")):
            if (rule := getattr(definition.context_inputs, name)) is not None:
                result["inputs"].append(
                    InputBinding(
                        source="context",
                        field=name,
                        domain=domain,
                        collection=False,
                        required=rule.required,
                        default=rule.default,
                    )
                )
    for name, field in definition.model_fields.items():
        if role := _key_role(field.rebuild_annotation()):
            result["inputs"].append(
                InputBinding(
                    source="parameter",
                    field=name,
                    domain=role[0],
                    collection=role[1],
                    required=field.is_required(),
                    default=None if field.is_required() or field.default_factory else field.default,
                )
            )
    return result
