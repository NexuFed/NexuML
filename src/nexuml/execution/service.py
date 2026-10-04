"""Shared Python entrypoints; CLI and API only transport these operations."""

import importlib.util
from typing import Any

from nexuml.core.components import ExecutionBackendDefinition
from nexuml.core.registry import get_component_registry
from nexuml.core.serialization import lower_model, restore_model_data
from nexuml.execution.schemas import NativeReference, ResourceSnapshot


def dependency_diagnostics(definition: type[ExecutionBackendDefinition]) -> list[str]:
    """Check installation support without importing optional runtimes.

    Returns:
        Missing dependency reasons, not cluster readiness claims.
    """
    return [
        f"Missing dependency {name}; install the backend's optional dependencies."
        for name in definition.dependencies
        if importlib.util.find_spec(name) is None
    ]


def catalog() -> list[dict[str, Any]]:
    """Describe execution definitions from the selected Python installation.

    Returns:
        Registered identities, actual schemas, provenance and installation capabilities.
    """
    rows = []
    for entry in get_component_registry().entries(kind="execution_backend"):
        definition = entry.definition_type
        diagnostics = dependency_diagnostics(definition)
        rows.append(
            {
                "type": entry.name,
                "version": entry.version,
                "label": definition.label or entry.name,
                "schema": definition.model_json_schema(),
                "import_target": entry.import_target,
                "available": not diagnostics,
                "diagnostics": diagnostics,
                "capabilities": dict(definition.capabilities),
                "presentation": dict(definition.presentation),
            }
        )
    return rows


def discover(execution: ExecutionBackendDefinition, **kwargs: Any) -> ResourceSnapshot:
    """Probe only the caller-selected backend and target.

    Returns:
        An advisory snapshot, including unavailable/unknown diagnostics.
    """
    get_component_registry().entry_for_type(type(execution))
    diagnostics = dependency_diagnostics(type(execution))
    if diagnostics:
        return ResourceSnapshot(
            backend=execution.component_name,
            source=execution.component_name,
            diagnostics=diagnostics,
        )
    return execution.discover(**kwargs)


def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]:
    """Revalidate a frozen selection before any worker allocation or submission.

    Returns:
        Backend-owned review information.

    Raises:
        ValueError: If dependencies, scenario semantics or launch inputs are invalid.
    """
    scenario = type(scenario).model_validate(
        restore_model_data(lower_model(scenario), type(scenario))
    )
    diagnostics = dependency_diagnostics(type(scenario.execution))
    if diagnostics:
        raise ValueError(" ".join(diagnostics))
    return scenario.execution.preflight(scenario, **kwargs)


def run(scenario: Any, **kwargs: Any) -> Any:
    """Launch through the same installation-owned dispatch in every interface.

    Returns:
        The backend's native result, never a second training result hierarchy.
    """
    scenario = type(scenario).model_validate(
        restore_model_data(lower_model(scenario), type(scenario))
    )
    review = preflight(scenario, **kwargs)
    return scenario.execution.run(scenario, review=review, **kwargs)


def inspect(reference: NativeReference) -> dict[str, Any]:
    """Inspect a native reference without resubmitting it.

    Returns:
        Actual supported native observation from the registered backend.
    """
    definition = get_component_registry().get_type(
        "execution_backend", reference.backend, reference.version
    )
    return definition().inspect(reference)


def cancel(reference: NativeReference) -> dict[str, Any]:
    """Request supported native cancellation through the registered backend.

    Returns:
        Confirmed cancellation or an explicit unsupported/unknown error.
    """
    definition = get_component_registry().get_type(
        "execution_backend", reference.backend, reference.version
    )
    return definition().cancel(reference)
