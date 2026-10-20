"""Installation-owned execution, with lazy optional runtime imports."""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nexuml.execution.ray import RayExecutionError, run_ray
    from nexuml.execution.service import catalog, discover, preflight, run, inspect, cancel

__all__ = [
    "RayExecutionError",
    "run_ray",
    "catalog",
    "discover",
    "preflight",
    "run",
    "inspect",
    "cancel",
]


def __getattr__(name: str) -> Any:
    if name in {"RayExecutionError", "run_ray"}:
        from nexuml.execution import ray

        return getattr(ray, name)
    if name in {"catalog", "discover", "preflight", "run", "inspect", "cancel"}:
        from nexuml.execution import service

        return getattr(service, name)
    raise AttributeError(name)
