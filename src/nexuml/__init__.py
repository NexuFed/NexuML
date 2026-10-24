"""NexuML - Modular deep learning pipeline framework."""

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from nexuml.core.factory import callback, optimizer, scheduler, strategy, writer

if TYPE_CHECKING:
    from nexuml.core.torch_adapter import NnModuleLayer, nn_module

__version__ = version("nexuml")

__all__ = [
    "NnModuleLayer",
    "callback",
    "nn_module",
    "optimizer",
    "scheduler",
    "strategy",
    "writer",
]


def __getattr__(name: str) -> Any:
    if name in {"NnModuleLayer", "nn_module"}:
        from nexuml.core import torch_adapter

        value = getattr(torch_adapter, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
