"""Core NexuML runtime: pipeline, layers, registry, compiler, and types."""

from typing import TYPE_CHECKING, Any

from nexuml.core.factory import callback, optimizer, scheduler, strategy, writer

if TYPE_CHECKING:
    from nexuml.core.post_train_layer import PostTrainFitLayer, PostTrainLayerNotFittedError
    from nexuml.core.torch_adapter import NnModuleLayer, TorchModuleAdapter, nn_module

__all__ = [
    "NnModuleLayer",
    "PostTrainFitLayer",
    "PostTrainLayerNotFittedError",
    "TorchModuleAdapter",
    "callback",
    "nn_module",
    "optimizer",
    "scheduler",
    "strategy",
    "writer",
]


def __getattr__(name: str) -> Any:
    if name in {"PostTrainFitLayer", "PostTrainLayerNotFittedError"}:
        from nexuml.core import post_train_layer as module
    elif name in {"NnModuleLayer", "TorchModuleAdapter", "nn_module"}:
        from nexuml.core import torch_adapter as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
