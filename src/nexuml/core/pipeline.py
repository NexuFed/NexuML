"""Compiled pipeline: the runtime model assembled from specs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.config import ResolvedConfig


class CompiledPipeline(nn.Module):
    """Executable pipeline assembled from compiled layer specs.

    Holds ordered stages of layers, executes sequentially over TensorDicts,
    and provides optimizer/scheduler factory methods.
    """

    def __init__(
        self,
        stages: nn.ModuleDict,
        loss_keys: dict[str, float],
        metric_keys: list[str],
        resolved_config: ResolvedConfig,
        optimizer_spec: dict[str, Any] | None = None,
        scheduler_spec: dict[str, Any] | None = None,
        input_sizes: dict[str, tuple] | None = None,
    ):
        super().__init__()
        self.stages = stages
        self.loss_keys = loss_keys
        self.metric_keys = metric_keys
        self.resolved_config = resolved_config
        self._optimizer_spec = optimizer_spec or {
            "type": "torch.optim.Adam",
            "params": {"lr": 1e-3},
        }
        self._scheduler_spec = scheduler_spec or {
            "type": "torch.optim.lr_scheduler.ConstantLR",
            "params": {"factor": 1.0, "total_iters": 0},
        }
        self.input_sizes = input_sizes or {}

    def iter_layers(self) -> Iterator[tuple[str, str, nn.Module]]:
        """Yield pipeline layers in execution order."""
        for stage_name, stage_layers in self.stages.items():
            if isinstance(stage_layers, nn.ModuleDict):
                for layer_name, layer in stage_layers.items():
                    yield stage_name, layer_name, layer
            elif isinstance(stage_layers, nn.ModuleList):
                for index, layer in enumerate(stage_layers):
                    yield stage_name, f"{index:02d}", layer
            elif isinstance(stage_layers, nn.Module):
                yield stage_name, stage_name, stage_layers

    def forward(
        self, x: TensorDict, y: TensorDict | None = None
    ) -> tuple[TensorDict, TensorDict | None]:
        for _stage_name, _layer_name, layer in self.iter_layers():
            x, y = layer(x, y)
        return x, y

    def forward_until(
        self,
        x: TensorDict,
        y: TensorDict | None = None,
        *,
        x_keys: Sequence[str] | None = None,
        y_keys: Sequence[str] | None = None,
    ) -> tuple[TensorDict, TensorDict | None]:
        """Run the pipeline until the requested keys are available.

        This is useful for feature export when the codebase does not yet model
        preprocessing layers separately from the rest of the network.

        Returns:
            Tuple of ``(x_out, y_out)`` with the requested keys present.

        Raises:
            KeyError: If the pipeline did not produce the requested keys.
        """
        required_x = list(x_keys or [])
        required_y = list(y_keys or [])
        if not required_x and not required_y:
            return self.forward(x, y)

        if _has_required_keys(x, y, required_x, required_y):
            return x, y

        for _stage_name, _layer_name, layer in self.iter_layers():
            x, y = layer(x, y)
            if _has_required_keys(x, y, required_x, required_y):
                return x, y

        missing_x = [key for key in required_x if key not in x.keys()]
        missing_y = [key for key in required_y if y is None or key not in y.keys()]
        raise KeyError(
            "Pipeline did not produce the requested export keys. "
            f"Missing x keys: {missing_x}; missing y keys: {missing_y}"
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        cls_path = cast(str, self._optimizer_spec["type"])
        params = cast(dict[str, Any], self._optimizer_spec.get("params", {}))
        optimizer_cls = _resolve_class(cls_path)
        return optimizer_cls(self.parameters(), **params)

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        cls_path = cast(str, self._scheduler_spec["type"])
        params = cast(dict[str, Any], self._scheduler_spec.get("params", {}))
        scheduler_cls = _resolve_class(cls_path)
        return scheduler_cls(optimizer, **params)

    def call_layer_hook(self, hook_name: str) -> None:
        """Propagate a lifecycle hook to all pipeline layers."""
        for stage_layers in self.stages.values():
            if isinstance(stage_layers, nn.ModuleDict):
                layers: list[nn.Module] = list(stage_layers.values())
            elif isinstance(stage_layers, PipelineLayer):
                layers = [stage_layers]
            elif isinstance(stage_layers, nn.ModuleList):
                layers = list(stage_layers)
            else:
                layers = [stage_layers]
            for layer in layers:
                hook = getattr(layer, hook_name, None)
                if callable(hook):
                    hook()


def _resolve_class(dotted_path: str) -> type:
    """Resolve a dotted class path like 'torch.optim.Adam'.

    Returns:
        The resolved class object.

    Raises:
        ValueError: If the path cannot be split into module and attribute.
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) == 2:
        import importlib

        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])
    raise ValueError(f"Cannot resolve class path: {dotted_path}")


def _has_required_keys(
    x: TensorDict,
    y: TensorDict | None,
    x_keys: Sequence[str],
    y_keys: Sequence[str],
) -> bool:
    return all(key in x.keys() for key in x_keys) and all(
        y is not None and key in y.keys() for key in y_keys
    )
