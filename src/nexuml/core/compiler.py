"""Compiler: transforms ScenarioSpec into a runnable CompiledPipeline."""

from __future__ import annotations

import inspect
import logging
from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.config import ResolvedConfig
from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.registry import LayerRegistry, get_registry
from nexuml.core.types import ScenarioSpec

logger = logging.getLogger(__name__)


def compile(
    scenario: ScenarioSpec,
    registry: LayerRegistry | None = None,
) -> CompiledPipeline:
    """Compile a ScenarioSpec into a runnable CompiledPipeline.

    Steps:
      1. Iterate pipeline stages in order
      2. For each LayerSpec: resolve meta_in, instantiate via registry, capture meta_out
      3. Run dummy forward for shape propagation
      4. Return assembled CompiledPipeline

    Returns:
        Compiled pipeline ready for training or inference.
    """
    if registry is None:
        registry = get_registry()

    # Track accumulated shapes and metadata
    pipeline_dims: dict[str, tuple] = {}
    metadata: dict[str, Any] = {}
    stages = nn.ModuleDict()

    # Initialize input dims from data spec. New scenarios should declare explicit
    # input_shapes; older synthetic scenarios still fall back to feature_shape.
    if scenario.data.input_shapes:
        pipeline_dims.update(
            {key: tuple(shape) for key, shape in scenario.data.input_shapes.items()}
        )
    else:
        feature_shape = tuple(scenario.data.params.get("feature_shape", (128,)))
        pipeline_dims[scenario.data.feature_key] = feature_shape

    for stage_name, layer_specs in scenario.pipeline.stages.items():
        if stage_name in scenario.data.skip_pipeline_stages:
            logger.info("Skipping pipeline stage '%s' per data.skip_pipeline_stages", stage_name)
            continue

        stage_layers = nn.ModuleDict()

        for i, spec in enumerate(layer_specs):
            # Resolve meta_in: inject metadata values into params
            resolved_params = dict(spec.params)
            if spec.meta_in:
                for param_name, meta_key in spec.meta_in.items():
                    if meta_key in metadata:
                        resolved_params[param_name] = metadata[meta_key]
                    else:
                        logger.warning(
                            f"meta_in key '{meta_key}' not found in metadata for "
                            f"{spec.type_key}. Available: {list(metadata.keys())}"
                        )

            # Auto-inject num_classes from data spec when the layer accepts it
            # but it wasn't explicitly provided (or was set to None).
            if scenario.data.num_classes is not None:
                layer_cls = registry.get(spec.type_key)
                sig = inspect.signature(layer_cls.__init__)
                if "num_classes" in sig.parameters and resolved_params.get("num_classes") is None:
                    resolved_params["num_classes"] = scenario.data.num_classes

            # Instantiate layer
            keys_in_val: list[str] = (
                list(spec.keys_in.values()) if isinstance(spec.keys_in, dict) else spec.keys_in
            )
            layer = registry.instantiate(
                spec.type_key,
                input_sizes=dict(pipeline_dims),
                keys_in=keys_in_val,
                keys_out=spec.keys_out,
                **resolved_params,
            )

            # Shape propagation via dummy forward
            updated_dims = _propagate_shapes(layer, pipeline_dims)
            pipeline_dims.update(updated_dims)

            # Capture meta_out
            if spec.meta_out:
                for attr_name, meta_key in spec.meta_out.items():
                    if hasattr(layer, attr_name):
                        metadata[meta_key] = getattr(layer, attr_name)
                    else:
                        logger.warning(
                            f"meta_out attribute '{attr_name}' not found on "
                            f"{spec.type_key} instance"
                        )

            layer_key = f"{i:02d}_{spec.type_key}"
            stage_layers[layer_key] = layer

        stages[stage_name] = stage_layers

    resolved_config = ResolvedConfig.from_scenario(scenario)

    return CompiledPipeline(
        stages=stages,
        loss_keys=scenario.training.loss_keys,
        metric_keys=scenario.training.metric_keys,
        resolved_config=resolved_config,
        optimizer_spec={
            "type": scenario.training.optimizer.type,
            "params": {**scenario.training.optimizer.params, "lr": scenario.training.lr},
        },
        scheduler_spec={
            "type": scenario.training.scheduler.type,
            "params": scenario.training.scheduler.params,
        },
        input_sizes=dict(pipeline_dims),
    )


def _propagate_shapes(
    layer: nn.Module,
    current_dims: dict[str, tuple],
) -> dict[str, tuple]:
    """Run a dummy forward pass to infer output shapes.

    Returns:
        Mapping of output tensor keys to their inferred shapes (excluding batch).
    """
    # Build dummy TensorDict from current known dimensions
    batch_size = 2
    td_data = {}
    for key, shape in current_dims.items():
        td_data[key] = torch.randn(batch_size, *shape)

    x = TensorDict(cast(Any, td_data), batch_size=[batch_size])
    y = None

    with torch.no_grad():
        if isinstance(layer, PipelineLayer):
            setattr(layer, "_shape_propagation_mode", True)
        try:
            x_out, _ = layer(x, y)
        finally:
            if isinstance(layer, PipelineLayer):
                setattr(layer, "_shape_propagation_mode", False)

    # Extract output shapes from keys_out
    updated: dict[str, tuple] = {}
    if isinstance(layer, PipelineLayer):
        for key in layer.keys_out:
            if key in x_out.keys():
                updated[key] = tuple(x_out[key].shape[1:])

    return updated
