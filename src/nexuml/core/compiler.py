"""Compiler: transforms ScenarioSpec into a runnable CompiledPipeline."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.types import ScenarioSpec

logger = logging.getLogger(__name__)


def compile(scenario: ScenarioSpec) -> CompiledPipeline:
    """Compile a ScenarioSpec into a runnable CompiledPipeline.

    Steps:
      1. Iterate pipeline stages in order
      2. For each LayerSpec: resolve meta_in, materialize its definition, capture meta_out
      3. Run dummy forward for shape propagation
      4. Return assembled CompiledPipeline

    Returns:
        Compiled pipeline ready for training or inference.

    Raises:
        TypeError: If a definition does not build a ``PipelineLayer``.
    """
    # Track accumulated shapes and metadata
    pipeline_dims: dict[str, tuple] = {}
    metadata: dict[str, Any] = {}
    stages = nn.ModuleDict()

    # Initialize input dims from data spec or a source definition's declared shape.
    if scenario.data.input_shapes:
        pipeline_dims.update(
            {key: tuple(shape) for key, shape in scenario.data.input_shapes.items()}
        )
    else:
        source = scenario.data.source
        if source is None and scenario.data.datasets:
            source = scenario.data.datasets[0].source
        feature_shape = tuple(getattr(source, "feature_shape", (128,)))
        pipeline_dims[scenario.data.feature_key] = feature_shape

    for stage_name, layer_specs in scenario.pipeline.stages.items():
        if stage_name in scenario.data.skip_pipeline_stages:
            logger.info("Skipping pipeline stage '%s' per data.skip_pipeline_stages", stage_name)
            continue

        stage_layers = nn.ModuleDict()

        for i, spec in enumerate(layer_specs):
            resolved_metadata: dict[str, Any] = {}
            if spec.meta_in:
                for param_name, meta_key in spec.meta_in.items():
                    if meta_key in metadata:
                        resolved_metadata[param_name] = metadata[meta_key]
                    else:
                        logger.warning(
                            f"meta_in key '{meta_key}' not found in metadata for "
                            f"{type(spec.component).__name__}. Available: {list(metadata.keys())}"
                        )

            keys_in_val: list[str] = (
                list(spec.keys_in.values()) if isinstance(spec.keys_in, dict) else spec.keys_in
            )
            context = LayerBuildContext(
                input_sizes=pipeline_dims,
                keys_in=keys_in_val,
                keys_out=spec.keys_out,
                label_key=spec.label_key,
                label_in_x=spec.label_in_x,
                num_classes=scenario.data.num_classes,
                metadata=resolved_metadata,
                delay_epochs=spec.delay_epochs,
                update_every_n_epochs=spec.update_every_n_epochs,
            )
            layer = spec.component.build(context)
            if not isinstance(layer, PipelineLayer):
                raise TypeError(
                    f"{type(spec.component).__name__}.build() must return PipelineLayer, "
                    f"got {type(layer).__name__}"
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
                            f"{type(spec.component).__name__} instance"
                        )

            layer_key = f"{i:02d}_{spec.component.component_name}"
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
