"""Compiler: transforms ScenarioSpec into a runnable CompiledPipeline."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.types import (
    DataSpec,
    InteractionContract,
    PipelineSpec,
    ScenarioSpec,
    TrainingSpec,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineCompileContext:
    """Input metadata shared by dataset, policy and algorithm-owned pipelines."""

    input_sizes: Mapping[str, tuple[int, ...]]
    input_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    num_classes: int | None = None
    skip_stages: tuple[str, ...] = ()


def compile_context_from_data(data: DataSpec) -> PipelineCompileContext:
    """Preserve the supervised compiler's existing input inference.

    Returns:
        Dataset-derived input context, without materializing a dataset.
    """
    if data.input_shapes:
        sizes = {key: tuple(shape) for key, shape in data.input_shapes.items()}
    else:
        source = data.source
        if source is None and data.datasets:
            source = data.datasets[0].source
        sizes = {data.feature_key: tuple(getattr(source, "feature_shape", (128,)))}
    return PipelineCompileContext(
        input_sizes=sizes,
        num_classes=data.num_classes,
        skip_stages=tuple(data.skip_pipeline_stages),
    )


def compile_context_from_interaction(contract: InteractionContract) -> PipelineCompileContext:
    """Translate explicit tensor observations, never inventing missing leaf shapes.

    Returns:
        Portable observation context, excluding vectorization dimensions.

    Raises:
        TypeError: If description is not an interaction contract.
        ValueError: If an observation declares an unsupported tensor dtype.
    """
    if not isinstance(contract, InteractionContract):
        raise TypeError("EnvironmentDefinition.describe() must return InteractionContract")
    supported = {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    }
    for key, leaf in contract.observations.items():
        if leaf.dtype not in supported:
            raise ValueError(f"Observation {key!r} has unsupported dtype {leaf.dtype!r}")
    return PipelineCompileContext(
        input_sizes={key: leaf.shape for key, leaf in contract.observations.items()},
        input_metadata={
            key: leaf.model_dump(mode="json") for key, leaf in contract.observations.items()
        },
    )


def compile(scenario: ScenarioSpec) -> CompiledPipeline:
    """Compile a ScenarioSpec into a runnable CompiledPipeline.

    Steps:
      1. Iterate pipeline stages in order
      2. For each LayerSpec: resolve meta_in, materialize its definition, capture meta_out
      3. Run dummy forward for shape propagation
      4. Return assembled CompiledPipeline

    Returns:
        Compiled pipeline ready for training or inference.
    """
    training = scenario.training if isinstance(scenario.training, TrainingSpec) else None
    if training is None:
        assert scenario.interaction is not None
        context = compile_context_from_interaction(scenario.interaction.environment.describe())
    else:
        context = compile_context_from_data(scenario.data)
    return compile_pipeline(
        scenario.pipeline,
        context=context,
        resolved_config=ResolvedConfig.from_scenario(scenario),
        training=training,
    )


def compile_pipeline(
    pipeline: PipelineSpec,
    *,
    context: PipelineCompileContext,
    resolved_config: ResolvedConfig,
    training: TrainingSpec | None = None,
) -> CompiledPipeline:
    """Build one pipeline implementation from data or interaction metadata.

    Omitting ``training`` disables the supervised loss/optimizer path.

    Returns:
        Compiled neural TensorDict graph.

    Raises:
        TypeError: If a definition does not build a pipeline layer.
    """
    pipeline_dims: dict[str, tuple] = dict(context.input_sizes)
    metadata: dict[str, Any] = {}
    stages = nn.ModuleDict()
    for stage_name, layer_specs in pipeline.stages.items():
        if stage_name in context.skip_stages:
            logger.info("Skipping pipeline stage '%s' per data.skip_pipeline_stages", stage_name)
            continue

        stage_layers = nn.ModuleDict()

        for i, spec in enumerate(layer_specs):
            resolved_metadata: dict[str, Any] = {}
            if context.input_metadata:
                resolved_metadata["input_metadata"] = dict(context.input_metadata)
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
            layer_context = LayerBuildContext(
                input_sizes=pipeline_dims,
                keys_in=keys_in_val,
                keys_out=spec.keys_out,
                label_key=spec.label_key,
                label_in_x=spec.label_in_x,
                num_classes=context.num_classes,
                metadata=resolved_metadata,
                delay_epochs=spec.delay_epochs,
                update_every_n_epochs=spec.update_every_n_epochs,
            )
            layer = spec.component.build(layer_context)
            if not isinstance(layer, PipelineLayer):
                raise TypeError(
                    f"{type(spec.component).__name__}.build() must return PipelineLayer, "
                    f"got {type(layer).__name__}"
                )

            # Shape propagation via dummy forward
            updated_dims = _propagate_shapes(layer, pipeline_dims, context.input_metadata)
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

    return CompiledPipeline(
        stages=stages,
        loss_keys=training.loss_keys if training is not None else {},
        metric_keys=training.metric_keys if training is not None else [],
        resolved_config=resolved_config,
        optimizer_spec=training.optimizer.model_copy(
            update={
                "kwargs": {
                    **training.optimizer.kwargs,
                    "lr": training.lr,
                }
            }
        ).model_dump()
        if training is not None
        else None,
        scheduler_spec=training.scheduler.model_dump() if training is not None else None,
        input_sizes=dict(pipeline_dims),
        supervised_training=training is not None,
    )


def _propagate_shapes(
    layer: nn.Module,
    current_dims: dict[str, tuple],
    input_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple]:
    """Run a dummy forward pass to infer output shapes.

    Returns:
        Mapping of output tensor keys to their inferred shapes (excluding batch).
    """
    # Build dummy TensorDict from current known dimensions
    batch_size = 2
    td_data = {}
    for key, shape in current_dims.items():
        leaf = input_metadata.get(key)
        td_data[key] = (
            torch.zeros((batch_size, *shape), dtype=getattr(torch, leaf["dtype"]))
            if leaf is not None
            else torch.randn(batch_size, *shape)
        )

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
