"""Mermaid diagram export for NexuML pipelines."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any, cast

import torch.nn as nn


_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z_]+")


def _sanitize_id(value: str) -> str:
    """Replace special characters with underscores to make valid Mermaid IDs.

    Returns:
        Sanitised identifier safe for Mermaid node names.
    """
    safe = _NON_ALNUM_RE.sub("_", value).strip("_")
    if safe and safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def _escape_label(label: str) -> str:
    """Escape backticks and quotes for Mermaid backtick-quoted labels.

    Returns:
        Label string with backticks and quotes replaced, newlines escaped.
    """
    return label.replace("`", "'").replace('"', "'").replace("\n", "<br/>")


class _MermaidBuilder:
    """Helper to assemble Mermaid flowchart syntax with indentation."""

    def __init__(self, direction: str = "TB") -> None:
        self._direction = direction
        self._lines: list[str] = []
        self._indent = 0
        self._counter = itertools.count(1)
        self._class_defs: dict[str, str] = {}

    def start(self) -> None:
        self._lines.append("---")
        self._lines.append("config:")
        self._lines.append("    layout: elk")
        self._lines.append("---")
        self._lines.append(f"flowchart {self._direction}")
        self._indent = 1

    def finish(self) -> str:
        current_indent = self._indent
        self._indent = 1
        for class_name, definition in self._class_defs.items():
            self._write(f"classDef {class_name} {definition}")
        self._indent = current_indent
        return "\n".join(self._lines)

    def unique_id(self, prefix: str) -> str:
        safe_prefix = _sanitize_id(prefix) or "node"
        return f"{safe_prefix}_{next(self._counter)}"

    def begin_subgraph(self, subgraph_id: str, title: str) -> None:
        self._write(f'subgraph {subgraph_id}["{_escape_label(title)}"]')
        self._indent += 1
        self._write(f"direction {self._direction}")

    def end_subgraph(self) -> None:
        self._indent -= 1
        self._write("end")

    def add_node(self, node_id: str, label: str, node_class: str | None = None) -> None:
        escaped = _escape_label(label)
        text = f'{node_id}["`{escaped}`"]' if escaped else f'{node_id}[" "]'
        if node_class:
            text += f":::{node_class}"
        self._write(text)

    def add_edge(self, source: str, target: str, label: str | None = None) -> None:
        edge = f"{source} --> {target}"
        if label:
            edge = f"{source} --> |{_escape_label(label)}| {target}"
        self._write(edge)

    def add_class_def(self, class_name: str, definition: str) -> None:
        self._class_defs[class_name] = definition

    def apply_class(self, node_id: str, node_class: str) -> None:
        """Apply a class to a node or subgraph via Mermaid `class` directive."""
        self._write(f"class {node_id} {node_class}")

    def _write(self, text: str) -> None:
        indent = "    " * self._indent
        self._lines.append(f"{indent}{text}".rstrip())


def _layer_node_id(stage_name: str, layer_name: str, counter: itertools.count) -> str:
    prefix = _sanitize_id(f"{stage_name}_{layer_name}")
    return f"{prefix}_{next(counter)}"


def _find_producer_for_consumer(
    all_layers: list[tuple[str, str, nn.Module]],
    consumer_idx: int,
    key: str,
) -> tuple[str, str, nn.Module] | None:
    """Find the most recent producer of *key* before the consumer at *consumer_idx*.

    Returns:
        ``(stage_name, layer_name, layer)`` of the producer, or ``None`` if
        the key is an unresolved input.
    """
    for i in range(consumer_idx - 1, -1, -1):
        stage_name, layer_name, layer = all_layers[i]
        keys_out = cast(list[str], getattr(layer, "keys_out", []))
        if key in keys_out:
            return stage_name, layer_name, layer
    return None


def _get_shape(pipeline: Any, key: str) -> tuple | None:
    """Get the shape for a tensor key from the pipeline's accumulated sizes.

    Returns:
        Shape tuple for *key*, or ``None`` if not recorded.
    """
    return pipeline.input_sizes.get(key)


def _render_module_children(
    builder: _MermaidBuilder,
    module: nn.Module,
    prefix: str,
    depth: int,
    show_params: bool,
    pipeline: Any,
    layer_params: dict,
) -> list[str]:
    """Recursively render child modules and return their node IDs in sequence.

    Returns:
        Ordered list of Mermaid node/subgraph IDs for the rendered children.
    """
    if depth <= 0:
        node_id = f"{prefix}_body"
        params = layer_params.get((prefix, "body"), 0)
        label = module.__class__.__name__
        if show_params and params > 0:
            label += f" ({params:,} params)"
        builder.add_node(node_id, label)
        return [node_id]

    children = list(module.named_children())
    if not children:
        node_id = f"{prefix}_body"
        params = layer_params.get((prefix, "body"), 0)
        label = module.__class__.__name__
        if show_params and params > 0:
            label += f" ({params:,} params)"
        builder.add_node(node_id, label)
        return [node_id]

    sequence_ids: list[str] = []
    for index, (child_name, child_module) in enumerate(children):
        child_params = sum(p.numel() for p in child_module.parameters())
        child_label = f"({child_name}): {child_module.__class__.__name__}"
        if show_params and child_params > 0:
            child_label += f" ({child_params:,} params)"
        child_id = builder.unique_id(f"{prefix}_{_sanitize_id(child_name)}_{index}")

        if depth > 1 and any(child_module.named_children()):
            builder.begin_subgraph(child_id, child_label)
            inner_ids = _render_module_children(
                builder, child_module, child_id, depth - 1, show_params, pipeline, layer_params
            )
            for a, b in zip(inner_ids, inner_ids[1:]):
                builder.add_edge(a, b)
            builder.end_subgraph()
            sequence_ids.append(child_id)
        else:
            builder.add_node(child_id, child_label)
            sequence_ids.append(child_id)
    return sequence_ids


def _render_module_subgraph(
    builder: _MermaidBuilder,
    stage_name: str,
    layer_name: str,
    layer: nn.Module,
    depth: int,
    show_params: bool,
    show_metrics: bool,
    pipeline: Any,
    layer_params: dict,
) -> str:
    """Render a layer as a subgraph or leaf node. Returns the top-level node/subgraph ID.

    Returns:
        Mermaid node or subgraph ID for the rendered layer.
    """
    node_id = _layer_node_id(stage_name, layer_name, builder._counter)

    classes: list[str] = []
    if hasattr(layer, "keys_out"):
        keys_out_list = cast(list[str], getattr(layer, "keys_out", []))
        loss_out = set(keys_out_list) & set(pipeline.loss_keys.keys())
        metric_out = set(keys_out_list) & set(pipeline.metric_keys)
        if loss_out:
            classes.append("loss")
        if metric_out and show_metrics:
            classes.append("metric")

    title = f"{layer_name}: {layer.__class__.__name__}"
    params = layer_params.get((stage_name, layer_name), 0)
    if show_params and params > 0:
        title += f" ({params:,} params)"

    classes_str = " ".join(classes) if classes else None
    children = list(layer.named_children())
    if depth > 0 and children:
        builder.begin_subgraph(node_id, title)
        inner_ids = _render_module_children(
            builder, layer, node_id, depth, show_params, pipeline, layer_params
        )
        for a, b in zip(inner_ids, inner_ids[1:]):
            builder.add_edge(a, b)
        builder.end_subgraph()
        if classes_str:
            builder.apply_class(node_id, classes_str)
    else:
        builder.add_node(node_id, title, node_class=classes_str)

    return node_id


def build_pipeline_mermaid_diagram(
    pipeline: Any,
    depth: int = 2,
    direction: str = "TB",
    show_params: bool = True,
    show_shapes: bool = True,
    show_metrics: bool = True,
) -> str:
    """Build a Mermaid flowchart string representing *pipeline*.

    Returns:
        Complete Mermaid flowchart as a string.
    """
    builder = _MermaidBuilder(direction=direction)
    builder.start()

    builder.add_class_def("loss", "fill:#ffcccc,stroke:#cc0000,stroke-width:2px")
    builder.add_class_def("metric", "fill:#ccccff,stroke:#0000cc,stroke-width:2px")
    builder.add_class_def("input", "fill:#e6f3ff,stroke:#0066cc,stroke-width:2px")
    builder.add_class_def("terminal", "fill:#fff0e6,stroke:#cc6600,stroke-width:2px")

    # Flatten execution-ordered list of all top-level layers
    all_layers: list[tuple[str, str, nn.Module]] = []
    for stage_name, stage_layers in pipeline.stages.items():
        if isinstance(stage_layers, nn.ModuleDict):
            for layer_name, layer in stage_layers.items():
                all_layers.append((stage_name, layer_name, layer))
        elif isinstance(stage_layers, nn.ModuleList):
            for index, layer in enumerate(stage_layers):
                all_layers.append((stage_name, f"{index:02d}", layer))
        elif isinstance(stage_layers, nn.Module):
            all_layers.append((stage_name, stage_name, stage_layers))

    # Compute parameter counts per layer and per stage
    layer_params: dict[tuple[str, str], int] = {}
    stage_params: dict[str, int] = {}
    for stage_name, layer_name, layer in all_layers:
        params = sum(p.numel() for p in layer.parameters())
        layer_params[(stage_name, layer_name)] = params
        stage_params[stage_name] = stage_params.get(stage_name, 0) + params

    # Input node
    input_id = "data"
    input_label = "Input Data"
    if show_shapes and pipeline.input_sizes:
        shapes_str = ", ".join(f"{k}: {tuple(v)}" for k, v in pipeline.input_sizes.items())
        input_label = f"Input Data<br/>{shapes_str}"
    builder.add_node(input_id, input_label, node_class="input")

    # Track module IDs for edge drawing — key by (stage_name, layer_name) to avoid collisions
    layer_node_ids: dict[tuple[str, str], str] = {}

    # Render stages as subgraphs
    for stage_name, stage_layers in pipeline.stages.items():
        stage_title = stage_name
        if show_params and stage_name in stage_params:
            stage_title += f" ({stage_params[stage_name]:,} params)"

        builder.begin_subgraph(builder.unique_id(stage_name), stage_title)

        if isinstance(stage_layers, nn.ModuleDict):
            for layer_name, layer in stage_layers.items():
                node_id = _render_module_subgraph(
                    builder,
                    stage_name,
                    layer_name,
                    layer,
                    depth,
                    show_params,
                    show_metrics,
                    pipeline,
                    layer_params,
                )
                layer_node_ids[(stage_name, layer_name)] = node_id
        elif isinstance(stage_layers, nn.ModuleList):
            for index, layer in enumerate(stage_layers):
                layer_name = f"{index:02d}"
                node_id = _render_module_subgraph(
                    builder,
                    stage_name,
                    layer_name,
                    layer,
                    depth,
                    show_params,
                    show_metrics,
                    pipeline,
                    layer_params,
                )
                layer_node_ids[(stage_name, layer_name)] = node_id
        elif isinstance(stage_layers, nn.Module):
            layer_name = stage_name
            node_id = _render_module_subgraph(
                builder,
                stage_name,
                layer_name,
                stage_layers,
                depth,
                show_params,
                show_metrics,
                pipeline,
                layer_params,
            )
            layer_node_ids[(stage_name, layer_name)] = node_id

        builder.end_subgraph()

    # Cross-module edges: for each consumer, find the most recent producer before it
    for consumer_idx, (consumer_stage, consumer_name, consumer_layer) in enumerate(all_layers):
        consumer_id = layer_node_ids[(consumer_stage, consumer_name)]

        keys_in: list[str] = []
        if hasattr(consumer_layer, "keys_in"):
            raw_keys_in = getattr(consumer_layer, "keys_in")
            if isinstance(raw_keys_in, dict):
                keys_in = cast(list[str], list(raw_keys_in.values()))
            else:
                keys_in = cast(list[str], list(raw_keys_in))

        for key in keys_in:
            # Find the most recent producer of this key before the consumer
            producer = _find_producer_for_consumer(all_layers, consumer_idx, key)
            if producer is None:
                # Input key — edge from data node
                label = key
                if show_shapes:
                    shape = pipeline.input_sizes.get(key)
                    if shape is not None:
                        label += f": {','.join(str(v) for v in shape)}"
                builder.add_edge(input_id, consumer_id, label)
            else:
                producer_stage, producer_name, producer_layer = producer
                producer_id = layer_node_ids[(producer_stage, producer_name)]
                label = key
                if show_shapes:
                    shape = _get_shape(pipeline, key)
                    if shape is not None:
                        label += f": {','.join(str(v) for v in shape)}"
                builder.add_edge(producer_id, consumer_id, label)

    # Orphan loss/metric keys -> terminal nodes
    loss_metric_keys = set(pipeline.loss_keys.keys()) | set(pipeline.metric_keys)
    for key in loss_metric_keys:
        last_producer: tuple[str, str, nn.Module] | None = None
        for i in range(len(all_layers) - 1, -1, -1):
            stage_name, layer_name, layer = all_layers[i]
            if hasattr(layer, "keys_out") and key in cast(
                list[str], getattr(layer, "keys_out", [])
            ):
                last_producer = (stage_name, layer_name, layer)
                break

        if last_producer is not None:
            is_consumed = False
            for _, _, layer in all_layers:
                if hasattr(layer, "keys_in"):
                    raw_keys_in = getattr(layer, "keys_in")
                    if isinstance(raw_keys_in, dict):
                        raw_keys_in = list(raw_keys_in.values())
                    if key in cast(list[str], raw_keys_in):
                        is_consumed = True
                        break

            if not is_consumed:
                producer_stage, producer_name, producer_layer = last_producer
                producer_id = layer_node_ids[(producer_stage, producer_name)]
                terminal_id = builder.unique_id(f"terminal_{key}")
                terminal_label = f"{key} (terminal)"
                builder.add_node(terminal_id, terminal_label, node_class="terminal")

                dims = tuple(pipeline.input_sizes.get(key, ()))
                dim_label = ",".join(str(v) for v in dims) if dims else "?"
                builder.add_edge(producer_id, terminal_id, f"{key}: {dim_label}")

    return builder.finish()


def export_mermaid_diagram(pipeline: Any, path: Path | str, **kwargs: Any) -> None:
    """Render a Mermaid diagram for *pipeline* and write it to *path* as Markdown."""
    diagram = build_pipeline_mermaid_diagram(pipeline, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# Pipeline Diagram\n\n```mermaid\n{diagram}\n```\n"
    path.write_text(content, encoding="utf-8")
