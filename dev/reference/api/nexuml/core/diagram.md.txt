# nexuml.core.diagram

*module*

Mermaid diagram export for NexuML pipelines.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/diagram.py#L1-L1)

## nexuml.core.diagram.build_pipeline_mermaid_diagram

*function*

```python
def build_pipeline_mermaid_diagram(pipeline: Any, depth: int = 2, direction: str = 'TB', show_params: bool = True, show_shapes: bool = True, show_metrics: bool = True) -> str
```

Build a Mermaid flowchart string representing *pipeline*.

**Returns**

- (`str`) — Complete Mermaid flowchart as a string.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/diagram.py#L241-L414)

## nexuml.core.diagram.export_mermaid_diagram

*function*

```python
def export_mermaid_diagram(pipeline: Any, path: Path | str, **kwargs: Any) -> None
```

Render a Mermaid diagram for *pipeline* and write it to *path* as Markdown.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/diagram.py#L417-L423)
