# Pipeline diagrams

NexuML can write a Mermaid representation of a compiled pipeline when diagram output is enabled in `LoggingSpec`.

## Configure diagram output

```python
from nexuml.core.types import DiagramSpec, LoggingSpec

logging = LoggingSpec(
    diagram=DiagramSpec(
        enabled=True,
        depth=2,
        direction="TB",
        show_params=True,
        show_shapes=True,
        show_metrics=True,
        output_dir=".experiments/diagrams",
    )
)
```

## Generate during build

```bash
nexuml resolve my-scenario
nexuml build configs/my-scenario.yaml
```

`build` compiles the pipeline and, when diagram output is enabled, writes `<output_dir>/<scenario-name>.md`. The command itself reports the compiled pipeline summary; it does not rely on printing the full Mermaid source to standard output.

## Generate during training

`nexuml train` performs the same configured diagram export before the training session begins. You do not need a separate `build` invocation just to create the diagram.

Diagram generation is diagnostic: a diagram-export failure is reported as a warning rather than turning a valid pipeline compile into a failure.

## Rendering

Material for MkDocs renders Mermaid code fences directly. The exported `.md` can also be rendered by any Mermaid-compatible tool.

## See also

- [Architecture](architecture.md)
- [TensorDict data flow](tensordict.md)
- [`nexuml.core.diagram`](../reference/api/nexuml/core/diagram.md)
