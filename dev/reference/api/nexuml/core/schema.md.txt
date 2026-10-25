# nexuml.core.schema

*module*

Schema utilities for resolved config YAML I/O.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/schema.py#L1-L1)

## nexuml.core.schema.dump_resolved_yaml

*function*

```python
def dump_resolved_yaml(config: ResolvedConfig, path: Path) -> None
```

Serialize a resolved config to YAML file.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/schema.py#L10-L12)

## nexuml.core.schema.load_resolved_yaml

*function*

```python
def load_resolved_yaml(path: Path) -> ResolvedConfig
```

Load a resolved config from YAML file.

**Returns**

- (`ResolvedConfig`) — Deserialized ``ResolvedConfig`` instance.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/schema.py#L15-L21)
