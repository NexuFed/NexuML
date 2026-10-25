# nexuml.data.loaders.registry

*module*

Loader backend protocol and typed materialization helpers.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/registry.py#L1-L1)

## nexuml.data.loaders.registry.LoaderBackend

*class*

```python
class LoaderBackend(Protocol)
```

Bases: `Protocol`

Runtime protocol implemented by loader backends.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/registry.py#L13-L24)

### nexuml.data.loaders.registry.LoaderBackend.create_loader

*method*

```python
def create_loader(module: Any, dataset: Any, *, split: str, shuffle: bool = False, sampler: WeightedRandomSampler | None = None) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/registry.py#L16-L24)

## nexuml.data.loaders.registry.get_loader_backend

*function*

```python
def get_loader_backend(definition: LoaderBackendDefinition) -> LoaderBackend
```

Materialize a loader backend definition.

**Returns**

- (`LoaderBackend`) — Runtime loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/registry.py#L27-L33)

## nexuml.data.loaders.registry.list_loader_backends

*function*

```python
def list_loader_backends() -> list[str]
```

List registered loader backend identities.

**Returns**

- (`list[str]`) — Stable loader backend names.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/registry.py#L36-L42)
