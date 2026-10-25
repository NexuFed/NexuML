# nexuml.data.loaders

*module*

Loader backends for NexuML data loading.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/loaders/__init__.py#L1-L1)

## nexuml.data.loaders.LoaderBackend

*class*

```python
class LoaderBackend(Protocol)
```

Bases: `Protocol`

Re-exported from: `nexuml.data.loaders.registry`

Runtime protocol implemented by loader backends.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/loaders/registry.py#L13-L24)

### nexuml.data.loaders.LoaderBackend.create_loader

*method*

```python
def create_loader(module: Any, dataset: Any, *, split: str, shuffle: bool = False, sampler: WeightedRandomSampler | None = None) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/loaders/registry.py#L16-L24)

## nexuml.data.loaders.get_loader_backend

*function*

```python
def get_loader_backend(definition: LoaderBackendDefinition) -> LoaderBackend
```

Re-exported from: `nexuml.data.loaders.registry`

Materialize a loader backend definition.

**Returns**

- (`LoaderBackend`) — Runtime loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/loaders/registry.py#L27-L33)

## nexuml.data.loaders.list_loader_backends

*function*

```python
def list_loader_backends() -> list[str]
```

Re-exported from: `nexuml.data.loaders.registry`

List registered loader backend identities.

**Returns**

- (`list[str]`) — Stable loader backend names.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/loaders/registry.py#L36-L42)

**Modules**

- `nexuml.data.loaders.dali_backend`
- `nexuml.data.loaders.dali_multimodal`
- `nexuml.data.loaders.dali_numpy`
- `nexuml.data.loaders.dali_numpy_external`
- `nexuml.data.loaders.dali_wav`
- `nexuml.data.loaders.definitions`
- `nexuml.data.loaders.registry`
- `nexuml.data.loaders.tensor_shards_backend`
- `nexuml.data.loaders.torch_backend`
