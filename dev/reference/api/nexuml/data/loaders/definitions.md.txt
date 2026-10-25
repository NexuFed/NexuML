# nexuml.data.loaders.definitions

*module*

Typed definitions for built-in loader backends.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L1-L1)

## nexuml.data.loaders.definitions.DaliLoader

*class* · *pydantic model*

```python
class DaliLoader(LoaderBackendDefinition)
```

Bases: `nexuml.core.components.LoaderBackendDefinition`

Use the NVIDIA DALI data loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L19-L26)

### nexuml.data.loaders.definitions.DaliLoader.kind

*attribute* · *pydantic field*

```python
kind = 'loader_backend'
```

Inherited from: `nexuml.core.components.LoaderBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L159-L159)

### nexuml.data.loaders.definitions.DaliLoader.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.data.loaders.definitions.DaliLoader.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.data.loaders.definitions.DaliLoader.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.data.loaders.definitions.DaliLoader.build

*method*

```python
def build()
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L23-L26)

## nexuml.data.loaders.definitions.TensorShardsLoader

*class* · *pydantic model*

```python
class TensorShardsLoader(LoaderBackendDefinition)
```

Bases: `nexuml.core.components.LoaderBackendDefinition`

Configure the windowed tensor-shard loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L29-L45)

### nexuml.data.loaders.definitions.TensorShardsLoader.drop_last

*attribute* · *pydantic field*

```python
drop_last: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L39-L39)

### nexuml.data.loaders.definitions.TensorShardsLoader.pin_memory

*attribute* · *pydantic field*

```python
pin_memory: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L38-L38)

### nexuml.data.loaders.definitions.TensorShardsLoader.prefetch_windows

*attribute* · *pydantic field*

```python
prefetch_windows: int = 2
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L34-L34)

### nexuml.data.loaders.definitions.TensorShardsLoader.prefetch_workers

*attribute* · *pydantic field*

```python
prefetch_workers: int = 2
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L35-L35)

### nexuml.data.loaders.definitions.TensorShardsLoader.seed

*attribute* · *pydantic field*

```python
seed: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L40-L40)

### nexuml.data.loaders.definitions.TensorShardsLoader.shards_per_window

*attribute* · *pydantic field*

```python
shards_per_window: int = 6
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L33-L33)

### nexuml.data.loaders.definitions.TensorShardsLoader.shuffle_samples

*attribute* · *pydantic field*

```python
shuffle_samples: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L37-L37)

### nexuml.data.loaders.definitions.TensorShardsLoader.shuffle_shards

*attribute* · *pydantic field*

```python
shuffle_shards: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L36-L36)

### nexuml.data.loaders.definitions.TensorShardsLoader.kind

*attribute* · *pydantic field*

```python
kind = 'loader_backend'
```

Inherited from: `nexuml.core.components.LoaderBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L159-L159)

### nexuml.data.loaders.definitions.TensorShardsLoader.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.data.loaders.definitions.TensorShardsLoader.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.data.loaders.definitions.TensorShardsLoader.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.data.loaders.definitions.TensorShardsLoader.build

*method*

```python
def build()
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L42-L45)

## nexuml.data.loaders.definitions.TorchLoader

*class* · *pydantic model*

```python
class TorchLoader(LoaderBackendDefinition)
```

Bases: `nexuml.core.components.LoaderBackendDefinition`

Use the standard PyTorch data loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L9-L16)

### nexuml.data.loaders.definitions.TorchLoader.kind

*attribute* · *pydantic field*

```python
kind = 'loader_backend'
```

Inherited from: `nexuml.core.components.LoaderBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L159-L159)

### nexuml.data.loaders.definitions.TorchLoader.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.data.loaders.definitions.TorchLoader.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.data.loaders.definitions.TorchLoader.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.data.loaders.definitions.TorchLoader.build

*method*

```python
def build()
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/loaders/definitions.py#L13-L16)
