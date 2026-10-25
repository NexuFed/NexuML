# nexuml.execution.schemas

*module*

Safe, sourced resource snapshots; unknown is distinct from observed zero.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L1-L1)

## nexuml.execution.schemas.NativeReference

*class* · *pydantic model*

```python
class NativeReference(BaseModel)
```

Bases: `BaseModel`

Exact native identity for safe observation and cancellation.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L50-L61)

### nexuml.execution.schemas.NativeReference.api_version

*attribute* · *pydantic field*

```python
api_version: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L58-L58)

### nexuml.execution.schemas.NativeReference.backend

*attribute* · *pydantic field*

```python
backend: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L54-L54)

### nexuml.execution.schemas.NativeReference.context

*attribute* · *pydantic field*

```python
context: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L56-L56)

### nexuml.execution.schemas.NativeReference.kind

*attribute* · *pydantic field*

```python
kind: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L59-L59)

### nexuml.execution.schemas.NativeReference.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L53-L53)

### nexuml.execution.schemas.NativeReference.name

*attribute* · *pydantic field*

```python
name: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L60-L60)

### nexuml.execution.schemas.NativeReference.namespace

*attribute* · *pydantic field*

```python
namespace: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L57-L57)

### nexuml.execution.schemas.NativeReference.uid

*attribute* · *pydantic field*

```python
uid: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L61-L61)

### nexuml.execution.schemas.NativeReference.version

*attribute* · *pydantic field*

```python
version: str = '1'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L55-L55)

## nexuml.execution.schemas.NodeResources

*class* · *pydantic model*

```python
class NodeResources(BaseModel)
```

Bases: `BaseModel`

Observable scheduling quantities for one node, without credentials.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L8-L16)

### nexuml.execution.schemas.NodeResources.accelerator_type

*attribute* · *pydantic field*

```python
accelerator_type: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L16-L16)

### nexuml.execution.schemas.NodeResources.allocatable

*attribute* · *pydantic field*

```python
allocatable: dict[str, float]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L13-L13)

### nexuml.execution.schemas.NodeResources.eligible

*attribute* · *pydantic field*

```python
eligible: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L15-L15)

### nexuml.execution.schemas.NodeResources.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L11-L11)

### nexuml.execution.schemas.NodeResources.name

*attribute* · *pydantic field*

```python
name: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L12-L12)

### nexuml.execution.schemas.NodeResources.unallocated

*attribute* · *pydantic field*

```python
unallocated: dict[str, float] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L14-L14)

## nexuml.execution.schemas.ResourceSnapshot

*class* · *pydantic model*

```python
class ResourceSnapshot(BaseModel)
```

Bases: `BaseModel`

Read-only advisory capacity in an explicit target scope.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L19-L47)

### nexuml.execution.schemas.ResourceSnapshot.allocatable

*attribute* · *pydantic field*

```python
allocatable: dict[str, float] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L36-L36)

### nexuml.execution.schemas.ResourceSnapshot.api_supported

*attribute* · *pydantic field*

```python
api_supported: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L31-L31)

### nexuml.execution.schemas.ResourceSnapshot.backend

*attribute* · *pydantic field*

```python
backend: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L23-L23)

### nexuml.execution.schemas.ResourceSnapshot.complete

*attribute* · *pydantic field*

```python
complete: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L29-L29)

### nexuml.execution.schemas.ResourceSnapshot.context

*attribute* · *pydantic field*

```python
context: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L26-L26)

### nexuml.execution.schemas.ResourceSnapshot.controller_healthy

*attribute* · *pydantic field*

```python
controller_healthy: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L33-L33)

### nexuml.execution.schemas.ResourceSnapshot.diagnostics

*attribute* · *pydantic field*

```python
diagnostics: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L47-L47)

### nexuml.execution.schemas.ResourceSnapshot.host_total

*attribute* · *pydantic field*

```python
host_total: dict[str, float] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L37-L37)

### nexuml.execution.schemas.ResourceSnapshot.matching_nodes

*attribute* · *pydantic field*

```python
matching_nodes: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L35-L35)

### nexuml.execution.schemas.ResourceSnapshot.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L22-L22)

### nexuml.execution.schemas.ResourceSnapshot.namespace

*attribute* · *pydantic field*

```python
namespace: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L27-L27)

### nexuml.execution.schemas.ResourceSnapshot.namespaces

*attribute* · *pydantic field*

```python
namespaces: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L43-L43)

### nexuml.execution.schemas.ResourceSnapshot.nodes

*attribute* · *pydantic field*

```python
nodes: list[NodeResources]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L41-L41)

### nexuml.execution.schemas.ResourceSnapshot.observed_at

*attribute* · *pydantic field*

```python
observed_at: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L28-L28)

### nexuml.execution.schemas.ResourceSnapshot.options

*attribute* · *pydantic field*

```python
options: dict[str, list[str]]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L46-L46)

### nexuml.execution.schemas.ResourceSnapshot.quota_remaining

*attribute* · *pydantic field*

```python
quota_remaining: dict[str, float] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L40-L40)

### nexuml.execution.schemas.ResourceSnapshot.ray_clusters

*attribute* · *pydantic field*

```python
ray_clusters: list[dict[str, str]]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L44-L44)

### nexuml.execution.schemas.ResourceSnapshot.reachable

*attribute* · *pydantic field*

```python
reachable: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L30-L30)

### nexuml.execution.schemas.ResourceSnapshot.roles

*attribute* · *pydantic field*

```python
roles: list[dict]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L45-L45)

### nexuml.execution.schemas.ResourceSnapshot.source

*attribute* · *pydantic field*

```python
source: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L24-L24)

### nexuml.execution.schemas.ResourceSnapshot.submission_allowed

*attribute* · *pydantic field*

```python
submission_allowed: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L32-L32)

### nexuml.execution.schemas.ResourceSnapshot.target

*attribute* · *pydantic field*

```python
target: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L25-L25)

### nexuml.execution.schemas.ResourceSnapshot.targets

*attribute* · *pydantic field*

```python
targets: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L42-L42)

### nexuml.execution.schemas.ResourceSnapshot.unallocated

*attribute* · *pydantic field*

```python
unallocated: dict[str, float] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L39-L39)

### nexuml.execution.schemas.ResourceSnapshot.units

*attribute* · *pydantic field*

```python
units: dict[str, str]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L38-L38)

### nexuml.execution.schemas.ResourceSnapshot.visible_nodes

*attribute* · *pydantic field*

```python
visible_nodes: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/schemas.py#L34-L34)
