# nexuml.api.operation_routes

*module*

Authenticated existing-operation controls and disk-backed WebSocket replay.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L1-L1)

## nexuml.api.operation_routes.BuildInput

*class* · *pydantic model*

```python
class BuildInput(ConfigInput)
```

Bases: `nexuml.api.config_routes.ConfigInput`

Explicit executable check with a bounded deadline.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L21-L24)

### nexuml.api.operation_routes.BuildInput.timeout

*attribute* · *pydantic field*

```python
timeout: float = 120
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L24-L24)

### nexuml.api.operation_routes.BuildInput.data

*attribute* · *pydantic field*

```python
data: dict[str, Any] | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L20-L20)

### nexuml.api.operation_routes.BuildInput.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L19-L19)

### nexuml.api.operation_routes.BuildInput.stage_order

*attribute* · *pydantic field*

```python
stage_order: list[str] | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L22-L22)

### nexuml.api.operation_routes.BuildInput.yaml

*attribute* · *pydantic field*

```python
yaml: str | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L21-L21)

## nexuml.api.operation_routes.DiscoveryInput

*class* · *pydantic model*

```python
class DiscoveryInput(ConfigInput)
```

Bases: `nexuml.api.config_routes.ConfigInput`

Explicit selected-scope discovery outside the resource-consuming slot.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L34-L37)

### nexuml.api.operation_routes.DiscoveryInput.timeout

*attribute* · *pydantic field*

```python
timeout: float = 10
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L37-L37)

### nexuml.api.operation_routes.DiscoveryInput.data

*attribute* · *pydantic field*

```python
data: dict[str, Any] | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L20-L20)

### nexuml.api.operation_routes.DiscoveryInput.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L19-L19)

### nexuml.api.operation_routes.DiscoveryInput.stage_order

*attribute* · *pydantic field*

```python
stage_order: list[str] | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L22-L22)

### nexuml.api.operation_routes.DiscoveryInput.yaml

*attribute* · *pydantic field*

```python
yaml: str | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L21-L21)

## nexuml.api.operation_routes.ExportInput

*class* · *pydantic model*

```python
class ExportInput(BaseModel)
```

Bases: `BaseModel`

Export from a completed operation and actual checkpoint, not placeholder weights.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L40-L46)

### nexuml.api.operation_routes.ExportInput.kind

*attribute* · *pydantic field*

```python
kind: Literal['train_package', 'safetensors', 'onnx'] = 'train_package'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L45-L45)

### nexuml.api.operation_routes.ExportInput.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L43-L43)

### nexuml.api.operation_routes.ExportInput.output

*attribute* · *pydantic field*

```python
output: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L46-L46)

### nexuml.api.operation_routes.ExportInput.source_id

*attribute* · *pydantic field*

```python
source_id: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L44-L44)

## nexuml.api.operation_routes.TrainInput

*class* · *pydantic model*

```python
class TrainInput(ConfigInput)
```

Bases: `nexuml.api.config_routes.ConfigInput`

Freeze a scenario, or use existing Trainer checkpoint resume semantics.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L27-L31)

### nexuml.api.operation_routes.TrainInput.template_revision

*attribute* · *pydantic field*

```python
template_revision: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L31-L31)

### nexuml.api.operation_routes.TrainInput.trainer_checkpoint

*attribute* · *pydantic field*

```python
trainer_checkpoint: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L30-L30)

### nexuml.api.operation_routes.TrainInput.data

*attribute* · *pydantic field*

```python
data: dict[str, Any] | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L20-L20)

### nexuml.api.operation_routes.TrainInput.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L19-L19)

### nexuml.api.operation_routes.TrainInput.stage_order

*attribute* · *pydantic field*

```python
stage_order: list[str] | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L22-L22)

### nexuml.api.operation_routes.TrainInput.yaml

*attribute* · *pydantic field*

```python
yaml: str | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/config_routes.py#L21-L21)

## nexuml.api.operation_routes.operation_router

*function*

```python
def operation_router(operations: Operations) -> APIRouter
```

Expose lightweight observation without a training loop or event queue.

**Returns**

- (`APIRouter`) — REST/WebSocket router over actual invocation ownership and persisted observations.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/operation_routes.py#L49-L220)
