# nexuml.core.config

*module*

Configuration serialization for full scenario documents.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L1-L1)

## nexuml.core.config.ResolvedConfig

*class* · *pydantic model*

```python
class ResolvedConfig(ScenarioSpec)
```

Bases: `nexuml.core.types.ScenarioSpec`

Fully resolved, portable scenario document.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L14-L49)

### nexuml.core.config.ResolvedConfig.name

*attribute* · *pydantic field*

```python
name: str = ''
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L17-L17)

### nexuml.core.config.ResolvedConfig.pipeline

*attribute* · *pydantic field*

```python
pipeline: PipelineSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L18-L18)

### nexuml.core.config.ResolvedConfig.callbacks

*attribute* · *pydantic field*

```python
callbacks: list[CallbackSpec]
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L505-L505)

### nexuml.core.config.ResolvedConfig.checkpoint

*attribute* · *pydantic field*

```python
checkpoint: CheckpointLoadSpec | None = None
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L507-L507)

### nexuml.core.config.ResolvedConfig.data

*attribute* · *pydantic field*

```python
data: DataSpec
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L502-L502)

### nexuml.core.config.ResolvedConfig.evaluation

*attribute* · *pydantic field*

```python
evaluation: EvaluationSpec
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L503-L503)

### nexuml.core.config.ResolvedConfig.execution

*attribute* · *pydantic field*

```python
execution: ExecutionBackendDefinition
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L509-L509)

### nexuml.core.config.ResolvedConfig.exports

*attribute* · *pydantic field*

```python
exports: list[ExportSpec]
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L508-L508)

### nexuml.core.config.ResolvedConfig.logging

*attribute* · *pydantic field*

```python
logging: LoggingSpec | None = None
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L504-L504)

### nexuml.core.config.ResolvedConfig.training

*attribute* · *pydantic field*

```python
training: TrainingSpec
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L501-L501)

### nexuml.core.config.ResolvedConfig.tuning

*attribute* · *pydantic field*

```python
tuning: TuningSpec | None = None
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L506-L506)

### nexuml.core.config.ResolvedConfig.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L46-L46)

### nexuml.core.config.ResolvedConfig.from_scenario

*method* · *classmethod*

```python
def from_scenario(scenario: ScenarioSpec) -> 'ResolvedConfig'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L44-L46)

### nexuml.core.config.ResolvedConfig.from_yaml

*method* · *classmethod*

```python
def from_yaml(yaml_str: str) -> 'ResolvedConfig'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L20-L24)

### nexuml.core.config.ResolvedConfig.load

*method* · *classmethod*

```python
def load(path: Path) -> 'ResolvedConfig'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L40-L42)

### nexuml.core.config.ResolvedConfig.save

*method*

```python
def save(path: Path) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L36-L38)

### nexuml.core.config.ResolvedConfig.to_scenario

*method*

```python
def to_scenario() -> ScenarioSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L48-L49)

### nexuml.core.config.ResolvedConfig.to_yaml

*method*

```python
def to_yaml() -> str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/config.py#L26-L34)

### nexuml.core.config.ResolvedConfig.restore_execution

*method* · *classmethod*

```python
def restore_execution(value: Any) -> ExecutionBackendDefinition
```

Inherited from: `nexuml.core.types.ScenarioSpec`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L511-L524)

### nexuml.core.config.ResolvedConfig.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L51-L53)

### nexuml.core.config.ResolvedConfig.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L58-L59)

### nexuml.core.config.ResolvedConfig.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/types.py#L48-L49)
