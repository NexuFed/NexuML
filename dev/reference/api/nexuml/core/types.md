# nexuml.core.types

*module*

Core type definitions for NexuML pipeline specifications.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L1-L1)

## nexuml.core.types.BatchSizeSpec

*attribute* · *module attribute*

```python
BatchSizeSpec = int | AutoBatchSizeSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L216-L216)

## nexuml.core.types.AutoBatchSizeSpec

*class* · *pydantic model*

```python
class AutoBatchSizeSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Structured config for runtime automatic training batch-size resolution.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L193-L213)

### nexuml.core.types.AutoBatchSizeSpec.candidates

*attribute* · *pydantic field*

```python
candidates: Literal['power_of_two'] = 'power_of_two'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L199-L199)

### nexuml.core.types.AutoBatchSizeSpec.margin

*attribute* · *pydantic field*

```python
margin: float = 0.8
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L201-L201)

### nexuml.core.types.AutoBatchSizeSpec.max

*attribute* · *pydantic field*

```python
max: int = 128
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L198-L198)

### nexuml.core.types.AutoBatchSizeSpec.min

*attribute* · *pydantic field*

```python
min: int = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L197-L197)

### nexuml.core.types.AutoBatchSizeSpec.mode

*attribute* · *pydantic field*

```python
mode: Literal['auto'] = 'auto'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L196-L196)

### nexuml.core.types.AutoBatchSizeSpec.safety

*attribute* · *pydantic field*

```python
safety: Literal['largest', 'previous_power_of_two', 'margin'] = 'previous_power_of_two'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L200-L200)

### nexuml.core.types.AutoBatchSizeSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.AutoBatchSizeSpec.validate_bounds

*method* · *pydantic validator*

```python
def validate_bounds() -> 'AutoBatchSizeSpec'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L203-L213)

### nexuml.core.types.AutoBatchSizeSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.AutoBatchSizeSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.AutoBatchSizeSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.AxisKeySpec

*class* · *pydantic model*

```python
class AxisKeySpec(BaseModel)
```

Bases: `BaseModel`

Typed axis-key declaration with source provenance.

``source`` indicates where the axis value is resolved from at runtime:

- ``"x"``: from the transformed pipeline output TensorDict (e.g. a
  reconstructed feature or anomaly score produced by a pipeline layer).
- ``"y"``: from the label TensorDict — most common for grouping axes such
  as ``machine``, ``section``, or ``target``.
- ``"metadata"``: from the dataset metadata DataFrame attached to the batch
  by ``_attach_eval_metadata``.  Use this source for non-tensor columns that
  are not part of the label TensorDict but are needed for grouped evaluation
  (e.g. ``basename``, ``domain``).  Only available during eval/test phases.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L24-L40)

### nexuml.core.types.AxisKeySpec.key

*attribute* · *pydantic field*

```python
key: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L39-L39)

### nexuml.core.types.AxisKeySpec.source

*attribute* · *pydantic field*

```python
source: Literal['x', 'y', 'metadata'] = 'y'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L40-L40)

## nexuml.core.types.CallbackSpec

*class* · *pydantic model*

```python
class CallbackSpec(FactorySpec)
```

Bases: `nexuml.core.types.FactorySpec`

Specification for a training callback.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L462-L463)

### nexuml.core.types.CallbackSpec.args

*attribute* · *pydantic field*

```python
args: list[object]
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L74-L74)

### nexuml.core.types.CallbackSpec.factory

*attribute* · *pydantic field*

```python
factory: str
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L73-L73)

### nexuml.core.types.CallbackSpec.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L75-L75)

### nexuml.core.types.CallbackSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.CallbackSpec.build

*method*

```python
def build(*runtime_args: object, **runtime_kwargs: object) -> object
```

Inherited from: `nexuml.core.types.FactorySpec`

Construct the configured object with any runtime-supplied arguments.

**Returns**

- (`object`) — Constructed runtime object.

**Raises**

- `TypeError` — If construction fails.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L110-L125)

### nexuml.core.types.CallbackSpec.from_factory

*method* · *classmethod*

```python
def from_factory(factory: object, *args: object, **kwargs: object) -> Self
```

Inherited from: `nexuml.core.types.FactorySpec`

Capture an importable factory and portable constructor arguments.

**Returns**

- (`Self`) — Validated factory specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L93-L100)

### nexuml.core.types.CallbackSpec.resolve

*method*

```python
def resolve() -> Callable[..., object]
```

Inherited from: `nexuml.core.types.FactorySpec`

Resolve the configured import target.

**Returns**

- (`Callable[..., object]`) — Imported callable.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L102-L108)

### nexuml.core.types.CallbackSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.CallbackSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.CallbackSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.CheckpointLoadSpec

*class* · *pydantic model*

```python
class CheckpointLoadSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Selective weight loading policy for resume/fine-tune flows.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L476-L484)

### nexuml.core.types.CheckpointLoadSpec.allow_missing

*attribute* · *pydantic field*

```python
allow_missing: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L482-L482)

### nexuml.core.types.CheckpointLoadSpec.allow_shape_mismatch

*attribute* · *pydantic field*

```python
allow_shape_mismatch: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L483-L483)

### nexuml.core.types.CheckpointLoadSpec.exclude

*attribute* · *pydantic field*

```python
exclude: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L481-L481)

### nexuml.core.types.CheckpointLoadSpec.freeze_loaded

*attribute* · *pydantic field*

```python
freeze_loaded: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L484-L484)

### nexuml.core.types.CheckpointLoadSpec.include

*attribute* · *pydantic field*

```python
include: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L480-L480)

### nexuml.core.types.CheckpointLoadSpec.source

*attribute* · *pydantic field*

```python
source: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L479-L479)

### nexuml.core.types.CheckpointLoadSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.CheckpointLoadSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.CheckpointLoadSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.CheckpointLoadSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.DVCLiveSpec

*class* · *pydantic model*

```python
class DVCLiveSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

DVCLive logger configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L417-L420)

### nexuml.core.types.DVCLiveSpec.dir

*attribute* · *pydantic field*

```python
dir: str = '.experiments/dvclive'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L420-L420)

### nexuml.core.types.DVCLiveSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.DVCLiveSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.DVCLiveSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.DVCLiveSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.DataSpec

*class* · *pydantic model*

```python
class DataSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for data configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L327-L349)

### nexuml.core.types.DataSpec.datasets

*attribute* · *pydantic field*

```python
datasets: list[DatasetSpec]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L335-L335)

### nexuml.core.types.DataSpec.feature_key

*attribute* · *pydantic field*

```python
feature_key: str = 'features'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L340-L340)

### nexuml.core.types.DataSpec.input_shapes

*attribute* · *pydantic field*

```python
input_shapes: dict[str, list[int]]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L337-L337)

### nexuml.core.types.DataSpec.loader

*attribute* · *pydantic field*

```python
loader: LoaderSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L336-L336)

### nexuml.core.types.DataSpec.merge_labels

*attribute* · *pydantic field*

```python
merge_labels: dict[str, dict] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L339-L339)

### nexuml.core.types.DataSpec.num_classes

*attribute* · *pydantic field*

```python
num_classes: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L338-L338)

### nexuml.core.types.DataSpec.preprocessing

*attribute* · *pydantic field*

```python
preprocessing: PreprocessingSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L342-L342)

### nexuml.core.types.DataSpec.skip_pipeline_stages

*attribute* · *pydantic field*

```python
skip_pipeline_stages: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L341-L341)

### nexuml.core.types.DataSpec.source

*attribute* · *pydantic field*

```python
source: DataSourceDefinition | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L330-L330)

### nexuml.core.types.DataSpec.targets

*attribute* · *pydantic field*

```python
targets: list[TargetSpec]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L331-L331)

### nexuml.core.types.DataSpec.test_split

*attribute* · *pydantic field*

```python
test_split: float = 0.15
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L334-L334)

### nexuml.core.types.DataSpec.train_split

*attribute* · *pydantic field*

```python
train_split: float = 0.7
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L332-L332)

### nexuml.core.types.DataSpec.val_split

*attribute* · *pydantic field*

```python
val_split: float = 0.15
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L333-L333)

### nexuml.core.types.DataSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.DataSpec.validate_splits

*method* · *pydantic validator*

```python
def validate_splits() -> 'DataSpec'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L344-L349)

### nexuml.core.types.DataSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.DataSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.DataSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.DatasetSpec

*class* · *pydantic model*

```python
class DatasetSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for a single dataset in the pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L253-L259)

### nexuml.core.types.DatasetSpec.max_samples

*attribute* · *pydantic field*

```python
max_samples: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L258-L258)

### nexuml.core.types.DatasetSpec.modality

*attribute* · *pydantic field*

```python
modality: str = 'audio'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L257-L257)

### nexuml.core.types.DatasetSpec.source

*attribute* · *pydantic field*

```python
source: DataSourceDefinition
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L256-L256)

### nexuml.core.types.DatasetSpec.split_type

*attribute* · *pydantic field*

```python
split_type: str = 'fit'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L259-L259)

### nexuml.core.types.DatasetSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.DatasetSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.DatasetSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.DatasetSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.DiagramSpec

*class* · *pydantic model*

```python
class DiagramSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Mermaid diagram export configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L437-L446)

### nexuml.core.types.DiagramSpec.depth

*attribute* · *pydantic field*

```python
depth: int = 2
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L441-L441)

### nexuml.core.types.DiagramSpec.direction

*attribute* · *pydantic field*

```python
direction: str = 'TB'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L442-L442)

### nexuml.core.types.DiagramSpec.enabled

*attribute* · *pydantic field*

```python
enabled: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L440-L440)

### nexuml.core.types.DiagramSpec.output_dir

*attribute* · *pydantic field*

```python
output_dir: str = '.experiments/diagrams'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L446-L446)

### nexuml.core.types.DiagramSpec.show_metrics

*attribute* · *pydantic field*

```python
show_metrics: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L445-L445)

### nexuml.core.types.DiagramSpec.show_params

*attribute* · *pydantic field*

```python
show_params: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L443-L443)

### nexuml.core.types.DiagramSpec.show_shapes

*attribute* · *pydantic field*

```python
show_shapes: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L444-L444)

### nexuml.core.types.DiagramSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.DiagramSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.DiagramSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.DiagramSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.DistanceEstimatorSpec

*class* · *pydantic model*

```python
class DistanceEstimatorSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for a distance estimator used in anomaly detection.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L352-L376)

### nexuml.core.types.DistanceEstimatorSpec.fallback_policy

*attribute* · *pydantic field*

```python
fallback_policy: Literal['error', 'parent', 'global', 'parent_or_global'] = 'error'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L357-L357)

### nexuml.core.types.DistanceEstimatorSpec.group_label_keys

*attribute* · *pydantic field*

```python
group_label_keys: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L355-L355)

### nexuml.core.types.DistanceEstimatorSpec.max_samples

*attribute* · *pydantic field*

```python
max_samples: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L360-L360)

### nexuml.core.types.DistanceEstimatorSpec.missing_label_policy

*attribute* · *pydantic field*

```python
missing_label_policy: Literal['error', 'skip', 'unknown'] = 'error'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L356-L356)

### nexuml.core.types.DistanceEstimatorSpec.retain_storage

*attribute* · *pydantic field*

```python
retain_storage: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L361-L361)

### nexuml.core.types.DistanceEstimatorSpec.storage_backend

*attribute* · *pydantic field*

```python
storage_backend: Literal['ram', 'memmap'] = 'memmap'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L358-L358)

### nexuml.core.types.DistanceEstimatorSpec.storage_path

*attribute* · *pydantic field*

```python
storage_path: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L359-L359)

### nexuml.core.types.DistanceEstimatorSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.DistanceEstimatorSpec.create_feature_store

*method*

```python
def create_feature_store() -> FeatureStore
```

Create the feature store configured for estimator fitting.

**Returns**

- (`FeatureStore`) — Configured feature store runtime.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L363-L376)

### nexuml.core.types.DistanceEstimatorSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.DistanceEstimatorSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.DistanceEstimatorSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.EvalAlgorithmSpec

*class* · *pydantic model*

```python
class EvalAlgorithmSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for a post-training evaluation algorithm.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L379-L400)

### nexuml.core.types.EvalAlgorithmSpec.algorithm

*attribute* · *pydantic field*

```python
algorithm: EvalAlgorithmDefinition
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L382-L382)

### nexuml.core.types.EvalAlgorithmSpec.axis_keys

*attribute* · *pydantic field*

```python
axis_keys: list[AxisKeySpec | str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L385-L385)

### nexuml.core.types.EvalAlgorithmSpec.enabled

*attribute* · *pydantic field*

```python
enabled: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L384-L384)

### nexuml.core.types.EvalAlgorithmSpec.feature_key

*attribute* · *pydantic field*

```python
feature_key: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L386-L386)

### nexuml.core.types.EvalAlgorithmSpec.label_key

*attribute* · *pydantic field*

```python
label_key: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L387-L387)

### nexuml.core.types.EvalAlgorithmSpec.name

*attribute* · *pydantic field*

```python
name: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L383-L383)

### nexuml.core.types.EvalAlgorithmSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.EvalAlgorithmSpec.contract_summary

*method*

```python
def contract_summary() -> dict[str, Any]
```

Return a summary of the key contract for diagnostics.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L389-L400)

### nexuml.core.types.EvalAlgorithmSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.EvalAlgorithmSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.EvalAlgorithmSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.EvaluationSpec

*class* · *pydantic model*

```python
class EvaluationSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for evaluation configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L403-L408)

### nexuml.core.types.EvaluationSpec.algorithms

*attribute* · *pydantic field*

```python
algorithms: list[EvalAlgorithmSpec]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L407-L407)

### nexuml.core.types.EvaluationSpec.metrics

*attribute* · *pydantic field*

```python
metrics: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L406-L406)

### nexuml.core.types.EvaluationSpec.test_result_metrics

*attribute* · *pydantic field*

```python
test_result_metrics: list[str] | Literal['all', 'none'] = 'none'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L408-L408)

### nexuml.core.types.EvaluationSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.EvaluationSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.EvaluationSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.EvaluationSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.ExportSpec

*class* · *pydantic model*

```python
class ExportSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Requested export artifact.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L487-L493)

### nexuml.core.types.ExportSpec.exclude

*attribute* · *pydantic field*

```python
exclude: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L493-L493)

### nexuml.core.types.ExportSpec.include

*attribute* · *pydantic field*

```python
include: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L492-L492)

### nexuml.core.types.ExportSpec.kind

*attribute* · *pydantic field*

```python
kind: Literal['train_package', 'onnx', 'safetensors'] = 'train_package'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L490-L490)

### nexuml.core.types.ExportSpec.output

*attribute* · *pydantic field*

```python
output: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L491-L491)

### nexuml.core.types.ExportSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.ExportSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.ExportSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.ExportSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.FactorySpec

*class* · *pydantic model*

```python
class FactorySpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Portable constructor configuration authored through a typed factory helper.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L70-L125)

### nexuml.core.types.FactorySpec.args

*attribute* · *pydantic field*

```python
args: list[object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L74-L74)

### nexuml.core.types.FactorySpec.factory

*attribute* · *pydantic field*

```python
factory: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L73-L73)

### nexuml.core.types.FactorySpec.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L75-L75)

### nexuml.core.types.FactorySpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.FactorySpec.build

*method*

```python
def build(*runtime_args: object, **runtime_kwargs: object) -> object
```

Construct the configured object with any runtime-supplied arguments.

**Returns**

- (`object`) — Constructed runtime object.

**Raises**

- `TypeError` — If construction fails.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L110-L125)

### nexuml.core.types.FactorySpec.from_factory

*method* · *classmethod*

```python
def from_factory(factory: object, *args: object, **kwargs: object) -> Self
```

Capture an importable factory and portable constructor arguments.

**Returns**

- (`Self`) — Validated factory specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L93-L100)

### nexuml.core.types.FactorySpec.resolve

*method*

```python
def resolve() -> Callable[..., object]
```

Resolve the configured import target.

**Returns**

- (`Callable[..., object]`) — Imported callable.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L102-L108)

### nexuml.core.types.FactorySpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.FactorySpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.FactorySpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.LayerSpec

*class* · *pydantic model*

```python
class LayerSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for a single layer in the pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L128-L139)

### nexuml.core.types.LayerSpec.component

*attribute* · *pydantic field*

```python
component: LayerDefinition
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L131-L131)

### nexuml.core.types.LayerSpec.delay_epochs

*attribute* · *pydantic field*

```python
delay_epochs: int = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L138-L138)

### nexuml.core.types.LayerSpec.keys_in

*attribute* · *pydantic field*

```python
keys_in: list[str] | dict[str, str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L132-L132)

### nexuml.core.types.LayerSpec.keys_out

*attribute* · *pydantic field*

```python
keys_out: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L133-L133)

### nexuml.core.types.LayerSpec.label_in_x

*attribute* · *pydantic field*

```python
label_in_x: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L135-L135)

### nexuml.core.types.LayerSpec.label_key

*attribute* · *pydantic field*

```python
label_key: str | list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L134-L134)

### nexuml.core.types.LayerSpec.meta_in

*attribute* · *pydantic field*

```python
meta_in: dict[str, str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L136-L136)

### nexuml.core.types.LayerSpec.meta_out

*attribute* · *pydantic field*

```python
meta_out: dict[str, str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L137-L137)

### nexuml.core.types.LayerSpec.update_every_n_epochs

*attribute* · *pydantic field*

```python
update_every_n_epochs: int = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L139-L139)

### nexuml.core.types.LayerSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.LayerSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.LayerSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.LayerSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.LoaderSpec

*class* · *pydantic model*

```python
class LoaderSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for the data loader.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L262-L279)

### nexuml.core.types.LoaderSpec.backend

*attribute* · *pydantic field*

```python
backend: LoaderBackendDefinition
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L265-L265)

### nexuml.core.types.LoaderSpec.batch_size

*attribute* · *pydantic field*

```python
batch_size: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L267-L267)

### nexuml.core.types.LoaderSpec.num_workers

*attribute* · *pydantic field*

```python
num_workers: int = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L268-L268)

### nexuml.core.types.LoaderSpec.persistent_workers

*attribute* · *pydantic field*

```python
persistent_workers: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L270-L270)

### nexuml.core.types.LoaderSpec.prefetch_factor

*attribute* · *pydantic field*

```python
prefetch_factor: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L269-L269)

### nexuml.core.types.LoaderSpec.shuffle_train

*attribute* · *pydantic field*

```python
shuffle_train: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L272-L272)

### nexuml.core.types.LoaderSpec.weighted_sampling

*attribute* · *pydantic field*

```python
weighted_sampling: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L271-L271)

### nexuml.core.types.LoaderSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.LoaderSpec.validate_batch_size

*method* · *pydantic validator*

```python
def validate_batch_size(value: int | None) -> int | None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L274-L279)

### nexuml.core.types.LoaderSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.LoaderSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.LoaderSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.LoggingSpec

*class* · *pydantic model*

```python
class LoggingSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Experiment logging configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L449-L459)

### nexuml.core.types.LoggingSpec.diagram

*attribute* · *pydantic field*

```python
diagram: DiagramSpec | None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L455-L455)

### nexuml.core.types.LoggingSpec.dvclive

*attribute* · *pydantic field*

```python
dvclive: DVCLiveSpec | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L453-L453)

### nexuml.core.types.LoggingSpec.experiment_name

*attribute* · *pydantic field*

```python
experiment_name: str = 'NexuML'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L457-L457)

### nexuml.core.types.LoggingSpec.log_system_metrics

*attribute* · *pydantic field*

```python
log_system_metrics: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L459-L459)

### nexuml.core.types.LoggingSpec.mlflow

*attribute* · *pydantic field*

```python
mlflow: MLflowSpec | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L454-L454)

### nexuml.core.types.LoggingSpec.run_name

*attribute* · *pydantic field*

```python
run_name: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L458-L458)

### nexuml.core.types.LoggingSpec.temp_artifact_dir

*attribute* · *pydantic field*

```python
temp_artifact_dir: str = '/tmp/mlruns'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L456-L456)

### nexuml.core.types.LoggingSpec.tensorboard

*attribute* · *pydantic field*

```python
tensorboard: TensorBoardSpec | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L452-L452)

### nexuml.core.types.LoggingSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.LoggingSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.LoggingSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.LoggingSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.MLflowSpec

*class* · *pydantic model*

```python
class MLflowSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

MLflow logger configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L423-L434)

### nexuml.core.types.MLflowSpec.artifact_location

*attribute* · *pydantic field*

```python
artifact_location: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L427-L427)

### nexuml.core.types.MLflowSpec.auto_nested_runs

*attribute* · *pydantic field*

```python
auto_nested_runs: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L432-L432)

### nexuml.core.types.MLflowSpec.experiment_name

*attribute* · *pydantic field*

```python
experiment_name: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L428-L428)

### nexuml.core.types.MLflowSpec.log_model

*attribute* · *pydantic field*

```python
log_model: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L429-L429)

### nexuml.core.types.MLflowSpec.parent_run_id

*attribute* · *pydantic field*

```python
parent_run_id: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L431-L431)

### nexuml.core.types.MLflowSpec.run_id

*attribute* · *pydantic field*

```python
run_id: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L430-L430)

### nexuml.core.types.MLflowSpec.synchronous

*attribute* · *pydantic field*

```python
synchronous: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L434-L434)

### nexuml.core.types.MLflowSpec.tags

*attribute* · *pydantic field*

```python
tags: dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L433-L433)

### nexuml.core.types.MLflowSpec.tracking_uri

*attribute* · *pydantic field*

```python
tracking_uri: str = 'file:./.experiments/mlflow'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L426-L426)

### nexuml.core.types.MLflowSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.MLflowSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.MLflowSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.MLflowSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.OptimizerSpec

*class* · *pydantic model*

```python
class OptimizerSpec(FactorySpec)
```

Bases: `nexuml.core.types.FactorySpec`

Specification for an optimizer.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L151-L155)

### nexuml.core.types.OptimizerSpec.factory

*attribute* · *pydantic field*

```python
factory: str = 'torch.optim.adam:Adam'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L154-L154)

### nexuml.core.types.OptimizerSpec.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L155-L155)

### nexuml.core.types.OptimizerSpec.args

*attribute* · *pydantic field*

```python
args: list[object]
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L74-L74)

### nexuml.core.types.OptimizerSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.OptimizerSpec.build

*method*

```python
def build(*runtime_args: object, **runtime_kwargs: object) -> object
```

Inherited from: `nexuml.core.types.FactorySpec`

Construct the configured object with any runtime-supplied arguments.

**Returns**

- (`object`) — Constructed runtime object.

**Raises**

- `TypeError` — If construction fails.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L110-L125)

### nexuml.core.types.OptimizerSpec.from_factory

*method* · *classmethod*

```python
def from_factory(factory: object, *args: object, **kwargs: object) -> Self
```

Inherited from: `nexuml.core.types.FactorySpec`

Capture an importable factory and portable constructor arguments.

**Returns**

- (`Self`) — Validated factory specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L93-L100)

### nexuml.core.types.OptimizerSpec.resolve

*method*

```python
def resolve() -> Callable[..., object]
```

Inherited from: `nexuml.core.types.FactorySpec`

Resolve the configured import target.

**Returns**

- (`Callable[..., object]`) — Imported callable.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L102-L108)

### nexuml.core.types.OptimizerSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.OptimizerSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.OptimizerSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.PipelineSpec

*class* · *pydantic model*

```python
class PipelineSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Ordered specification of pipeline stages, each containing layer specs.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L142-L145)

### nexuml.core.types.PipelineSpec.stages

*attribute* · *pydantic field*

```python
stages: dict[str, list[LayerSpec]]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L145-L145)

### nexuml.core.types.PipelineSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.PipelineSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.PipelineSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.PipelineSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.PreprocessingSpec

*class* · *pydantic model*

```python
class PreprocessingSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Optional preprocessing stage contract.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L310-L324)

### nexuml.core.types.PreprocessingSpec.enabled

*attribute* · *pydantic field*

```python
enabled: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L313-L313)

### nexuml.core.types.PreprocessingSpec.include_labels

*attribute* · *pydantic field*

```python
include_labels: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L321-L321)

### nexuml.core.types.PreprocessingSpec.label_prefix

*attribute* · *pydantic field*

```python
label_prefix: str = 'label__'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L322-L322)

### nexuml.core.types.PreprocessingSpec.overwrite

*attribute* · *pydantic field*

```python
overwrite: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L324-L324)

### nexuml.core.types.PreprocessingSpec.path

*attribute* · *pydantic field*

```python
path: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L316-L316)

### nexuml.core.types.PreprocessingSpec.source_view

*attribute* · *pydantic field*

```python
source_view: str = 'raw'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L314-L314)

### nexuml.core.types.PreprocessingSpec.target_view

*attribute* · *pydantic field*

```python
target_view: str = 'prepared'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L315-L315)

### nexuml.core.types.PreprocessingSpec.until_x_keys

*attribute* · *pydantic field*

```python
until_x_keys: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L317-L317)

### nexuml.core.types.PreprocessingSpec.until_y_keys

*attribute* · *pydantic field*

```python
until_y_keys: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L318-L318)

### nexuml.core.types.PreprocessingSpec.writer

*attribute* · *pydantic field*

```python
writer: WriterSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L323-L323)

### nexuml.core.types.PreprocessingSpec.x_keys

*attribute* · *pydantic field*

```python
x_keys: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L319-L319)

### nexuml.core.types.PreprocessingSpec.y_keys

*attribute* · *pydantic field*

```python
y_keys: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L320-L320)

### nexuml.core.types.PreprocessingSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.PreprocessingSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.PreprocessingSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.PreprocessingSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.ScenarioSpec

*class* · *pydantic model*

```python
class ScenarioSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Complete scenario specification combining all components.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L496-L524)

### nexuml.core.types.ScenarioSpec.callbacks

*attribute* · *pydantic field*

```python
callbacks: list[CallbackSpec]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L505-L505)

### nexuml.core.types.ScenarioSpec.checkpoint

*attribute* · *pydantic field*

```python
checkpoint: CheckpointLoadSpec | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L507-L507)

### nexuml.core.types.ScenarioSpec.data

*attribute* · *pydantic field*

```python
data: DataSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L502-L502)

### nexuml.core.types.ScenarioSpec.evaluation

*attribute* · *pydantic field*

```python
evaluation: EvaluationSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L503-L503)

### nexuml.core.types.ScenarioSpec.execution

*attribute* · *pydantic field*

```python
execution: ExecutionBackendDefinition
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L509-L509)

### nexuml.core.types.ScenarioSpec.exports

*attribute* · *pydantic field*

```python
exports: list[ExportSpec]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L508-L508)

### nexuml.core.types.ScenarioSpec.logging

*attribute* · *pydantic field*

```python
logging: LoggingSpec | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L504-L504)

### nexuml.core.types.ScenarioSpec.name

*attribute* · *pydantic field*

```python
name: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L499-L499)

### nexuml.core.types.ScenarioSpec.pipeline

*attribute* · *pydantic field*

```python
pipeline: PipelineSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L500-L500)

### nexuml.core.types.ScenarioSpec.training

*attribute* · *pydantic field*

```python
training: TrainingSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L501-L501)

### nexuml.core.types.ScenarioSpec.tuning

*attribute* · *pydantic field*

```python
tuning: TuningSpec | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L506-L506)

### nexuml.core.types.ScenarioSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.ScenarioSpec.restore_execution

*method* · *classmethod*

```python
def restore_execution(value: Any) -> ExecutionBackendDefinition
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L511-L524)

### nexuml.core.types.ScenarioSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.ScenarioSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.ScenarioSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.SchedulerSpec

*class* · *pydantic model*

```python
class SchedulerSpec(FactorySpec)
```

Bases: `nexuml.core.types.FactorySpec`

Specification for a learning rate scheduler.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L161-L186)

### nexuml.core.types.SchedulerSpec.factory

*attribute* · *pydantic field*

```python
factory: str = 'torch.optim.lr_scheduler:ConstantLR'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L164-L164)

### nexuml.core.types.SchedulerSpec.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L165-L165)

### nexuml.core.types.SchedulerSpec.warmup

*attribute* · *pydantic field*

```python
warmup: str | int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L166-L166)

### nexuml.core.types.SchedulerSpec.args

*attribute* · *pydantic field*

```python
args: list[object]
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L74-L74)

### nexuml.core.types.SchedulerSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.SchedulerSpec.resolve_warmup

*method*

```python
def resolve_warmup(max_epochs: int) -> int
```

Resolve the warmup spec to an integer number of epochs.

Accepts an integer (used directly), a percentage string like ``"5%"``
(computed as a fraction of *max_epochs*), or ``None`` (returns 0).
The result is always at least 1 when a non-zero warmup is specified.

**Returns**

- (`int`) — Resolved warmup epoch count.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L168-L186)

### nexuml.core.types.SchedulerSpec.build

*method*

```python
def build(*runtime_args: object, **runtime_kwargs: object) -> object
```

Inherited from: `nexuml.core.types.FactorySpec`

Construct the configured object with any runtime-supplied arguments.

**Returns**

- (`object`) — Constructed runtime object.

**Raises**

- `TypeError` — If construction fails.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L110-L125)

### nexuml.core.types.SchedulerSpec.from_factory

*method* · *classmethod*

```python
def from_factory(factory: object, *args: object, **kwargs: object) -> Self
```

Inherited from: `nexuml.core.types.FactorySpec`

Capture an importable factory and portable constructor arguments.

**Returns**

- (`Self`) — Validated factory specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L93-L100)

### nexuml.core.types.SchedulerSpec.resolve

*method*

```python
def resolve() -> Callable[..., object]
```

Inherited from: `nexuml.core.types.FactorySpec`

Resolve the configured import target.

**Returns**

- (`Callable[..., object]`) — Imported callable.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L102-L108)

### nexuml.core.types.SchedulerSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.SchedulerSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.SchedulerSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.SpecModel

*class* · *pydantic model*

```python
class SpecModel(BaseModel)
```

Bases: `BaseModel`

Shared base model for scenario specifications.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L43-L67)

### nexuml.core.types.SpecModel.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.SpecModel.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.SpecModel.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.SpecModel.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.StrategySpec

*class* · *pydantic model*

```python
class StrategySpec(FactorySpec)
```

Bases: `nexuml.core.types.FactorySpec`

Specification for an importable Lightning strategy.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L189-L190)

### nexuml.core.types.StrategySpec.args

*attribute* · *pydantic field*

```python
args: list[object]
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L74-L74)

### nexuml.core.types.StrategySpec.factory

*attribute* · *pydantic field*

```python
factory: str
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L73-L73)

### nexuml.core.types.StrategySpec.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L75-L75)

### nexuml.core.types.StrategySpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.StrategySpec.build

*method*

```python
def build(*runtime_args: object, **runtime_kwargs: object) -> object
```

Inherited from: `nexuml.core.types.FactorySpec`

Construct the configured object with any runtime-supplied arguments.

**Returns**

- (`object`) — Constructed runtime object.

**Raises**

- `TypeError` — If construction fails.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L110-L125)

### nexuml.core.types.StrategySpec.from_factory

*method* · *classmethod*

```python
def from_factory(factory: object, *args: object, **kwargs: object) -> Self
```

Inherited from: `nexuml.core.types.FactorySpec`

Capture an importable factory and portable constructor arguments.

**Returns**

- (`Self`) — Validated factory specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L93-L100)

### nexuml.core.types.StrategySpec.resolve

*method*

```python
def resolve() -> Callable[..., object]
```

Inherited from: `nexuml.core.types.FactorySpec`

Resolve the configured import target.

**Returns**

- (`Callable[..., object]`) — Imported callable.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L102-L108)

### nexuml.core.types.StrategySpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.StrategySpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.StrategySpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.TargetSpec

*class* · *pydantic model*

```python
class TargetSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for a synthetic target.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L243-L250)

### nexuml.core.types.TargetSpec.key

*attribute* · *pydantic field*

```python
key: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L247-L247)

### nexuml.core.types.TargetSpec.num_classes

*attribute* · *pydantic field*

```python
num_classes: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L248-L248)

### nexuml.core.types.TargetSpec.num_outputs

*attribute* · *pydantic field*

```python
num_outputs: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L249-L249)

### nexuml.core.types.TargetSpec.positive_fraction

*attribute* · *pydantic field*

```python
positive_fraction: float | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L250-L250)

### nexuml.core.types.TargetSpec.type

*attribute* · *pydantic field*

```python
type: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L246-L246)

### nexuml.core.types.TargetSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.TargetSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.TargetSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.TargetSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.TensorBoardSpec

*class* · *pydantic model*

```python
class TensorBoardSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

TensorBoard logger configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L411-L414)

### nexuml.core.types.TensorBoardSpec.log_dir

*attribute* · *pydantic field*

```python
log_dir: str = '.experiments/tensorboard'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L414-L414)

### nexuml.core.types.TensorBoardSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.TensorBoardSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.TensorBoardSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.TensorBoardSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.TrainingSpec

*class* · *pydantic model*

```python
class TrainingSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for training configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L219-L240)

### nexuml.core.types.TrainingSpec.accelerator

*attribute* · *pydantic field*

```python
accelerator: str = 'auto'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L230-L230)

### nexuml.core.types.TrainingSpec.batch_size

*attribute* · *pydantic field*

```python
batch_size: BatchSizeSpec = 64
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L228-L228)

### nexuml.core.types.TrainingSpec.devices

*attribute* · *pydantic field*

```python
devices: str | int = 'auto'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L231-L231)

### nexuml.core.types.TrainingSpec.loss_keys

*attribute* · *pydantic field*

```python
loss_keys: dict[str, float]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L224-L224)

### nexuml.core.types.TrainingSpec.lr

*attribute* · *pydantic field*

```python
lr: float = 0.001
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L229-L229)

### nexuml.core.types.TrainingSpec.max_epochs

*attribute* · *pydantic field*

```python
max_epochs: int = 10
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L227-L227)

### nexuml.core.types.TrainingSpec.metric_keys

*attribute* · *pydantic field*

```python
metric_keys: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L225-L225)

### nexuml.core.types.TrainingSpec.optimizer

*attribute* · *pydantic field*

```python
optimizer: OptimizerSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L222-L222)

### nexuml.core.types.TrainingSpec.precision

*attribute* · *pydantic field*

```python
precision: str | int = '32-true'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L233-L233)

### nexuml.core.types.TrainingSpec.progress_bar_keys

*attribute* · *pydantic field*

```python
progress_bar_keys: list[str] | Literal['all', 'none']
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L226-L226)

### nexuml.core.types.TrainingSpec.scheduler

*attribute* · *pydantic field*

```python
scheduler: SchedulerSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L223-L223)

### nexuml.core.types.TrainingSpec.strategy

*attribute* · *pydantic field*

```python
strategy: str | StrategySpec = 'auto'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L232-L232)

### nexuml.core.types.TrainingSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.TrainingSpec.validate_batch_size

*method* · *pydantic validator*

```python
def validate_batch_size(value: BatchSizeSpec) -> BatchSizeSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L235-L240)

### nexuml.core.types.TrainingSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.TrainingSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.TrainingSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.TuningSpec

*class* · *pydantic model*

```python
class TuningSpec(SpecModel)
```

Bases: `nexuml.core.types.SpecModel`

Specification for hyperparameter tuning with Optuna.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L466-L473)

### nexuml.core.types.TuningSpec.directions

*attribute* · *pydantic field*

```python
directions: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L470-L470)

### nexuml.core.types.TuningSpec.metric_key

*attribute* · *pydantic field*

```python
metric_key: str = 'val/loss'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L471-L471)

### nexuml.core.types.TuningSpec.n_trials

*attribute* · *pydantic field*

```python
n_trials: int = 50
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L469-L469)

### nexuml.core.types.TuningSpec.prune

*attribute* · *pydantic field*

```python
prune: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L473-L473)

### nexuml.core.types.TuningSpec.storage

*attribute* · *pydantic field*

```python
storage: str = '.experiments/optuna/optuna.log'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L472-L472)

### nexuml.core.types.TuningSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.TuningSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.TuningSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.TuningSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)

## nexuml.core.types.WriterSpec

*class* · *pydantic model*

```python
class WriterSpec(FactorySpec)
```

Bases: `nexuml.core.types.FactorySpec`

Specification for an importable dataset export backend.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L288-L307)

### nexuml.core.types.WriterSpec.factory

*attribute* · *pydantic field*

```python
factory: str = 'nexuml.data.export.numpy_files:NumpyBackend'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L291-L291)

### nexuml.core.types.WriterSpec.args

*attribute* · *pydantic field*

```python
args: list[object]
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L74-L74)

### nexuml.core.types.WriterSpec.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

Inherited from: `nexuml.core.types.FactorySpec`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L75-L75)

### nexuml.core.types.WriterSpec.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L46-L46)

### nexuml.core.types.WriterSpec.backend_name

*method*

```python
def backend_name() -> str
```

Return the stable registry name for the configured backend class.

**Returns**

- (`str`) — Registered export backend name.

**Raises**

- `TypeError` — If the configured factory is not an export backend class.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L293-L307)

### nexuml.core.types.WriterSpec.build

*method*

```python
def build(*runtime_args: object, **runtime_kwargs: object) -> object
```

Inherited from: `nexuml.core.types.FactorySpec`

Construct the configured object with any runtime-supplied arguments.

**Returns**

- (`object`) — Constructed runtime object.

**Raises**

- `TypeError` — If construction fails.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L110-L125)

### nexuml.core.types.WriterSpec.from_factory

*method* · *classmethod*

```python
def from_factory(factory: object, *args: object, **kwargs: object) -> Self
```

Inherited from: `nexuml.core.types.FactorySpec`

Capture an importable factory and portable constructor arguments.

**Returns**

- (`Self`) — Validated factory specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L93-L100)

### nexuml.core.types.WriterSpec.resolve

*method*

```python
def resolve() -> Callable[..., object]
```

Inherited from: `nexuml.core.types.FactorySpec`

Resolve the configured import target.

**Returns**

- (`Callable[..., object]`) — Imported callable.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L102-L108)

### nexuml.core.types.WriterSpec.from_dict

*method* · *classmethod*

```python
def from_dict(data: dict[str, Any]) -> 'SpecModel'
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L51-L53)

### nexuml.core.types.WriterSpec.get

*method*

```python
def get(key: str, default: Any = None) -> Any
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L58-L59)

### nexuml.core.types.WriterSpec.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

Inherited from: `nexuml.core.types.SpecModel`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/types.py#L48-L49)
