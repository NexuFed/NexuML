# nexuml.core.components

*module*

Typed component definitions and explicit runtime build contexts.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L1-L1)

## nexuml.core.components.FeatureKey

*attribute* · *module attribute*

```python
FeatureKey = Annotated[str, _KeyRole('x')]
```

A feature-tensor key; values and generated schemas remain ordinary strings.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L16-L16)

## nexuml.core.components.LabelKey

*attribute* · *module attribute*

```python
LabelKey = Annotated[str, _KeyRole('y')]
```

A label-tensor key, also usable inside optional and list annotations.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L18-L18)

## nexuml.core.components.ComponentDefinition

*class* · *pydantic model*

```python
class ComponentDefinition(BaseModel, ABC)
```

Bases: `BaseModel`, `ABC`

Immutable semantic configuration for a registered component.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L22-L35)

### nexuml.core.components.ComponentDefinition.component_name

*attribute* · *class attribute*

```python
component_name: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.core.components.ComponentDefinition.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.core.components.ComponentDefinition.kind

*attribute* · *class attribute*

```python
kind: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L33-L33)

### nexuml.core.components.ComponentDefinition.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

## nexuml.core.components.ContextInput

*class* · *dataclass*

```python
class ContextInput
```

One consumed context slot; a default replaces missing, null or empty values.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L81-L100)

### nexuml.core.components.ContextInput.default

*attribute* · *class attribute* · *instance attribute*

```python
default: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L86-L86)

### nexuml.core.components.ContextInput.required

*attribute* · *class attribute* · *instance attribute*

```python
required: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L85-L85)

### nexuml.core.components.ContextInput.resolve

*method*

```python
def resolve(value: str | None, name: str) -> str | None
```

Resolve a context key without rejecting historically accepted empty strings.

**Returns**

- (`str | None`) — Explicit key or its declared fallback.

**Raises**

- `ValueError` — If a required key is absent and has no fallback.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L88-L100)

## nexuml.core.components.DataSourceDefinition

*class* · *pydantic model*

```python
class DataSourceDefinition(ComponentDefinition)
```

Bases: `nexuml.core.components.ComponentDefinition`

Semantic configuration that materializes a dataset.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L132-L141)

### nexuml.core.components.DataSourceDefinition.fixed_label_keys

*attribute* · *class attribute*

```python
fixed_label_keys: tuple[str, ...] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L137-L137)

### nexuml.core.components.DataSourceDefinition.kind

*attribute* · *pydantic field*

```python
kind = 'data_source'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L135-L135)

### nexuml.core.components.DataSourceDefinition.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.core.components.DataSourceDefinition.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.core.components.DataSourceDefinition.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

### nexuml.core.components.DataSourceDefinition.build

*method* · *abstract*

```python
def build() -> Any
```

Build the mutable dataset runtime.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L139-L141)

## nexuml.core.components.EvalAlgorithmDefinition

*class* · *pydantic model*

```python
class EvalAlgorithmDefinition(ComponentDefinition)
```

Bases: `nexuml.core.components.ComponentDefinition`

Semantic configuration that materializes an evaluation algorithm.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L144-L153)

### nexuml.core.components.EvalAlgorithmDefinition.context_inputs

*attribute* · *class attribute*

```python
context_inputs: EvalContextInputs | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L149-L149)

### nexuml.core.components.EvalAlgorithmDefinition.kind

*attribute* · *pydantic field*

```python
kind = 'eval_algorithm'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L147-L147)

### nexuml.core.components.EvalAlgorithmDefinition.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.core.components.EvalAlgorithmDefinition.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.core.components.EvalAlgorithmDefinition.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

### nexuml.core.components.EvalAlgorithmDefinition.build

*method* · *abstract*

```python
def build(context: EvalBuildContext) -> Any
```

Build the mutable evaluation runtime.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L151-L153)

## nexuml.core.components.EvalBuildContext

*class* · *dataclass*

```python
class EvalBuildContext
```

Runtime routing values supplied to an evaluation definition.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L73-L78)

### nexuml.core.components.EvalBuildContext.feature_key

*attribute* · *class attribute* · *instance attribute*

```python
feature_key: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L77-L77)

### nexuml.core.components.EvalBuildContext.label_key

*attribute* · *class attribute* · *instance attribute*

```python
label_key: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L78-L78)

## nexuml.core.components.EvalContextInputs

*class* · *dataclass*

```python
class EvalContextInputs
```

Consumed evaluation slots; an omitted slot is unused, not a guessed route.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L103-L118)

### nexuml.core.components.EvalContextInputs.feature_key

*attribute* · *class attribute* · *instance attribute*

```python
feature_key: ContextInput | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L107-L107)

### nexuml.core.components.EvalContextInputs.label_key

*attribute* · *class attribute* · *instance attribute*

```python
label_key: ContextInput | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L108-L108)

### nexuml.core.components.EvalContextInputs.resolve

*method*

```python
def resolve(context: EvalBuildContext) -> EvalBuildContext
```

Return runtime keys using the same rules exposed to tooling.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L110-L118)

## nexuml.core.components.ExecutionBackendDefinition

*class* · *pydantic model*

```python
class ExecutionBackendDefinition(ComponentDefinition)
```

Bases: `nexuml.core.components.ComponentDefinition`

Installation-owned placement settings and execution behavior.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L166-L222)

### nexuml.core.components.ExecutionBackendDefinition.capabilities

*attribute* · *class attribute*

```python
capabilities: dict[str, bool] = {}
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L172-L172)

### nexuml.core.components.ExecutionBackendDefinition.dependencies

*attribute* · *class attribute*

```python
dependencies: tuple[str, ...] = ()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L171-L171)

### nexuml.core.components.ExecutionBackendDefinition.kind

*attribute* · *pydantic field*

```python
kind = 'execution_backend'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L169-L169)

### nexuml.core.components.ExecutionBackendDefinition.label

*attribute* · *class attribute*

```python
label: str = ''
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L170-L170)

### nexuml.core.components.ExecutionBackendDefinition.presentation

*attribute* · *class attribute*

```python
presentation: dict[str, list[str]] = {}
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L173-L173)

### nexuml.core.components.ExecutionBackendDefinition.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.core.components.ExecutionBackendDefinition.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.core.components.ExecutionBackendDefinition.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

### nexuml.core.components.ExecutionBackendDefinition.cancel

*method*

```python
def cancel(reference: Any) -> dict[str, Any]
```

Refuse to confuse submitter termination with native cancellation.

**Raises**

- `NotImplementedError` — When native cancellation cannot be confirmed.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L208-L214)

### nexuml.core.components.ExecutionBackendDefinition.discover

*method*

```python
def discover(**kwargs: Any) -> Any
```

Return an explicit unknown snapshot when a library has no capacity probe.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L190-L198)

### nexuml.core.components.ExecutionBackendDefinition.inspect

*method*

```python
def inspect(reference: Any) -> dict[str, Any]
```

Refuse native observation that the backend has not implemented.

**Raises**

- `NotImplementedError` — When no native inspection is available.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L200-L206)

### nexuml.core.components.ExecutionBackendDefinition.preflight

*method*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

Validate a request without allocating workers.

**Returns**

- (`dict[str, Any]`) — Backend-specific review information.

**Raises**

- `ValueError` — If requested resume or exports are unsupported.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L175-L188)

### nexuml.core.components.ExecutionBackendDefinition.run

*method* · *abstract*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

Execute the canonical lifecycle and return its native result.

Shared dispatch supplies ``review`` from preflight and caller-specific keyword
options (including an optional observer). Definitions accept these via kwargs.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L216-L222)

## nexuml.core.components.LayerBuildContext

*class* · *dataclass*

```python
class LayerBuildContext
```

Runtime values supplied by the pipeline compiler to a layer definition.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L38-L70)

### nexuml.core.components.LayerBuildContext.delay_epochs

*attribute* · *class attribute* · *instance attribute*

```python
delay_epochs: int = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L52-L52)

### nexuml.core.components.LayerBuildContext.input_sizes

*attribute* · *instance attribute*

```python
input_sizes: Mapping[str, tuple[int, ...]]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L42-L42)

### nexuml.core.components.LayerBuildContext.keys_in

*attribute* · *instance attribute*

```python
keys_in: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L43-L43)

### nexuml.core.components.LayerBuildContext.keys_out

*attribute* · *instance attribute*

```python
keys_out: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L44-L44)

### nexuml.core.components.LayerBuildContext.label_in_x

*attribute* · *class attribute* · *instance attribute*

```python
label_in_x: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L46-L46)

### nexuml.core.components.LayerBuildContext.label_key

*attribute* · *class attribute* · *instance attribute*

```python
label_key: str | list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L45-L45)

### nexuml.core.components.LayerBuildContext.metadata

*attribute* · *class attribute* · *instance attribute*

```python
metadata: Mapping[str, Any] = field(default_factory=dict)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L48-L48)

### nexuml.core.components.LayerBuildContext.num_classes

*attribute* · *class attribute* · *instance attribute*

```python
num_classes: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L47-L47)

### nexuml.core.components.LayerBuildContext.shared_inputs

*attribute* · *class attribute* · *instance attribute*

```python
shared_inputs: list[str] | str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L51-L51)

### nexuml.core.components.LayerBuildContext.shared_outputs

*attribute* · *class attribute* · *instance attribute*

```python
shared_outputs: list[str] | str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L50-L50)

### nexuml.core.components.LayerBuildContext.shared_storage

*attribute* · *class attribute* · *instance attribute*

```python
shared_storage: Any = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L49-L49)

### nexuml.core.components.LayerBuildContext.update_every_n_epochs

*attribute* · *class attribute* · *instance attribute*

```python
update_every_n_epochs: int = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L53-L53)

### nexuml.core.components.LayerBuildContext.runtime_kwargs

*method*

```python
def runtime_kwargs(**overrides: Any) -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L55-L70)

## nexuml.core.components.LayerDefinition

*class* · *pydantic model*

```python
class LayerDefinition(ComponentDefinition)
```

Bases: `nexuml.core.components.ComponentDefinition`

Semantic configuration that materializes a pipeline layer.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L121-L129)

### nexuml.core.components.LayerDefinition.kind

*attribute* · *pydantic field*

```python
kind = 'layer'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L124-L124)

### nexuml.core.components.LayerDefinition.requires_post_train_fit

*attribute* · *class attribute*

```python
requires_post_train_fit: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L125-L125)

### nexuml.core.components.LayerDefinition.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.core.components.LayerDefinition.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.core.components.LayerDefinition.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

### nexuml.core.components.LayerDefinition.build

*method* · *abstract*

```python
def build(context: LayerBuildContext) -> Any
```

Build the mutable pipeline layer runtime.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L127-L129)

## nexuml.core.components.LoaderBackendDefinition

*class* · *pydantic model*

```python
class LoaderBackendDefinition(ComponentDefinition)
```

Bases: `nexuml.core.components.ComponentDefinition`

Semantic configuration that materializes a loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L156-L163)

### nexuml.core.components.LoaderBackendDefinition.kind

*attribute* · *pydantic field*

```python
kind = 'loader_backend'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L159-L159)

### nexuml.core.components.LoaderBackendDefinition.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.core.components.LoaderBackendDefinition.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.core.components.LoaderBackendDefinition.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

### nexuml.core.components.LoaderBackendDefinition.build

*method* · *abstract*

```python
def build() -> Any
```

Build the loader backend runtime.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L161-L163)
