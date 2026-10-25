# nexuml

*module*

NexuML - Modular deep learning pipeline framework.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/__init__.py#L1-L1)

## nexuml.NnModuleLayer

*class* · *pydantic model*

```python
class NnModuleLayer(LayerDefinition)
```

Bases: `nexuml.core.components.LayerDefinition`

Re-exported from: `nexuml.core.torch_adapter`

Portable definition for an importable one-input/one-output PyTorch module.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L22-L69)

### nexuml.NnModuleLayer.args

*attribute* · *pydantic field*

```python
args: list[object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L27-L27)

### nexuml.NnModuleLayer.factory

*attribute* · *pydantic field*

```python
factory: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L26-L26)

### nexuml.NnModuleLayer.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L28-L28)

### nexuml.NnModuleLayer.kind

*attribute* · *pydantic field*

```python
kind = 'layer'
```

Inherited from: `nexuml.core.components.LayerDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L124-L124)

### nexuml.NnModuleLayer.requires_post_train_fit

*attribute* · *class attribute*

```python
requires_post_train_fit: bool = False
```

Inherited from: `nexuml.core.components.LayerDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L125-L125)

### nexuml.NnModuleLayer.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.NnModuleLayer.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.NnModuleLayer.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

### nexuml.NnModuleLayer.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L46-L69)

## nexuml.callback

*function*

```python
def callback(factory: Callable[P, Callback], *args: P.args, **kwargs: P.kwargs) -> CallbackSpec
```

Re-exported from: `nexuml.core.factory`

Configure an importable Lightning callback factory.

**Returns**

- (`CallbackSpec`) — Portable callback specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L127-L135)

## nexuml.nn_module

*function*

```python
def nn_module(factory: Callable[P, nn.Module], *args: P.args, **kwargs: P.kwargs) -> NnModuleLayer
```

Re-exported from: `nexuml.core.torch_adapter`

Create a portable direct-module definition from an importable factory.

**Returns**

- (`NnModuleLayer`) — Validated universal layer definition.

**Raises**

- `TypeError` — If *factory* is a live module instance.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L75-L92)

## nexuml.optimizer

*function*

```python
def optimizer(factory: Callable[Concatenate[Any, P], Optimizer], *args: P.args, **kwargs: P.kwargs) -> OptimizerSpec
```

Re-exported from: `nexuml.core.factory`

Configure an optimizer factory; model parameters are supplied at runtime.

**Returns**

- (`OptimizerSpec`) — Portable optimizer specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L138-L148)

## nexuml.scheduler

*function*

```python
def scheduler(factory: Callable[Concatenate[Optimizer, P], LRScheduler], *args: P.args, **kwargs: P.kwargs) -> SchedulerSpec
```

Re-exported from: `nexuml.core.factory`

Configure a scheduler factory; its optimizer is supplied at runtime.

**Returns**

- (`SchedulerSpec`) — Portable scheduler specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L151-L163)

## nexuml.strategy

*function*

```python
def strategy(factory: Callable[P, Strategy], *args: P.args, **kwargs: P.kwargs) -> StrategySpec
```

Re-exported from: `nexuml.core.factory`

Configure an importable Lightning strategy factory.

**Returns**

- (`StrategySpec`) — Portable strategy specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L166-L174)

## nexuml.writer

*function*

```python
def writer(factory: Callable[P, ExportBackend], *args: P.args, **kwargs: P.kwargs) -> WriterSpec
```

Re-exported from: `nexuml.core.factory`

Configure an importable dataset export backend factory.

**Returns**

- (`WriterSpec`) — Portable writer specification.

**Raises**

- `TypeError` — If *factory* is not an export backend class.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L177-L192)

**Modules**

- `nexuml.api`
- `nexuml.cli`
- `nexuml.core`
- `nexuml.data`
- `nexuml.evaluation`
- `nexuml.execution`
- `nexuml.storage`
- `nexuml.tracking`
- `nexuml.training`
- `nexuml.tuning`
