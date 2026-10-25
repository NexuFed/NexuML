# nexuml.core.factory

*module*

Portable importable-factory helpers for typed Python configuration.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L1-L1)

## nexuml.core.factory.JsonValue

*attribute*

```python
JsonValue
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L26-L26)

## nexuml.core.factory.P

*attribute* · *module attribute*

```python
P = ParamSpec('P')
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L124-L124)

## nexuml.core.factory.callback

*function*

```python
def callback(factory: Callable[P, Callback], *args: P.args, **kwargs: P.kwargs) -> CallbackSpec
```

Configure an importable Lightning callback factory.

**Returns**

- (`CallbackSpec`) — Portable callback specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L127-L135)

## nexuml.core.factory.factory_target

*function*

```python
def factory_target(factory: object, *, label: str = 'factory') -> str
```

Return and verify the portable target for an importable callable.

**Returns**

- (`str`) — Portable ``module:name`` target.

**Raises**

- `TypeError` — If *factory* is not callable.
- `ValueError` — If *factory* cannot be re-imported safely.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L88-L121)

## nexuml.core.factory.factory_values

*function*

```python
def factory_values(factory: object, args: tuple[object, ...], kwargs: Mapping[str, object], *, label: str) -> dict[str, object]
```

Build validated Pydantic field values for a factory helper.

**Returns**

- (`dict[str, object]`) — Factory target and constructor arguments.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L195-L211)

## nexuml.core.factory.normalize_json_value

*function*

```python
def normalize_json_value(value: Any, path: str) -> JsonValue
```

Normalize one constructor value to portable JSON data.

**Returns**

- (`JsonValue`) — Normalized JSON value.

**Raises**

- `ValueError` — If the value cannot be represented portably.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L29-L54)

## nexuml.core.factory.optimizer

*function*

```python
def optimizer(factory: Callable[Concatenate[Any, P], Optimizer], *args: P.args, **kwargs: P.kwargs) -> OptimizerSpec
```

Configure an optimizer factory; model parameters are supplied at runtime.

**Returns**

- (`OptimizerSpec`) — Portable optimizer specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L138-L148)

## nexuml.core.factory.resolve_factory

*function*

```python
def resolve_factory(target: str, *, label: str = 'factory') -> Callable[..., object]
```

Resolve a portable ``module:name`` target.

**Returns**

- (`Callable[..., object]`) — Imported callable.

**Raises**

- `TypeError` — If the imported target is not callable.
- `ValueError` — If the target is malformed or cannot be imported.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L57-L85)

## nexuml.core.factory.scheduler

*function*

```python
def scheduler(factory: Callable[Concatenate[Optimizer, P], LRScheduler], *args: P.args, **kwargs: P.kwargs) -> SchedulerSpec
```

Configure a scheduler factory; its optimizer is supplied at runtime.

**Returns**

- (`SchedulerSpec`) — Portable scheduler specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L151-L163)

## nexuml.core.factory.strategy

*function*

```python
def strategy(factory: Callable[P, Strategy], *args: P.args, **kwargs: P.kwargs) -> StrategySpec
```

Configure an importable Lightning strategy factory.

**Returns**

- (`StrategySpec`) — Portable strategy specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L166-L174)

## nexuml.core.factory.writer

*function*

```python
def writer(factory: Callable[P, ExportBackend], *args: P.args, **kwargs: P.kwargs) -> WriterSpec
```

Configure an importable dataset export backend factory.

**Returns**

- (`WriterSpec`) — Portable writer specification.

**Raises**

- `TypeError` — If *factory* is not an export backend class.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L177-L192)
