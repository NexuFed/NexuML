# nexuml.api.contracts

*module*

Static catalog descriptions from native definitions, never runtime construction.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L1-L1)

## nexuml.api.contracts.ComponentIO

*class*

```python
class ComponentIO(TypedDict)
```

Bases: `TypedDict`

Known input bindings and labels; unknown is distinct from known-empty.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L25-L30)

### nexuml.api.contracts.ComponentIO.inputs

*attribute* · *instance attribute*

```python
inputs: list[InputBinding]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L28-L28)

### nexuml.api.contracts.ComponentIO.inputs_known

*attribute* · *instance attribute*

```python
inputs_known: bool
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L29-L29)

### nexuml.api.contracts.ComponentIO.labels

*attribute* · *instance attribute*

```python
labels: list[str] | None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L30-L30)

## nexuml.api.contracts.InputBinding

*class*

```python
class InputBinding(TypedDict)
```

Bases: `TypedDict`

A definition-local key field or surrounding evaluation context slot.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L14-L22)

### nexuml.api.contracts.InputBinding.collection

*attribute* · *instance attribute*

```python
collection: bool
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L20-L20)

### nexuml.api.contracts.InputBinding.default

*attribute* · *instance attribute*

```python
default: str | list[str] | None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L22-L22)

### nexuml.api.contracts.InputBinding.domain

*attribute* · *instance attribute*

```python
domain: Literal['x', 'y']
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L19-L19)

### nexuml.api.contracts.InputBinding.field

*attribute* · *instance attribute*

```python
field: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L18-L18)

### nexuml.api.contracts.InputBinding.required

*attribute* · *instance attribute*

```python
required: bool
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L21-L21)

### nexuml.api.contracts.InputBinding.source

*attribute* · *instance attribute*

```python
source: Literal['parameter', 'context']
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L17-L17)

## nexuml.api.contracts.component_io

*function*

```python
def component_io(definition: type[ComponentDefinition]) -> ComponentIO
```

Describe public field annotations and shared context rules without instances.

**Returns**

- (`ComponentIO`) — Static bindings, contract completeness and fixed source labels.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/contracts.py#L50-L88)
