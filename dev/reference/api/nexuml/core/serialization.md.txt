# nexuml.core.serialization

*module*

Generic component lowering and restoration at config boundaries.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/serialization.py#L1-L1)

## nexuml.core.serialization.lower_component

*function*

```python
def lower_component(component: ComponentDefinition) -> dict[str, Any]
```

Lower a concrete definition to its stable serialized identity.

**Returns**

- (`dict[str, Any]`) — Portable component identity and validated parameters.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/serialization.py#L16-L27)

## nexuml.core.serialization.lower_model

*function*

```python
def lower_model(model: BaseModel) -> dict[str, Any]
```

Recursively lower a Pydantic model and all component definitions.

**Returns**

- (`dict[str, Any]`) — Portable plain model data.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/serialization.py#L51-L57)

## nexuml.core.serialization.restore_component

*function*

```python
def restore_component(kind: str, value: Mapping[str, Any]) -> ComponentDefinition
```

Restore a concrete definition from stable serialized data.

**Returns**

- (`ComponentDefinition`) — Validated concrete component definition.

**Raises**

- `TypeError` — If serialized parameters are not a mapping.
- `ValueError` — If identity fields are missing or invalid.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/serialization.py#L30-L48)

## nexuml.core.serialization.restore_model_data

*function*

```python
def restore_model_data(data: Mapping[str, Any], model_type: type[BaseModel]) -> dict[str, Any]
```

Restore definitions, preserving full document locations on component validation errors.

**Returns**

- (`dict[str, Any]`) — Model data containing concrete component definition instances.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/serialization.py#L74-L89)
