# nexuml.core.registry

*module*

Identity registry for typed component definitions.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L1-L1)

## nexuml.core.registry.ComponentEntry

*class* · *dataclass*

```python
class ComponentEntry
```

Registered stable identity and its concrete definition type.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L9-L17)

### nexuml.core.registry.ComponentEntry.definition_type

*attribute* · *instance attribute*

```python
definition_type: type[ComponentDefinition]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L16-L16)

### nexuml.core.registry.ComponentEntry.import_target

*attribute* · *instance attribute*

```python
import_target: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L17-L17)

### nexuml.core.registry.ComponentEntry.kind

*attribute* · *instance attribute*

```python
kind: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L13-L13)

### nexuml.core.registry.ComponentEntry.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L14-L14)

### nexuml.core.registry.ComponentEntry.version

*attribute* · *instance attribute*

```python
version: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L15-L15)

## nexuml.core.registry.ComponentRegistry

*class*

```python
class ComponentRegistry
```

Map stable component identities to concrete definition types.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L20-L134)

### nexuml.core.registry.ComponentRegistry.errors

*property*

```python
errors: list[DiscoveryError]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L30-L32)

### nexuml.core.registry.ComponentRegistry.ensure_loaded

*method*

```python
def ensure_loaded() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L132-L134)

### nexuml.core.registry.ComponentRegistry.entries

*method*

```python
def entries(kind: str | None = None) -> tuple[ComponentEntry, ...]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L101-L106)

### nexuml.core.registry.ComponentRegistry.entry_for_type

*method*

```python
def entry_for_type(definition_type: type[ComponentDefinition]) -> ComponentEntry
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L93-L99)

### nexuml.core.registry.ComponentRegistry.get_entry

*method*

```python
def get_entry(kind: str, name: str, version: str = '1') -> ComponentEntry
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L78-L88)

### nexuml.core.registry.ComponentRegistry.get_type

*method*

```python
def get_type(kind: str, name: str, version: str = '1') -> type[ComponentDefinition]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L90-L91)

### nexuml.core.registry.ComponentRegistry.register

*method*

```python
def register(name: str, definition_type: type[ComponentDefinition], *, kind: str | None = None, version: str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L34-L76)

### nexuml.core.registry.ComponentRegistry.scan

*method*

```python
def scan(package_paths: list[str] | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L108-L130)

## nexuml.core.registry.get_component_registry

*function*

```python
def get_component_registry() -> ComponentRegistry
```

Return the process-wide component identity registry.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/registry.py#L140-L145)
