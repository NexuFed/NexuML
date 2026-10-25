# nexuml.core.scenario_registry

*module*

Scenario registry with decorator-based discovery.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L1-L1)

## nexuml.core.scenario_registry.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L17-L17)

## nexuml.core.scenario_registry.ScenarioRegistry

*class*

```python
class ScenarioRegistry
```

Registry for scenario functions with automatic discovery.

Supports only explicitly decorated scenario functions.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L20-L88)

### nexuml.core.scenario_registry.ScenarioRegistry.errors

*property*

```python
errors: builtins.list[DiscoveryError]
```

Discovery failures from the last scan (import + registration).

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L32-L35)

### nexuml.core.scenario_registry.ScenarioRegistry.ensure_loaded

*method*

```python
def ensure_loaded() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L86-L88)

### nexuml.core.scenario_registry.ScenarioRegistry.get

*method*

```python
def get(key: str) -> Callable[[], ScenarioSpec]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L50-L55)

### nexuml.core.scenario_registry.ScenarioRegistry.list

*method*

```python
def list() -> dict[str, Callable[[], ScenarioSpec]]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L57-L59)

### nexuml.core.scenario_registry.ScenarioRegistry.register

*method*

```python
def register(key: str, fn: Callable[[], ScenarioSpec]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L37-L48)

### nexuml.core.scenario_registry.ScenarioRegistry.scan

*method*

```python
def scan(package_paths: builtins.list[str] | None = None) -> None
```

Scan packages for scenario functions.

Collects only explicitly decorated scenarios via Scanner.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L61-L84)

## nexuml.core.scenario_registry.get_scenario_registry

*function*

```python
def get_scenario_registry() -> ScenarioRegistry
```

Return the default global ScenarioRegistry, creating it if needed.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/core/scenario_registry.py#L94-L99)
