# nexuml.cli.overrides

*module*

CLI override utilities: apply key.path=value overrides to a ScenarioSpec.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/overrides.py#L1-L1)

## nexuml.cli.overrides.console

*attribute* · *module attribute*

```python
console = Console()
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/overrides.py#L11-L11)

## nexuml.cli.overrides.apply_overrides

*function*

```python
def apply_overrides(scenario: Any, overrides: list[str]) -> Any
```

Apply ``key.path=value`` overrides to a ScenarioSpec in-place.

Dot-separated paths traverse Pydantic model attributes and dict keys.
Values are coerced to int/float/bool/list before assignment.
Prints a Rich table summarising each change.

**Returns**

- (`Any`) — The mutated *scenario* instance.

**Raises**

- `ValueError` — If an override string is malformed or has an empty path segment.
- `KeyError` — If a path segment does not exist on the scenario.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/overrides.py#L40-L98)
