# Registry inspection

NexuML uses discovery for two reasons: CLI inspection/recipe lookup and restoration of stable component identities from persisted configuration. Normal Python scenarios import and instantiate concrete definition classes directly.

## List registered components

```bash
nexuml registry list layers
nexuml registry list data
nexuml registry list eval
nexuml registry list scenarios
```

Component rows include the stable name/version identity, concrete definition type, and Pydantic fields. Scenario rows show the registered recipe function.

Use verbose mode to surface discovery/import failures:

```bash
nexuml registry list layers --verbose
```

## Correct extension shape

A layer registry entry is a `LayerDefinition`, not a mutable `PipelineLayer` runtime:

```python
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("scaled_relu")
class ScaledReLU(LayerDefinition):
    scale: float = 1.0

    def build(self, context: LayerBuildContext):
        return _ScaledReLURuntime(scale=self.scale, **context.runtime_kwargs())
```

Python then uses `ScaledReLU(scale=2.0)` directly. The `scaled_relu@1` identity is used for discovery/YAML persistence.

## Loader and other runtime backends

Use `nexuml backend list` for loader/export/training/tracking backend catalogs. `nexuml registry list` intentionally focuses on the user-facing layer/data/eval/scenario commands.

## Troubleshooting a missing component

1. Check the library source:
   ```bash
   nexuml library list
   ```
2. Check the relevant registry with `--verbose`.
3. Confirm the definition class/function is decorated with the correct role decorator.
4. Confirm the module and optional dependencies can import.
5. For a local package, confirm its root is registered with `nexuml library add PATH`.

Discovery runs fresh for each process; there is no persistent component cache.

## See also

- [Decorators](decorators.md)
- [Library discovery](../explanation/library-discovery.md)
- [Manage local library roots](../how-to/library-cli.md)
