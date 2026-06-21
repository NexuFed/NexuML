# Decorators and discovery

NexuML uses a decorator-based registry system. You mark Python classes or functions with decorators, and NexuML discovers and loads them automatically from installed packages.

## The four decorators

### `@scenario("key")`

Makes a scenario available to CLI commands by name.

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec

@scenario("my-classifier")
def my_classifier() -> ScenarioSpec:
    return ScenarioSpec(...)
```

After installation, `nexuml registry list scenarios` will show `my-classifier`, and `nexuml resolve my-classifier` will call this function.

### `@layer("key")`

Registers a `PipelineLayer` class so it can be referenced by `LayerSpec(type_key="key")`.

```python
from nexuml.core.discovery import layer
from nexuml.core.pipeline import PipelineLayer

@layer("my-backbone")
class MyBackbone(PipelineLayer):
    def forward(self, x):
        ...
```

In a scenario:

```python
LayerSpec(type_key="my-backbone", keys_in={"x": "image"}, keys_out=["features"])
```

### `@data_source("key")`

Registers a dataset source so it can be referenced by `DataSpec(source_type="key")`.

```python
from nexuml.core.discovery import data_source

@data_source("my-dataset")
class MyDataset:
    ...
```

In a scenario:

```python
DataSpec(source_type="my-dataset")
```

### `@eval_algorithm("key")`

Registers an evaluation algorithm so it can be referenced by `EvaluationSpec(type="key")`.

```python
from nexuml.core.discovery import eval_algorithm

@eval_algorithm("classification")
class ClassificationEval:
    ...
```

In a scenario:

```python
EvaluationSpec(type="classification")
```

## How discovery works

When a NexuML CLI command runs, it scans for registered components from three sources:

1. **Installed packages via entry points.** Any Python package that declares a `nexuml.libraries` entry point group is scanned. This is how the base library (`nexuml_library`) is discovered.

2. **Local library roots.** Directories added with `nexuml library add <path>` are scanned for packages. Useful during development before packaging.

3. **Direct imports.** Importing a module with decorated classes or functions registers them immediately.

## Verify discovery

After installing or adding a library, verify that its components are discovered:

```bash
# List registered scenarios
nexuml registry list scenarios

# List registered layers
nexuml registry list layers

# List registered data sources
nexuml registry list data

# List registered eval algorithms
nexuml registry list eval
```

If a component is missing, check that:
- The package is installed in the same environment as `nexuml`.
- The package declares the `nexuml.libraries` entry point (for installed packages).
- The local root path is correctly added (`nexuml library list` to verify).

## Development workflow

During development, before packaging a library, use local roots:

```bash
nexuml library add /path/to/my-library/src
nexuml registry list scenarios   # should now include your scenarios
```

When ready to distribute, add the entry point to `pyproject.toml`:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

## Next steps

- [Decorator reference](../reference/decorators.md) — compact reference for all decorators
- [Library discovery explanation](../explanation/library-discovery.md) — deep dive on the discovery mechanism
- [Add a custom layer](../how-to/custom-layer.md) — implement and register a layer
- [Add a custom data source](../how-to/custom-data-source.md)
- [Register a library](../how-to/register-library.md) — entry-point distribution
- [Manage local library roots](../how-to/library-cli.md) — `nexuml library add/delete/list`
- [Registry inspection reference](../reference/registry.md)
