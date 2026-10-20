# Build A Custom Library

A NexuML library is an importable package containing typed component definitions and scenario recipes.

```text
my_library/
├── pyproject.toml
└── src/my_library/
    ├── __init__.py
    ├── layers/
    ├── data/
    ├── evaluation/
    └── scenarios/
```

## Components

Create public definitions decorated with stable identities:

```python
@layer("my_encoder")
class MyEncoder(LayerDefinition):
    width: int = 64

    def build(self, context: LayerBuildContext):
        return _MyEncoderRuntime(width=self.width, **context.runtime_kwargs())
```

Use `DataSourceDefinition`, `EvalAlgorithmDefinition`, and `LoaderBackendDefinition` for their respective roles. Mutable tensors, modules, datasets, and evaluation accumulators belong on private runtime classes.

Python scenarios import definitions directly:

```python
LayerSpec(
    component=MyEncoder(width=128),
    keys_in=["features"],
    keys_out=["encoded"],
)
```

The decorator identity is used for discovery, CLI inspection, and YAML restoration, not normal Python construction.

## Discovery

Follow [Register a library](register-library.md) to declare the package entry point,
install it and verify discovery. For an uninstalled source tree, use
[local library roots](library-cli.md) instead; registration does not install dependencies.

The registered path can be the package directory itself (containing `__init__.py`),
a directory containing packages, or a project with packages under `src/`.
Registering a package preserves its qualified module names and relative imports;
for example, `library/config/` is imported as `library.config`, not top-level `config`.

## Persistence

Resolved YAML stores exact kind-specific identities as `type`, `version`, and `params`. It never stores the concrete module path. The package or local root must therefore be discoverable when YAML is restored.

## Guides

- [Add a custom layer](custom-layer.md)
- [Add a custom data source](custom-data-source.md)
- [Add a custom eval algorithm](custom-eval-algorithm.md)
- [Decorators and discovery](../learn/decorators-and-discovery.md)
