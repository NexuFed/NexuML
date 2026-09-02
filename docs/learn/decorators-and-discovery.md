# Decorators And Discovery

NexuML decorators attach stable persisted identities to typed definitions. Python code imports and constructs the definition class directly; discovery is needed for CLI listing and for restoring YAML.

## Component Decorators

`@layer`, `@data_source`, and `@eval_algorithm` decorate the corresponding definition role:

```python
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("scaled_relu")
class ScaledReLU(LayerDefinition):
    scale: float = 1.0

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _ScaledReLURuntime(scale=self.scale, **context.runtime_kwargs())


class _ScaledReLURuntime(PipelineLayer):
    def __init__(self, scale: float, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def forward_tensor(self, x, y=None):
        return x.relu() * self.scale
```

Use the class directly in Python:

```python
LayerSpec(
    component=ScaledReLU(scale=2.0),
    keys_in=["features"],
    keys_out=["activated"],
)
```

The decorator registers `scaled_relu@1`, allowing YAML restoration without persisting a Python import path.

## Scenario Decorator

`@scenario` remains a recipe registry for CLI lookup:

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec


@scenario("my-experiment")
def my_experiment() -> ScenarioSpec:
    return ScenarioSpec(name="my-experiment")
```

## Discovery Sources

Each CLI run scans fresh from:

1. The built-in `nexuml_library` package.
2. Installed packages declaring a `nexuml.libraries` entry point.
3. Local roots configured with `nexuml library add <path>`.

One broken module is recorded as a `DiscoveryError`; unrelated valid components remain available.

## Verify Discovery

```bash
nexuml registry list layers
nexuml registry list data
nexuml registry list eval
nexuml registry list scenarios
```

The component commands show the stable name, version, concrete type, and Pydantic fields. Use `--verbose` to inspect discovery failures.

For distribution:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

## See Also

- [Decorator reference](../reference/decorators.md)
- [Add a custom layer](../how-to/custom-layer.md)
- [Add a custom data source](../how-to/custom-data-source.md)
- [Add a custom eval algorithm](../how-to/custom-eval-algorithm.md)
