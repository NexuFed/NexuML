# Add A Custom Layer

A custom layer has one public immutable definition and, when runtime state is needed, one private `PipelineLayer` implementation in the same module.

## Define The Component

```python
import torch

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("scaled_relu")
class ScaledReLU(LayerDefinition):
    """Scale a ReLU activation."""

    scale: float = 1.0

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _ScaledReLURuntime(scale=self.scale, **context.runtime_kwargs())


class _ScaledReLURuntime(PipelineLayer):
    def __init__(self, scale: float, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def forward_tensor(self, x: torch.Tensor, y=None) -> torch.Tensor:
        return torch.relu(x) * self.scale
```

`ScaledReLU` declares semantic configuration. `_ScaledReLURuntime` owns mutable execution state. Input shapes, TensorDict keys, labels, shared storage, and scheduling come from `LayerBuildContext`, not definition fields.

Pydantic validates definition values immediately:

```python
ScaledReLU(scale=2.0)
ScaledReLU.model_json_schema()
```

## Use It In Python

```python
from nexuml.core.types import LayerSpec
from my_library.layers.scaled_relu import ScaledReLU

layer = LayerSpec(
    component=ScaledReLU(scale=2.0),
    keys_in=["features"],
    keys_out=["activated"],
)
```

Do not look the component up by its decorator name in Python. The registered name is for discovery, CLI inspection, and persisted YAML identity.

## Register The Library

For local development:

```bash
nexuml library add /path/to/my_library
nexuml registry list layers
```

For an installed package:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

The registry output reports `scaled_relu`, version `1`, its concrete type, and the `scale` field sourced from `model_json_schema()`.

## Persistence

A resolved config stores stable identity rather than a module path:

```yaml
component:
  type: scaled_relu
  version: '1'
  params:
    scale: 2.0
```

The library must be discoverable when that YAML is restored.

## See Also

- [Discovery decorators](../reference/decorators.md)
- [Custom data source](custom-data-source.md)
- [Custom library](custom-library.md)
