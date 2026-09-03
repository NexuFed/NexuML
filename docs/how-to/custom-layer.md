# Add A Custom Layer

Use `nn_module(...)` for an ordinary importable PyTorch module with one tensor input and one tensor output. Register a typed `LayerDefinition` only when the layer needs NexuML-specific context, labels, metadata, lifecycle behavior, or richer routing.

## Prefer A Direct Module

```python
import torch

from nexuml import nn_module
from nexuml.core.types import LayerSpec

layer = LayerSpec(
    component=nn_module(torch.nn.Dropout, p=0.5),
    keys_in=["features"],
    keys_out=["regularized"],
)
```

The factory symbol remains navigable and needs no component decorator or custom library registration. `nn_module` uses `ParamSpec`, so static checkers can validate constructor arguments when the factory provides annotations. NexuML does not inspect the constructor at runtime.

Resolved YAML uses the one core `NnModule` identity:

```yaml
component:
  type: NnModule
  version: '1'
  params:
    factory: torch.nn.modules.dropout:Dropout
    args: []
    kwargs:
      p: 0.5
```

The factory must be a top-level importable class or function, and its dependency must be installed when the config is compiled. Arguments may contain only null, booleans, integers, finite floats, strings, lists or tuples, and string-key mappings composed from those values. Live module instances, lambdas, closures, local definitions, tensors, devices, dtypes, callables, and other process-local values are rejected.

Resolved config is trusted input: compiling it imports and invokes the recorded Python factory. Syntax validation does not make external code safe.

The old `IdentityLayer`, `Dropout`, and `Flatten` component identities are intentionally removed. Use `nn_module(torch.nn.Identity)`, `nn_module(torch.nn.Dropout, ...)`, and `nn_module(torch.nn.Flatten, start_dim=1, end_dim=-1)`.

## Register Richer Behavior

A registered custom layer has one public immutable definition and, when runtime state is needed, one private `PipelineLayer` implementation in the same module.

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
