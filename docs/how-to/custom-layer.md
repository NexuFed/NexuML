# Add a custom layer

This guide walks through adding a custom `PipelineLayer` to a library and using it from a `ScenarioSpec`.

## Prerequisites

- NexuML installed (`uv sync`)
- A library root registered with `nexuml library add` or installed via entry-point
- For a new library: see [Register a library](register-library.md)

## 1. Create the layer file

Place the layer under `layers/<category>/` inside your library package:

```
my_library/
└── src/
    └── my_library/
        ├── __init__.py
        └── layers/
            ├── __init__.py
            └── feature/
                ├── __init__.py
                └── my_normalizer.py
```

```python
# my_library/src/my_library/layers/feature/my_normalizer.py
import torch
from nexuml.core.discovery import layer
from nexuml.core.base_layer import PipelineLayer


@layer("my_normalizer")
class MyNormalizer(PipelineLayer):
    """L2-normalize a feature tensor."""

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(features, dim=-1)
```

## 2. Expose through package imports (optional)

If you want IDE completion and clean imports, re-export from `__init__.py`:

```python
# my_library/src/my_library/layers/__init__.py
from my_library.layers.feature.my_normalizer import MyNormalizer

__all__ = ["MyNormalizer"]
```

The `@layer` decorator attaches metadata to the class regardless of import path. Discovery scans module-by-module so you don't need explicit re-exports for registration to work — they help IDEs.

## 3. Register the local root

If your library is not installed as a package:

```bash
nexuml library add my_library
```

Or install it:

```bash
uv pip install --link-mode=copy -e my_library
```

## 4. Verify registration

```bash
nexuml registry list layers
```

You should see `my_normalizer` in the output:

```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Type Key       ┃ Module                                                  ┃ Constructor Params ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ my_normalizer  │ my_library.layers.feature.my_normalizer                 │                    │
└────────────────┴─────────────────────────────────────────────────────────┴────────────────────┘
```

If the layer doesn't appear, run with `--verbose` to see import errors:

```bash
nexuml registry list layers --verbose
```

## 5. Use in a scenario

Reference the layer by `type_key` in a `LayerSpec`:

```python
from nexuml.core.types import ScenarioSpec, PipelineSpec, LayerSpec, DataSpec, TrainingSpec

ScenarioSpec(
    name="normalized_ae",
    data=DataSpec(source_type="synthetic", params={"feature_shape": [64], "num_samples": 500}),
    training=TrainingSpec(lr=1e-3, max_epochs=5, loss_keys={"reconstruction_loss": 1.0}),
    pipeline=PipelineSpec(stages={
        "preprocess": [
            LayerSpec(
                type_key="my_normalizer",
                keys_in=["features"],
                keys_out=["features"],
            ),
        ],
        "encode": [
            LayerSpec(
                type_key="linear_encoder",
                keys_in=["features"],
                keys_out=["z"],
                params={"input_dim": 64, "output_dim": 8},
            ),
        ],
    }),
)
```

Or train from a trusted file:

```bash
nexuml train --scenario-file scenario.py
```

## Layer constructor parameters

If your layer needs constructor arguments, declare them in `__init__`:

```python
@layer("scaled_relu")
class ScaledReLU(PipelineLayer):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) * self.scale
```

Pass them via `LayerSpec.params`:

```python
LayerSpec(type_key="scaled_relu", keys_in=["x"], keys_out=["x"], params={"scale": 2.0})
```

## Expected output

After running `nexuml train --scenario-file scenario.py`, training proceeds normally with your custom layer in the pipeline.

## Implementation map

- `src/nexuml/core/discovery.py` — `@layer` decorator, `Scanner`
- `src/nexuml/core/base_layer.py` — `PipelineLayer` base class
- `src/nexuml/core/registry.py` — layer registry

## See also

- [Discovery decorators](../reference/decorators.md)
- [Custom composed scenario](custom-scenario.md)
- [Registry inspection](../reference/registry.md)
