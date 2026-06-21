# Automatic batch size

NexuML can automatically find the largest training batch size that fits in GPU memory using binary search with CUDA forward/backward probing.

!!! warning "CUDA required"
    Automatic batch-size probing requires a CUDA-capable GPU. Attempting it on CPU raises a `RuntimeError`.

## Prerequisites

- NexuML installed (`uv sync`)
- CUDA-capable GPU
- `data.loader.batch_size` must **not** be set (an explicit loader batch size disables auto-probing)

## Basic configuration

```python
from nexuml.core.types import ScenarioSpec, TrainingSpec, AutoBatchSizeSpec

ScenarioSpec(
    name="my_scenario",
    training=TrainingSpec(
        batch_size=AutoBatchSizeSpec(
            mode="auto",
            min=8,
            max=256,
            candidates="power_of_two",
            safety="previous_power_of_two",
            margin=0.8,
        ),
        lr=1e-3,
        max_epochs=50,
        loss_keys={"reconstruction_loss": 1.0},
    ),
    ...
)
```

## `AutoBatchSizeSpec` fields

| Field | Type | Default (spec) | Description |
|---|---|---|---|
| `mode` | `"auto"` | `"auto"` | Must be `"auto"` to enable probing |
| `min` | `int` | `1` | Minimum candidate batch size |
| `max` | `int` | `128` | Maximum candidate batch size |
| `candidates` | `"power_of_two"` | `"power_of_two"` | Candidate generation strategy |
| `safety` | `"largest"` \| `"previous_power_of_two"` \| `"margin"` | `"previous_power_of_two"` | Selection policy after probing |
| `margin` | `float (0,1]` | `0.8` | Memory fraction threshold (used only with `safety="margin"`) |

## How probing works

1. NexuML generates power-of-two candidates between `min` and `max` (e.g. `[8, 16, 32, 64, 128, 256]`).
2. Starting from the largest candidate, it runs a single forward + backward pass.
3. If CUDA out-of-memory (OOM) occurs, it backs off to the next smaller candidate.
4. The selected batch size is determined by `safety`:
   - `"largest"` — use the largest candidate that did not OOM.
   - `"previous_power_of_two"` — use one step smaller than the largest that fit (leaves headroom for longer sequences or larger validation batches).
   - `"margin"` — use the largest candidate whose peak memory is below `margin × GPU_total`.

## Safety policy comparison

| Policy | Behaviour | When to use |
|---|---|---|
| `"largest"` | Maximum throughput, no headroom | Fixed-length inputs, tight GPU |
| `"previous_power_of_two"` | One step smaller than largest | Variable-length sequences, safer default |
| `"margin"` | Peak memory ≤ `margin × total` | When you have a memory estimate to target |

## Defaults: spec vs. library

The `AutoBatchSizeSpec` defaults and the library's `DEFAULT_AUTO_BATCH_SIZE` differ:

| Setting | `AutoBatchSizeSpec` default | `DEFAULT_AUTO_BATCH_SIZE` (library) |
|---|---|---|
| `min` | `1` | `8` |
| `max` | `128` | `128` |
| `safety` | `"previous_power_of_two"` | `"margin"` |
| `margin` | `0.8` | `0.8` |

`DEFAULT_AUTO_BATCH_SIZE` is defined in `nexuml_library.scenarios.training.defaults` and is the value used by pre-built library scenarios when no explicit `batch_size` is specified. If you define your own `AutoBatchSizeSpec`, the spec defaults apply.

## Precedence

An explicit loader batch size always takes precedence and disables auto-probing:

```python
from nexuml.core.types import LoaderSpec

# This DISABLES auto-probing even if training.batch_size is AutoBatchSizeSpec
DataSpec(loader=LoaderSpec(batch_size=64))
```

Set `data.loader.batch_size=None` (the default) to allow probing.

## Example

```python
from nexuml.core.types import (
    ScenarioSpec, TrainingSpec, AutoBatchSizeSpec, DataSpec, LoggingSpec
)

ScenarioSpec(
    name="auto_batch_demo",
    data=DataSpec(
        source_type="synthetic",
        params={"feature_shape": [128], "num_samples": 2000},
    ),
    training=TrainingSpec(
        lr=1e-3,
        max_epochs=10,
        batch_size=AutoBatchSizeSpec(mode="auto", min=8, max=512, safety="margin"),
        loss_keys={"reconstruction_loss": 1.0},
    ),
)
```

Run:

```bash
nexuml train --scenario-file auto_batch_demo.py
```

Expected output during startup:

```
Probing batch sizes: [8, 16, 32, 64, 128, 256, 512]
  batch_size=512: OOM
  batch_size=256: OK (peak 14.2 GB / 16.0 GB)
  batch_size=128: OK (peak 7.3 GB / 16.0 GB)
Selected batch size: 128 (margin policy, 0.8 threshold)
```

## Implementation map

- `src/nexuml/data/auto_batch.py` — CUDA probing logic
- `src/nexuml/core/types.py` — `AutoBatchSizeSpec`
- `library/src/nexuml_library/scenarios/training/defaults.py` — `DEFAULT_AUTO_BATCH_SIZE`

## See also

- [Train a model](train.md)
- [CLI lifecycle](cli-lifecycle.md)
