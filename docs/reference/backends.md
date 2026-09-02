# Backends

`nexuml backend list` is the runtime catalog for several independent extension points. "Backend" is not one universal interface; each category belongs to a different concern.

```bash
nexuml backend list
nexuml backend list data-loader
nexuml backend list data-export
```

!!! important "Catalog vs dependency health"
    The command lists registered backend definitions/implementations. A listed optional backend can still require a third-party package at materialization time. For example, `DaliLoader` is registered even when `nvidia.dali` is not importable.

## Data loaders

| Name | Role |
| --- | --- |
| `torch` | standard PyTorch loading |
| `dali` | NVIDIA DALI loader; optional platform-specific runtime |
| `tensor_shards` | windowed loading from materialized tensor shards |

See [Data loading](../how-to/data-loading.md).

## Dataset export

Built-in data-export names currently include:

- `numpy`
- `numpy_mmap`
- `torch`
- `tensordict_memmap`
- `webdataset`
- `tensor_shards`

See [Dataset export](../how-to/export-dataset.md).

## Training / execution

The catalog exposes `lightning` and `ray` in the training category. Conceptually, Lightning remains the canonical session/training lifecycle and Ray changes distributed placement/setup around that lifecycle.

See [Execution modes](../how-to/training-backends/index.md).

## Tracking

The catalog includes TensorBoard, DVCLive, and MLflow integrations. Their dependencies/configuration are documented in [Tracking and logging](../how-to/tracking.md).

## Pipeline export

The catalog includes:

- `package` — primary NexuML train-package artifact;
- `safetensors` — weight export;
- `onnx` — inference graph export for supported pipelines.

See [Export and reload](../how-to/export.md).

## Evaluation temporary storage

The current alpha code has a naming inconsistency: the backend catalog/internal storage factory uses `memory`, while `DistanceEstimatorSpec.storage_backend` currently declares the literal `ram` alongside `memmap`. Treat that spelling as unstable until the implementation is unified rather than building new user configuration around the mismatch.

The generated Python API is the exact reference for the currently installed version.
