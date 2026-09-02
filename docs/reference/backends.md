# Backends

NexuML uses multiple independent backend registries — one per concern. There is no single unified "backend" concept. Use `nexuml backend list` to see what is available in your environment.

```bash
nexuml backend list
```

```
                          Available Backends
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Category        ┃ Name              ┃ Implementation                         ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ data-export     │ numpy             │ nexuml.data.export.numpy_files         │
│ data-export     │ numpy_mmap        │ nexuml.data.export.numpy_mmap          │
│ data-export     │ tensordict_memmap │ nexuml.data.export.tensordict_memmap   │
│ data-export     │ torch             │ nexuml.data.export.torch_files         │
│ data-export     │ webdataset        │ nexuml.data.export.webdataset          │
│ data-loader     │ dali              │ nexuml.data.loaders.dali_backend       │
│ data-loader     │ torch             │ nexuml.data.loaders.torch_backend      │
│ eval-storage    │ memmap            │ nexuml.evaluation.storage              │
│ eval-storage    │ memory            │ nexuml.evaluation.storage              │
│ pipeline-export │ onnx              │ nexuml.core.export.export_onnx         │
│ pipeline-export │ package           │ nexuml.core.export.export_package      │
│ pipeline-export │ safetensors       │ nexuml.core.export.export_safetensors  │
│ tracking        │ dvclive           │ nexuml.tracking.logger                 │
│ tracking        │ mlflow            │ nexuml.tracking.logger                 │
│ tracking        │ tensorboard       │ nexuml.tracking.logger                 │
│ training        │ lightning         │ nexuml.training.lightning.NexuSession  │
└─────────────────┴───────────────────┴────────────────────────────────────────┘
```

## Backend categories

### `data-export` — dataset export backends

Selected with `--backend` on `nexuml export-dataset`.

| Name | Description |
|---|---|
| `numpy` | One `.npy` file per key per sample. Portable and readable everywhere. |
| `numpy_mmap` | Memory-mapped `.npy` files. Efficient for large datasets. |
| `torch` | PyTorch `.pt` tensors per sample. |
| `tensordict_memmap` | TensorDict memory-mapped format. Best for TensorDict-native pipelines. |
| `webdataset` | WebDataset `.tar` shards. Best for streaming and distributed training. |

```bash
nexuml export-dataset my-scenario --backend numpy_mmap --output ./cache/
```

Default backend: `numpy`.

### `data-loader` — dataloader backends

Selected via `data.loader.backend` in `DataSpec.loader` (`LoaderSpec`).

| Name | Description |
|---|---|
| `torch` | Standard PyTorch `DataLoader`. Always available. |
| `dali` | NVIDIA DALI accelerated loader. Registered only if the DALI import succeeds. |

```python
from nexuml.core.types import DataSpec, LoaderSpec
from nexuml.data.loaders.definitions import TorchLoader
from my_library.data import MyDataset

DataSpec(
    source=MyDataset(),
    loader=LoaderSpec(backend=TorchLoader(), num_workers=4),
)
```

!!! note "DALI availability"
    The `dali` data-loader backend is registered only when `nexuml.data.loaders.dali_backend` can be imported. If DALI is not installed, the backend is silently absent — it does not appear in `nexuml backend list`. Install DALI separately following NVIDIA's instructions and then verify with `nexuml backend list`.

### `training` — training backends

| Name | Description |
|---|---|
| `lightning` | PyTorch Lightning `NexuSession`. The only implemented training backend. |

The training backend is selected automatically. `NexuSession` wraps the compiled pipeline as a `LightningModule` and drives training with the Lightning `Trainer`.

### `tracking` — experiment tracking backends

Selected via `LoggingSpec` fields in the scenario.

| Name | Config field | Description |
|---|---|---|
| `tensorboard` | `logging.tensorboard` | TensorBoard scalar logging |
| `dvclive` | `logging.dvclive` | DVCLive metrics and plots |
| `mlflow` | `logging.mlflow` | MLflow experiment tracking (runs, artifacts, params) |

See [Tracking and logging](../how-to/tracking.md) for configuration details.

### `eval-storage` — evaluation result storage backends

Selected via `DistanceEstimatorSpec.storage_backend`.

| Name | Description |
|---|---|
| `memory` | In-memory storage. Fast but lost after the process exits. |
| `memmap` | Memory-mapped file. Persists to disk and survives restarts. |

```python
from nexuml.core.types import DistanceEstimatorSpec

DistanceEstimatorSpec(storage_backend="memmap", storage_path=".experiments/eval_storage/")
```

### `pipeline-export` — trained pipeline export backends

| Name | Function | Description |
|---|---|---|
| `package` | `export_package` | Default: state dict + config YAML + metadata JSON |
| `safetensors` | `export_safetensors` | SafeTensors format for weights |
| `onnx` | `export_onnx` | ONNX for cross-framework inference |

See [Model export and reload](../how-to/export.md).

## Custom backends

### Custom data-export backend

```python
from nexuml.data.export import register_export_backend, ExportBackend

class MyExportBackend(ExportBackend):
    def write(self, key: str, tensor, split: str, index: int, output_dir: str) -> None:
        ...

register_export_backend("my_backend", MyExportBackend)
```

After registration, use `--backend my_backend` with `nexuml export-dataset`.

### Custom data-loader backend

```python
from nexuml.core.components import LoaderBackendDefinition
from nexuml.core.discovery import loader_backend
from nexuml.data.loaders import LoaderBackend

@loader_backend("my_loader")
class MyLoader(LoaderBackendDefinition):
    queue_depth: int = 2

    def build(self) -> LoaderBackend:
        return _MyLoaderBackend(queue_depth=self.queue_depth)
```

Use via `LoaderSpec(backend=MyLoader(queue_depth=4))`.

!!! note "Kubernetes training"
    No Kubernetes training-execution backend is implemented. The `training` category exposes only `lightning`. If a future change implements Kubernetes execution, this page will be updated.

## Implementation map

- `src/nexuml/data/export/` — data-export backend implementations
- `src/nexuml/data/loaders/` — data-loader backend implementations
- `src/nexuml/training/lightning.py` — `NexuSession` (training backend)
- `src/nexuml/tracking/logger.py` — tracking backends
- `src/nexuml/evaluation/storage.py` — eval-storage backends
- `src/nexuml/core/export.py` — pipeline-export backend functions

## See also

- [Dataset export](../how-to/export-dataset.md)
- [Model export and reload](../how-to/export.md)
- [Tracking and logging](../how-to/tracking.md)
