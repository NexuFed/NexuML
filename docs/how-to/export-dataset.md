# Export a dataset

`nexuml export-dataset` extracts a dataset view to disk for fast re-use, offline analysis, or training a different model on pre-extracted features.

## Prerequisites

- NexuML installed (`uv sync`)
- A registered scenario or config YAML
- Sufficient disk space for the export

## Raw dataset export

Export the raw (pre-pipeline) feature and label tensors:

```bash
nexuml export-dataset my-scenario \
  --output ./exported_data/ \
  --backend numpy \
  --split train \
  --split val
```

Full option reference:

| Option | Description |
|---|---|
| `SCENARIO_NAME` | Registered scenario name |
| `--config` / `-c PATH` | Config YAML path (alternative to scenario name) |
| `--output` / `-o PATH` | Export directory (default: `exported_dataset`) |
| `--backend TEXT` | Export backend name (default: `numpy`) |
| `--split TEXT` | Split to export (`train`, `val`, `test`). Repeatable. |
| `--x-key TEXT` | TensorDict x keys to persist. Repeatable. Default: all. |
| `--y-key TEXT` | Label TensorDict keys to persist. Repeatable. Default: all. |
| `--labels` / `--no-labels` | Include label TensorDict in the export (default: `--labels`) |
| `--dtype TEXT` | Optional storage dtype (e.g. `float16`) passed to the backend |

## Preprocessed dataset export

Run the compiled pipeline up to a preprocessing boundary and export the intermediate tensors:

```bash
nexuml export-dataset my-scenario \
  --output ./preprocessed_data/ \
  --backend numpy_mmap \
  --split train \
  --split val \
  --split test \
  --preprocess \
  --preprocess-until-key z
```

| Option | Description |
|---|---|
| `--preprocess` / `--no-preprocess` | Run the pipeline before exporting (default: `--no-preprocess`) |
| `--preprocess-until-key TEXT` | TensorDict key marking the preprocessing boundary. Repeatable. |

When `--preprocess` is set, NexuML runs the compiled pipeline with `forward_until` semantics: each layer is executed in order and processing stops once all `--preprocess-until-key` keys exist in the TensorDict. This lets you cache partially-processed representations (e.g. spectrogram features) without running the full model.

## Available export backends

| Backend | Description |
|---|---|
| `numpy` | One `.npy` file per key per sample |
| `numpy_mmap` | Memory-mapped `.npy` files for large datasets |
| `torch` | PyTorch `.pt` tensors |
| `tensordict_memmap` | TensorDict memory-mapped format |
| `webdataset` | WebDataset `.tar` shards |

Select with `--backend`. See [Backends](../reference/backends.md) for details.

## Export layout reference

After export, the output directory contains:

```
exported_data/
├── config.yaml          # ExportConfig: backend, x_keys, y_keys, dtype, splits
├── metadata.parquet     # Per-sample metadata (metadata.csv as fallback)
├── train/
│   └── data/            # Per-sample or per-shard files (layout is backend-specific)
├── val/
│   └── data/
└── test/
    └── data/
```

### `config.yaml` (ExportConfig)

```yaml
backend: numpy
x_keys: [features]
y_keys: [target]
key_specs:
  features: {dtype: float32, shape: [64]}
  target: {dtype: int64, shape: []}
extra:
  transform_applied: false      # true when --preprocess was used
splits:
  train:
    num_samples: 700
    label_prefix: "label__"
  val:
    num_samples: 150
```

Key fields:

| Field | Description |
|---|---|
| `backend` | Backend used for this export |
| `x_keys` | Feature keys in the TensorDict |
| `y_keys` | Label keys; stored with `label__` prefix in the data directory |
| `key_specs` | dtype and shape per key |
| `extra.transform_applied` | `true` when `--preprocess` was used |
| `splits.<name>.label_prefix` | Prefix applied to label keys in the stored files |

## Reuse with `ExportedDataset`

Load an exported dataset in a new scenario using the `ExportedDataset` data source (registered as `"ExportedDataset"`):

### Raw export reuse

```python
from nexuml.core.types import ScenarioSpec, DataSpec, TrainingSpec

ScenarioSpec(
    name="train_on_export",
    data=DataSpec(
        source_type="ExportedDataset",
        params={
            "path": "./exported_data/",
            "x_keys": ["features"],
            "y_keys": ["target"],
        },
    ),
    training=TrainingSpec(lr=1e-3, max_epochs=10, loss_keys={"classification_loss": 1.0}),
    ...
)
```

### Preprocessed export reuse

```python
DataSpec(
    source_type="ExportedDataset",
    params={
        "path": "./preprocessed_data/",
        "x_keys": ["z"],            # the pre-extracted embedding key
        "y_keys": ["target"],
        "preprocessed": True,       # skip pipeline stages up to the export boundary
    },
)
```

When `preprocessed=True`, the `skip_pipeline_stages` list in the scenario is populated to skip stages that were already applied before export.

## Full example

```bash
# 1. Export training and validation splits
nexuml export-dataset synthetic-ae-tutorial \
  --output ./cache/synthetic_ae/ \
  --backend numpy_mmap \
  --split train \
  --split val \
  --preprocess \
  --preprocess-until-key z

# 2. Verify the layout
ls ./cache/synthetic_ae/
# config.yaml  metadata.parquet  train/  val/

# 3. Train a classifier on the cached embeddings
# (scenario file uses DataSpec(source_type="ExportedDataset", ...))
nexuml train --scenario-file classifier_on_cache.py
```

## Implementation map

- `src/nexuml/data/export/` — export backend implementations
- `src/nexuml/cli/main.py` — `export-dataset` command
- `library/src/nexuml_library/data/exported.py` — `ExportedDataset` data source

## See also

- [Backends](../reference/backends.md)
- [Model export and reload](export.md)
- [Trusted scenario files](scenario-file.md)
