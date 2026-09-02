# Choose a data loader

`DataSpec` describes the dataset view. `LoaderSpec` describes common loading policy such as batch size, workers, prefetching, and shuffling. Its `backend` is a typed `LoaderBackendDefinition`.

NexuML currently provides three loader definitions:

- `TorchLoader` — standard PyTorch loading;
- `DaliLoader` — NVIDIA DALI, including native file routes where supported;
- `TensorShardsLoader` — windowed loading from materialized tensor shards.

Inspect registered definitions with:

```bash
nexuml backend list data-loader
```

This lists the backend definitions known to NexuML. It does not prove that every optional third-party runtime (for example NVIDIA DALI) can import on the current machine.

## Portable choice for your own scenario

Select the Torch backend explicitly when your scenario should work without DALI:

```python
from nexuml.core.types import DataSpec, LoaderSpec
from nexuml.data.loaders.definitions import TorchLoader

DataSpec(
    source=MyDataset(...),
    loader=LoaderSpec(
        backend=TorchLoader(),
        num_workers=4,
    ),
)
```

When `LoaderSpec.batch_size` is `None`, the effective batch size comes from `TrainingSpec.batch_size`. Setting an explicit loader batch size takes precedence and disables automatic batch-size probing.

!!! note "Current 0.2 default"
    `LoaderSpec` currently defaults to `DaliLoader()`. Because DALI is an optional platform-specific dependency, portable user-authored scenarios should select `TorchLoader()` explicitly unless DALI is intentionally required.

## NVIDIA DALI

Install the optional integration on a compatible Linux/CUDA environment:

```bash
uv pip install "nexuml[dali]" --index https://pypi.nvidia.com
python -c "import nvidia.dali"
```

Then select it explicitly:

```python
from nexuml.data.loaders.definitions import DaliLoader

LoaderSpec(
    backend=DaliLoader(),
    num_workers=4,
)
```

The DALI runtime has two broad paths:

- file-backed datasets with metadata can use native readers/decoders for supported audio, image, video, text, NumPy, and WebDataset data;
- datasets that cannot use the native route may fall back to the Torch loader after the DALI backend itself has been initialized.

A file-backed `NexuDataset` can expose metadata such as a `file` column and optional DALI sample-contract hints (`dali_x_keys`, layout, sequence length). The [audio tutorial](../tutorials.md) is intended to demonstrate this path once its 0.2 migration is complete.

## Tensor shards

For large pre-materialized tensors, use the typed tensor-shard loader:

```python
from nexuml.data.loaders.definitions import TensorShardsLoader

LoaderSpec(
    backend=TensorShardsLoader(
        shards_per_window=6,
        prefetch_windows=2,
        prefetch_workers=2,
    ),
    batch_size=64,
    num_workers=0,
)
```

`PreprocessingSpec(writer="tensor_shards")` can materialize a dataset view into non-prebatched tensor shards first. The runtime loader then creates training batches independently from those storage shards.

See [Export a dataset](export-dataset.md) for the general export formats.

## Automatic batch size

Automatic probing lives on `TrainingSpec.batch_size`. Leave `LoaderSpec.batch_size=None` so the selected training batch size can flow into the loader. See [Automatic batch size](auto-batch-size.md).
