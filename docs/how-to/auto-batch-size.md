# Automatic batch size

NexuML can probe candidate training batch sizes on CUDA before the full run starts.

## Configure it

```python
from nexuml.core.types import AutoBatchSizeSpec, DataSpec, LoaderSpec, TrainingSpec
from nexuml.data.loaders.definitions import TorchLoader

training = TrainingSpec(
    batch_size=AutoBatchSizeSpec(
        min=8,
        max=256,
        candidates="power_of_two",
        safety="previous_power_of_two",
        margin=0.8,
    )
)

data = DataSpec(
    source=MyDataset(...),
    loader=LoaderSpec(
        backend=TorchLoader(),
        batch_size=None,
    ),
)
```

`data.loader.batch_size` must remain `None`. An explicit loader batch size takes precedence over `TrainingSpec.batch_size` and therefore disables the auto probe.

## CUDA is required

The runtime raises an error when automatic batch-size probing is requested without CUDA.

## How probing works

For the current `power_of_two` candidate strategy, NexuML:

1. generates the bounded candidate set between `min` and `max`;
2. probes the candidates in ascending order with one forward/backward batch on CUDA;
3. records success, CUDA OOM, peak memory, and `over_margin` where applicable;
4. selects a final size from the successful candidates according to `safety`;
5. rebuilds the final data module with the selected training batch size.

This is an exhaustive candidate probe, **not a binary search**.

## Safety policies

- `largest` — choose the largest successful candidate.
- `previous_power_of_two` — choose the next lower successful candidate when one exists.
- `margin` — candidates whose measured peak allocation exceeds `margin × total GPU memory` are excluded; choose the largest remaining candidate.

The probe result, attempts, device metadata, and selected batch size are included in runtime metadata.

## Defaults

`AutoBatchSizeSpec` itself defaults to `min=1`, `max=128`, `safety="previous_power_of_two"`, and `margin=0.8`.

The optional base library also defines `DEFAULT_AUTO_BATCH_SIZE` for its scenario helpers. Those library defaults are a convenience policy, not the core model defaults.

## See also

- [Train](train.md)
- [Data loading](data-loading.md)
- [Scenario/config reference](../reference/scenario-spec.md)
