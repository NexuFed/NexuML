# Predictive-maintenance bearing datasets (NEX-233)

These sources implement the NexuML 0.2 immutable `DataSourceDefinition` pattern.
They build private, lazy, metadata-backed `NexuDataset` runtimes under
`nexuml_library.data.maintenance.bearings` without changing the core.

## Sources and access

| Definition | Original source | Data | Download behavior |
| --- | --- | --- | --- |
| `PaderbornBearingDataset` | [Paderborn KAt](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download) | healthy/faulty MATLAB vibration | `download=True` by default; extract locally or supply a direct approved HTTPS ZIP `archive_url`. Original RAR distributions need manual extraction. |
| `PRONOSTIABearingDataset` | FEMTO-ST / IEEE PHM 2012 | 6-column acceleration CSVs under `Learning_set/Bearing*/`; optional truncated `Test_set/` | local files or explicit approved `archive_url` |
| `XJTUSYBearingDataset` | [XJTU-SY official page](https://biaowang.tech/xjtu-sy-bearing-datasets/) | 2-column acceleration CSVs under `Bearing1_1/` etc. | local files or explicit approved `archive_url`; download mirrors on official site are not assumed to be direct reproducible links |
| `IMSBearingDataset` | [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | timestamp-named whitespace recordings in `1st_test/`, `2nd_test/`, `3rd_test/` | official NASA HTTPS ZIP as default `archive_url` |

All definitions have `download=True` as a default **policy**, but downloads do not
occur when the local dataset is already present. For sources without a verified
machine-retrievable ZIP URL, the default produces an actionable error instead of
scraping a website or silently using fabricated data. Set `download=False` to
require an existing local copy. Extraction checks archive entries for path traversal
and symlinks. Never commit downloaded datasets to the repository.

## Example

```python
from nexuml_library.data.maintenance.bearings import (
    PaderbornBearingDataset,
    PRONOSTIABearingDataset,
    XJTUSYBearingDataset,
    IMSBearingDataset,
)

# Definitions are immutable and registry-discoverable.
paderborn = PaderbornBearingDataset(root="/data/paderborn", download=False)
xjtu = XJTUSYBearingDataset(root="/data/xjtu_sy", window_size=2048)
runtime = xjtu.build()
print(runtime.meta[["group_id", "split", "health", "rul", "rul_valid"]].head())
features, targets = runtime[0]

# In a NexuML recipe, KEEP the dataset's group-level split decisions:
from nexuml.core.types import DataSpec, DatasetSpec

data = DataSpec(
    datasets=[
        DatasetSpec(
            source=xjtu,
            modality="vibration",
            split_type="keep",
        ),
    ],
    input_shapes={"features": [2, 2048]},
)
```

## Labels, splits, and provenance

Each row includes `file`, `dataset_name`, `source_url`, `bearing_id`,
`run_id`, `group_id`, `operating_condition`, `record_index`,
`elapsed_seconds`, `sample_rate_hz`, `rul_unit`, `label_source`,
`split`, and the numeric labels `health`, `rul`, `rul_valid`.

- `health=0` means an observed healthy endpoint; `health=1` means an
  observed faulty/final endpoint; `health=-1` means no endpoint annotation.
  The latter must be excluded from endpoint-supervised objectives.
- `rul` is in **seconds**. For complete run-to-failure recordings it is
  computed as last observed acquisition time minus current acquisition time,
  not a separately supplied per-record ground-truth value.
- `rul_valid=1` identifies an observed complete trajectory for which that
  convention applies. When `rul_valid=0`, `rul=0` is a **masked placeholder,
  not a RUL annotation**. Training/evaluation must mask it before computing a
  RUL loss or metric. Paderborn always has `rul_valid=0`.
- PRONOSTIA `Test_set` is incomplete and remains the official `test`
  partition, with `rul_valid=0` unless a separate, verified run-to-failure
  completion source is introduced. `Learning_set` yields complete runs.
- IMS includes all four bearings per physical test, but RUL is only assigned
  to bearings known to fail in that experiment. All channels from the same
  experiment share a `group_id` to prevent leakage.
- Paderborn samples are **training only**. Complete runs are assigned train,
  val and test by **group**, reproducibly via `seed`. Pass an explicit
  `split_map={"xjtu_sy:Bearing1_1": "test"}` to control groups; official
  challenge partitions cannot be overridden.
- NexuML's generic sample-level random splitter must **not** be used on these
  datasets. Retain their `split` values by using `DatasetSpec(split_type="keep")`.
  All preprocessing statistics must be fitted on training groups only.

## Features and limitations

The lazy runtime loads float32 vibration data into `features` shaped
`[channels, window_size]`. It crops from the beginning of each recorded
snapshot and zero-pads shorter snapshots. It does **not** resample or normalize.

Channel counts differ: Paderborn and IMS currently expose one channel, while
FEMTO-ST and XJTU-SY expose two. Do not pool incompatible raw shapes into a
single training batch. Align the channels/sampling rate and learn a shared
representation using **separate reusable library transforms** in the next
experiment stage. The IMS 1st_test data have two sensors per bearing, but this
initial adapter selects their first sensor for consistent IMS dimensions.

The source collections differ in physical lifetime definitions. In particular,
a PRONOSTIA challenge test snapshot is not the end of life. Recheck the
meaning of failure/end-of-test conventions before interpreting RUL scores.
The independent full-run split and RUL calibration experiment is the next
NEX-233 milestone.

## Tests

```bash
pytest tests/library/test_predictive_maintenance_datasets.py -v
ruff check .
ruff format --check .
ty check
```
