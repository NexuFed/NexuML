# nexuml_library.scenarios.data.dcase

*module*

DCASE Task 2 data specification builders.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L1-L1)

## nexuml_library.scenarios.data.dcase.DCASE_MACHINE_SPECS

*attribute* · *module attribute*

```python
DCASE_MACHINE_SPECS: tuple[MachineSpec, ...] = tuple((MachineSpec(machine_type=machine_type, year=year, data_type=data_type) for year, data_types in _DCASE_MACHINES.items() for data_type, machine_types in data_types.items() for machine_type in machine_types))
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L103-L108)

## nexuml_library.scenarios.data.dcase.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L14-L14)

## nexuml_library.scenarios.data.dcase.MachineSpec

*class* · *pydantic model*

```python
class MachineSpec(BaseModel)
```

Bases: `BaseModel`

Identifies a DCASE machine type together with the year it was released.

The year determines which dataset subdirectory the files live in
(e.g. year=2023 → ``dcase2023t2/``).  ``data_type`` is either
``"dev"`` (labelled development set) or ``"eval"`` (evaluation set).

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L17-L36)

### nexuml_library.scenarios.data.dcase.MachineSpec.data_type

*attribute* · *pydantic field*

```python
data_type: str = 'dev'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L27-L27)

### nexuml_library.scenarios.data.dcase.MachineSpec.machine_type

*attribute* · *pydantic field*

```python
machine_type: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L25-L25)

### nexuml_library.scenarios.data.dcase.MachineSpec.year

*attribute* · *pydantic field*

```python
year: int = 2023
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L26-L26)

### nexuml_library.scenarios.data.dcase.MachineSpec.dataset_name

*property*

```python
dataset_name: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L35-L36)

### nexuml_library.scenarios.data.dcase.MachineSpec.key

*property*

```python
key: str
```

Unique machine identity across years/data splits.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L30-L32)

## nexuml_library.scenarios.data.dcase.dcase_data

*function*

```python
def dcase_data(data_root: str | Path = 'DCASET2', machine_types: list[str] | None = None, machine_specs: list[MachineSpec] | None = None, train_machine_specs: list[MachineSpec] | None = None, test_machine_specs: list[MachineSpec] | None = None, years: list[int] | None = None, data_types: list[str] | None = None, dedupe_machine_types: bool = False, download: bool = False, sample_rate: int = 16000, clip_num_samples: int = 160000, batch_size: int = 64, num_workers: int = 4, validate_machine_specs: bool = True) -> DataSpec
```

DCASE Task 2 anomaly detection data spec.

Pass ``machine_specs`` for explicit multi-year control, or ``machine_types``
to filter the built-in DCASE machine catalog.

Set ``NEXUML_DATA_ROOT`` so scenario files can use logical dataset
paths without cluster-specific paths in code.

**Parameters**

- `data_root` (`str | Path`) (default: `'DCASET2'`) — Logical root. Resolved via ``NEXUML_DATA_ROOT`` if the path does not exist as-is.
- `machine_types` (`list[str] | None`) (default: `None`) — Names of machine types to keep from the catalog.
- `machine_specs` (`list[MachineSpec] | None`) (default: `None`) — Explicit per-machine year + data_type (multi-year) for both fit and test.
- `train_machine_specs` (`list[MachineSpec] | None`) (default: `None`) — Explicit machine specs used only for fit.
- `test_machine_specs` (`list[MachineSpec] | None`) (default: `None`) — Explicit machine specs used only for test.
- `years` (`list[int] | None`) (default: `None`) — Optional year filter.
- `data_types` (`list[str] | None`) (default: `None`) — Optional data type filter, e.g. ["eval"] for additional/eval machines.
- `dedupe_machine_types` (`bool`) (default: `False`) — Keep at most one spec per raw machine type, preferring eval/newer year.
- `download` (`bool`) (default: `False`) — Download/extract missing Zenodo zips from the DCASE manifest.
- `sample_rate` (`int`) (default: `16000`) — Audio sample rate in Hz.
- `clip_num_samples` (`int`) (default: `160000`) — Samples per audio clip.
- `batch_size` (`int`) (default: `64`) — DataLoader batch size.
- `num_workers` (`int`) (default: `4`) — DataLoader workers.
- `validate_machine_specs` (`bool`) (default: `True`) — Validate explicit machine_specs against the catalog. Set to False to allow custom/experimental machines not in the catalog.

**Returns**

- `DataSpec` (`DataSpec`) — DCASE dataset specification with fit and test splits.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L187-L344)

## nexuml_library.scenarios.data.dcase.dcase_machine_specs

*function*

```python
def dcase_machine_specs(machine_types: list[str] | None = None, years: list[int] | None = None, data_types: list[str] | None = None, dedupe_machine_types: bool = False) -> list[MachineSpec]
```

Return known DCASE machine specs from the static scenario catalog.

When ``dedupe_machine_types`` is true, keep at most one spec per raw machine
type, preferring eval over dev and newer years over older years.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/dcase.py#L151-L184)
