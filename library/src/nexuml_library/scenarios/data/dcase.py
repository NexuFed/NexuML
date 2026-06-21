"""DCASE Task 2 data specification builders."""

from __future__ import annotations

import logging
from pydantic import BaseModel
from pathlib import Path

from nexuml.core.types import DataSpec, DatasetSpec, LoaderSpec
from nexuml_library.scenarios.data.roots import resolve_data_root

logger = logging.getLogger(__name__)


class MachineSpec(BaseModel):
    """Identifies a DCASE machine type together with the year it was released.

    The year determines which dataset subdirectory the files live in
    (e.g. year=2023 → ``dcase2023t2/``).  ``data_type`` is either
    ``"dev"`` (labelled development set) or ``"eval"`` (evaluation set).
    """

    machine_type: str
    year: int = 2023
    data_type: str = "dev"

    @property
    def key(self) -> str:
        """Unique machine identity across years/data splits."""
        return f"{self.dataset_name}:{self.data_type}:{self.machine_type}"

    @property
    def dataset_name(self) -> str:
        return f"DCASE{self.year}T2"


_DCASE_MACHINES: dict[int, dict[str, list[str]]] = {
    2020: {
        "dev": ["fan", "pump", "slider", "ToyCar", "ToyConveyor", "valve"],
        "eval": ["fan", "pump", "slider", "ToyCar", "ToyConveyor", "valve"],
    },
    2021: {
        "dev": ["fan", "gearbox", "pump", "slider", "ToyCar", "ToyTrain", "valve"],
        "eval": ["fan", "gearbox", "pump", "slider", "ToyCar", "ToyTrain", "valve"],
    },
    2022: {
        "dev": ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"],
        "eval": ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"],
    },
    2023: {
        "dev": ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"],
        "eval": ["bandsaw", "grinder", "shaker", "ToyDrone", "ToyNscale", "ToyTank", "Vacuum"],
    },
    2024: {
        "dev": ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"],
        "eval": [
            "3DPrinter",
            "AirCompressor",
            "BrushlessMotor",
            "HairDryer",
            "HoveringDrone",
            "RoboticArm",
            "Scanner",
            "ToothBrush",
            "ToyCircuit",
        ],
    },
    2025: {
        "dev": ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"],
        "eval": [
            "AutoTrash",
            "BandSealer",
            "CoffeeGrinder",
            "HomeCamera",
            "Polisher",
            "ScrewFeeder",
            "ToyPet",
            "ToyRCCar",
        ],
    },
    2026: {
        "dev": [
            "bearingEmu",
            "fan",
            "gearboxEmu",
            "sliderEmu",
            "ToyCar",
            "ToyCarEmu",
            "valveEmu",
        ],
        "eval": [
            "BlowerDustCollector",
            "Sander",
            "SewingMachine",
            "ToothBrush",
            "ToyDrone",
        ],
    },
}

DCASE_MACHINE_SPECS: tuple[MachineSpec, ...] = tuple(
    MachineSpec(machine_type=machine_type, year=year, data_type=data_type)
    for year, data_types in _DCASE_MACHINES.items()
    for data_type, machine_types in data_types.items()
    for machine_type in machine_types
)

_DCASE_SECTION_IDS = [f"{section_id:02d}" for section_id in range(7)]


def _validate_machine_specs(
    specs: list[MachineSpec],
    *,
    allow_unknown: bool = False,
) -> list[MachineSpec]:
    """Validate machine specs against the known DCASE catalog.

    Returns the validated specs. Raises ValueError if any spec is not in the
    catalog and ``allow_unknown`` is False.

    Task 4.3: Address catalog gap for 2023/2024 dev machines by validating
    explicit machine_specs. Users can pass allow_unknown=True to bypass
    validation for custom/experimental machines.

    Returns:
        list[MachineSpec]: The validated machine specs.

    Raises:
        ValueError: If any spec is not in the catalog and ``allow_unknown``
            is False.
    """
    if allow_unknown or not specs:
        return specs

    catalog_keys = {spec.key for spec in DCASE_MACHINE_SPECS}
    invalid = [spec for spec in specs if spec.key not in catalog_keys]
    if invalid:
        invalid_summary = [
            f"{s.machine_type} (year={s.year}, data_type={s.data_type})" for s in invalid
        ]
        raise ValueError(
            f"Machine specs not found in DCASE catalog: {', '.join(invalid_summary)}. "
            f"Catalog contains {len(catalog_keys)} entries. "
            "Pass allow_unknown=True to bypass validation for custom machines."
        )
    return specs


def dcase_machine_specs(
    machine_types: list[str] | None = None,
    years: list[int] | None = None,
    data_types: list[str] | None = None,
    dedupe_machine_types: bool = False,
) -> list[MachineSpec]:
    """Return known DCASE machine specs from the static scenario catalog.

    When ``dedupe_machine_types`` is true, keep at most one spec per raw machine
    type, preferring eval over dev and newer years over older years.
    """
    specs = list(DCASE_MACHINE_SPECS)
    if machine_types is not None:
        machine_set = set(machine_types)
        specs = [s for s in specs if s.machine_type in machine_set]
    if years is not None:
        year_set = set(years)
        specs = [s for s in specs if s.year in year_set]
    if data_types is not None:
        data_type_set = set(data_types)
        specs = [s for s in specs if s.data_type in data_type_set]
    if not dedupe_machine_types:
        return specs

    priority = {"dev": 0, "eval": 1}
    by_machine: dict[str, MachineSpec] = {}
    for spec in specs:
        current = by_machine.get(spec.machine_type)
        if current is None or (priority.get(spec.data_type, -1), spec.year) > (
            priority.get(current.data_type, -1),
            current.year,
        ):
            by_machine[spec.machine_type] = spec
    return sorted(by_machine.values(), key=lambda s: (s.year, s.data_type, s.machine_type))


def dcase_data(
    data_root: str | Path = "DCASET2",
    machine_types: list[str] | None = None,
    machine_specs: list[MachineSpec] | None = None,
    train_machine_specs: list[MachineSpec] | None = None,
    test_machine_specs: list[MachineSpec] | None = None,
    years: list[int] | None = None,
    data_types: list[str] | None = None,
    dedupe_machine_types: bool = False,
    download: bool = False,
    sample_rate: int = 16000,
    clip_num_samples: int = 160000,
    batch_size: int = 64,
    num_workers: int = 4,
    validate_machine_specs: bool = True,
) -> DataSpec:
    """DCASE Task 2 anomaly detection data spec.

    Pass ``machine_specs`` for explicit multi-year control, or ``machine_types``
    to filter the built-in DCASE machine catalog.

    Set ``NEXUML_DATA_ROOT`` so scenario files can use logical dataset
    paths without cluster-specific paths in code.

    Args:
        data_root: Logical root. Resolved via ``NEXUML_DATA_ROOT`` if the
            path does not exist as-is.
        machine_types: Names of machine types to keep from the catalog.
        machine_specs: Explicit per-machine year + data_type (multi-year) for both fit and test.
        train_machine_specs: Explicit machine specs used only for fit.
        test_machine_specs: Explicit machine specs used only for test.
        years: Optional year filter.
        data_types: Optional data type filter, e.g. ["eval"] for additional/eval machines.
        dedupe_machine_types: Keep at most one spec per raw machine type,
            preferring eval/newer year.
        download: Download/extract missing Zenodo zips from the DCASE manifest.
        sample_rate: Audio sample rate in Hz.
        clip_num_samples: Samples per audio clip.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        validate_machine_specs: Validate explicit machine_specs against the catalog.
            Set to False to allow custom/experimental machines not in the catalog.

    Returns:
        DataSpec: DCASE dataset specification with fit and test splits.
    """
    root = resolve_data_root(str(data_root) if isinstance(data_root, Path) else data_root)

    if train_machine_specs is not None or test_machine_specs is not None:
        train_specs = train_machine_specs or machine_specs or []
        test_specs = test_machine_specs or machine_specs or []
        specs = [*train_specs, *test_specs]
        if validate_machine_specs:
            _validate_machine_specs(specs)
    elif machine_specs is not None:
        train_specs = machine_specs
        test_specs = machine_specs
        specs = machine_specs
        if validate_machine_specs:
            _validate_machine_specs(specs)
    else:
        specs = dcase_machine_specs(
            machine_types=machine_types,
            years=years,
            data_types=data_types,
            dedupe_machine_types=dedupe_machine_types,
        )
        if not specs:
            logger.error(
                "No DCASE machines matched filters: machine_types=%s years=%s data_types=%s",
                machine_types,
                years,
                data_types,
            )
        train_specs = specs
        test_specs = specs

    all_machine_types = sorted({s.key for s in specs})

    datasets: list[DatasetSpec] = []
    for spec in train_specs:
        common = {
            "data_root": str(root),
            "dataset_name": spec.dataset_name,
            "data_type": spec.data_type,
            "machine_type": spec.machine_type,
            "download": download,
            "sample_rate": sample_rate,
            "clip_num_samples": clip_num_samples,
            "machine_types": all_machine_types,
            "section_ids": _DCASE_SECTION_IDS,
        }
        if spec.year == 2020:
            common["section_keyword"] = "id"
        datasets.append(
            DatasetSpec(
                type_key="DCASE2026T2Dataset" if spec.year == 2026 else "DCASET2Dataset",
                params=common,
                modality="audio",
                split_type="fit",
            )
        )
    for spec in test_specs:
        common = {
            "data_root": str(root),
            "dataset_name": spec.dataset_name,
            "data_type": spec.data_type,
            "machine_type": spec.machine_type,
            "download": download,
            "sample_rate": sample_rate,
            "clip_num_samples": clip_num_samples,
            "machine_types": all_machine_types,
            "section_ids": _DCASE_SECTION_IDS,
        }
        if spec.year == 2020:
            common["section_keyword"] = "id"
        datasets.append(
            DatasetSpec(
                type_key="DCASE2026T2Dataset" if spec.year == 2026 else "DCASET2Dataset",
                params={**common, "train": False},
                modality="audio",
                split_type="test",
            )
        )

    return DataSpec(
        source_type="dcase",
        datasets=datasets,
        loader=LoaderSpec(
            backend="dali",
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
        params={"data_root": str(root)},
        input_shapes={"waveform": [clip_num_samples]},
        num_classes=len(all_machine_types) * 2 if all_machine_types else None,
        merge_labels={
            "class": {
                "columns": ["machine", "target"],
                "logits": False,
                "include_dataset": False,
            },
            "class_logits": {
                "columns": ["machine", "target"],
                "logits": True,
                "include_dataset": False,
            },
            "anomaly": {
                "columns": ["y_true"],
                "logits": False,
                "include_dataset": False,
            },
            "domain": {
                "columns": ["target"],
                "logits": False,
                "include_dataset": False,
            },
        },
    )
