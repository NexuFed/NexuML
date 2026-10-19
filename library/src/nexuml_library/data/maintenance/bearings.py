"""Metadata-backed industrial bearing sources for predictive-maintenance research.

The public definitions are immutable NexuML components; the private runtime
indexes source files and lazily loads fixed-length vibration windows.
"""

from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset

_SOURCE_URLS = {
    "paderborn": "https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download",
    "pronostia": "https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM",
    "xjtu_sy": "https://biaowang.tech/xjtu-sy-bearing-datasets/",
    "ims": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
}
_IMS_ARCHIVE = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
_SPLITS = frozenset({"train", "val", "test"})
_PADERBORN_HEALTHY = frozenset(f"K{i:03d}" for i in range(1, 7))
_IMS_FAILED_BEARINGS = {"1st_test": (3, 4), "2nd_test": (1,), "3rd_test": (3,)}


def _filename_order(file: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", file.stem)
    return (int(numbers[-1]) if numbers else -1, file.name)


def _safe_extract(archive: Path, root: Path) -> None:
    """Extract ZIP files without allowing traversal or links outside the root.

    Raises:
        ValueError: If an entry escapes the extraction root or is a symbolic link.
    """
    destination = root.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
            if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"ZIP symlink is not permitted: {entry.filename}")
        zipped.extractall(destination)


def _ensure_files(root: Path, kind: str, download: bool, archive_url: str | None) -> None:
    if kind == "paderborn":
        suffix = ".mat"
    elif kind in {"pronostia", "xjtu_sy"}:
        suffix = ".csv"
    else:
        suffix = ""

    def present() -> bool:
        if kind == "ims":
            return any(
                path.is_file() and re.fullmatch(r"\d{4}(?:\.\d{2}){5}", path.name)
                for path in root.rglob("*")
            )
        return any(root.rglob(f"*{suffix}"))

    if root.exists() and present():
        return
    if not download or archive_url is None:
        action = (
            "Enable download with an approved archive_url, or extract the original "
            f"dataset into {root}. Official source: {_SOURCE_URLS[kind]}"
        )
        raise FileNotFoundError(f"{kind} data not found at {root}. {action}")

    parsed = urlparse(archive_url)
    if parsed.scheme != "https" or not parsed.path.lower().endswith(".zip"):
        raise ValueError("Automatic retrieval requires a direct HTTPS .zip archive URL")
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / ".nexuml-download.zip"
    partial = root / ".nexuml-download.zip.part"
    try:
        with urlopen(archive_url, timeout=90) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(archive_path)
        _safe_extract(archive_path, root)
        # Some official distributions wrap the data in another ZIP.
        if not present():
            nested = list(root.rglob("*.zip"))
            for item in nested:
                if item != archive_path:
                    _safe_extract(item, item.parent)
                    if present():
                        break
        if not present():
            raise FileNotFoundError(f"Archive did not contain a supported {kind} layout")
    finally:
        partial.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)


def _run_rows(
    kind: str,
    files: list[Path],
    *,
    bearing: str,
    group: str,
    condition: str,
    times: list[float],
    sample_rate: int,
    complete: bool,
    split: str | None = None,
    channel: int | None = None,
) -> list[dict]:
    if len(files) != len(times):
        raise ValueError("File/time count mismatch")
    # A single snapshot, or a non-increasing clock, is not evidence of
    # a completed run-to-failure sequence.
    complete = complete and len(files) > 1 and times[-1] > times[0]
    rows = []
    final_time = max(times) if times else 0.0
    for index, (file, elapsed) in enumerate(zip(files, times)):
        health = -1
        if complete:
            if index == 0:
                health = 0
            elif index == len(files) - 1:
                health = 1
        rows.append(
            {
                "file": str(file),
                "basename": file.name,
                "dataset_name": kind,
                "source_url": _SOURCE_URLS[kind],
                "bearing_id": bearing,
                "run_id": f"{kind}:{bearing}",
                "group_id": group,
                "operating_condition": condition,
                "record_index": index,
                "elapsed_seconds": elapsed,
                "sample_rate_hz": sample_rate,
                "health": health,
                "rul": max(0.0, final_time - elapsed) if complete else 0.0,
                "rul_valid": int(complete),
                "rul_unit": "seconds",
                "label_source": "trajectory-derived" if complete else "unavailable",
                "channel": channel if channel is not None else -1,
                "split": split or "unassigned",
            }
        )
    return rows


def _paderborn_rows(root: Path) -> list[dict]:
    rows = []
    for file in sorted(root.rglob("*.mat")):
        matched = re.search(
            r"(?<![A-Z0-9])(K(?:\d{3}|[AI]\d{2}))(?![A-Z0-9])",
            file.stem,
            re.IGNORECASE,
        )
        if matched is None:
            continue
        bearing = matched.group(1).upper()
        condition = file.stem.split(f"_{matched.group(1)}")[0]
        health = 0 if bearing in _PADERBORN_HEALTHY else 1
        rows.append(
            {
                "file": str(file),
                "basename": file.name,
                "dataset_name": "paderborn",
                "source_url": _SOURCE_URLS["paderborn"],
                "bearing_id": bearing,
                "run_id": f"paderborn:{bearing}",
                "group_id": f"paderborn:{bearing}",
                "operating_condition": condition,
                "record_index": _filename_order(file)[0],
                "elapsed_seconds": 0.0,
                "sample_rate_hz": 64000,
                "health": health,
                "rul": 0.0,
                "rul_valid": 0,
                "rul_unit": "seconds",
                "label_source": "unavailable",
                "channel": -1,
                "split": "train",
            }
        )
    return rows


def _pronostia_rows(root: Path) -> list[dict]:
    rows = []
    for directory in sorted(root.rglob("Bearing*")):
        if not directory.is_dir() or not re.fullmatch(r"Bearing\d+_\d+", directory.name):
            continue
        files = sorted(directory.glob("acc_*.csv"), key=_filename_order)
        if not files:
            continue
        parents = {part.lower() for part in directory.parts}
        complete = "learning_set" in parents or "full_test_set" in parents
        # The PHM challenge Test_set is truncated. Full_Test_Set, if
        # supplied, is a complete but still official held-out test run.
        official_test = "test_set" in parents or "full_test_set" in parents
        group = f"pronostia:{directory.name}"
        first_index = _filename_order(files[0])[0]
        times = [float(_filename_order(file)[0] - first_index) * 10.0 for file in files]
        rows.extend(
            _run_rows(
                "pronostia",
                files,
                bearing=directory.name,
                group=group,
                condition=directory.name.split("_")[0],
                times=times,
                sample_rate=25600,
                complete=complete,
                split="test" if official_test else None,
            )
        )
    return rows


def _xjtu_rows(root: Path) -> list[dict]:
    rows = []
    for directory in sorted(root.rglob("Bearing*")):
        if not directory.is_dir() or not re.fullmatch(r"Bearing[123]_[1-5]", directory.name):
            continue
        files = sorted(directory.glob("*.csv"), key=_filename_order)
        if not files:
            continue
        indices = [_filename_order(file)[0] for file in files]
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError(f"Non-unique or unordered XJTU-SY records: {directory}")
        rows.extend(
            _run_rows(
                "xjtu_sy",
                files,
                bearing=directory.name,
                group=f"xjtu_sy:{directory.name}",
                condition=directory.parent.name,
                times=[float(i - indices[0]) * 60.0 for i in indices],
                sample_rate=25600,
                complete=True,
            )
        )
    return rows


def _ims_rows(root: Path) -> list[dict]:
    from datetime import datetime

    rows = []
    for test_name, failed in _IMS_FAILED_BEARINGS.items():
        directories = [p for p in root.rglob(test_name) if p.is_dir()]
        for directory in directories:
            files = [
                p
                for p in directory.iterdir()
                if p.is_file() and re.fullmatch(r"\d{4}(?:\.\d{2}){5}", p.name)
            ]
            if not files:
                continue
            files.sort(key=lambda p: p.name)
            times = [datetime.strptime(file.name, "%Y.%m.%d.%H.%M.%S") for file in files]
            relative = [(value - times[0]).total_seconds() for value in times]
            for bearing in range(1, 5):
                rows.extend(
                    _run_rows(
                        "ims",
                        files,
                        bearing=f"{test_name}:bearing{bearing}",
                        group=f"ims:{test_name}",
                        condition=test_name,
                        times=relative,
                        sample_rate=20000,
                        complete=bearing in failed,
                        channel=(bearing - 1) * (2 if test_name == "1st_test" else 1),
                    )
                )
            break
    return rows


def _set_group_splits(rows: list[dict], seed: int, split_map: dict[str, str] | None) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    if split_map is not None:
        unknown = set(split_map) - groups.keys()
        if unknown:
            raise ValueError(f"Unknown group_id in split_map: {sorted(unknown)}")
        if set(split_map.values()) - _SPLITS:
            raise ValueError("split_map values must be train, val or test")

    candidates = sorted(
        group
        for group, records in groups.items()
        if all(row["split"] == "unassigned" for row in records)
    )
    if split_map is None:
        rng = np.random.default_rng(seed)
        rng.shuffle(candidates)
        n = len(candidates)
        n_val = max(1, n // 5) if n >= 2 else 0
        n_test = max(1, n // 5) if n >= 3 else 0
        assigned = {
            group: ("val" if index < n_val else "test" if index < n_val + n_test else "train")
            for index, group in enumerate(candidates)
        }
    else:
        assigned = {group: split_map.get(group, "train") for group in candidates}

    for group, records in groups.items():
        names = {row["split"] for row in records}
        if len(names) != 1:
            raise ValueError(f"Conflicting official splits within experiment {group}")
        official = next(iter(names))
        if official != "unassigned" and split_map is not None:
            proposed = split_map.get(group, official)
            if proposed != official:
                raise ValueError(f"Cannot override official split for {group}")
        split = official if official != "unassigned" else assigned[group]
        for row in records:
            row["split"] = split

    if any(len({row["split"] for row in records}) != 1 for records in groups.values()):
        raise AssertionError("Split leakage across an experiment")


def _read_csv(file: Path, *, columns: tuple[int, ...]) -> np.ndarray:
    with file.open(encoding="utf-8-sig") as stream:
        first = stream.readline().strip().split(",")
    try:
        [float(value) for value in first]
        skiprows = 0
    except ValueError:
        skiprows = 1
    data = np.loadtxt(file, delimiter=",", skiprows=skiprows, dtype=np.float32, ndmin=2)
    if data.shape[1] <= max(columns):
        raise ValueError(f"Expected columns {columns} in {file}, got {data.shape[1]}")
    return np.asarray(data[:, columns].T, dtype=np.float32)


def _mat_signal(file: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Paderborn .mat loading requires scipy") from exc

    record = loadmat(file, simplify_cells=True)

    def find(item: object) -> np.ndarray | None:
        if isinstance(item, dict):
            record_dict = cast(dict[str, object], item)
            if "vibration" in record_dict:
                vibration = record_dict["vibration"]
                if isinstance(vibration, dict):
                    vibration_dict = cast(dict[str, object], vibration)
                    if "y_values" in vibration_dict:
                        value = np.asarray(vibration_dict["y_values"])
                        if np.issubdtype(value.dtype, np.number):
                            return value.astype(np.float32).reshape(1, -1)
            for key, value in record_dict.items():
                if not str(key).startswith("__"):
                    result = find(value)
                    if result is not None:
                        return result
        if isinstance(item, (list, tuple, np.ndarray)) and not (
            isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.number)
        ):
            for value in np.asarray(item, dtype=object).flat:
                result = find(value)
                if result is not None:
                    return result
        return None

    signal = find(record)
    if signal is None:
        raise ValueError(f"No vibration.y_values signal found in {file}")
    return signal


class _BearingDatasetRuntime(NexuDataset):
    """Lazy file-backed vibration dataset, preserving run-level provenance."""

    def __init__(
        self,
        *,
        root: str | Path,
        kind: Literal["paderborn", "pronostia", "xjtu_sy", "ims"],
        download: bool,
        archive_url: str | None,
        window_size: int,
        seed: int,
        split_map: dict[str, str] | None,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.kind = kind
        root_path = Path(root).expanduser()
        _ensure_files(root_path, kind, download, archive_url)
        scanners = {
            "paderborn": _paderborn_rows,
            "pronostia": _pronostia_rows,
            "xjtu_sy": _xjtu_rows,
            "ims": _ims_rows,
        }
        rows = scanners[kind](root_path)
        if not rows:
            raise ValueError(f"No supported {kind} recordings found in {root_path}")
        _set_group_splits(rows, seed=seed, split_map=split_map)
        super().__init__(
            meta=pd.DataFrame.from_records(rows),
            label_names=["health", "rul", "rul_valid"],
            do_split=False,
            modality="vibration",
        )

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        file = Path(str(row["file"]))
        if self.kind == "paderborn":
            signal = _mat_signal(file)
        elif self.kind == "ims":
            matrix = np.loadtxt(file, dtype=np.float32, ndmin=2)
            channel = int(row["channel"])
            if channel >= matrix.shape[1]:
                raise ValueError(f"Missing IMS channel {channel} in {file}")
            signal = matrix[:, channel : channel + 1].T.copy()
        elif self.kind == "pronostia":
            signal = _read_csv(file, columns=(4, 5))
        else:
            signal = _read_csv(file, columns=(0, 1))
        features = torch.from_numpy(np.ascontiguousarray(signal[:, : self.window_size]))
        if features.shape[1] < self.window_size:
            features = F.pad(features, (0, self.window_size - features.shape[1]))
        return TensorDict({"features": features}, batch_size=[])


@data_source("PaderbornBearingDataset")
class PaderbornBearingDataset(DataSourceDefinition):
    """Paderborn healthy/faulty endpoints; no inferred RUL supervision."""

    root: str | Path = "data/paderborn"
    download: bool = True
    archive_url: str | None = None
    window_size: int = 2048
    seed: int = 42
    split_map: dict[str, str] | None = None

    def build(self) -> NexuDataset:
        return _BearingDatasetRuntime(kind="paderborn", **self.model_dump())


@data_source("PRONOSTIABearingDataset")
class PRONOSTIABearingDataset(DataSourceDefinition):
    """FEMTO-ST PHM2012 snapshots; challenge Test_set is truncated."""

    root: str | Path = "data/pronostia"
    download: bool = True
    archive_url: str | None = None
    window_size: int = 2048
    seed: int = 42
    split_map: dict[str, str] | None = None

    def build(self) -> NexuDataset:
        return _BearingDatasetRuntime(kind="pronostia", **self.model_dump())


@data_source("XJTUSYBearingDataset")
class XJTUSYBearingDataset(DataSourceDefinition):
    """XJTU-SY run-to-failure vibration snapshots."""

    root: str | Path = "data/xjtu_sy"
    download: bool = True
    archive_url: str | None = None
    window_size: int = 2048
    seed: int = 42
    split_map: dict[str, str] | None = None

    def build(self) -> NexuDataset:
        return _BearingDatasetRuntime(kind="xjtu_sy", **self.model_dump())


@data_source("IMSBearingDataset")
class IMSBearingDataset(DataSourceDefinition):
    """NASA IMS tests, keeping entire test rigs together across splits."""

    root: str | Path = "data/ims"
    download: bool = True
    archive_url: str | None = _IMS_ARCHIVE
    window_size: int = 2048
    seed: int = 42
    split_map: dict[str, str] | None = None

    def build(self) -> NexuDataset:
        return _BearingDatasetRuntime(kind="ims", **self.model_dump())
