"""Native CLI smoke tests for recent RUL architectures and battery capacity."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.battery_capacity import BatteryCapacityWindows
from nexuml_library.data.maintenance.bimodal_rul import LabeledBimodalRULWindows
from nexuml_library.data.maintenance.patchformer_flow import AutoPatchFormerFlowEndpoints
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows
from nexuml_library.data.maintenance.rul_flow_calibration import (
    AutoPatchFormerFlowRULFeatures,
)


def _run_nexuml(
    tmp_path: Path,
    scenario_name: str,
    *,
    env: dict[str, str],
) -> Path:
    """Build and train a native scenario for one epoch without custom loops.

    Returns:
        Path to the native resolved YAML configuration.
    """
    path = tmp_path / f"{scenario_name}.yaml"
    for command in (
        ["nexuml", "resolve", scenario_name, "--output", str(path)],
        ["nexuml", "build", str(path)],
        ["nexuml", "train", "--config", str(path), "--max-epochs", "1"],
    ):
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return path


def _synthetic_battery(root: Path) -> None:
    """Create realistic-format observed capacities for the source contract."""
    for cell_id, offset in (
        ("B0005", 0.0),
        ("B0006", 0.01),
        ("B0007", 0.02),
        ("B0018", 0.03),
    ):
        cycles = np.arange(1, 45)
        capacities = 2.0 - 0.008 * cycles - offset
        pd.DataFrame(
            {"cycle": cycles, "capacity": capacities}
        ).to_csv(root / f"{cell_id}.csv", index=False)


def _synthetic_bimodal(path: Path) -> None:
    """Create explicit machine-level train/validation/test supervised CSV."""
    records = []
    for name, split in (("a", "train"), ("b", "val"), ("c", "test")):
        for step in range(1, 32):
            records.append(
                {
                    "asset_id": name,
                    "cycle": step,
                    "split": split,
                    "temperature": float(step) + (1 if name == "a" else 2),
                    "load": float(step) * 0.5,
                    "rul": float(32 - step),
                }
            )
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def _synthetic_cmapss(root: Path) -> None:
    """Create a small complete FD001 file trio for official-source protocol tests."""
    train = [
        [engine, cycle, *([float(engine + cycle / 100)] * 24)]
        for engine in range(1, 9)
        for cycle in range(1, 37)
    ]
    test = [
        [engine, cycle, *([float(engine + cycle / 100)] * 24)]
        for engine in (1, 2)
        for cycle in range(1, 13)
    ]
    np.savetxt(root / "train_FD001.txt", np.asarray(train))
    np.savetxt(root / "test_FD001.txt", np.asarray(test))
    np.savetxt(root / "RUL_FD001.txt", [5, 7])


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("maintenance-rul-dmm-ja-adapted", "cmapss"),
        ("maintenance-rul-patchformer-adapted", "cmapss"),
        ("maintenance-rul-mamba-adapted", "cmapss"),
        ("maintenance-rul-controlled-fd003-patchformer-s42", "cmapss"),
        ("maintenance-rul-controlled-fd001-safety-lstm-s43", "cmapss"),
        ("maintenance-rul-controlled-fd003-dmm-ja-s44", "cmapss"),
        ("maintenance-rul-patchformer-flow-fd001-s42", "flow"),
        ("maintenance-rul-patchformer-flow-time-fd001-s42", "flow_calibration"),
        ("maintenance-rul-patchformer-hybrid-fd003-s44", "flow_calibration"),
        ("maintenance-rul-dmm-ja-bimodal", "bimodal"),
        ("maintenance-battery-patchformer-nasa", "battery"),
        ("maintenance-battery-rul-mamba-nasa", "battery"),
    ],
)
def test_advanced_registry_cli_can_resolve_without_preloaded_data(
    tmp_path: Path, name: str, kind: str,
) -> None:
    """Every new paper-derived scenario is discoverable by the public CLI."""
    config = tmp_path / "registered.yaml"
    env = os.environ.copy()
    env.pop("NEXUML_RUL_PREPARED_FILE", None)
    result = subprocess.run(
        ["nexuml", "resolve", name, "--output", str(config)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    source = ResolvedConfig.load(config).to_scenario().data.datasets[0].source
    if kind == "cmapss":
        assert isinstance(source, AutoCMapSSRULWindows)
    elif kind == "bimodal":
        assert isinstance(source, LabeledBimodalRULWindows)
    elif kind == "flow":
        assert isinstance(source, AutoPatchFormerFlowEndpoints)
    elif kind == "flow_calibration":
        assert isinstance(source, AutoPatchFormerFlowRULFeatures)
    else:
        assert isinstance(source, BatteryCapacityWindows)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("maintenance-rul-patchformer-adapted", "cmapss"),
        ("maintenance-rul-dmm-ja-bimodal", "bimodal"),
        ("maintenance-battery-patchformer-nasa", "battery"),
    ],
)
def test_advanced_native_cli_builds_and_trains_without_custom_runner(
    tmp_path: Path,
    name: str,
    kind: str,
) -> None:
    """Complete one-epoch data→fit→test pipeline using public commands only."""
    env = os.environ.copy()
    env.pop("NEXUML_RUL_PREPARED_FILE", None)
    source_root = tmp_path / "data"
    source_root.mkdir()
    if kind == "cmapss":
        _synthetic_cmapss(source_root)
        env["NEXUML_CMAPSS_ROOT"] = str(source_root)
    elif kind == "bimodal":
        trace = source_root / "bimodal.csv"
        _synthetic_bimodal(trace)
        env["NEXUML_RUL_BIMODAL_CSV"] = str(trace)
    else:
        _synthetic_battery(source_root)
        env["NEXUML_BATTERY_CAPACITY_ROOT"] = str(source_root)
    config = _run_nexuml(tmp_path, name, env=env)
    assert config.is_file()
