"""CPU smoke test exercising the real NexuML resolve/build/train CLI path."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows


@pytest.mark.parametrize("scenario_name", ("maintenance-rul-feature-mlp", "maintenance-rul-all"))
def test_native_nexuml_rul_cli_build_and_train(tmp_path: Path, scenario_name: str) -> None:
    """A labeled engine-disjoint scenario trains through CLI, never an ad-hoc loop."""
    root = tmp_path / "source"
    root.mkdir()
    rows: list[list[float]] = []
    for asset in range(1, 6):
        for step in range(1, 5):
            values = [
                (asset + 0.1 * step + channel * 0.001)
                for channel in range(24)
            ]
            rows.append([float(asset), float(step), *values])
    np.savetxt(root / "train_FD001.txt", np.asarray(rows))
    np.savetxt(
        root / "test_FD001.txt",
        np.asarray([[1.0, float(step), *([0.1 * step] * 24)]
                    for step in range(1, 5)]),
    )
    np.savetxt(root / "RUL_FD001.txt", [2.0])
    path = tmp_path / "config.yaml"
    env = os.environ.copy()
    env.pop("NEXUML_RUL_PREPARED_FILE", None)
    env["NEXUML_CMAPSS_ROOT"] = str(root)
    for command in (
        ["nexuml", "resolve", scenario_name, "--output", str(path)],
        ["nexuml", "build", str(path)],
        ["nexuml", "train", "--config", str(path), "--max-epochs", "1"],
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=env,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
    restored = ResolvedConfig.load(path).to_scenario()
    assert restored.data.input_shapes["rul_history"] == [30, 24]
    source = restored.data.datasets[0].source
    assert isinstance(source, AutoCMapSSRULWindows)
    assert source.download is True
    assert source.root == str(root)
    assert Path(source.artifact_path).is_file()


def test_native_resolve_with_empty_prepared_override(tmp_path: Path) -> None:
    """The public CLI must ignore blank overrides and resolve without raw data."""
    config = tmp_path / "rul-all.yaml"
    env = os.environ.copy()
    env["NEXUML_RUL_PREPARED_FILE"] = ""
    result = subprocess.run(
        ["nexuml", "resolve", "maintenance-rul-all", "--output", str(config)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    scenario = ResolvedConfig.load(config).to_scenario()
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoCMapSSRULWindows)
    assert source.download is True
    assert Path(source.artifact_path) != tmp_path
    assert not Path(source.artifact_path).exists()
