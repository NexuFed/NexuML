"""Paired FD001/FD003 Goel reproduction and RUL safety experiment contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import ScenarioSpec
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.layers.model.rul_safety import RULAsymmetricSquaredLoss
from nexuml_library.scenarios.maintenance import rul as rul_scenarios


def _scenario(
    subset: Literal["FD001", "FD003"],
    architecture: Literal["lstm", "cnn"],
    kind: Literal["goel", "safety"],
    seed: int,
) -> ScenarioSpec:
    """Produce a named paired experiment without accessing held-out data.

    Returns:
        Native reproducibility scenario with a dedicated checkpoint.
    """
    if kind == "goel":
        func = (
            rul_scenarios.maintenance_rul_goel2026_lstm_protocol
            if architecture == "lstm"
            else rul_scenarios.maintenance_rul_goel2026_cnn_protocol
        )
        return func(subset=subset, seed=seed)
    return rul_scenarios._maintenance_rul_safety(
        architecture, subset=subset, seed=seed
    )


@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("architecture", ["lstm", "cnn"])
def test_paired_repro_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subset: Literal["FD001", "FD003"],
    seed: int,
    architecture: Literal["lstm", "cnn"],
) -> None:
    """Paired objectives share data/split while keeping independent weights."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    control = _scenario(subset, architecture, "goel", seed)
    safety = _scenario(subset, architecture, "safety", seed)
    control_source = control.data.datasets[0].source
    safety_source = safety.data.datasets[0].source
    assert isinstance(control_source, AutoCMapSSRULWindows)
    assert isinstance(safety_source, AutoCMapSSRULWindows)
    assert control_source == safety_source
    assert control_source.subset == subset
    assert control_source.seed == seed
    assert control_source.sensor_ids is not None
    assert len(control_source.sensor_ids) == (14 if subset == "FD001" else 15)
    assert (6 in control_source.sensor_ids) == (subset == "FD003")
    assert f"cmapss-{subset}-seed{seed}-" in control_source.artifact_path
    assert control.data.input_shapes["rul_history"] == [
        30, 14 if subset == "FD001" else 15
    ]
    assert control.pipeline.stages["RULModel"] == safety.pipeline.stages["RULModel"]
    assert control.training.optimizer == safety.training.optimizer
    assert control.training.loss_keys == safety.training.loss_keys
    assert control.name != safety.name
    assert control.callbacks[1].kwargs["dirpath"] != safety.callbacks[1].kwargs["dirpath"]
    assert control.callbacks[0].kwargs["seed"] == seed
    assert safety.callbacks[0].kwargs["seed"] == seed
    assert isinstance(
        safety.pipeline.stages["RULLoss"][0].component,
        RULAsymmetricSquaredLoss,
    )
    for selected in (control, safety):
        evaluator = selected.evaluation.algorithms[2].algorithm
        assert isinstance(evaluator, RULPerEngineAudit)
        assert evaluator.artifact_path == control_source.artifact_path
        assert selected.evaluation.algorithms[1].label_key == "rul_raw"
        assert selected.evaluation.algorithms[0].label_key == "rul"
    compiled = compile_pipeline(safety)
    assert compiled.resolved_config.to_scenario().name == safety.name


@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("architecture", ["lstm", "cnn"])
@pytest.mark.parametrize("kind", ["goel", "safety"])
def test_named_repro_scenario_is_registered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subset: Literal["FD001", "FD003"],
    seed: int,
    architecture: Literal["lstm", "cnn"],
    kind: Literal["goel", "safety"],
) -> None:
    """All intended variants have stable native registry function names."""
    if subset == "FD001" and seed == 42:
        pytest.skip("FD001 seed42 uses unchanged public scenario identifiers")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    name = f"maintenance_rul_repro_{subset.lower()}_{architecture}_{kind}_s{seed}"
    entry = getattr(rul_scenarios, name)
    assert callable(entry)
    selected = entry()
    expected = _scenario(subset, architecture, kind, seed)
    assert selected.name == expected.name
    assert selected.data == expected.data


def _tiny_fd003(root: Path) -> None:
    """Make valid FD003-like tables for offline public CLI verification."""
    train = [
        [engine, cycle, *([float(engine + cycle * 0.01)] * 24)]
        for engine in range(1, 9)
        for cycle in range(1, 36)
    ]
    test = [
        [engine, cycle, *([float(engine + cycle * 0.01)] * 24)]
        for engine in (1, 2)
        for cycle in range(1, 12)
    ]
    np.savetxt(root / "train_FD003.txt", np.asarray(train))
    np.savetxt(root / "test_FD003.txt", np.asarray(test))
    np.savetxt(root / "RUL_FD003.txt", [4, 136])


@pytest.mark.parametrize(
    "scenario_name",
    [
        "maintenance-rul-repro-fd003-lstm-goel-s42",
        "maintenance-rul-repro-fd003-lstm-safety-s42",
    ],
)
def test_native_fd003_resolve_build_train(
    tmp_path: Path,
    scenario_name: str,
) -> None:
    """FD003 Goel and safety variants run via the unmodified NexuML CLI."""
    root = tmp_path / "fd003"
    root.mkdir()
    _tiny_fd003(root)
    config = tmp_path / "fd003.yaml"
    env = os.environ.copy()
    env.pop("NEXUML_RUL_PREPARED_FILE", None)
    env["NEXUML_CMAPSS_ROOT"] = str(root)
    for command in (
        ["nexuml", "resolve", scenario_name, "--output", str(config)],
        ["nexuml", "build", str(config)],
        ["nexuml", "train", "--config", str(config), "--max-epochs", "1"],
    ):
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
    source = ResolvedConfig.load(config).to_scenario().data.datasets[0].source
    assert isinstance(source, AutoCMapSSRULWindows)
    assert source.subset == "FD003"
    assert len(source.sensor_ids or []) == 15
    assert Path(source.artifact_path).is_file()
