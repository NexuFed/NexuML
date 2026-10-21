"""CPU smoke test exercising the real NexuML resolve/build/train CLI path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from nexuml.core.compiler import compile as compile_pipeline
from nexuml_library.evaluation.maintenance.rul_prepare import (
    prepare_cmapss,
    save_rul_artifact,
)
from nexuml_library.scenarios.maintenance.rul import maintenance_rul_benchmark


def test_native_nexuml_rul_cli_build_and_train(tmp_path: Path) -> None:
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
    payload = prepare_cmapss(root, window=3, seed=42)
    artifact = tmp_path / "prepared.pt"
    save_rul_artifact(payload, artifact)
    scenario = maintenance_rul_benchmark(
        artifact,
        architecture="feature_mlp",
        hidden_dim=12,
        depth=1,
        batch_size=4,
        max_epochs=1,
        patience=2,
        seed=42,
        checkpoint_root=tmp_path / "checkpoints",
    )
    path = tmp_path / "config.yaml"
    compile_pipeline(scenario).resolved_config.save(path)
    for command in (
        ["nexuml", "build", str(path)],
        ["nexuml", "train", "--config", str(path)],
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
