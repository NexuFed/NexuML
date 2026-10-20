"""Two real CPU torchrun ranks exercise the canonical remote Lightning lifecycle."""

import os
import subprocess
import sys
from pathlib import Path

from nexuml.core.config import ResolvedConfig
from nexuml_library.scenarios.asd.synthetic_linear_ae import synthetic_linear_ae_reconstruction


def test_two_cpu_ranks_share_one_native_lifecycle(tmp_path):
    scenario = synthetic_linear_ae_reconstruction(
        feature_shape=(8,),
        num_samples=16,
        hidden_dims=[8],
        latent_dim=2,
        batch_size=4,
        max_epochs=1,
    )
    scenario.training.accelerator = "cpu"
    scenario.training.devices = 2
    scenario.evaluation.algorithms = []
    scenario.logging = None
    path = tmp_path / "worker.yaml"
    ResolvedConfig.from_scenario(scenario).save(path)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "-m",
            "nexuml.execution.worker",
            "--config",
            str(path),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": f"{root / 'src'}{os.pathsep}{root / 'library/src'}",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-16000:]
    output = result.stdout + result.stderr
    assert "All distributed processes registered" in output
    assert "max_epochs=1" in output
