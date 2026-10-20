"""Execute the canonical tutorial using installed wheels, never repository source imports."""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    "NEXUML_INSTALLED_DIST_TEST" not in os.environ,
    reason="installed distribution environment was not requested",
)


def test_documented_synthetic_cpu_first_run(tmp_path):
    root = Path(__file__).resolve().parents[2]
    guide = (root / "docs/src/content/docs/start/first-run.mdx").read_text()
    commands = []
    for fence in re.findall(r"```(?:bash|sh|shell)\n(.*?)```", guide, re.DOTALL):
        for line in fence.replace("\\\n", " ").splitlines():
            if line.startswith("nexuml "):
                commands.append(shlex.split(line)[1:])
    assert [cmd[0] for cmd in commands] == ["registry", "resolve", "build", "train"]
    assert "synthetic-linear-ae-reconstruction" in commands[1]
    assert "--output" in commands[1]
    train = commands[-1]
    assert train[train.index("--max-epochs") + 1] == "1"
    assert "training.accelerator=cpu" in train and "training.devices=1" in train
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / ".config"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    env.pop("PYTHONPATH", None)
    output = []
    for args in commands:
        result = subprocess.run(
            [sys.executable, "-I", "-m", "nexuml", *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        output.append(result.stdout + result.stderr)
    assert "synthetic-linear-ae-reconstruction" in output[0]
    assert "Pipeline compiled successfully" in output[2]
    assert "max_epochs=1" in output[3]
    config = tmp_path / commands[1][commands[1].index("--output") + 1]
    assert config.is_file()
