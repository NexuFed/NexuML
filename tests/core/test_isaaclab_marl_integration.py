"""Explicit native D gate; CPU shape/API tests never stand in for these jobs."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.requires_simulator
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("family", ["pendulum", "handover"])
@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_native_grouped_updates_resume_export_and_fresh_replay(tmp_path, family, algorithm):
    assert shutil.disk_usage(ROOT).free > 4 * 2**30
    fixture = ROOT / "tests/fixtures/isaaclab_marl_job.py"
    for stage in ("interrupt", "resume", "exhausted", "infer", "replay"):
        assert shutil.disk_usage(ROOT).free > 4 * 2**30
        command = [
            sys.executable,
            *(["-I"] if stage == "infer" else [str(ROOT / "scripts/isaaclab/run.py")]),
            str(fixture),
            family,
            algorithm,
            stage,
            str(tmp_path),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NEXUML_LOGS_ROOT": str(tmp_path / "native-logs"),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        (tmp_path / f"{stage}.log").write_text(result.stdout)
        assert result.returncode == 0, result.stdout
        expected = {
            "interrupt": "GROUPED_CURRENT_CHECKPOINT_INTERRUPTED_AT_PHYSICAL_FRAMES",
            "resume": "GROUPED_NATIVE_ACTOR_CRITIC_UPDATES_COUNTERS_EXPORT_AND_CLEANUP_VERIFIED",
            "exhausted": "GROUPED_NATIVE_ACTOR_CRITIC_UPDATES_COUNTERS_EXPORT_AND_CLEANUP_VERIFIED",
            "infer": "GROUPED_EXACT_ACTOR_ONLY_CLEAN_INFERENCE_AND_SHARING_VERIFIED",
            "replay": "GROUPED_FRESH_NATIVE_DECENTRALIZED_REPLAY_VERIFIED",
        }[stage]
        assert expected in result.stdout
        if stage in {"interrupt", "resume"}:
            assert "NATIVE_GROUP_FINAL_OBSERVATIONS_AND_CACHED_RESET_VERIFIED" in result.stdout
            assert "NATIVE_LOCAL_ACTOR_AND_DECLARED_CRITIC_DEPENDENCIES_VERIFIED" in result.stdout
