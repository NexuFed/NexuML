"""Actual old-metadata policy load and fresh CLI replay; no training substitute."""

import json
import os
import subprocess
import sys

from nexuml.core import export
from nexuml.core.compiler import compile
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import PolicyEvaluationSpec
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo


def test_legacy_package_without_provenance_replays_with_unknown_source_frames(
    tmp_path, monkeypatch
):
    scenario = cartpole_ppo(total_frames=64, frames_per_batch=16)
    scenario.logging = None
    scenario.policy_evaluation = PolicyEvaluationSpec(episodes=1, max_steps_per_episode=2, seed=7)
    assert scenario.interaction is not None
    policy = CompiledPolicy.from_pipeline(
        compile(scenario), scenario.interaction.environment.describe()
    )
    original = export._package_payload
    removed = {"training_frames", "policy_files_sha256", "checkpoint"}

    def legacy_payload(pipeline, metadata, **kwargs):
        for key in removed:
            metadata.pop(key, None)
        return original(pipeline, metadata, **kwargs)

    monkeypatch.setattr(export, "_package_payload", legacy_payload)
    package = export.export_package(policy, tmp_path / "legacy")
    _, _, metadata = export.load_inference_package(package)
    assert removed.isdisjoint(metadata)
    assert removed.isdisjoint(json.loads((package / "metadata.json").read_text()))
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "nexuml.cli.main",
            "evaluate",
            str(package),
            "--device",
            "cpu",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "source physical frames: unavailable" in result.stdout
    assert "Episodes: 1/1" in result.stdout
