import os
import shutil
import subprocess
from pathlib import Path

import pytest
from ruamel.yaml import YAML


WORKFLOW = YAML(typ="safe").load(
    (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
)


def test_studio_ci_preserves_platform_and_installation_coverage():
    for event in ("push", "pull_request"):
        assert {"main", "dev"} <= set(WORKFLOW["on"][event]["branches"])
    job = WORKFLOW["jobs"]["studio-package"]
    assert {"package", "studio-static"} <= set(job["needs"])
    assert job["strategy"]["fail-fast"] is False
    assert set(job["strategy"]["matrix"]["os"]) == {
        "ubuntu-latest",
        "macos-latest",
        "windows-latest",
    }
    assert set(job["strategy"]["matrix"]["mode"]) == {"tool", "project"}


def test_required_ci_checks_every_job_and_fails_closed(tmp_path):
    if not shutil.which("bash"):
        pytest.skip("CI gate uses bash")
    jobs = WORKFLOW["jobs"]
    gate = jobs["required-ci"]
    assert set(gate["needs"]) == set(jobs) - {"required-ci"}
    assert gate["if"] == "${{ always() }}"
    step = gate["steps"][0]
    assert set(step["env"].values()) == {
        f"${{{{ needs.{name}.result }}}}" for name in gate["needs"]
    }
    results = dict.fromkeys(step["env"], "success")
    env = {**os.environ, "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")}

    def run(values):
        return subprocess.run(
            ["bash", "-e", "-c", step["run"]],
            env={**env, **values},
            capture_output=True,
            text=True,
            check=False,
        ).returncode

    assert run(results) == 0
    for name in results:
        for outcome in ("failure", "skipped", "cancelled"):
            assert run({**results, name: outcome}) == 1, (name, outcome)
