import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML


WORKFLOW = YAML(typ="safe").load(
    (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
)


def test_studio_ci_builds_once_and_tests_both_trusted_linux_installations(tmp_path):
    assert "pull_request" not in WORKFLOW["on"]
    assert "pull_request_target" not in WORKFLOW["on"]
    assert WORKFLOW["on"]["push"]["branches"] == ["**"]
    tool_paths = next(
        step["run"]
        for step in WORKFLOW["jobs"]["package"]["steps"]
        if step.get("name") == "Configure writable uv tool paths"
    )
    assert "UV_TOOL_DIR=$RUNNER_TEMP/" in tool_paths
    assert "UV_TOOL_BIN_DIR=$RUNNER_TEMP/" in tool_paths
    assert WORKFLOW["jobs"]["test"]["runs-on"] == "ika-runner-gpu"
    assert any(
        "assert torch.cuda.is_available()" in step.get("run", "")
        for step in WORKFLOW["jobs"]["test"]["steps"]
    )
    job = WORKFLOW["jobs"]["studio-package"]
    assert {"package", "studio-build"} <= set(job["needs"])
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"]["os"] == ["ika-runner"]
    assert set(job["strategy"]["matrix"]["mode"]) == {"tool", "project"}
    build = WORKFLOW["jobs"]["studio-build"]
    assert (
        sum("npm pack --ignore-scripts --json" in step.get("run", "") for step in build["steps"])
        == 1
    )
    assert not any("npm pack" in step.get("run", "") for step in job["steps"])
    assert any(step.get("with", {}).get("name") == "studio-distribution" for step in job["steps"])
    smoke = Path(__file__).resolve().parents[2] / "studio/scripts/package-smoke.mjs"
    assert '"prepack"' not in smoke.read_text()
    setup = smoke.parents[2] / "scripts/studio-test-runtime.py"
    result = subprocess.run(
        [
            sys.executable,
            str(setup),
            "--wheel",
            "missing.whl",
            "--library-wheel",
            "missing.whl",
            "--mode",
            "tool",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "must be a new directory" in result.stderr


def test_release_namespaces_and_studio_tag_validation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    python = YAML(typ="safe").load((root / ".github/workflows/release.yml").read_text())
    assert "startsWith(github.event.release.tag_name, 'v')" in python["jobs"]["validate"]["if"]
    studio = YAML(typ="safe").load((root / ".github/workflows/publish-studio.yml").read_text())
    publish = studio["jobs"]["publish"]
    assert publish["if"] == "startsWith(github.event.release.tag_name, 'studio-v')"
    assert publish["environment"]["name"] == "npm"
    assert publish["permissions"]["id-token"] == "write"
    assert publish["runs-on"] == "ubuntu-latest"  # npm OIDC rejects self-hosted runners.
    scripts = "\n".join(step.get("run", "") for step in publish["steps"])
    assert "npm publish" in scripts and "--ignore-scripts" in scripts and "--provenance" in scripts
    assert "npm ci" not in scripts and "npm pack" not in scripts and "npm run build" not in scripts
    if not shutil.which("node"):
        pytest.skip("Tag validator uses Node")
    source = next(
        step["run"] for step in publish["steps"] if step.get("name", "").startswith("Validate tag")
    )
    validator = source.split("<<'JS'\n", 1)[1].split("\nJS", 1)[0]
    (tmp_path / "studio").mkdir()
    (tmp_path / "studio/package.json").write_text(
        json.dumps({"name": "@nexufed/nexuml-studio", "version": "0.1.0"})
    )
    for tag, accepted in [
        ("studio-v0.1.0", True),
        ("v0.1.0", False),
        ("studio-v0.1.1", False),
        ("studio-v0.1.0-rc.1", False),
    ]:
        result = subprocess.run(
            ["node", "--input-type=module", "-e", validator],
            cwd=tmp_path,
            env={**os.environ, "RELEASE_TAG": tag},
            capture_output=True,
            check=False,
        )
        assert (result.returncode == 0) is accepted, (tag, result.stderr)


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


def test_python_release_tag_push_tests_before_release_publication():
    root = Path(__file__).resolve().parents[2]
    workflow = YAML(typ="safe").load((root / ".github/workflows/release.yml").read_text())
    assert workflow["on"]["push"] == {"tags": ["v*"]}
    assert workflow["on"]["release"] == {"types": ["published"]}
    assert "workflow_dispatch" in workflow["on"]
    jobs = workflow["jobs"]
    assert jobs["validate"]["if"].startswith(
        "(github.event_name == 'push' && !github.event.deleted) ||"
    )
    for name in ("publish-testpypi-core", "publish-testpypi-library", "verify-testpypi"):
        assert jobs[name]["if"] == (
            "github.event_name == 'push' || github.event_name == 'workflow_dispatch'"
        )
    for name in ("publish-pypi-core", "publish-pypi-library", "github-release"):
        assert jobs[name]["if"] == "github.event_name == 'release'"
    validation = "\n".join(step.get("run", "") for step in jobs["validate"]["steps"])
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in validation
    assert 'select(.name == "CI"' in validation
    assert "head_sha=$GITHUB_SHA&status=success" in validation
    assert 'select(.name == "python-distributions" and .expired == false)' in validation


@pytest.mark.parametrize("event", ["push", "release", "workflow_dispatch"])
def test_python_release_version_validation(tmp_path, event):
    root = Path(__file__).resolve().parents[2]
    workflow = YAML(typ="safe").load((root / ".github/workflows/release.yml").read_text())
    source = next(
        step["run"]
        for step in workflow["jobs"]["validate"]["steps"]
        if step.get("name") == "Validate release version"
    )
    validator = source.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    (tmp_path / "library").mkdir()
    for directory in (tmp_path, tmp_path / "library"):
        (directory / "pyproject.toml").write_text('[project]\nversion = "0.3.0"\n')

    def run(ref):
        return subprocess.run(
            [sys.executable, "-c", validator, event, ref],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    for ref in ("v0.3.0", "v0.3.1", "v0.3.0rc1", "v00.3.0", "studio-v0.3.0", "main"):
        result = run(ref)
        assert (result.returncode == 0) is (event == "workflow_dispatch" or ref == "v0.3.0")
    (tmp_path / "library/pyproject.toml").write_text('[project]\nversion = "0.4.0"\n')
    assert run("v0.3.0").returncode != 0
