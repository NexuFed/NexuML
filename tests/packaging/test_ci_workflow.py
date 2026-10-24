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


def test_shared_release_triggers_and_studio_channels():
    root = Path(__file__).resolve().parents[2]
    python = YAML(typ="safe").load((root / ".github/workflows/release.yml").read_text())
    assert "startsWith(github.event.release.tag_name, 'v')" in python["jobs"]["validate"]["if"]
    studio = YAML(typ="safe").load((root / ".github/workflows/publish-studio.yml").read_text())
    publish = studio["jobs"]["publish"]
    assert studio["on"] == python["on"]
    assert publish["if"] == python["jobs"]["validate"]["if"]
    assert publish["environment"]["name"] == "npm"
    assert publish["permissions"]["id-token"] == "write"
    assert publish["runs-on"] == "ubuntu-latest"  # npm OIDC rejects self-hosted runners.
    scripts = "\n".join(step.get("run", "") for step in publish["steps"])
    assert "npm publish" in scripts and "--ignore-scripts" in scripts and "--provenance" in scripts
    assert "npm ci" not in scripts and "npm pack" not in scripts and "npm run build" not in scripts
    assert "python scripts/check-release-versions.py" in scripts
    assert 'args+=(--tag "$RELEASE_TAG")' in scripts
    assert "git merge-base --is-ancestor HEAD origin/main" in scripts
    assert "a.name === 'studio-distribution' && !a.expired" in scripts
    assert "artifacts[0].workflow_run.head_sha, process.env.GITHUB_SHA" in scripts
    assert "npm install --global npm@11.21.0" in scripts
    candidate = next(step for step in publish["steps"] if "npm publish" in step.get("run", ""))
    assert (
        candidate["if"] == "github.event_name == 'push' || github.event_name == 'workflow_dispatch'"
    )
    assert "--tag next" in candidate["run"]
    promote = next(step for step in publish["steps"] if "npm dist-tag add" in step.get("run", ""))
    assert promote["if"] == "github.event_name == 'release'"
    assert "npm publish" not in promote["run"]
    assert 'npm dist-tag add "$package" latest' in promote["run"]
    assert any(
        step.get("run") == "python scripts/check-release-versions.py"
        for step in WORKFLOW["jobs"]["static"]["steps"]
    )


@pytest.fixture
def release_versions(tmp_path):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "library").mkdir()
    (tmp_path / "studio").mkdir()
    script = tmp_path / "scripts/check-release-versions.py"
    shutil.copyfile(root / "scripts/check-release-versions.py", script)
    for path in ("pyproject.toml", "library/pyproject.toml"):
        (tmp_path / path).write_text('[project]\nversion = "0.3.0"\n')
    (tmp_path / "studio/package.json").write_text(json.dumps({"version": "0.3.0"}))
    (tmp_path / "studio/package-lock.json").write_text(
        json.dumps({"version": "0.3.0", "packages": {"": {"version": "0.3.0"}}})
    )
    return script


def test_shared_release_versions_and_tags(release_versions):
    for args, accepted in [
        ([], True),
        (["--tag", "v0.3.0"], True),
        (["--tag", "v0.3.1"], False),
        (["--tag", "v00.3.0"], False),
        (["--tag", "v0.3.0rc1"], False),
        (["--tag", "studio-v0.3.0"], False),
        (["--tag", "main"], False),
    ]:
        result = subprocess.run(
            [sys.executable, str(release_versions), *args], capture_output=True, text=True
        )
        assert (result.returncode == 0) is accepted, result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "library/pyproject.toml",
        "studio/package.json",
        "studio/package-lock.json",
        "lock-root",
    ],
)
def test_shared_release_rejects_version_drift(release_versions, path):
    root = release_versions.parents[1]
    target = root / ("studio/package-lock.json" if path == "lock-root" else path)
    if path.endswith(".toml"):
        target.write_text('[project]\nversion = "0.4.0"\n')
    else:
        value = json.loads(target.read_text())
        if path == "lock-root":
            value["packages"][""]["version"] = "0.4.0"
        else:
            value["version"] = "0.4.0"
        target.write_text(json.dumps(value))
    result = subprocess.run([sys.executable, str(release_versions)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Package versions must match" in result.stderr


@pytest.mark.parametrize("registry", ["matching", "different", "missing", "empty"])
def test_studio_promotion_requires_the_tested_registry_integrity(tmp_path, registry):
    if not shutil.which("node") or not shutil.which("bash"):
        pytest.skip("Promotion verifier uses Node and bash")
    root = Path(__file__).resolve().parents[2]
    workflow = YAML(typ="safe").load((root / ".github/workflows/publish-studio.yml").read_text())
    source = next(
        step["run"]
        for step in workflow["jobs"]["publish"]["steps"]
        if "npm dist-tag add" in step.get("run", "")
    )
    (tmp_path / "release-studio").mkdir()
    (tmp_path / "release-studio/pack.json").write_text(json.dumps([{"integrity": "sha512-tested"}]))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npm = bin_dir / "npm"
    npm.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['NPM_CALLS'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'view':\n"
        "    print(os.environ['REGISTRY_RESPONSE'])\n"
        "    sys.exit(1 if os.environ['REGISTRY_MODE'] == 'missing' else 0)\n"
    )
    npm.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    response = (
        '"sha512-tested"'
        if registry == "matching"
        else ('"sha512-other"' if registry == "different" else "")
    )
    result = subprocess.run(
        ["bash", "-e", "-c", source],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "RELEASE_VERSION": "0.3.0",
            "NPM_CALLS": str(calls),
            "REGISTRY_MODE": registry,
            "REGISTRY_RESPONSE": response,
        },
    )
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert commands[0] == ["view", "@nexufed/nexuml-studio@0.3.0", "dist.integrity", "--json"]
    if registry == "matching":
        assert result.returncode == 0, result.stderr
        assert commands[1:] == [["dist-tag", "add", "@nexufed/nexuml-studio@0.3.0", "latest"]]
    else:
        assert result.returncode != 0
        assert len(commands) == 1


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
