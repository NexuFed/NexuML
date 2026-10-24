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
    assert any(
        step.get("with", {}).get("name") == "studio-distribution${{ env.ARTIFACT_SUFFIX }}"
        for step in job["steps"]
    )
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
    assert studio["jobs"]["validate"]["if"] == python["jobs"]["validate"]["if"]
    assert publish["needs"] == "validate"
    assert publish["environment"]["name"] == "npm"
    assert publish["permissions"]["id-token"] == "write"
    assert publish["runs-on"] == "ubuntu-latest"  # npm OIDC rejects self-hosted runners.
    scripts = "\n".join(step.get("run", "") for step in publish["steps"])
    assert "npm publish" in scripts and "--ignore-scripts" in scripts and "--provenance" in scripts
    assert "npm ci" not in scripts and "npm pack" not in scripts and "npm run build" not in scripts
    assert any(
        step.get("uses") == "./.github/actions/release-artifact" for step in publish["steps"]
    )
    for workflow in (python, studio):
        validation = workflow["jobs"]["validate"]
        assert validation["timeout-minutes"] == 120
        assert "environment" not in validation
        assert "id-token" not in validation.get("permissions", {})
        assert any(
            "scripts/release_candidate.py resolve" in step.get("run", "")
            for step in validation["steps"]
        )
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
    shutil.copyfile(root / "scripts/release_versions.py", tmp_path / "scripts/release_versions.py")
    for path in ("pyproject.toml", "library/pyproject.toml"):
        (tmp_path / path).write_text('[project]\nversion = "0.3.0"\n')
    (tmp_path / "studio/package.json").write_text(json.dumps({"version": "0.3.0"}))
    (tmp_path / "studio/package-lock.json").write_text(
        json.dumps({"version": "0.3.0", "packages": {"": {"version": "0.3.0"}}})
    )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "nexuml"\nversion = "0.3.0"\nsource = { editable = "." }\n'
        '[[package]]\nname = "nexuml-library"\nversion = "0.3.0"\n'
        'source = { editable = "library" }\n'
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
    (tmp_path / "release-dist/studio").mkdir(parents=True)
    (tmp_path / "release-dist/studio/pack.json").write_text(
        json.dumps([{"integrity": "sha512-tested"}])
    )
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
    assert "scripts/release_candidate.py resolve" in validation
    assert "--require-testpypi" in validation
    assert jobs["publish-pypi-core"]["needs"] == "validate"
    assert "publish-pypi-library" in jobs["github-release"]["needs"]
    assert "publish-pypi-core" in jobs["publish-pypi-library"]["needs"]
    for name in (
        "publish-testpypi-core",
        "publish-testpypi-library",
        "publish-pypi-core",
        "publish-pypi-library",
        "github-release",
    ):
        download = next(
            step
            for step in jobs[name]["steps"]
            if step.get("uses") == "./.github/actions/release-artifact"
        )
        assert download["with"]["manifest-id"] == "${{ needs.validate.outputs.manifest-id }}"
        assert download["with"]["artifact-id"] == "${{ needs.validate.outputs.python-id }}"
        assert download["with"]["group"] == "python"
        assert not any("uv build" in step.get("run", "") for step in jobs[name]["steps"])
    verification = jobs["verify-testpypi"]
    assert verification["env"]["NEXUML_EXPECTED_VERSION"] == "${{ needs.validate.outputs.version }}"
    scripts = "\n".join(step.get("run", "") for step in verification["steps"])
    assert "--group wheels" in scripts
    assert "record-testpypi" in scripts
    assert scripts.index("--group wheels") < scripts.index("uv pip install")
    assert scripts.index("test_installed_distributions.py -k library") < scripts.index(
        "record-testpypi"
    )
    assert not any(step.get("continue-on-error") for step in verification["steps"])
    assert any(
        step.get("with", {}).get("name") == "testpypi-verification-${{ github.run_attempt }}"
        for step in verification["steps"]
    )


def test_tag_ci_prepares_every_package_checkout_before_use():
    root = Path(__file__).resolve().parents[2]
    assert WORKFLOW["on"]["push"]["tags"] == ["v*"]
    assert WORKFLOW["env"]["ARTIFACT_SUFFIX"] == (
        "${{ startsWith(github.ref, 'refs/tags/') && format('-{0}', github.run_attempt) || '' }}"
    )
    jobs = WORKFLOW["jobs"]
    assert jobs["release-identity"]["runs-on"] == "ubuntu-latest"
    for name in ("static", "test", "studio-static", "docs-checks"):
        assert jobs[name]["needs"] == "release-identity"
    docs = YAML(typ="safe").load((root / ".github/workflows/docs-checks.yml").read_text())
    for job in [
        jobs[name]
        for name in ("static", "test", "package", "studio-static", "studio-build", "studio-package")
    ] + [docs["jobs"]["check"]]:
        steps = job["steps"]
        prepared = next(
            i
            for i, step in enumerate(steps)
            if step.get("uses") == "./.github/actions/prepare-release"
        )
        assert steps[prepared]["if"] == "startsWith(github.ref, 'refs/tags/')"
        for i, step in enumerate(steps):
            command = step.get("run", "")
            if any(
                value in command
                for value in (
                    "uv pip install",
                    "uv build",
                    "npm ci",
                    "run check",
                    "studio-test-runtime.py",
                )
            ):
                assert i > prepared
            if step.get("with", {}).get("cache"):
                assert "!startsWith(github.ref" in step["with"]["cache"]
    gate = jobs["required-ci"]
    assert gate["runs-on"] == "ubuntu-latest"
    assert gate["steps"][0]["name"] == "Verify required jobs"
    assert any(
        "scripts/release_candidate.py create" in step.get("run", "") for step in gate["steps"]
    )
    manifest = gate["steps"][-1]
    assert manifest["with"]["name"] == "release-candidate-${{ github.run_attempt }}"
    assert manifest["with"]["retention-days"] == 7
    action = YAML(typ="safe").load(
        (root / ".github/actions/release-artifact/action.yml").read_text()
    )
    assert "--group" in action["runs"]["steps"][-1]["run"]
    assert "scripts/release_candidate.py verify" in action["runs"]["steps"][-1]["run"]
