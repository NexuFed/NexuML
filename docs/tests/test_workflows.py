"""Keep untrusted documentation checks separate from institute and publication jobs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return YAML(typ="safe").load((ROOT / ".github/workflows" / name).read_text())


def test_untrusted_checks_are_read_only_and_hosted():
    checks = load("docs-checks.yml")
    assert "pull_request" in checks["on"] and "workflow_call" in checks["on"]
    assert "pull_request_target" not in checks["on"]
    inputs = checks["on"]["workflow_call"]["inputs"]
    assert set(inputs) == {"version", "trusted-source"}
    assert inputs["trusted-source"] == {"type": "boolean", "default": False}
    assert checks["permissions"] == {"contents": "read"}
    assert "secrets" not in checks["on"]["workflow_call"]
    job = checks["jobs"]["check"]
    assert "permissions" not in job
    checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v5")
    assert checkout["with"] == {"fetch-depth": 0, "persist-credentials": False}
    assert any(
        step.get("run") == "npm --prefix docs run setup:browser -- --with-deps"
        for step in checks["jobs"]["check"]["steps"]
    )
    ci = load("ci.yml")
    assert "pull_request" not in ci["on"] and "pull_request_target" not in ci["on"]
    assert "docs-checks" in ci["jobs"]["required-ci"]["needs"]
    assert ci["jobs"]["docs-checks"]["needs"] == "release-identity"
    assert ci["jobs"]["docs-checks"]["with"]["trusted-source"] is True


@pytest.mark.parametrize("event_name", ["pull_request", "push", "workflow_dispatch"])
@pytest.mark.parametrize("same_repository", [False, True])
@pytest.mark.parametrize("trusted_source", [False, True])
def test_docs_runner_trust_boundary(event_name, same_repository, trusted_source):
    runner = load("docs-checks.yml")["jobs"]["check"]["runs-on"]
    assert runner == (
        "${{ ((github.event_name == 'pull_request' && "
        "github.event.pull_request.head.repo.full_name == github.repository) || "
        "(github.event_name != 'pull_request' && inputs.trusted-source)) "
        "&& 'ika-runner' || 'ubuntu-latest' }}"
    )
    event = SimpleNamespace()
    if event_name == "pull_request":
        event.pull_request = SimpleNamespace(
            head=SimpleNamespace(
                repo=SimpleNamespace(full_name="owner/repo" if same_repository else "fork/repo")
            )
        )
    github = SimpleNamespace(event_name=event_name, event=event, repository="owner/repo")
    expression = (
        runner[3:-2]
        .replace("inputs.trusted-source", "trusted_source")
        .replace("&&", "and")
        .replace("||", "or")
    )
    expected_trust = same_repository if event_name == "pull_request" else trusted_source
    assert eval(expression, {"__builtins__": {}}, locals()) == (
        "ika-runner" if expected_trust else "ubuntu-latest"
    )


def test_durable_staging_precedes_only_the_deploy_queue():
    workflow = load("docs.yml")
    assert "concurrency" not in workflow
    assert workflow["on"]["push"]["branches"] == ["main", "dev"]
    jobs = workflow["jobs"]
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    identity = jobs["release-identity"]
    assert identity == load("ci.yml")["jobs"]["release-identity"]
    assert identity["runs-on"] == (
        "${{ startsWith(github.ref, 'refs/tags/') && 'ubuntu-latest' || 'ika-runner' }}"
    )
    assert identity["steps"][0]["with"] == {
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    assert identity["steps"][1]["with"] == {"python-version": "3.12"}
    assert identity["steps"][-1]["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert identity["steps"][-1]["run"] == "python scripts/release_candidate.py eligibility"
    assert jobs["validate"]["needs"] == "release-identity"
    assert jobs["validate"]["with"]["trusted-source"] is True
    assert jobs["validate"]["with"]["version"] == "${{ github.ref_name }}"
    assert jobs["stage"]["env"]["VERSION"] == "${{ github.ref_name }}"
    stage_script = next(
        step["run"] for step in jobs["stage"]["steps"] if "Durably" in step.get("name", "")
    )
    assert "if version == 'main':" in stage_script
    assert "f'refs/remotes/origin/{version}'" in stage_script
    assert jobs["stage"]["needs"] == "validate"
    assert jobs["stage"]["runs-on"] == "ika-runner"
    assert "concurrency" not in jobs["stage"]
    assert jobs["stage"]["permissions"] == {"contents": "write"}
    assert jobs["deploy"]["needs"] == "stage"
    assert jobs["deploy"]["runs-on"] == "ubuntu-latest"
    assert jobs["deploy"]["concurrency"]["cancel-in-progress"] is False
    assert jobs["deploy"]["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    steps = jobs["deploy"]["steps"]
    assert any(step.get("with", {}).get("ref") == "gh-pages" for step in steps)
    assert any(str(step.get("uses", "")).startswith("actions/deploy-pages@") for step in steps)
