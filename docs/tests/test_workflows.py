"""Keep untrusted documentation checks separate from institute and publication jobs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return YAML(typ="safe").load((ROOT / ".github/workflows" / name).read_text())


def test_docs_checks_are_read_only_and_skip_untrusted_forks():
    checks = load("docs-checks.yml")
    assert "pull_request" in checks["on"] and "workflow_call" in checks["on"]
    assert "pull_request_target" not in checks["on"]
    inputs = checks["on"]["workflow_call"]["inputs"]
    assert set(inputs) == {"version"}
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
    assert set(ci["jobs"]["docs-checks"]["with"]) == {"version"}


@pytest.mark.parametrize("event_name", ["pull_request", "push", "workflow_dispatch"])
@pytest.mark.parametrize("same_repository", [False, True])
def test_docs_runner_trust_boundary(event_name, same_repository):
    job = load("docs-checks.yml")["jobs"]["check"]
    assert job["runs-on"] == "ika-runner"
    assert job["if"] == (
        "github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository"
    )
    event = SimpleNamespace()
    if event_name == "pull_request":
        event.pull_request = SimpleNamespace(
            head=SimpleNamespace(
                repo=SimpleNamespace(full_name="owner/repo" if same_repository else "fork/repo")
            )
        )
    github = SimpleNamespace(event_name=event_name, event=event, repository="owner/repo")
    expression = job["if"].replace("||", "or")
    expected_run = event_name != "pull_request" or same_repository
    assert eval(expression, {"__builtins__": {}}, {"github": github}) is expected_run


def test_durable_staging_precedes_only_the_deploy_queue():
    workflow = load("docs.yml")
    assert "concurrency" not in workflow
    assert workflow["on"]["push"]["branches"] == ["main", "dev"]
    assert workflow["on"]["push"]["tags"] == ["v*", "!v*-rc.*"]
    assert "workflow_dispatch" in workflow["on"]
    jobs = workflow["jobs"]
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    identity = jobs["release-identity"]
    assert identity == load("ci.yml")["jobs"]["release-identity"]
    assert identity["runs-on"] == "ika-runner"
    assert identity["steps"][0]["with"] == {
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    assert identity["steps"][1]["with"] == {"python-version": "3.12"}
    assert identity["steps"][-1]["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert identity["steps"][-1]["run"] == "python scripts/release_candidate.py eligibility"
    assert jobs["validate"]["needs"] == "release-identity"
    assert jobs["validate"]["if"] == "github.event_name == 'push'"
    assert jobs["validate"]["with"] == {"version": "${{ github.ref_name }}"}
    assert jobs["stage"]["env"]["VERSION"] == "${{ github.ref_name }}"
    stage_script = next(
        step["run"] for step in jobs["stage"]["steps"] if "Durably" in step.get("name", "")
    )
    assert "if version == 'main':" in stage_script
    assert "f'refs/remotes/origin/{version}'" in stage_script
    assert jobs["stage"]["needs"] == "validate"
    assert jobs["stage"]["runs-on"] == "ika-runner"
    assert "concurrency" not in jobs["stage"]
    assert jobs["stage"]["permissions"] == {"contents": "write", "actions": "write"}
    dispatch = jobs["stage"]["steps"][-1]
    assert dispatch["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert dispatch["run"] == 'gh workflow run docs.yml --repo "$GITHUB_REPOSITORY" --ref main'
    assert jobs["deploy"]["needs"] == "release-identity"
    assert jobs["deploy"]["runs-on"] == "ika-runner"
    assert jobs["deploy"]["concurrency"]["cancel-in-progress"] is False
    assert jobs["deploy"]["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    steps = jobs["deploy"]["steps"]
    assert any(step.get("with", {}).get("ref") == "gh-pages" for step in steps)
    assert any(str(step.get("uses", "")).startswith("actions/deploy-pages@") for step in steps)


@pytest.mark.parametrize("event_name", ["push", "workflow_dispatch"])
@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/dev", "refs/tags/v0.2.3"])
def test_pages_dispatch_only_deploys_from_reviewed_main(event_name, ref):
    job = load("docs.yml")["jobs"]["deploy"]
    assert job["if"] == (
        "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'"
    )
    github = SimpleNamespace(event_name=event_name, ref=ref)
    assert eval(job["if"].replace("&&", "and"), {"__builtins__": {}}, {"github": github}) is (
        event_name == "workflow_dispatch" and ref == "refs/heads/main"
    )


def test_pages_transport_is_unique_and_live_content_is_verified():
    steps = load("docs.yml")["jobs"]["deploy"]["steps"]
    upload = next(step for step in steps if "upload-pages-artifact@" in step.get("uses", ""))
    deploy = next(step for step in steps if "deploy-pages@" in step.get("uses", ""))
    assert upload["with"]["name"] == "github-pages-${{ github.run_id }}-${{ github.run_attempt }}"
    assert deploy["with"]["artifact_name"] == upload["with"]["name"]
    verify = steps[-1]
    assert verify["env"] == {"PAGE_URL": "${{ steps.deployment.outputs.page_url }}"}
    assert verify["run"] == (
        "python source/docs/scripts/verify_live.py "
        '--directory "${{ steps.assembled.outputs.path }}" --url "$PAGE_URL"'
    )
