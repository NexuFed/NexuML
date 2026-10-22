"""Keep untrusted documentation checks separate from institute and publication jobs."""

from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return YAML(typ="safe").load((ROOT / ".github/workflows" / name).read_text())


def test_untrusted_checks_are_read_only_and_hosted():
    checks = load("docs-checks.yml")
    assert "pull_request" in checks["on"] and "workflow_call" in checks["on"]
    assert checks["permissions"] == {"contents": "read"}
    assert any(
        step.get("run") == "npm --prefix docs run setup:browser -- --with-deps"
        for step in checks["jobs"]["check"]["steps"]
    )
    assert all(job.get("runs-on") == "ubuntu-latest" for job in checks["jobs"].values())
    ci = load("ci.yml")
    assert "pull_request" not in ci["on"] and "pull_request_target" not in ci["on"]
    assert "docs-checks" in ci["jobs"]["required-ci"]["needs"]


def test_durable_staging_precedes_only_the_deploy_queue():
    workflow = load("docs.yml")
    assert "concurrency" not in workflow
    jobs = workflow["jobs"]
    assert jobs["stage"]["needs"] == "validate"
    assert "concurrency" not in jobs["stage"]
    assert jobs["stage"]["permissions"] == {"contents": "write"}
    assert jobs["deploy"]["needs"] == "stage"
    assert jobs["deploy"]["concurrency"]["cancel-in-progress"] is False
    assert jobs["deploy"]["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    steps = jobs["deploy"]["steps"]
    assert any(step.get("with", {}).get("ref") == "gh-pages" for step in steps)
    assert any(str(step.get("uses", "")).startswith("actions/deploy-pages@") for step in steps)
