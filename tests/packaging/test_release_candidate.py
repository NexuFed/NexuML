"""Synthetic publication files and read-only GitHub fixtures; never contact a registry."""

import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "release_candidate", Path(__file__).resolve().parents[2] / "scripts/release_candidate.py"
)
assert SPEC is not None and SPEC.loader is not None
rc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc)


REPO = "NexuFed/NexuML"
REF = "refs/tags/v0.2.2"
VERSION = "0.2.2"
PREFIX = f"repos/{REPO}"
START = "2026-10-24T01:00:00Z"
CREATED = "2026-10-24T01:05:00Z"
SECOND = "2026-10-24T02:00:00Z"


def zipped(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


def tarred(members):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


class FakeAPI:
    def __init__(self):
        self.values = {}
        self.calls = []

    def api(self, path, *, pages=False, binary=False):
        self.calls.append((path, pages, binary))
        assert path in self.values, f"Unexpected GET: {path}"
        value = self.values[path]
        return copy.deepcopy(value)


def workflow(
    commit,
    *,
    run_id=20,
    attempt=1,
    status="completed",
    conclusion="success",
    branch="v0.2.2",
    workflow=rc.CI_WORKFLOW,
    event="push",
    start=START,
):
    return {
        "id": run_id,
        "run_attempt": attempt,
        "head_sha": commit,
        "head_branch": branch,
        "head_repository": {"full_name": REPO},
        "repository": {"full_name": REPO},
        "path": workflow,
        "event": event,
        "status": status,
        "conclusion": conclusion,
        "run_started_at": start,
    }


def artifact(commit, name, artifact_id, *, run_id=20, created=CREATED):
    return {
        "id": artifact_id,
        "name": name,
        "expired": False,
        "created_at": created,
        "workflow_run": {"id": run_id, "head_sha": commit, "head_branch": "v0.2.2"},
    }


def publish_attempt(api, run, artifacts, *, job="CI"):
    run_id, number = run["id"], run["run_attempt"]
    api.values[f"{PREFIX}/actions/runs/{run_id}/attempts/{number}"] = run
    api.values[f"{PREFIX}/actions/runs/{run_id}/attempts/{number}/jobs?per_page=100"] = [
        {
            "jobs": [
                {
                    "name": job,
                    "status": "completed",
                    "conclusion": "success",
                    "run_id": run_id,
                    "run_attempt": number,
                }
            ]
        }
    ]
    api.values[f"{PREFIX}/actions/runs/{run_id}/artifacts?per_page=100"] = [
        {"artifacts": artifacts}
    ]


def distributions(root, dist, commit, version=VERSION):
    files = {}
    for directory, name in (("nexuml", "nexuml"), ("nexuml-library", "nexuml_library")):
        metadata = (
            f"Metadata-Version: 2.3\nName: {name.replace('_', '-')}\nVersion: {version}\n\n"
        ).encode()
        files[f"{directory}/{name}-{version}-py3-none-any.whl"] = zipped(
            {
                f"{name}-{version}.dist-info/METADATA": metadata,
                f"{name}/__init__.py": b"",
            }
        )
        files[f"{directory}/{name}-{version}.tar.gz"] = tarred(
            {
                f"{name}-{version}/PKG-INFO": metadata,
                f"{name}-{version}/pyproject.toml": f'[project]\nversion = "{version}"\n'.encode(),
            }
        )
    package = json.loads((root / "studio/package.json").read_bytes())
    package["version"] = version
    filename = f"nexufed-nexuml-studio-{version}.tgz"
    tarball = tarred({"package/package.json": rc.json_bytes(package)})
    integrity = "sha512-" + base64.b64encode(hashlib.sha512(tarball).digest()).decode()
    files[f"studio/{filename}"] = tarball
    files["studio/pack.json"] = rc.json_bytes(
        [
            {
                "name": package["name"],
                "version": version,
                "filename": filename,
                "integrity": integrity,
            }
        ]
    )
    files["studio/commit.txt"] = (commit + "\n").encode()
    for name, raw in files.items():
        path = dist / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return files


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "library").mkdir()
    (root / "studio").mkdir()
    for name in ("pyproject.toml", "library/pyproject.toml"):
        (root / name).write_text('[project]\nversion = "0.2.1"\n')
    package = {
        "name": "@nexufed/nexuml-studio",
        "version": "0.2.1",
        "repository": {
            "type": "git",
            "url": "git+https://example.invalid/repo.git",
            "directory": "studio",
        },
        "license": "Apache-2.0",
        "bin": {"nexuml-studio": "bin/studio.mjs"},
        "engines": {"node": ">=22.13.0"},
        "publishConfig": {"access": "public", "provenance": True},
    }
    (root / "studio/package.json").write_bytes(rc.json_bytes(package))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = rc.git(root, "rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(root), "tag", "v0.2.2"], check=True)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GITHUB_REF", REF)
    monkeypatch.setenv("GITHUB_SHA", commit)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_RUN_ID", "20")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    event = tmp_path / "event.json"
    event.write_text('{"deleted": false}')
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(rc, "__file__", str(root / "scripts/release_candidate.py"))
    dist = tmp_path / "dist"
    files = distributions(root, dist, commit)
    for name in ("pyproject.toml", "library/pyproject.toml"):
        (root / name).write_text('[project]\nversion = "0.2.2"\n')
    package["version"] = VERSION
    (root / "studio/package.json").write_bytes(rc.json_bytes(package))
    manifest = rc.create(root, dist, 102, 103)
    raw = rc.json_bytes(manifest)
    api = FakeAPI()
    api.values[f"{PREFIX}/git/ref/tags/v0.2.2"] = {"object": {"type": "commit", "sha": commit}}
    run = workflow(commit)
    api.values[f"{PREFIX}/actions/workflows/ci.yml/runs?branch=v0.2.2&event=push&per_page=100"] = [
        {"workflow_runs": [run]}
    ]
    artifacts = [
        artifact(commit, name, artifact_id)
        for name, artifact_id in (
            ("release-candidate-1", 101),
            ("python-distributions-1", 102),
            ("studio-distribution-1", 103),
        )
    ]
    publish_attempt(api, run, artifacts)
    api.values[f"{PREFIX}/actions/artifacts/101/zip"] = zipped({"release-candidate.json": raw})
    api.values[f"{PREFIX}/actions/artifacts/102/zip"] = zipped(
        {name: data for name, data in files.items() if not name.startswith("studio/")}
    )
    api.values[f"{PREFIX}/actions/artifacts/103/zip"] = zipped(
        {
            name.removeprefix("studio/"): data
            for name, data in files.items()
            if name.startswith("studio/")
        }
    )
    return {
        "root": root,
        "dist": dist,
        "manifest": manifest,
        "raw": raw,
        "api": api,
        "commit": commit,
        "run": run,
        "artifacts": artifacts,
        "files": files,
    }


def scan(c):
    return rc.scan_candidates(c["api"], REPO, REF, VERSION, c["commit"])


def test_create_is_deterministic_and_checks_real_metadata(candidate):
    c = candidate
    assert rc.json_bytes(rc.create(c["root"], c["dist"], 102, 103)) == c["raw"]
    assert len(c["manifest"]["files"]) == 5
    assert c["manifest"]["version"] == VERSION
    assert rc.git(c["root"], "show", "HEAD:pyproject.toml") == '[project]\nversion = "0.2.1"'
    rc.verify_files(c["root"], c["dist"], c["manifest"])


@pytest.mark.parametrize("python_id,studio_id", [(0, 103), (-1, 103), (102, 102), (True, 103)])
def test_create_rejects_invalid_artifact_ids(candidate, python_id, studio_id):
    with pytest.raises(rc.CandidateError, match="positive and distinct"):
        rc.create(candidate["root"], candidate["dist"], python_id, studio_id)


@pytest.mark.parametrize(
    "filename", ["pyproject.toml", "library/pyproject.toml", "studio/package.json"]
)
def test_create_requires_stamped_source(candidate, filename):
    c = candidate
    path = c["root"] / filename
    path.write_text(path.read_text().replace(VERSION, "0.2.1"))
    with pytest.raises(rc.CandidateError, match="Prepare"):
        rc.create(c["root"], c["dist"], 102, 103)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ref", "refs/tags/v0.2.3"),
        ("commit", "f" * 40),
        ("repository", "wrong/repo"),
        ("workflow", rc.RELEASE_WORKFLOW),
        ("schema", True),
        ("run_attempt", 0),
    ],
)
def test_manifest_identity_rejected(candidate, field, value):
    manifest = copy.deepcopy(candidate["manifest"])
    manifest[field] = value
    with pytest.raises(rc.CandidateError):
        rc.validate_manifest(manifest, REPO, REF, VERSION, candidate["commit"])


def test_hash_and_manifest_path_rejected(candidate):
    c = candidate
    wheel = c["dist"] / rc.filenames(VERSION)["python"][0]
    wheel.write_bytes(wheel.read_bytes() + b"modified")
    with pytest.raises(rc.CandidateError, match="SHA-256"):
        rc.verify_files(c["root"], c["dist"], c["manifest"], "python")
    c["manifest"]["files"]["../outside"] = "0" * 64
    with pytest.raises(rc.CandidateError, match="unsafe filenames"):
        rc.validate_manifest(c["manifest"], REPO, REF, VERSION, c["commit"])


@pytest.mark.parametrize(
    "kind",
    [
        "wheel-name",
        "wheel-version",
        "sdist-version",
        "sdist-project",
        "studio-license",
        "studio-version",
        "pack-integrity",
        "pack-filename",
        "commit",
    ],
)
def test_wrong_package_metadata_rejected_even_with_matching_hash(candidate, kind):
    c = candidate
    manifest = copy.deepcopy(c["manifest"])
    names = rc.filenames(VERSION)
    name = names["python"][0]
    if kind.startswith("wheel"):
        package_name = "wrong" if kind == "wheel-name" else "nexuml"
        version = "9.9.9" if kind == "wheel-version" else VERSION
        metadata = f"Name: {package_name}\nVersion: {version}\n".encode()
        raw = zipped({f"nexuml-{VERSION}.dist-info/METADATA": metadata})
    elif kind.startswith("sdist"):
        name = names["python"][1]
        version = "9.9.9" if kind == "sdist-version" else VERSION
        project_version = "9.9.9" if kind == "sdist-project" else VERSION
        raw = tarred(
            {
                f"nexuml-{VERSION}/PKG-INFO": f"Name: nexuml\nVersion: {version}\n".encode(),
                f"nexuml-{VERSION}/pyproject.toml": (
                    f'[project]\nversion = "{project_version}"\n'
                ).encode(),
            }
        )
    elif kind.startswith("studio"):
        name = names["studio"][0]
        package = json.loads((c["root"] / "studio/package.json").read_bytes())
        package["license" if kind == "studio-license" else "version"] = "wrong"
        raw = tarred({"package/package.json": rc.json_bytes(package)})
        manifest["npm_integrity"] = (
            "sha512-" + base64.b64encode(hashlib.sha512(raw).digest()).decode()
        )
    elif kind.startswith("pack"):
        path = c["dist"] / "studio/pack.json"
        pack = json.loads(path.read_bytes())
        pack[0]["integrity" if kind == "pack-integrity" else "filename"] = "wrong"
        path.write_bytes(rc.json_bytes(pack))
        with pytest.raises(rc.CandidateError, match="pack.json"):
            rc.verify_files(c["root"], c["dist"], manifest, "studio")
        return
    else:
        (c["dist"] / "studio/commit.txt").write_text("wrong")
        with pytest.raises(rc.CandidateError, match="commit.txt"):
            rc.verify_files(c["root"], c["dist"], manifest, "studio")
        return
    (c["dist"] / name).write_bytes(raw)
    manifest["files"][name] = rc.digest(raw)
    with pytest.raises(rc.CandidateError, match="mismatch|differs"):
        rc.verify_files(c["root"], c["dist"], manifest)


def test_selective_verification_needs_no_source_stamp_or_other_packages(candidate, tmp_path):
    c = candidate
    (c["root"] / "pyproject.toml").write_text('[project]\nversion = "0.2.1"\n')
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in rc.filenames(VERSION)["python"]:
        if name.endswith(".whl"):
            shutil.copy(c["dist"] / name, wheels / Path(name).name)
    rc.verify_files(c["root"], wheels, c["manifest"], "wheels")
    shutil.rmtree(c["dist"] / "nexuml")
    shutil.rmtree(c["dist"] / "nexuml-library")
    rc.verify_files(c["root"], c["dist"], c["manifest"], "studio")


@pytest.mark.parametrize(
    "name", ["../escape.json", "/escape.json", "nested/data.json", "a\\b.json", "./data.json"]
)
def test_evidence_zip_never_extracts_unsafe_paths(candidate, name):
    c = candidate
    c["api"].values[f"{PREFIX}/actions/artifacts/101/zip"] = zipped({name: c["raw"]})
    with pytest.raises(rc.CandidateError, match="plain JSON"):
        scan(c)


@pytest.mark.parametrize("name", ["../evil", "/evil", "nexuml/../../evil", "nexuml/extra.whl"])
def test_distribution_zip_allowlist(candidate, name):
    c = candidate
    c["api"].values[f"{PREFIX}/actions/artifacts/102/zip"] = zipped({name: b"evil"})
    with pytest.raises(rc.CandidateError, match="member paths"):
        rc.download(c["api"], REPO, c["dist"], c["manifest"])


def test_zip_duplicate_symlink_and_size_limits(candidate, monkeypatch):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        info = zipfile.ZipInfo("data.json")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "target")
    with pytest.raises(rc.CandidateError, match="Unsafe"):
        rc.zip_members(output.getvalue(), ["data.json"])
    with pytest.raises(rc.CandidateError, match="oversized"):
        rc.zip_members(zipped({"data.json": b"large"}), ["data.json"], limit=1)
    monkeypatch.setattr(rc, "MAX_ARCHIVE", 1)
    with pytest.raises(rc.CandidateError, match="too large"):
        rc.zip_members(zipped({"data.json": b""}), ["data.json"])


def test_symlink_destination_rejected(candidate, tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "download").mkdir()
    (tmp_path / "download/nexuml").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(rc.CandidateError, match="Symlink"):
        rc.write_members(tmp_path / "download", {"nexuml/file": b"bad"})
    assert not list((tmp_path / "outside").iterdir())


def test_two_tags_same_commit_and_paginated_selection(candidate):
    c = candidate
    other = workflow(c["commit"], run_id=30, branch="v0.2.3")
    branch = workflow(c["commit"], run_id=40, branch="main")
    key = f"{PREFIX}/actions/workflows/ci.yml/runs?branch=v0.2.2&event=push&per_page=100"
    c["api"].values[key] = [{"workflow_runs": [branch, other]}, {"workflow_runs": [c["run"]]}]
    manifest, raw, artifact_id = scan(c)
    assert manifest == c["manifest"] and raw == c["raw"] and artifact_id == 101
    assert all("/runs/30/" not in path and "/runs/40/" not in path for path, _, _ in c["api"].calls)


def test_pending_to_success_with_injected_clock(candidate):
    c = candidate
    c["run"]["status"] = "in_progress"
    c["run"]["conclusion"] = None
    now = [0]
    sleeps = []

    def sleep(seconds):
        now[0] += seconds
        sleeps.append(seconds)
        c["run"].update(status="completed", conclusion="success")

    result = rc.wait_candidate(
        c["api"], REPO, REF, VERSION, c["commit"], clock=lambda: now[0], sleep=sleep
    )
    assert result[0] == c["manifest"]
    assert sleeps == [30]


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", "skipped"])
def test_failed_tag_ci_does_not_poll(candidate, conclusion):
    c = candidate
    c["run"]["conclusion"] = conclusion
    with pytest.raises(rc.CandidateError, match="failed/cancelled"):
        rc.wait_candidate(
            c["api"], REPO, REF, VERSION, c["commit"], sleep=lambda _: pytest.fail("must not sleep")
        )


@pytest.mark.parametrize("absent", [False, True])
def test_pending_or_absent_timeout_is_bounded(candidate, absent):
    c = candidate
    if absent:
        c["api"].values[
            f"{PREFIX}/actions/workflows/ci.yml/runs?branch=v0.2.2&event=push&per_page=100"
        ] = [{"workflow_runs": []}]
    else:
        c["run"].update(status="queued", conclusion=None)
    now = [0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    with pytest.raises(rc.CandidateError, match="Timed out"):
        rc.wait_candidate(
            c["api"], REPO, REF, VERSION, c["commit"], timeout=65, clock=lambda: now[0], sleep=sleep
        )
    assert sleeps == [30, 30, 5]


@pytest.mark.parametrize(
    "problem",
    [
        "expired",
        "missing",
        "duplicate",
        "wrong-id",
        "wrong-origin",
        "old-created-at",
        "attempt-origin",
        "failed-gate",
        "wrong-recorded-attempt",
    ],
)
def test_artifact_and_attempt_identity_fail_closed(candidate, problem):
    c = candidate
    if problem == "expired":
        c["artifacts"][1]["expired"] = True
    elif problem == "missing":
        c["artifacts"].pop(1)
    elif problem == "duplicate":
        extra = copy.deepcopy(c["artifacts"][1])
        extra["id"] = 999
        c["artifacts"].append(extra)
    elif problem == "wrong-id":
        c["artifacts"][1]["id"] = 999
    elif problem == "wrong-origin":
        c["artifacts"][1]["workflow_run"]["id"] = 999
    elif problem == "old-created-at":
        c["artifacts"][1]["created_at"] = "2026-10-23T01:00:00Z"
    elif problem == "attempt-origin":
        altered = copy.deepcopy(c["run"])
        altered["path"] = rc.RELEASE_WORKFLOW
        c["api"].values[f"{PREFIX}/actions/runs/20/attempts/1"] = altered
    elif problem == "failed-gate":
        c["api"].values[f"{PREFIX}/actions/runs/20/attempts/1/jobs?per_page=100"][0]["jobs"][0][
            "conclusion"
        ] = "failure"
    else:
        manifest = copy.deepcopy(c["manifest"])
        manifest["run_attempt"] = 2
        c["api"].values[f"{PREFIX}/actions/artifacts/101/zip"] = zipped(
            {"candidate.json": rc.json_bytes(manifest)}
        )
    with pytest.raises(rc.CandidateError):
        scan(c)


def add_second_attempt(c, *, success=True, created="2026-10-24T02:05:00Z"):
    current = workflow(
        c["commit"], attempt=2, start=SECOND, conclusion="success" if success else "failure"
    )
    c["api"].values[
        f"{PREFIX}/actions/workflows/ci.yml/runs?branch=v0.2.2&event=push&per_page=100"
    ] = [{"workflow_runs": [current]}]
    artifacts = c["artifacts"] + [
        artifact(c["commit"], name, artifact_id, created=created)
        for name, artifact_id in (
            ("release-candidate-2", 201),
            ("python-distributions-2", 202),
            ("studio-distribution-2", 203),
        )
    ]
    publish_attempt(c["api"], current, artifacts)
    manifest = copy.deepcopy(c["manifest"])
    manifest.update(run_attempt=2, artifacts={"python": 202, "studio": 203})
    c["api"].values[f"{PREFIX}/actions/artifacts/201/zip"] = zipped(
        {"candidate.json": rc.json_bytes(manifest)}
    )
    return current, artifacts


def test_duplicate_successful_historical_attempts_rejected(candidate):
    add_second_attempt(candidate)
    with pytest.raises(rc.CandidateError, match="multiple successful"):
        scan(candidate)


def test_original_successful_candidate_survives_failed_rerun(candidate):
    add_second_attempt(candidate, success=False)
    assert scan(candidate)[2] == 101


def test_failed_attempt_can_be_retried_but_old_bytes_cannot_be_reused(candidate):
    c = candidate
    c["run"]["conclusion"] = "failure"
    _, artifacts = add_second_attempt(c, created=CREATED)
    with pytest.raises(rc.CandidateError, match="outside recorded run attempt"):
        scan(c)
    for item in artifacts[3:]:
        item["created_at"] = "2026-10-24T02:05:00Z"
    assert scan(c)[0]["run_attempt"] == 2


def test_retagging_rejected_including_annotated_tags(candidate):
    c = candidate
    api = c["api"]
    api.values[f"{PREFIX}/git/ref/tags/v0.2.2"] = {"object": {"type": "tag", "sha": "a" * 40}}
    api.values[f"{PREFIX}/git/tags/{'a' * 40}"] = {"object": {"type": "commit", "sha": c["commit"]}}
    rc.remote_tag(api, REPO, REF, c["commit"])
    api.values[f"{PREFIX}/git/tags/{'a' * 40}"]["object"]["sha"] = "f" * 40
    with pytest.raises(rc.CandidateError, match="Remote tag changed"):
        rc.wait_candidate(
            api, REPO, REF, VERSION, c["commit"], sleep=lambda _: pytest.fail("must not sleep")
        )


def verification(c, *, attempt=1, event="push", conclusion="success"):
    api = c["api"]
    run = workflow(
        c["commit"],
        run_id=50,
        attempt=attempt,
        workflow=rc.RELEASE_WORKFLOW,
        event=event,
        conclusion=conclusion,
    )
    api.values[
        f"{PREFIX}/actions/workflows/release.yml/runs?head_sha={c['commit']}&per_page=100"
    ] = [{"workflow_runs": [run]}]
    evidence = rc.verification_record(c["manifest"], c["raw"], 50, attempt)
    artifacts = [artifact(c["commit"], f"testpypi-verification-{attempt}", 501, run_id=50)]
    publish_attempt(api, run, artifacts, job="verify-testpypi")
    api.values[f"{PREFIX}/actions/artifacts/501/zip"] = zipped(
        {"verification.json": rc.json_bytes(evidence)}
    )
    return run, artifacts, evidence


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_matching_testpypi_evidence(candidate, event):
    verification(candidate, event=event)
    rc.require_testpypi(candidate["api"], candidate["manifest"], candidate["raw"])


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "failed",
        "wrong-job",
        "expired",
        "wrong-candidate",
        "wrong-ci-attempt",
        "wrong-verification-attempt",
        "wrong-ref",
        "wrong-repo",
    ],
)
def test_missing_or_mismatched_testpypi_evidence(candidate, problem):
    c = candidate
    run, artifacts, evidence = verification(c)
    if problem == "missing":
        artifacts.clear()
    elif problem == "failed":
        run["conclusion"] = "failure"
    elif problem == "wrong-job":
        c["api"].values[f"{PREFIX}/actions/runs/50/attempts/1/jobs?per_page=100"][0]["jobs"][0][
            "name"
        ] = "unrelated"
    elif problem == "expired":
        artifacts[0]["expired"] = True
    else:
        field = {
            "wrong-candidate": "candidate_sha256",
            "wrong-ci-attempt": "ci_run_attempt",
            "wrong-verification-attempt": "verification_run_attempt",
            "wrong-ref": "ref",
            "wrong-repo": "repository",
        }[problem]
        evidence[field] = "wrong"
        c["api"].values[f"{PREFIX}/actions/artifacts/501/zip"] = zipped(
            {"verification.json": rc.json_bytes(evidence)}
        )
    with pytest.raises(rc.CandidateError):
        rc.require_testpypi(c["api"], c["manifest"], c["raw"])


def test_historical_testpypi_evidence_remains_bound_to_original_attempt(candidate):
    c = candidate
    first, artifacts, _ = verification(c)
    second = workflow(
        c["commit"],
        run_id=50,
        attempt=2,
        workflow=rc.RELEASE_WORKFLOW,
        conclusion="failure",
        start=SECOND,
    )
    c["api"].values[
        f"{PREFIX}/actions/workflows/release.yml/runs?head_sha={c['commit']}&per_page=100"
    ] = [{"workflow_runs": [second]}]
    publish_attempt(c["api"], second, artifacts, job="verify-testpypi")
    assert first["run_attempt"] == 1
    rc.require_testpypi(c["api"], c["manifest"], c["raw"])
    artifacts[0]["created_at"] = "2026-10-24T02:05:00Z"
    with pytest.raises(rc.CandidateError, match="outside recorded run attempt"):
        rc.require_testpypi(c["api"], c["manifest"], c["raw"])


def test_resolve_verify_and_record_cli(candidate, monkeypatch, tmp_path):
    c = candidate
    monkeypatch.setattr(rc, "GitHub", lambda: c["api"])
    monkeypatch.setattr(rc, "integrated", lambda *args: None)
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    manifest_path = tmp_path / "candidate.json"
    dist = tmp_path / "download"
    rc.main(["resolve", "--manifest", str(manifest_path), "--dist", str(dist)])
    assert manifest_path.read_bytes() == c["raw"]
    assert "run-id=20\n" in output.read_text()
    assert "manifest-id=101\n" in output.read_text()
    assert f"tarball={dist}/studio/nexufed-nexuml-studio-{VERSION}.tgz\n" in output.read_text()
    c["api"].calls.clear()
    shutil.rmtree(dist / "nexuml")
    shutil.rmtree(dist / "nexuml-library")
    rc.main(["verify", "--manifest", str(manifest_path), "--dist", str(dist), "--group", "studio"])
    assert not any(path.endswith(("/102/zip", "/103/zip")) for path, _, _ in c["api"].calls)
    evidence = tmp_path / "evidence.json"
    monkeypatch.setenv("GITHUB_RUN_ID", "50")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    rc.main(["record-testpypi", "--manifest", str(manifest_path), "--output", str(evidence)])
    assert json.loads(evidence.read_bytes()) == rc.verification_record(
        c["manifest"], c["raw"], 50, 2
    )
    manifest_path.write_bytes(c["raw"] + b"\n")
    with pytest.raises(SystemExit, match="1"):
        rc.main(
            ["verify", "--manifest", str(manifest_path), "--dist", str(dist), "--group", "studio"]
        )


@pytest.mark.parametrize("event", ["workflow_dispatch", "release"])
def test_publishers_accept_only_existing_tags(candidate, monkeypatch, event):
    c = candidate
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    assert rc.context(c["root"])[2:] == (VERSION, c["commit"])
    with pytest.raises(rc.CandidateError, match="Only tag pushes"):
        rc.context(c["root"], push=True)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    with pytest.raises(rc.CandidateError, match="existing full"):
        rc.context(c["root"])


def test_deleted_tag_and_wrong_sha_rejected(candidate, monkeypatch):
    c = candidate
    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    with pytest.raises(rc.CandidateError, match="GITHUB_SHA"):
        rc.context(c["root"])
    monkeypatch.setenv("GITHUB_SHA", c["commit"])
    Path(rc.os.environ["GITHUB_EVENT_PATH"]).write_text('{"deleted": true}')
    with pytest.raises(rc.CandidateError, match="Deleted"):
        rc.context(c["root"])


def test_branch_eligibility_has_no_api_calls(candidate, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setattr(rc, "GitHub", lambda: candidate["api"])
    rc.main([])
    assert "Branch CI" in capsys.readouterr().out
    assert candidate["api"].calls == []


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "protection",
        "ancestry",
        "wrong-event",
        "wrong-repo",
        "wrong-workflow",
        "wrong-branch",
        "wrong-commit",
        "old-attempt",
    ],
)
def test_integrated_requires_exact_current_main_ci(candidate, monkeypatch, problem):
    c = candidate
    api = c["api"]
    main = workflow(c["commit"], run_id=10, branch="main", attempt=2)
    api.values[f"{PREFIX}/branches/main"] = {"protected": problem != "protection"}
    key = f"{PREFIX}/actions/workflows/ci.yml/runs?head_sha={c['commit']}&event=push&per_page=100"
    api.values[key] = [{"workflow_runs": [c["run"]]}, {"workflow_runs": [main]}]
    publish_attempt(api, main, [])
    if problem == "wrong-event":
        main["event"] = "workflow_dispatch"
    elif problem == "wrong-repo":
        main["head_repository"]["full_name"] = "fork/repo"
    elif problem == "wrong-workflow":
        main["path"] = rc.RELEASE_WORKFLOW
    elif problem == "wrong-branch":
        main["head_branch"] = "dev"
    elif problem == "wrong-commit":
        main["head_sha"] = "f" * 40
    elif problem == "old-attempt":
        api.values[f"{PREFIX}/actions/runs/10/attempts/2/jobs?per_page=100"][0]["jobs"][0][
            "conclusion"
        ] = "failure"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 1 if problem == "ancestry" and "merge-base" in command else 0
        )

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    if problem:
        with pytest.raises(rc.CandidateError):
            rc.integrated(api, c["root"], REPO, REF, c["commit"])
    else:
        rc.integrated(api, c["root"], REPO, REF, c["commit"])
    assert "fetch" in commands[0] and "--no-tags" in commands[0]


def test_gh_transport_is_get_only_and_paginates(monkeypatch):
    commands = []

    def check_output(command):
        commands.append(command)
        return b'[{"artifacts": []}]'

    monkeypatch.setattr(rc.subprocess, "check_output", check_output)
    assert rc.listing(rc.GitHub(), f"{PREFIX}/actions/runs/20/artifacts", "artifacts") == []
    assert commands == [
        [
            "gh",
            "api",
            "--method",
            "GET",
            f"{PREFIX}/actions/runs/20/artifacts?per_page=100",
            "--paginate",
            "--slurp",
        ]
    ]


@pytest.mark.parametrize("problem", ["moved-history", "wrong-repository", "wrong-workflow"])
def test_tag_history_and_workflow_origin_are_not_silently_ignored(candidate, problem):
    c = candidate
    if problem == "moved-history":
        c["run"]["head_sha"] = "f" * 40
    elif problem == "wrong-repository":
        c["run"]["head_repository"]["full_name"] = "fork/repo"
    else:
        c["run"]["path"] = rc.RELEASE_WORKFLOW
    with pytest.raises(rc.CandidateError, match="moved|origin mismatch"):
        scan(c)


def test_separate_successful_run_is_ambiguous_not_latest_wins(candidate):
    c = candidate
    second = workflow(c["commit"], run_id=30)
    key = f"{PREFIX}/actions/workflows/ci.yml/runs?branch=v0.2.2&event=push&per_page=100"
    c["api"].values[key].append({"workflow_runs": [second]})
    artifacts = [
        artifact(c["commit"], name, artifact_id, run_id=30)
        for name, artifact_id in (
            ("release-candidate-1", 301),
            ("python-distributions-1", 302),
            ("studio-distribution-1", 303),
        )
    ]
    publish_attempt(c["api"], second, artifacts)
    manifest = copy.deepcopy(c["manifest"])
    manifest.update(run_id=30, artifacts={"python": 302, "studio": 303})
    c["api"].values[f"{PREFIX}/actions/artifacts/301/zip"] = zipped(
        {
            "candidate.json": rc.json_bytes(manifest),
        }
    )
    with pytest.raises(rc.CandidateError, match="multiple successful"):
        scan(c)


@pytest.mark.parametrize("field", ["run_id", "run_attempt"])
def test_successful_job_must_belong_to_requested_attempt(candidate, field):
    c = candidate
    jobs = c["api"].values[f"{PREFIX}/actions/runs/20/attempts/1/jobs?per_page=100"]
    jobs[0]["jobs"][0][field] = 999
    with pytest.raises(rc.CandidateError, match="gate did not succeed"):
        scan(c)


@pytest.mark.parametrize("malformed", [None, [], "not an object", 1])
def test_manifest_must_be_json_object(candidate, malformed):
    with pytest.raises(rc.CandidateError, match="JSON object"):
        rc.validate_manifest(malformed, REPO, REF, VERSION, candidate["commit"])


def test_duplicate_zip_members_rejected():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("candidate.json", "{}")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("candidate.json", "{}")
    with pytest.raises(rc.CandidateError, match="Duplicate"):
        rc.zip_members(output.getvalue(), ["candidate.json"])


def test_unsafe_tar_metadata_rejected_without_extraction(candidate, tmp_path):
    path = tmp_path / "unsafe.tgz"
    path.write_bytes(tarred({"../escape": b"bad", "package/package.json": b"{}"}))
    with pytest.raises(rc.CandidateError, match="Unsafe"):
        rc.tar_metadata(path, ["package/package.json"])
    assert not (tmp_path.parent / "escape").exists()


def test_create_cli_makes_output_directory(candidate, tmp_path):
    path = tmp_path / "release-candidate/candidate.json"
    rc.main(
        [
            "create",
            "--dist",
            str(candidate["dist"]),
            "--python-id",
            "102",
            "--studio-id",
            "103",
            "--output",
            str(path),
        ]
    )
    assert path.read_bytes() == candidate["raw"]


def test_production_resolve_checks_evidence_before_downloading(candidate, monkeypatch, tmp_path):
    c = candidate
    run, _, _ = verification(c)
    run["conclusion"] = "failure"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "release")
    monkeypatch.setattr(rc, "GitHub", lambda: c["api"])
    monkeypatch.setattr(rc, "integrated", lambda *args: None)
    with pytest.raises(SystemExit, match="1"):
        rc.main(
            [
                "resolve",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--dist",
                str(tmp_path / "download"),
                "--require-testpypi",
            ]
        )
    assert not any(path.endswith(("/102/zip", "/103/zip")) for path, _, _ in c["api"].calls)


def test_truncated_api_search_cannot_hide_earlier_successful_candidates(candidate):
    c = candidate
    key = f"{PREFIX}/actions/workflows/ci.yml/runs?branch=v0.2.2&event=push&per_page=100"
    c["api"].values[key][0]["total_count"] = 1001
    with pytest.raises(rc.CandidateError, match="truncated"):
        scan(c)


def test_ci_repository_owner_not_just_head_repository_must_match(candidate):
    candidate["run"]["repository"]["full_name"] = "fork/repo"
    with pytest.raises(rc.CandidateError, match="origin mismatch"):
        scan(candidate)


def test_selected_group_rejects_extra_publishable_files(candidate):
    c = candidate
    (c["dist"] / "nexuml/unexpected.whl").write_bytes(b"not the tested package")
    with pytest.raises(rc.CandidateError, match="unexpected python"):
        rc.verify_files(c["root"], c["dist"], c["manifest"], "python")
    rc.verify_files(c["root"], c["dist"], c["manifest"], "studio")
