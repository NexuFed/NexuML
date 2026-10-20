"""Read-only release eligibility and immutable, tag-specific artifact validation."""

import argparse
import base64
from datetime import datetime
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import time
import tomllib
import zipfile

from release_versions import resolve_tag, stable_version


CI_WORKFLOW = ".github/workflows/ci.yml"
RELEASE_WORKFLOW = ".github/workflows/release.yml"
METADATA_KEYS = ("name", "version", "repository", "license", "bin", "engines", "publishConfig")
MAX_FILE = 512 * 1024 * 1024
MAX_ARCHIVE = 512 * 1024 * 1024
MAX_UNPACKED = 1024 * 1024 * 1024
MAX_METADATA = 2 * 1024 * 1024


class CandidateError(ValueError):
    """An identity or integrity check failed; publication must stop."""


def require(condition, message):
    """Reject invalid release evidence with an actionable error.

    Raises:
        CandidateError: The evidence fails the specified condition.
    """
    if not condition:
        raise CandidateError(message)


def positive(value):
    """Return whether a JSON value is a positive integer, excluding booleans."""
    return type(value) is int and value > 0


def digest(data):
    """Return the SHA-256 identity of exact artifact bytes."""
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    """Return deterministic, human-readable evidence bytes."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def git(root, *args):
    """Return requested Git identity or tagged metadata from the checkout."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


class GitHub:
    """Only GET requests, including paginated lists and bounded binary downloads."""

    def api(self, path, *, pages=False, binary=False):
        command = ["gh", "api", "--method", "GET", path]
        if pages:
            command += ["--paginate", "--slurp"]
        if binary:
            with tempfile.TemporaryFile() as output:
                subprocess.run(command, stdout=output, check=True)
                require(output.tell() <= MAX_ARCHIVE, f"Artifact archive too large: {path}")
                output.seek(0)
                return output.read()
        return json.loads(subprocess.check_output(command))


def listing(api, path, key):
    """Return every page rather than selecting the newest run or artifact."""
    separator = "&" if "?" in path else "?"
    pages = api.api(f"{path}{separator}per_page=100", pages=True)
    items = [item for page in pages for item in page[key]]
    require(
        max((page.get("total_count", 0) for page in pages), default=0) <= len(items),
        "GitHub truncated the evidence listing (including its 1,000-run search cap); "
        "cannot prove a unique candidate, use a fresh version tag",
    )
    return items


def context(root, *, push=False):
    """Return an existing stable tag identity and its exact checked-out commit."""
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo),
        "Set GITHUB_REPOSITORY to owner/repository",
    )
    ref = os.environ.get("GITHUB_REF", "")
    require(
        ref.startswith("refs/tags/"),
        "Publication requires an existing full refs/tags/vX.Y.Z ref; dispatch with --ref vX.Y.Z",
    )
    stable_version(ref.removeprefix("refs/tags/"))
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    require(
        event in ({"push"} if push else {"push", "workflow_dispatch", "release"}),
        "Only tag pushes or tag publisher events are eligible",
    )
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path:
        require(
            not json.loads(Path(event_path).read_bytes()).get("deleted", False),
            "Deleted tags cannot be released",
        )
    version, commit = resolve_tag(root, ref)
    require(
        not os.environ.get("GITHUB_SHA") or os.environ["GITHUB_SHA"] == commit,
        "GITHUB_SHA differs from the peeled tag commit",
    )
    return repo, ref, version, commit


def remote_tag(api, repo, ref, commit):
    """Peel the current remote ref and reject moved or deleted tags."""
    obj = api.api(f"repos/{repo}/git/ref/{ref.removeprefix('refs/')}")["object"]
    seen = set()
    while obj["type"] == "tag":
        require(obj["sha"] not in seen and len(seen) < 16, "Invalid annotated tag chain")
        seen.add(obj["sha"])
        obj = api.api(f"repos/{repo}/git/tags/{obj['sha']}")["object"]
    require(
        obj["type"] == "commit" and obj["sha"] == commit,
        "Remote tag changed or does not identify the candidate commit; use a fresh version tag",
    )


def origin(run, repo, workflow, commit, branch, events):
    """Return whether workflow provenance matches, not just its source commit."""
    return (
        (run.get("head_repository") or {}).get("full_name") == repo
        and (run.get("repository") or {}).get("full_name") == repo
        and run.get("path") == workflow
        and run.get("head_sha") == commit
        and run.get("head_branch") == branch
        and run.get("event") in events
    )


def jobs_success(api, repo, run_id, attempt, name):
    """Return whether the requested attempt has exactly one successful named job."""
    jobs = listing(api, f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")
    matches = [job for job in jobs if job.get("name") == name]
    return (
        len(matches) == 1
        and matches[0].get("conclusion") == "success"
        and matches[0].get("status") == "completed"
        and matches[0].get("run_id") == run_id
        and matches[0].get("run_attempt") == attempt
    )


def integrated(api, root, repo, ref, commit):
    """Require protected-main ancestry and exact current main-push CI evidence.

    Raises:
        CandidateError: No exact-commit main-push CI run has a successful current gate.
    """
    remote_tag(api, repo, ref, commit)
    subprocess.run(
        ["git", "-C", str(root), "fetch", "--no-tags", "origin", "main:refs/remotes/origin/main"],
        check=True,
    )
    require(
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "origin/main"],
            check=False,
        ).returncode
        == 0,
        "Tag commit is not contained in origin/main",
    )
    require(
        api.api(f"repos/{repo}/branches/main").get("protected") is True,
        "main must be protected before releasing",
    )
    runs = listing(
        api,
        f"repos/{repo}/actions/workflows/ci.yml/runs?head_sha={commit}&event=push",
        "workflow_runs",
    )
    for run in runs:
        if (
            origin(run, repo, CI_WORKFLOW, commit, "main", {"push"})
            and run.get("status") == "completed"
            and run.get("conclusion") == "success"
        ):
            if jobs_success(api, repo, run["id"], run["run_attempt"], "CI"):
                print(
                    f"Protected main CI: run {run['id']} attempt {run['run_attempt']} "
                    f"commit {commit}"
                )
                return
    raise CandidateError(
        "No successful exact-commit main push ci.yml run with a successful current-attempt CI job"
    )


def filenames(version):
    """Return the only five allowed publication filenames."""
    return {
        "python": [
            f"{directory}/{name}-{version}{suffix}"
            for directory, name in (("nexuml", "nexuml"), ("nexuml-library", "nexuml_library"))
            for suffix in ("-py3-none-any.whl", ".tar.gz")
        ],
        "studio": [f"studio/nexufed-nexuml-studio-{version}.tgz"],
    }


def validate_manifest(manifest, repo, ref, version, commit):
    """Validate candidate data before it can select files or API identities."""
    require(isinstance(manifest, dict), "Candidate manifest must be a JSON object")
    require(
        set(manifest)
        == {
            "schema",
            "repository",
            "ref",
            "version",
            "commit",
            "workflow",
            "run_id",
            "run_attempt",
            "artifacts",
            "files",
            "npm_integrity",
        },
        "Unexpected candidate manifest schema/keys",
    )
    require(
        type(manifest["schema"]) is int and manifest["schema"] == 1, "Unsupported candidate schema"
    )
    for key, expected in (
        ("repository", repo),
        ("ref", ref),
        ("version", version),
        ("commit", commit),
        ("workflow", CI_WORKFLOW),
    ):
        require(manifest[key] == expected, f"Candidate {key} does not match requested tag identity")
    require(
        positive(manifest["run_id"]) and positive(manifest["run_attempt"]),
        "Candidate run/attempt must be positive integers",
    )
    artifacts = manifest["artifacts"]
    require(
        isinstance(artifacts, dict)
        and set(artifacts) == {"python", "studio"}
        and all(positive(value) for value in artifacts.values())
        and len(set(artifacts.values())) == 2,
        "Candidate artifact IDs must be positive and distinct",
    )
    expected = filenames(version)
    require(
        isinstance(manifest["files"], dict)
        and set(manifest["files"]) == set(expected["python"] + expected["studio"]),
        "Candidate contains missing, unexpected, or unsafe filenames",
    )
    require(
        all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in manifest["files"].values()
        ),
        "Invalid candidate SHA-256",
    )
    require(
        isinstance(manifest["npm_integrity"], str)
        and re.fullmatch(r"sha512-[A-Za-z0-9+/]{86}==", manifest["npm_integrity"]),
        "Invalid npm SHA-512 integrity",
    )


def safe_name(name):
    """Return whether a path is relative, canonical, and free of traversal."""
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and str(path) == name.rstrip("/")
    )


def zip_members(raw, allowed, *, limit=MAX_FILE):
    """Return allowlisted regular zip member bytes without extracting paths."""
    require(len(raw) <= MAX_ARCHIVE, "Artifact archive too large")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        require(
            len(infos) <= 100 and len({info.filename for info in infos}) == len(infos),
            "Duplicate or excessive artifact zip members",
        )
        require(
            {info.filename for info in infos} == set(allowed),
            "Artifact zip has missing, unexpected, or unsafe member paths",
        )
        require(
            sum(info.file_size for info in infos) <= MAX_UNPACKED,
            "Artifact zip expands beyond size limit",
        )
        for info in infos:
            require(
                safe_name(info.filename)
                and not info.is_dir()
                and info.file_size <= limit
                and ((info.external_attr >> 16) & 0o170000) != 0o120000,
                "Unsafe or oversized artifact zip member",
            )
        return {info.filename: archive.read(info) for info in infos}


def artifact_json(api, repo, artifact_id):
    """Return parsed JSON and exact bytes from one bounded immutable evidence file."""
    raw = api.api(f"repos/{repo}/actions/artifacts/{artifact_id}/zip", binary=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
    require(
        len(names) == 1
        and safe_name(names[0])
        and "/" not in names[0]
        and names[0].endswith(".json"),
        "Evidence artifact must contain one plain JSON file",
    )
    data = zip_members(raw, names, limit=MAX_METADATA)[names[0]]
    return json.loads(data), data


def tar_metadata(path, wanted):
    """Return bounded metadata without executing or extracting a tarball.

    Raises:
        CandidateError: Requested packed metadata is not a readable regular file.
    """
    found = {}
    seen = set()
    total = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            total += member.size
            require(
                safe_name(member.name)
                and member.name not in seen
                and len(seen) < 100000
                and member.size <= MAX_FILE
                and total <= MAX_UNPACKED,
                f"Unsafe, duplicate, or oversized tar member in {path.name}",
            )
            seen.add(member.name)
            if member.name in wanted:
                require(
                    member.isfile() and member.size <= MAX_METADATA,
                    f"Invalid packed metadata: {member.name}",
                )
                stream = archive.extractfile(member)
                if stream is None:
                    raise CandidateError(f"Missing packed metadata: {member.name}")
                found[member.name] = stream.read()
    require(set(found) == set(wanted), f"Missing packed metadata in {path.name}")
    return found


def python_metadata(path, version):
    """Check wheel/sdist identity and the rebuildable sdist project version."""
    name = "nexuml-library" if path.name.startswith("nexuml_library-") else "nexuml"
    normalized = name.replace("-", "_")
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            metadata = f"{normalized}-{version}.dist-info/METADATA"
            require(
                [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                == [metadata],
                f"Missing or duplicate wheel METADATA: {path.name}",
            )
            require(archive.getinfo(metadata).file_size <= MAX_METADATA, "Wheel METADATA too large")
            raw = archive.read(metadata)
    else:
        prefix = f"{normalized}-{version}/"
        packed = tar_metadata(path, [prefix + "PKG-INFO", prefix + "pyproject.toml"])
        require(
            tomllib.loads(packed[prefix + "pyproject.toml"].decode())["project"]["version"]
            == version,
            f"sdist project.version mismatch: {path.name}",
        )
        raw = packed[prefix + "PKG-INFO"]
    metadata = BytesParser().parsebytes(raw)
    require(
        metadata.get_all("Name") == [name] and metadata.get_all("Version") == [version],
        f"Python Name/Version mismatch: {path.name}",
    )


def studio_source(root, commit, version):
    """Return tagged publication fields with only the expected version replaced."""
    source = json.loads(git(root, "show", f"{commit}:studio/package.json"))
    source["version"] = version
    require(
        all(key in source for key in METADATA_KEYS),
        "Tagged Studio publication metadata is incomplete",
    )
    return {key: source[key] for key in METADATA_KEYS}


def verify_files(root, dist, manifest, group="all"):
    """Check selected package hashes and metadata without downloading other groups."""
    version = manifest["version"]
    names = filenames(version)
    groups = (
        ("python", "studio") if group == "all" else ("python",) if group == "wheels" else (group,)
    )
    for selected in groups:
        if group == "wheels":
            expected = {Path(name).name for name in names["python"] if name.endswith(".whl")}
            directories = [dist]
        else:
            expected = set(names[selected])
            if selected == "studio":
                expected.update({"studio/pack.json", "studio/commit.txt"})
            directories = [dist / name for name in {Path(name).parts[0] for name in expected}]
        actual = {
            str(path.relative_to(dist))
            for directory in directories
            for path in directory.rglob("*")
            if not path.is_dir()
        }
        require(actual == expected, f"Missing or unexpected {selected} distribution files")
        for name in names[selected]:
            if group == "wheels" and not name.endswith(".whl"):
                continue
            path = dist / (Path(name).name if group == "wheels" else name)
            require(path.is_file() and not path.is_symlink(), f"Missing distribution: {path}")
            require(path.stat().st_size <= MAX_FILE, f"Distribution too large: {path}")
            require(
                digest(path.read_bytes()) == manifest["files"][name],
                f"SHA-256 mismatch: {path.name}",
            )
            if selected == "python":
                python_metadata(path, version)
            else:
                integrity = (
                    "sha512-"
                    + base64.b64encode(hashlib.sha512(path.read_bytes()).digest()).decode()
                )
                require(integrity == manifest["npm_integrity"], "npm tarball integrity mismatch")
                packed = json.loads(
                    tar_metadata(path, ["package/package.json"])["package/package.json"]
                )
                require(isinstance(packed, dict), "Packed Studio metadata must be a JSON object")
                require(
                    {key: packed.get(key) for key in METADATA_KEYS}
                    == studio_source(root, manifest["commit"], version),
                    "Packed Studio publication metadata differs from tagged source",
                )
                pack_path = dist / "studio/pack.json"
                commit_path = dist / "studio/commit.txt"
                require(
                    pack_path.stat().st_size <= MAX_METADATA
                    and commit_path.stat().st_size <= MAX_METADATA,
                    "Studio sidecar too large",
                )
                pack = json.loads(pack_path.read_bytes())
                require(
                    isinstance(pack, list) and len(pack) == 1 and isinstance(pack[0], dict),
                    "pack.json must contain exactly one npm package",
                )
                for key, expected in (
                    ("name", "@nexufed/nexuml-studio"),
                    ("version", version),
                    ("filename", path.name),
                    ("integrity", integrity),
                ):
                    require(pack[0].get(key) == expected, f"npm pack.json {key} mismatch")
                require(
                    commit_path.read_text().strip() == manifest["commit"],
                    "Studio commit.txt mismatch",
                )


def create(root, dist, python_id, studio_id):
    """Return a manifest binding tested files to their tag and CI artifact IDs."""
    repo, ref, version, commit = context(root, push=True)
    for path in ("pyproject.toml", "library/pyproject.toml"):
        require(
            tomllib.loads((root / path).read_text())["project"]["version"] == version,
            f"Prepare the tag checkout before building: {path}",
        )
    require(
        json.loads((root / "studio/package.json").read_bytes())["version"] == version,
        "Prepare Studio before building the candidate",
    )
    names = filenames(version)
    expected = set(names["python"] + names["studio"] + ["studio/pack.json", "studio/commit.txt"])
    require(
        {str(path.relative_to(dist)) for path in dist.rglob("*") if not path.is_dir()} == expected,
        "Candidate directory must contain only the five publication files and Studio sidecars",
    )
    manifest = {
        "schema": 1,
        "repository": repo,
        "ref": ref,
        "version": version,
        "commit": commit,
        "workflow": CI_WORKFLOW,
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "artifacts": {"python": python_id, "studio": studio_id},
        "files": {
            name: digest((dist / name).read_bytes()) for name in names["python"] + names["studio"]
        },
        "npm_integrity": json.loads((dist / "studio/pack.json").read_bytes())[0]["integrity"],
    }
    validate_manifest(manifest, repo, ref, version, commit)
    verify_files(root, dist, manifest)
    return manifest


def timestamp(value):
    """Return a parsed GitHub timestamp for attempt-specific artifact provenance."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def attempts(api, repo, run, workflow, commit, branch, events):
    """Return all historical attempts instead of silently substituting a rerun."""
    require(
        positive(run.get("id")) and positive(run.get("run_attempt")),
        "Invalid workflow run identity",
    )
    result = []
    for number in range(1, run["run_attempt"] + 1):
        attempt = api.api(f"repos/{repo}/actions/runs/{run['id']}/attempts/{number}")
        require(
            attempt.get("id") == run["id"]
            and attempt.get("run_attempt") == number
            and origin(attempt, repo, workflow, commit, branch, events),
            "Workflow run attempt origin mismatch",
        )
        result.append(attempt)
    return result


def run_artifacts(api, repo, run_id):
    """Return all original artifact identities for a workflow run."""
    artifacts = listing(api, f"repos/{repo}/actions/runs/{run_id}/artifacts", "artifacts")
    require(
        len({artifact["id"] for artifact in artifacts}) == len(artifacts),
        "Duplicate artifact IDs from GitHub",
    )
    return artifacts


def bound_artifact(artifacts, name, attempt, next_attempt=None, artifact_id=None):
    """Return a live immutable artifact bound to its run and attempt window."""
    matches = [artifact for artifact in artifacts if artifact.get("name") == name]
    require(
        len(matches) == 1,
        f"Missing or ambiguous original artifact {name}; use a fresh tag, do not rebuild",
    )
    artifact = matches[0]
    require(
        positive(artifact.get("id")) and not artifact.get("expired", True),
        f"Expired or invalid original artifact {name}; use a fresh version tag",
    )
    require(
        artifact_id is None or artifact["id"] == artifact_id, f"Artifact ID/origin mismatch: {name}"
    )
    run = artifact.get("workflow_run", {})
    require(
        run.get("id") == attempt["id"]
        and run.get("head_sha") == attempt["head_sha"]
        and run.get("head_branch") == attempt["head_branch"],
        f"Artifact workflow origin mismatch: {name}",
    )
    created = timestamp(artifact["created_at"])
    require(
        created >= timestamp(attempt["run_started_at"])
        and (next_attempt is None or created < timestamp(next_attempt["run_started_at"])),
        f"Artifact created_at is outside recorded run attempt: {name}",
    )
    return artifact


def successful_attempt(api, repo, attempt, job):
    """Return whether an attempt and its named gate job both completed successfully."""
    return (
        attempt.get("status") == "completed"
        and attempt.get("conclusion") == "success"
        and jobs_success(api, repo, attempt["id"], attempt["run_attempt"], job)
    )


def scan_candidates(api, repo, ref, version, commit):
    """Return the unique successful tag-push identity, or None while CI is pending."""
    tag = ref.removeprefix("refs/tags/")
    runs = listing(
        api,
        f"repos/{repo}/actions/workflows/ci.yml/runs?branch={tag}&event=push",
        "workflow_runs",
    )
    candidates = []
    terminal = False
    for run in runs:
        if run.get("head_branch") != tag or run.get("event") != "push":
            continue
        require(
            origin(run, repo, CI_WORKFLOW, run.get("head_sha"), tag, {"push"}),
            "Tag CI workflow/repository origin mismatch",
        )
        require(
            run.get("head_sha") == commit,
            "Tag CI records a different commit: the tag was moved; use a fresh version tag",
        )
        history = attempts(
            api, repo, run, CI_WORKFLOW, commit, ref.removeprefix("refs/tags/"), {"push"}
        )
        artifacts = run_artifacts(api, repo, run["id"])
        for index, attempt in enumerate(history):
            if attempt.get("status") != "completed":
                continue
            if not successful_attempt(api, repo, attempt, "CI"):
                if index == len(history) - 1:
                    terminal = True
                continue
            number = attempt["run_attempt"]
            following = history[index + 1] if index + 1 < len(history) else None
            candidate = bound_artifact(artifacts, f"release-candidate-{number}", attempt, following)
            manifest, raw = artifact_json(api, repo, candidate["id"])
            validate_manifest(manifest, repo, ref, version, commit)
            require(
                manifest["run_id"] == run["id"] and manifest["run_attempt"] == number,
                "Candidate recorded run/attempt mismatch",
            )
            for group, name in (
                ("python", "python-distributions"),
                ("studio", "studio-distribution"),
            ):
                bound_artifact(
                    artifacts, f"{name}-{number}", attempt, following, manifest["artifacts"][group]
                )
            candidates.append((manifest, raw, candidate["id"]))
    require(
        len(candidates) <= 1,
        "Ambiguous tag: multiple successful candidate identities (including historical attempts); "
        "use a fresh version tag",
    )
    if candidates:
        return candidates[0]
    require(
        not terminal,
        "Tag CI failed/cancelled or its CI gate did not succeed; rerun the failed original tag CI",
    )
    return None


def wait_candidate(
    api,
    repo,
    ref,
    version,
    commit,
    *,
    timeout=6900,
    interval=30,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Return a candidate after bounded polling with an injectable clock and sleep."""
    deadline = clock() + timeout
    while True:
        remote_tag(api, repo, ref, commit)
        candidate = scan_candidates(api, repo, ref, version, commit)
        if candidate is not None:
            print(
                f"Selected {ref}: CI run {candidate[0]['run_id']} "
                f"attempt {candidate[0]['run_attempt']}"
            )
            return candidate
        remaining = deadline - clock()
        require(
            remaining > 0,
            f"Timed out waiting for tag-push CI candidate for {ref}; "
            "retry publication after CI completes",
        )
        print(f"Waiting for {ref} tag-push CI (at most {int(remaining)}s remaining)", flush=True)
        sleep(min(interval, remaining))


def verification_record(manifest, raw, run_id, run_attempt):
    """Return evidence binding exact candidate bytes to a verification attempt."""
    require(
        positive(run_id) and positive(run_attempt),
        "Verification run/attempt must be positive integers",
    )
    return {
        "schema": 1,
        "repository": manifest["repository"],
        "ref": manifest["ref"],
        "commit": manifest["commit"],
        "candidate_sha256": digest(raw),
        "ci_run_id": manifest["run_id"],
        "ci_run_attempt": manifest["run_attempt"],
        "verification_run_id": run_id,
        "verification_run_attempt": run_attempt,
    }


def require_testpypi(api, manifest, raw):
    """Require matching immutable evidence from a successful TestPyPI attempt.

    Raises:
        CandidateError: No successful TestPyPI attempt verifies this exact candidate.
    """
    repo, ref, commit = (manifest[key] for key in ("repository", "ref", "commit"))
    runs = listing(
        api, f"repos/{repo}/actions/workflows/release.yml/runs?head_sha={commit}", "workflow_runs"
    )
    for run in runs:
        if not origin(
            run,
            repo,
            RELEASE_WORKFLOW,
            commit,
            ref.removeprefix("refs/tags/"),
            {"push", "workflow_dispatch"},
        ):
            continue
        history = attempts(
            api,
            repo,
            run,
            RELEASE_WORKFLOW,
            commit,
            ref.removeprefix("refs/tags/"),
            {"push", "workflow_dispatch"},
        )
        artifacts = run_artifacts(api, repo, run["id"])
        for index, attempt in enumerate(history):
            if not successful_attempt(api, repo, attempt, "verify-testpypi"):
                continue
            number = attempt["run_attempt"]
            following = history[index + 1] if index + 1 < len(history) else None
            artifact = bound_artifact(
                artifacts, f"testpypi-verification-{number}", attempt, following
            )
            evidence, _ = artifact_json(api, repo, artifact["id"])
            require(
                evidence == verification_record(manifest, raw, run["id"], number),
                "TestPyPI verification evidence does not match the exact candidate/run/attempt",
            )
            return
    raise CandidateError(
        "Missing successful exact-candidate TestPyPI verification; "
        "finish the tag TestPyPI workflow first"
    )


def write_members(dist, members):
    """Write only allowlisted package bytes, rejecting symlink destinations."""
    dist = dist.resolve()
    dist.mkdir(parents=True, exist_ok=True)
    for name, raw in members.items():
        require(safe_name(name), "Unsafe download destination")
        path = dist / name
        require(
            not any(
                parent.is_symlink() for parent in (path, *path.parents) if parent != dist.parent
            ),
            "Symlink in download destination",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def download(api, repo, dist, manifest):
    """Download the recorded artifact IDs with fixed zip member allowlists."""
    names = filenames(manifest["version"])
    raw = api.api(
        f"repos/{repo}/actions/artifacts/{manifest['artifacts']['python']}/zip", binary=True
    )
    write_members(dist, zip_members(raw, names["python"]))
    raw = api.api(
        f"repos/{repo}/actions/artifacts/{manifest['artifacts']['studio']}/zip", binary=True
    )
    allowed = [Path(names["studio"][0]).name, "pack.json", "commit.txt"]
    write_members(
        dist, {f"studio/{name}": value for name, value in zip_members(raw, allowed).items()}
    )


def emit(values):
    """Print evidence and optionally expose validated single-line workflow outputs."""
    for key, value in values.items():
        print(f"{key}={value}")
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            for key, value in values.items():
                require(
                    "\n" not in str(value) and "\r" not in str(value), "Invalid workflow output"
                )
                output.write(f"{key}={value}\n")


def main(argv=None):
    """Run credential-free eligibility, candidate, and promotion-evidence commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("eligibility")
    build = commands.add_parser("create")
    build.add_argument("--dist", type=Path, required=True)
    build.add_argument("--python-id", type=int, required=True)
    build.add_argument("--studio-id", type=int, required=True)
    build.add_argument("--output", type=Path, required=True)
    for name in ("resolve", "verify", "record-testpypi"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "record-testpypi":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--dist", type=Path, required=True)
            if name == "resolve":
                command.add_argument("--require-testpypi", action="store_true")
            else:
                command.add_argument(
                    "--group", choices=("python", "studio", "all", "wheels"), default="all"
                )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    api = GitHub()
    try:
        if args.command in (None, "eligibility"):
            if os.environ.get("GITHUB_REF", "").startswith("refs/heads/"):
                print("Branch CI: no release candidate or publication eligibility")
                emit(
                    {
                        "version": tomllib.loads((root / "pyproject.toml").read_text())["project"][
                            "version"
                        ],
                        "commit": git(root, "rev-parse", "HEAD"),
                    }
                )
                return
            repo, ref, version, commit = context(root, push=True)
            integrated(api, root, repo, ref, commit)
            emit({"version": version, "commit": commit})
        elif args.command == "create":
            manifest = create(root, args.dist, args.python_id, args.studio_id)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(json_bytes(manifest))
            emit({"version": manifest["version"], "candidate-sha256": digest(json_bytes(manifest))})
        else:
            repo, ref, version, commit = context(root)
            if args.command == "record-testpypi":
                raw = args.manifest.read_bytes()
                manifest = json.loads(raw)
                validate_manifest(manifest, repo, ref, version, commit)
                require(
                    os.environ.get("GITHUB_EVENT_NAME") in {"push", "workflow_dispatch"},
                    "Only TestPyPI tag publication can record verification",
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(
                    json_bytes(
                        verification_record(
                            manifest,
                            raw,
                            int(os.environ["GITHUB_RUN_ID"]),
                            int(os.environ["GITHUB_RUN_ATTEMPT"]),
                        )
                    )
                )
                return
            integrated(api, root, repo, ref, commit)
            candidate = (
                wait_candidate(api, repo, ref, version, commit)
                if args.command == "resolve"
                else scan_candidates(api, repo, ref, version, commit)
            )
            require(
                candidate is not None, "Recorded candidate CI is no longer successful/available"
            )
            manifest, raw, manifest_id = candidate
            if args.command == "resolve":
                if args.require_testpypi:
                    require_testpypi(api, manifest, raw)
                download(api, repo, args.dist, manifest)
                verify_files(root, args.dist, manifest)
                args.manifest.parent.mkdir(parents=True, exist_ok=True)
                args.manifest.write_bytes(raw)
            else:
                require(
                    args.manifest.read_bytes() == raw,
                    "Local manifest is not the original immutable candidate",
                )
                verify_files(root, args.dist, manifest, args.group)
            remote_tag(api, repo, ref, commit)
            outputs = {
                "run-id": manifest["run_id"],
                "manifest-id": manifest_id,
                "python-id": manifest["artifacts"]["python"],
                "studio-id": manifest["artifacts"]["studio"],
                "version": version,
                "candidate-sha256": digest(raw),
            }
            if args.command == "resolve" or args.group in {"studio", "all"}:
                outputs["tarball"] = str(args.dist / filenames(version)["studio"][0])
            emit(outputs)
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        OSError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as error:
        parser.exit(1, f"release candidate: {error}\n")


if __name__ == "__main__":
    main()
