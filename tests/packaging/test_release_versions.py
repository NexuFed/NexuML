"""Release identity and stamping checks, without project installs or registry writes."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "release_versions", ROOT / "scripts/release_versions.py"
)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def run(root, *command, check=True):
    return subprocess.run(command, cwd=root, text=True, capture_output=True, check=check)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    for tool in ("uv", "npm", "git"):
        if not shutil.which(tool):
            pytest.skip(f"{tool} is required for temporary release-checkout tests")
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    root = tmp_path / "checkout"
    (root / "library").mkdir(parents=True)
    (root / "studio").mkdir()
    (root / "pyproject.toml").write_text("""# Historical 0.2.1 must stay unchanged.
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nexuml"
version = "0.2.1"
requires-python = ">=3.12"
dependencies = ["six==1.17.0"]

[project.optional-dependencies]
library = ["nexuml-library>=0.2,<0.3"]

[tool.uv.workspace]
members = ["library"]

[tool.uv.sources]
nexuml = { workspace = true }
nexuml-library = { workspace = true }
""")
    (root / "library/pyproject.toml").write_text("""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nexuml-library"
version = "0.2.1"
requires-python = ">=3.12"
dependencies = ["nexuml>=0.2,<0.3"]
""")
    # Seed a real third-party resolution so uv never needs registry access.
    lock = (ROOT / "uv.lock").read_text()
    six = next(block for block in lock.split("[[package]]") if '\nname = "six"\n' in block)
    (root / "uv.lock").write_text(
        'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n[[package]]' + six
    )
    run(root, "uv", "lock")
    package = {
        "name": "@nexufed/nexuml-studio",
        "version": "0.2.1",
        "dependencies": {"example": "0.2.1"},
        "nexuml": {"backend": ">=0.2,<0.3", "interfaceVersion": "1"},
        "scripts": {
            name: 'node -e "process.exit(91)"'
            for name in (
                "preversion",
                "version",
                "postversion",
            )
        },
    }
    (root / "studio/package.json").write_text(json.dumps(package, indent=2) + "\n")
    (root / "studio/package-lock.json").write_text(
        json.dumps(
            {
                "name": package["name"],
                "version": "0.2.1",
                "lockfileVersion": 3,
                "requires": True,
                "packages": {
                    "": {
                        "name": package["name"],
                        "version": "0.2.1",
                        "dependencies": package["dependencies"],
                    },
                    "node_modules/example": {
                        "version": "0.2.1",
                        "resolved": "https://example.org/example-0.2.1.tgz",
                        "integrity": "sha512-unchanged",
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )
    (root / "CHANGELOG.md").write_text("Released 0.2.1; historical v0.2.1\n")
    (root / "compatibility.json").write_text('{"backend": "0.2", "interface": "1"}\n')
    run(root, "git", "init", "--quiet")
    run(root, "git", "config", "user.name", "Release Test")
    run(root, "git", "config", "user.email", "release@example.org")
    run(root, "git", "add", ".")
    run(root, "git", "commit", "--quiet", "-m", "Development metadata")
    return root


@pytest.mark.parametrize("tag", ["v0.0.0", "v0.2.2", "v12.34.567"])
def test_stable_version(tag):
    assert release.stable_version(tag) == tag[1:]


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "0.2.2",
        "refs/tags/v0.2.2",
        "v01.2.3",
        "v1.02.3",
        "v1.2.03",
        "v1.2",
        "v1.2.3rc1",
        "v1.2.3+build",
        "studio-v1.2.3",
        "main",
        "v1.2.3\n",
        " v1.2.3",
        "v١.2.3",
        "v1.٢.3",
        "v1.2.３",
    ],
)
def test_invalid_stable_version(tag):
    with pytest.raises(ValueError, match="Expected stable"):
        release.stable_version(tag)


@pytest.mark.parametrize("annotated", [False, True])
def test_prepare_release_is_idempotent_and_preserves_metadata(checkout, annotated):
    root = checkout
    tag_args = ("-a", "-m", "Release") if annotated else ()
    run(root, "git", "tag", *tag_args, "v0.2.2")
    ref = "refs/tags/v0.2.2"
    commit = run(root, "git", "rev-parse", "HEAD").stdout.strip()
    refs = run(root, "git", "show-ref").stdout
    before = {path: (root / path).read_bytes() for path in release.TARGETS}
    metadata = release._without_release_versions(release._read_metadata(root))
    assert release.resolve_tag(root, ref) == ("0.2.2", commit)
    for attempt in range(2):
        result = run(
            root,
            sys.executable,
            str(ROOT / "scripts/prepare-release.py"),
            "--root",
            str(root),
            "--ref",
            ref,
        )
        assert result.stdout == "0.2.2\n"
        assert release.validate_versions(root, "v0.2.2") == "0.2.2"
        assert release._without_release_versions(release._read_metadata(root)) == metadata
        after = {path: (root / path).read_bytes() for path in release.TARGETS}
        if attempt == 0:
            for path in release.TARGETS[:2]:
                assert after[path] == before[path].replace(
                    b'version = "0.2.1"',
                    b'version = "0.2.2"',
                    1,
                )
            prepared = after
        else:
            assert after == prepared
        assert not (root / ".venv").exists()
        assert not (root / "studio/node_modules").exists()
        assert run(root, "git", "rev-parse", "HEAD").stdout.strip() == commit
        assert run(root, "git", "show-ref").stdout == refs
        assert run(root, "git", "diff", "--name-only").stdout.splitlines() == sorted(
            release.TARGETS,
        )
        run(root, "uv", "lock", "--check")


@pytest.mark.parametrize(
    "ref",
    [
        "main",
        "v0.2.2",
        "refs/heads/v0.2.2",
        "refs/tags/v0.2.2",
        "refs/tags/v01.2.3",
    ],
)
def test_invalid_or_absent_ref_fails_before_writes(checkout, ref):
    run(checkout, "git", "branch", "v0.2.2")
    with pytest.raises(ValueError):
        release.prepare_release(checkout, ref)
    assert run(checkout, "git", "status", "--porcelain").stdout == ""


def test_deleted_tag_and_wrong_checkout(checkout):
    run(checkout, "git", "tag", "v0.2.2")
    run(checkout, "git", "commit", "--allow-empty", "--quiet", "-m", "Later commit")
    with pytest.raises(ValueError, match="not checkout HEAD"):
        release.resolve_tag(checkout, "refs/tags/v0.2.2")
    run(checkout, "git", "tag", "-d", "v0.2.2")
    with pytest.raises(ValueError, match="does not exist"):
        release.resolve_tag(checkout, "refs/tags/v0.2.2")


@pytest.mark.parametrize(
    "field",
    [
        "pyproject.toml",
        "library/pyproject.toml",
        "studio/package.json",
        "studio/package-lock.json",
        "npm-lock-root",
        "nexuml",
        "nexuml-library",
    ],
)
def test_validator_rejects_each_drifting_field(checkout, field):
    if field in ("nexuml", "nexuml-library"):
        path = checkout / "uv.lock"
        path.write_text(
            path.read_text().replace(
                f'name = "{field}"\nversion = "0.2.1"',
                f'name = "{field}"\nversion = "0.2.2"',
                1,
            )
        )
    elif field.endswith(".toml"):
        path = checkout / field
        path.write_text(path.read_text().replace('version = "0.2.1"', 'version = "0.2.2"'))
    else:
        path = checkout / ("studio/package-lock.json" if field == "npm-lock-root" else field)
        data = json.loads(path.read_text())
        target = data["packages"][""] if field == "npm-lock-root" else data
        target["version"] = "0.2.2"
        path.write_text(json.dumps(data))
    result = run(
        checkout,
        sys.executable,
        str(ROOT / "scripts/check-release-versions.py"),
        "--root",
        str(checkout),
        check=False,
    )
    assert result.returncode != 0
    assert "Package versions must match" in result.stderr


def test_branch_and_explicit_tag_validation(checkout):
    for args in ((), ("--tag", "v0.2.1")):
        result = run(
            checkout,
            sys.executable,
            str(ROOT / "scripts/check-release-versions.py"),
            "--root",
            str(checkout),
            *args,
        )
        assert result.stdout == "0.2.1\n"
    with pytest.raises(ValueError, match="must match"):
        release.validate_versions(checkout, "v0.2.2")


def test_missing_or_noneditable_workspace_entries_are_rejected(checkout):
    path = checkout / "uv.lock"
    path.write_text(path.read_text().replace('editable = "library"', 'virtual = "library"'))
    with pytest.raises(ValueError, match="one editable.*nexuml-library"):
        release.validate_versions(checkout)


def test_stale_lock_is_rejected_before_stamping(checkout):
    run(checkout, "git", "tag", "v0.2.2")
    path = checkout / "pyproject.toml"
    path.write_text(path.read_text().replace("six==1.17.0", "six==1.16.0"))
    before = {path: (checkout / path).read_bytes() for path in release.TARGETS}
    with pytest.raises(ValueError, match="uv lock --check failed"):
        release.prepare_release(checkout, "refs/tags/v0.2.2")
    assert {path: (checkout / path).read_bytes() for path in release.TARGETS} == before


@pytest.mark.parametrize("target", ["uv.lock", "studio/package-lock.json", "pyproject.toml"])
def test_preparation_rejects_unowned_changes(checkout, monkeypatch, target):
    run(checkout, "git", "tag", "v0.2.2")
    original = release._run

    def mutate(root, *command):
        original(root, *command)
        if command[:2] == ("npm", "version"):
            path = checkout / target
            text = path.read_text()
            if target == "uv.lock":
                text = text.replace("sha256:", "sha256:tampered", 1)
            elif target.endswith(".json"):
                text = text.replace("sha512-unchanged", "sha512-tampered")
            else:
                text = text.replace("six==1.17.0", "six>=1.17.0")
            path.write_text(text)

    monkeypatch.setattr(release, "_run", mutate)
    with pytest.raises(ValueError, match=f"non-version metadata in {target}"):
        release.prepare_release(checkout, "refs/tags/v0.2.2")


def test_repository_branch_metadata_is_valid():
    assert (
        release.validate_versions(ROOT)
        == tomllib.loads(
            (ROOT / "pyproject.toml").read_text(),
        )["project"]["version"]
    )
