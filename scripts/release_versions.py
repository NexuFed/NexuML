"""Shared stable release identity and disposable-checkout preparation."""

import copy
import json
from pathlib import Path
import re
import subprocess
import tomllib


PROJECTS = {"nexuml": ".", "nexuml-library": "library"}
TARGETS = (
    "pyproject.toml",
    "library/pyproject.toml",
    "uv.lock",
    "studio/package.json",
    "studio/package-lock.json",
)


def stable_version(tag: str) -> str:
    """Return X.Y.Z from an ASCII stable vX.Y.Z tag.

    Raises:
        ValueError: The tag is not an ASCII stable version.
    """
    if not re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag):
        raise ValueError(f"Expected stable vX.Y.Z tag: {tag!r}")
    return tag[1:]


def resolve_tag(root: Path, ref: str) -> tuple[str, str]:
    """Return the normalized version and peeled commit for a full stable tag ref.

    Raises:
        ValueError: The ref is invalid, absent, or does not point to checkout HEAD.
    """
    if not ref.startswith("refs/tags/"):
        raise ValueError(f"Expected full refs/tags/vX.Y.Z release ref, not {ref!r}")
    version = stable_version(ref.removeprefix("refs/tags/"))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Release tag {ref!r} or checkout HEAD does not exist") from error
    if commit != head:
        raise ValueError(f"Release tag {ref!r} resolves to {commit}, not checkout HEAD {head}")
    return version, commit


def _read_metadata(root: Path) -> dict:
    metadata = {}
    for path in TARGETS:
        text = (root / path).read_text()
        metadata[path] = json.loads(text) if path.endswith(".json") else tomllib.loads(text)
    return metadata


def _workspace_packages(lock: dict) -> dict:
    packages = {}
    for name, editable in PROJECTS.items():
        matches = [package for package in lock["package"] if package["name"] == name]
        if len(matches) != 1 or matches[0].get("source") != {"editable": editable}:
            raise ValueError(f"uv.lock must contain one editable {name!r} at {editable!r}")
        packages[name] = matches[0]
    return packages


def validate_versions(root: Path, tag: str | None = None) -> str:
    """Return the shared stable project/npm/workspace-lock version.

    Raises:
        ValueError: Versions differ, are unstable, or do not match the optional tag.
    """
    metadata = _read_metadata(root)
    versions = {path: metadata[path]["project"]["version"] for path in TARGETS[:2]}
    for path in TARGETS[3:]:
        versions[path] = metadata[path]["version"]
    versions['studio/package-lock.json packages[""]'] = metadata["studio/package-lock.json"][
        "packages"
    ][""]["version"]
    for name, package in _workspace_packages(metadata["uv.lock"]).items():
        versions[f"uv.lock {name}"] = package["version"]
    version = versions["pyproject.toml"]
    if len(set(versions.values())) != 1:
        raise ValueError(f"Package versions must match: {versions}")
    stable_version(f"v{version}")
    if tag is not None and stable_version(tag) != version:
        raise ValueError(f"Tag {tag!r} must match package version v{version}")
    return version


def _without_release_versions(metadata: dict) -> dict:
    """Return metadata masking only owned identities and uv workspace references."""
    metadata = copy.deepcopy(metadata)
    for path in TARGETS[:2]:
        del metadata[path]["project"]["version"]
    for path in TARGETS[3:]:
        del metadata[path]["version"]
    del metadata["studio/package-lock.json"]["packages"][""]["version"]

    def mask_workspace_version(value):
        if isinstance(value, dict):
            name = value.get("name")
            if name in PROJECTS and value.get("source") == {"editable": PROJECTS[name]}:
                value.pop("version", None)
            for child in value.values():
                mask_workspace_version(child)
        elif isinstance(value, list):
            for child in value:
                mask_workspace_version(child)

    mask_workspace_version(metadata["uv.lock"])
    return metadata


def _run(root: Path, *command: str) -> None:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"{' '.join(command)} failed:\n{result.stderr or result.stdout}")


def prepare_release(root: Path, ref: str) -> str:
    """Return the stamped release version without syncing, committing, or tagging.

    Raises:
        ValueError: Tag/metadata validation, a version command, or preservation checks fail.
    """
    version, commit = resolve_tag(root, ref)
    validate_versions(root)
    # Check the incoming lock before any writes; never repair stale dependency metadata.
    _run(root, "uv", "lock", "--check")
    before = _without_release_versions(_read_metadata(root))
    for name in PROJECTS:
        _run(root, "uv", "version", version, "--package", name, "--frozen")
    _run(root, "uv", "lock")
    _run(
        root / "studio",
        "npm",
        "version",
        version,
        "--no-git-tag-version",
        "--ignore-scripts",
        "--allow-same-version",
    )
    validate_versions(root, f"v{version}")
    after = _without_release_versions(_read_metadata(root))
    for path in TARGETS:
        if before[path] != after[path]:
            raise ValueError(f"Release preparation changed non-version metadata in {path}")
    if resolve_tag(root, ref) != (version, commit):
        raise ValueError("Release tag changed during preparation")
    return version
