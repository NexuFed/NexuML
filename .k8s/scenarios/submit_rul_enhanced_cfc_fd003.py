#!/usr/bin/env python3
"""Stage an immutable NEX-233 source snapshot and submit three native RUL Jobs.

This experiment-only launcher follows privacy_measure_theory's content-hashed
JuiceFS staging. It never copies a virtualenv, dataset, checkpoint or .git.
The Kubernetes worker creates its own ephemeral uv environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("job_rul_enhanced_cfc_fd003.yaml.template")
RENDERED = Path(__file__).with_name("rendered")
DEFAULT_STAGE = Path("/mnt/juicefs/job-staging/nexuml-nex233")
NAMESPACE = "lbecker"
SEEDS = (42, 43, 44)
SOURCE_DIRS = ("src", "library/src", "scripts")
ROOT_FILES = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "library/pyproject.toml",
    "library/README.md",
    "library/LICENSE",
)
SOURCE_SUFFIXES = frozenset((".py", ".sh", ".json", ".toml", ".yaml", ".yml", ".txt"))
EXCLUDED_DIRS = frozenset(
    ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", "rendered")
)
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def source_files(root: Path = ROOT) -> list[Path]:
    """Return the exact executable/source metadata file set to transfer."""
    files: list[Path] = []
    for name in SOURCE_DIRS:
        directory = root / name
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"missing or symbolic source directory: {directory}")
        for file in directory.rglob("*"):
            relative = file.relative_to(root)
            if any(part in EXCLUDED_DIRS for part in relative.parts):
                continue
            if file.is_symlink():
                raise ValueError(f"source snapshot must not contain symlinks: {file}")
            if file.is_file() and file.suffix in SOURCE_SUFFIXES:
                files.append(file)
    for name in ROOT_FILES:
        file = root / name
        if not file.is_file() or file.is_symlink():
            raise ValueError(f"missing or symbolic workspace metadata file: {file}")
        files.append(file)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def digest_source(root: Path = ROOT) -> str:
    """Fingerprint file paths and actual bytes without using Git metadata."""
    digest = hashlib.sha256()
    for file in source_files(root):
        relative = file.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with file.open("rb") as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def stage_snapshot(staging_root: Path, digest: str, *, dry_run: bool) -> Path:
    """Copy and verify the source exactly once, with an atomic publication."""
    target = staging_root / digest
    sentinel = target / "NEX233_SOURCE_DIGEST.txt"
    if target.exists():
        if not sentinel.is_file() or sentinel.read_text(encoding="utf-8").strip() != digest:
            raise ValueError(f"partial or conflicting source snapshot: {target}")
        if digest_source(target) != digest:
            raise ValueError(f"staged source fingerprint changed: {target}")
        return target
    if dry_run:
        return target
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / f".{digest}.partial-{os.getpid()}"
    temporary.mkdir(exist_ok=False)
    try:
        names = [file.relative_to(ROOT).as_posix() for file in source_files()]
        subprocess.run(
            [
                "rsync",
                "-a",
                "--from0",
                "--files-from=-",
                str(ROOT) + "/",
                str(temporary) + "/",
            ],
            input=("\0".join(names) + "\0").encode("utf-8"),
            check=True,
        )
        if digest_source(temporary) != digest or digest_source() != digest:
            raise ValueError("source changed during staging or copy verification failed")
        (temporary / "NEX233_SOURCE_DIGEST.txt").write_text(digest + "\n", encoding="utf-8")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def render_manifest(seed: int, digest: str, source_subpath: str) -> str:
    """Render one standard Kubernetes Job with the immutable source path."""
    if seed not in SEEDS:
        raise ValueError(f"unexpected RUL comparison seed: {seed}")
    scenario = (
        "maintenance-rul-enhanced-cfc-fd003-compare-six"
        + (f"-s{seed}" if seed != 42 else "")
    )
    replacements = {
        "__SEED__": str(seed),
        "__SCENARIO__": scenario,
        "__SOURCE_DIGEST__": digest,
        "__STAGE_SUBPATH__": source_subpath,
    }
    rendered = TEMPLATE.read_text(encoding="utf-8")
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    if re.search(r"__[A-Z_]+__", rendered):
        raise ValueError("unresolved NEX-233 Kubernetes Job template variable")
    return rendered


def existing_job(name: str, namespace: str) -> bool:
    """Return whether a Job already exists without touching it."""
    process = subprocess.run(
        ["kubectl", "-n", namespace, "get", "job", name, "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        if "NotFound" in process.stderr or "not found" in process.stderr:
            return False
        raise RuntimeError(f"kubectl job lookup failed: {process.stderr.strip()}")
    json.loads(process.stdout)
    return True


def main() -> None:
    """Stage the local source and render or submit the selected comparison Jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not NAME_PATTERN.fullmatch(args.namespace):
        parser.error("invalid Kubernetes namespace")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(
        seed not in SEEDS for seed in args.seeds
    ):
        parser.error("seeds must be unique and drawn from 42, 43, 44")
    staging_root = args.staging_root.expanduser().absolute()
    if not staging_root.is_relative_to(Path("/mnt/juicefs")):
        parser.error("staging root must be under the cluster's /mnt/juicefs PVC mount")
    if staging_root == Path("/mnt/juicefs"):
        parser.error("staging root must be a dedicated subdirectory")
    digest = digest_source()
    target = stage_snapshot(staging_root, digest, dry_run=args.dry_run)
    subpath = target.relative_to("/mnt/juicefs").as_posix()
    RENDERED.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        job = f"nexuml-rul-fd003-cfc-six-s{seed}"
        manifest = RENDERED / f"{job}.yaml"
        manifest.write_text(render_manifest(seed, digest, subpath), encoding="utf-8")
        if args.dry_run:
            print(f"[dry-run] {job}: {manifest} (no source copy or cluster changes)")
            continue
        if existing_job(job, args.namespace):
            print(f"[skip existing job] {job}: delete explicitly before replacing")
            continue
        subprocess.run(
            ["kubectl", "-n", args.namespace, "apply", "--dry-run=client", "-f", str(manifest)],
            check=True,
        )
        subprocess.run(
            ["kubectl", "-n", args.namespace, "apply", "-f", str(manifest)],
            check=True,
        )
        print(f"[submitted] {job} snapshot={digest}", flush=True)
    print(f"source_sha256={digest} staging={target}")


if __name__ == "__main__":
    main()
