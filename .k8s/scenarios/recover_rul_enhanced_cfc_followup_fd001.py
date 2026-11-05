#!/usr/bin/env python3
"""Recover six failed FD001 CfC follow-up Jobs using proven local artifacts.

Copy only the seed-42 direct-adaptation checkpoint, its checkpoint audit,
the hash-matched CfC endpoint normalizer, and the prepared FD001 RUL dataset.
Preserve all original rendered manifests and the source snapshot digest.
Kubernetes Job replacement is opt-in and never touches FD003 Jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact

DIRECT_RUN = "rul-cmapss-FD001-patchformer-direct-rul-adaptation-seed42"
CONTROLLED_RUN = "rul-FD001-controlled-patchformer-seed42"
ENDPOINT_FILE = "rul-FD001-patchformer-sequence-cfc-flow-seed42-endpoints.pt"
PREPARED_FILE = "cmapss-FD001-seed42-controlled-w30-s42.pt"
IMAGE = "ghcr.io/nexufed/nexuml:sha-f8d5490-cuda12.8.1"
DEFAULT_SHARED = Path("/mnt/juicefs_prisma/models/nexuml/.experiments")
RENDERED = Path(__file__).with_name("rendered")
CONFIGURATIONS = ("baseline", "all-no-trend")
SEEDS = (42, 43, 44)
_HEX = re.compile(r"[a-f0-9]{64}")


def job_names() -> tuple[str, ...]:
    """Return only the six predeclared FD001 follow-up Job names.

    Returns:
        Kubernetes Job names in deterministic order.
    """
    return tuple(
        f"nexuml-rul-cfc-followup-fd001-{configuration}-s{seed}"
        for configuration in CONFIGURATIONS
        for seed in SEEDS
    )


def file_hash(path: Path) -> str:
    """Hash one file without loading its full contents into memory.

    Returns:
        Hexadecimal SHA-256 checksum.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_artifacts(local: Path, shared: Path) -> tuple[list[tuple[Path, Path]], dict[str, Any]]:
    """Validate provenance and construct minimal, non-destructive transfer list.

    Args:
        local: Existing 3.7 GB local experiments directory.
        shared: Persistent Prisma experiments directory mounted by Kubernetes.

    Returns:
        Exact source/destination pairs and rewritten Pod-compatible audit.

    Raises:
        FileNotFoundError: If checkpoint, endpoint or prepared dataset is missing.
        ValueError: If upstream artifacts disagree or destination would be overwritten.
    """
    src_rul = local / "rul"
    dst_rul = shared / "rul"
    audit_src = src_rul / "audits" / f"{DIRECT_RUN}.checkpoint.json"
    endpoint_src = src_rul / "prepared" / ENDPOINT_FILE
    prepared_src = src_rul / "prepared" / PREPARED_FILE
    for source in (audit_src, endpoint_src, prepared_src):
        if not source.is_file():
            raise FileNotFoundError(f"Missing local FD001 prerequisite: {source}")
    audit = json.loads(audit_src.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError("Upstream checkpoint audit must be a JSON object")
    best = audit.get("best_model_path")
    if not isinstance(best, str) or not best:
        raise ValueError("Direct-adaptation audit has no best_model_path")
    recorded = Path(best)
    if recorded.parent.name != DIRECT_RUN or recorded.name == "last.ckpt":
        raise ValueError(f"Unexpected FD001 source checkpoint: {recorded}")
    checkpoint_src = src_rul / "checkpoints" / DIRECT_RUN / recorded.name
    if not checkpoint_src.is_file():
        raise FileNotFoundError(f"Direct-adaptation checkpoint missing: {checkpoint_src}")
    if audit.get("loaded_strict_model_state") is not True:
        raise ValueError("FD001 upstream checkpoint audit lacks strict-state confirmation")

    endpoint_settings = read_prepared_artifact(endpoint_src)["settings"]
    if not isinstance(endpoint_settings, dict):
        raise ValueError("Endpoint settings missing")
    checksum = file_hash(checkpoint_src)
    if (
        endpoint_settings.get("subset") != "FD001"
        or endpoint_settings.get("seed") != 42
        or endpoint_settings.get("feature_normalization") != "train_endpoints_only"
        or endpoint_settings.get("frozen_patchformer_upstream_run") != DIRECT_RUN
        or endpoint_settings.get("frozen_patchformer_checkpoint_sha256") != checksum
        or endpoint_settings.get("held_out_test_engine_labels_used") is not False
        or Path(str(endpoint_settings.get("source_rul_artifact"))).resolve()
        != prepared_src.resolve()
    ):
        raise ValueError("FD001 endpoint provenance does not match local checkpoint/data")
    controlled = dst_rul / "audits" / f"{CONTROLLED_RUN}.checkpoint.json"
    if controlled.exists():
        raise ValueError(
            f"{controlled} would shadow the direct-adaptation checkpoint; "
            "do not combine unverified upstream checkpoints"
        )

    checkpoint_dst = dst_rul / "checkpoints" / DIRECT_RUN / checkpoint_src.name
    endpoint_dst = dst_rul / "prepared" / ENDPOINT_FILE
    prepared_dst = dst_rul / "prepared" / PREPARED_FILE
    audit_dst = dst_rul / "audits" / audit_src.name
    pairs = [
        (checkpoint_src, checkpoint_dst),
        (endpoint_src, endpoint_dst),
        (prepared_src, prepared_dst),
    ]
    for source, destination in pairs:
        if destination.exists() and file_hash(source) != file_hash(destination):
            raise ValueError(f"Refusing to overwrite different shared artifact: {destination}")

    rewritten = dict(audit)
    rewritten["best_model_path"] = (
        f"/mnt/nexuml-experiments/rul/checkpoints/{DIRECT_RUN}/{checkpoint_src.name}"
    )
    if audit_dst.exists():
        loaded = json.loads(audit_dst.read_text(encoding="utf-8"))
        if loaded != rewritten:
            raise ValueError(f"Refusing to replace conflicting shared audit: {audit_dst}")
    return pairs, rewritten


def patched_manifest(name: str, original: str) -> str:
    """Update only the FD001 checkpoint preflight, preserving source SHA-256.

    Args:
        name: One of the six authorized FD001 Job names.
        original: Original rendered Job manifest, including source snapshot.

    Returns:
        Recovery manifest without changes to the job training configuration.

    Raises:
        ValueError: For other Jobs, missing attestations or invalid templates.
    """
    if name not in job_names():
        raise ValueError(f"Refusing to modify non-FD001 Job: {name}")
    expected_name = f"  name: {name}\n"
    old = f'run = "{CONTROLLED_RUN}"'
    new = f'run = "{DIRECT_RUN}"'
    if (
        original.count(expected_name) != 1
        or original.count(old) != 1
        or original.count(f"image: {IMAGE}") != 1
        or original.count("nexuml-study: rul-cfc-followup") < 1
        or "nvidia.com/gpu: \"1\"" not in original
        or "subPath: job-staging/nexuml-nex233/" not in original
        or "NEX233_SOURCE_DIGEST.txt" not in original
    ):
        raise ValueError(f"Incompatible original FD001 Job manifest: {name}")
    found = re.search(r'nexuml\.ika\.rub\.de/source-sha256: "([a-f0-9]{64})"', original)
    if found is None or _HEX.fullmatch(found.group(1)) is None:
        raise ValueError(f"Source digest missing from original Job: {name}")
    return original.replace(old, new, 1)


def plan_manifests(rendered_root: Path) -> dict[str, str]:
    """Read and strictly validate all six original rendered Job definitions.

    Returns:
        Authorized Job names mapped to patched YAML content.

    Raises:
        FileNotFoundError: If any original manifest is unavailable.
        ValueError: If source provenance or template shape is inconsistent.
    """
    manifests = {
        name: patched_manifest(
            name, (rendered_root / f"{name}.yaml").read_text(encoding="utf-8")
        )
        for name in job_names()
    }
    digests = {
        re.search(r'nexuml\.ika\.rub\.de/source-sha256: "([a-f0-9]{64})"', text).group(1)
        for text in manifests.values()
    }
    if len(digests) != 1:
        raise ValueError("FD001 manifests were not staged from the same source snapshot")
    return manifests


def publish_file(source: Path, destination: Path) -> None:
    """Copy a verified file atomically without replacing conflicting content.

    Raises:
        ValueError: If the copy differs from its trusted source.
    """
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError(f"Shared file changed since preflight: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        shutil.copyfile(source, temporary)
        if file_hash(source) != file_hash(temporary):
            raise ValueError(f"Failed copy integrity check: {destination}")
        if destination.exists():
            if file_hash(destination) != file_hash(source):
                raise ValueError(f"Shared destination appeared with different bytes: {destination}")
        else:
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_audit(shared: Path, audit: dict[str, Any]) -> None:
    """Write the Pod-compatible audit without overwriting other provenance.

    Raises:
        ValueError: If another audit differs from the planned contents.
    """
    destination = shared / "rul" / "audits" / f"{DIRECT_RUN}.checkpoint.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != audit:
            raise ValueError(f"Shared audit differs from expected: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as temp:
        temporary = Path(temp.name)
        json.dump(audit, temp, indent=2)
        temp.write("\n")
    try:
        if destination.exists():
            if json.loads(destination.read_text(encoding="utf-8")) != audit:
                raise ValueError(f"Shared audit changed during migration: {destination}")
        else:
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_manifests(
    rendered_root: Path, manifests: dict[str, str],
) -> Path:
    """Write independent recovery copies, never altering rendered originals.

    Returns:
        Directory of six recovery YAMLs.

    Raises:
        ValueError: If a previously rendered recovery YAML differs.
    """
    output = rendered_root / "fd001-direct-adaptation-recovery"
    for name, content in manifests.items():
        path = output / f"{name}.yaml"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Conflicting previous recovery manifest: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in manifests.items():
        path = output / f"{name}.yaml"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return output


def _kubectl(namespace: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Execute kubectl and capture failure output.

    Returns:
        Completed kubectl process.
    """
    return subprocess.run(
        ["kubectl", "-n", namespace, *arguments],
        check=False, capture_output=True, text=True,
    )


def restart_jobs(namespace: str, manifest_dir: Path) -> None:
    """Recreate only failed or absent FD001 Jobs after full cluster preflight.

    Raises:
        RuntimeError: If any Job is not failed, or Kubernetes rejects a manifest.
    """
    missing: set[str] = set()
    for name in job_names():
        queried = _kubectl(namespace, "get", "job", name, "-o", "json")
        if queried.returncode:
            if "(NotFound)" in queried.stderr or "not found" in queried.stderr:
                missing.add(name)
                continue
            raise RuntimeError(f"Cannot check Job {name}: {queried.stderr}")
        record = json.loads(queried.stdout)
        failed = any(
            condition.get("type") == "Failed" and condition.get("status") == "True"
            for condition in record.get("status", {}).get("conditions", [])
        )
        if not failed or record.get("status", {}).get("succeeded", 0):
            raise RuntimeError(f"Refusing to replace non-failed Job: {name}")
    for name in job_names():
        path = manifest_dir / f"{name}.yaml"
        result = _kubectl(namespace, "create", "--dry-run=client", "-f", str(path))
        if result.returncode:
            raise RuntimeError(f"Invalid recovery manifest {name}: {result.stderr}")
    for name in job_names():
        if name not in missing:
            deleted = _kubectl(
                namespace, "delete", "job", name, "--cascade=foreground", "--wait=true"
            )
            if deleted.returncode:
                raise RuntimeError(f"Cannot delete failed Job {name}: {deleted.stderr}")
        created = _kubectl(namespace, "create", "-f", str(manifest_dir / f"{name}.yaml"))
        if created.returncode:
            raise RuntimeError(f"Could not create replacement {name}: {created.stderr}")
        print(f"[restarted] {name}", flush=True)


def main() -> None:
    """Preview or safely execute the selective FD001 artifact migration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-experiments", type=Path, default=Path(".experiments"))
    parser.add_argument("--shared-experiments", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--rendered-dir", type=Path, default=RENDERED)
    parser.add_argument("--namespace", default="lbecker")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.restart and not args.apply:
        parser.error("--restart requires --apply")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.namespace) is None:
        parser.error("Invalid namespace")
    local = args.local_experiments.expanduser().absolute()
    shared = args.shared_experiments.expanduser().absolute()
    if not local.is_dir() or not shared.is_dir():
        parser.error("Existing local and shared experiments directories are required")
    if local.resolve() == shared.resolve():
        parser.error("Local and shared experiments paths are identical")
    pairs, audit = checked_artifacts(local, shared)
    manifests = plan_manifests(args.rendered_dir)
    print("FD001 upstream endpoint checksum matches direct-adaptation checkpoint.")
    print("Six original manifests share the same immutable source snapshot.")
    for source, destination in pairs:
        print(f"[transfer] {source} -> {destination}")
    print(
        "[audit] best_model_path -> "
        f"{audit['best_model_path']} (Pod-visible Prisma mount)"
    )
    output = args.rendered_dir / "fd001-direct-adaptation-recovery"
    print(f"[manifests] {output} (the originals remain untouched)")
    if not args.apply:
        print("[dry-run] No files or Jobs changed. Add --apply to migrate.")
        return
    for source, destination in pairs:
        publish_file(source, destination)
    publish_audit(shared, audit)
    output = publish_manifests(args.rendered_dir, manifests)
    if args.restart:
        restart_jobs(args.namespace, output)
    else:
        print("Artifacts migrated. Review six YAMLs, then add --apply --restart.")


if __name__ == "__main__":
    main()
