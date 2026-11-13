"""Stage and submit the FD003 recovery experiment to GH200 only.

Run from the configured cluster-connected workstation, with main NexuML core
and the NEX-233 feature worktree available locally. Planning is read-only;
``--submit`` stages two immutable Git commits and creates one Kubernetes Job.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

NAMESPACE = "lbecker"
JUICEFS_ROOT = Path("/mnt/juicefs_prisma")
DEFAULT_EXPERIMENT_ROOT = JUICEFS_ROOT / "models/nexuml/.experiments"
DEFAULT_STAGING_ROOT = JUICEFS_ROOT / "job-staging/nexuml/within-machine-recovery"
DEFAULT_FEATURE_ROOT = Path("/workspaces/NexuFed/NexuML-predictive")
DEFAULT_CORE_ROOT = Path("/workspaces/NexuFed/NexuML")
DEFAULT_JOB = "nexuml-rul-fd003-recovery-gh200"
IMAGE = "harbor.ika.rub.de/library/pytorch-template-lb-multiarch:latest"


def _git(worktree: Path, *args: str) -> str:
    """Run a read-only Git query for the selected worktree.

    Returns:
        Stripped Git command output.

    Raises:
        subprocess.CalledProcessError: If the worktree is not usable.
    """
    return subprocess.check_output(
        ["git", "-C", str(worktree), *args], text=True
    ).strip()


def _revision(worktree: Path, branch: str) -> str:
    """Identify an immutable committed source revision.

    Returns:
        Full 40-character Git commit hash.

    Raises:
        ValueError: If the worktree branch or committed state is not suitable.
    """
    if _git(worktree, "branch", "--show-current") != branch:
        raise ValueError(f"Expected branch {branch} at {worktree}")
    revision = _git(worktree, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Invalid Git revision at {worktree}")
    return revision


def _stage_subpath(staging_root: Path, stage_name: str) -> str:
    """Resolve the PVC-relative staged source path.

    Returns:
        Kubernetes volume subPath for the immutable staged source.

    Raises:
        ValueError: If the staging destination escapes the Prisma mount.
    """
    try:
        relative = staging_root.resolve().relative_to(JUICEFS_ROOT)
    except ValueError as exc:
        raise ValueError("Staging root must be inside /mnt/juicefs_prisma") from exc
    return str(relative / stage_name)


def _runtime_command(experiment_root: Path) -> str:
    """Construct the GH200-specific Python and CUDA bootstrap.

    Returns:
        Shell script passed to the pinned Job container.
    """
    return "\n".join(
        [
            "set -euo pipefail",
            "test -f /workspace/.nexuml-recovery-sources.json",
            "export UV_CACHE_DIR=/tmp/uv-cache",
            "export PYTHONPATH=/workspace/core/src:/workspace/feature/library/src",
            "uv venv --python 3.12 --system-site-packages /tmp/nexuml-venv",
            "export PATH=/tmp/nexuml-venv/bin:$PATH",
            "uv pip install --python /tmp/nexuml-venv/bin/python "
            "-e /workspace/core -e /workspace/feature/library",
            "python -c 'import torch; "
            "print(\"CUDA:\", torch.cuda.is_available(), "
            "\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() "
            "else \"none\", flush=True); "
            "assert torch.cuda.is_available()'",
            "python -m nexuml_library.evaluation.maintenance.rul_machine_flow_recover "
            f"--experiment-root {shlex.quote(str(experiment_root))} "
            "--rebuild --retrain-missing",
        ]
    )


def build_manifest(
    *, job: str, subpath: str, experiment_root: Path
) -> dict[str, Any]:
    """Build a single-GPU, immutable-source Job pinned strictly to GH200.

    Returns:
        Kubernetes batch/v1 Job manifest.

    Raises:
        ValueError: If the requested Job name or output path is invalid.
    """
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", job) or len(job) > 63:
        raise ValueError("Job name must be a valid Kubernetes DNS label")
    if not experiment_root.resolve().is_relative_to(JUICEFS_ROOT):
        raise ValueError("Experiment root must be inside /mnt/juicefs_prisma")
    if not subpath.startswith("job-staging/") or ".." in Path(subpath).parts:
        raise ValueError("Unsafe staged source subPath")
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job,
            "namespace": NAMESPACE,
            "labels": {"kueue.x-k8s.io/queue-name": "jobs"},
        },
        "spec": {
            "suspend": False,
            "backoffLimit": 1,
            "template": {
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname": "gh200"},
                    "tolerations": [{
                        "key": "gpu-recovery",
                        "operator": "Equal",
                        "value": "true",
                        "effect": "NoSchedule",
                    }],
                    "runtimeClassName": "nvidia",
                    "priorityClassName": "normal",
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": "regcred"}],
                    "containers": [{
                        "name": "recovery",
                        "image": IMAGE,
                        "imagePullPolicy": "Always",
                        "command": ["/bin/bash", "-lc"],
                        "args": [_runtime_command(experiment_root)],
                        "envFrom": [{"configMapRef": {"name": "env-config"}}],
                        "resources": {
                            "requests": {
                                "cpu": "8",
                                "memory": "64Gi",
                                "nvidia.com/gpu": "1",
                            },
                            "limits": {
                                "memory": "64Gi",
                                "nvidia.com/gpu": "1",
                            },
                        },
                        "volumeMounts": [
                            {"name": "juicefs", "mountPath": "/mnt/juicefs_prisma"},
                            {"name": "juicefs", "mountPath": "/workspace", "subPath": subpath},
                            {"name": "shm", "mountPath": "/dev/shm"},
                            {"name": "temporary", "mountPath": "/tmp"},
                        ],
                    }],
                    "volumes": [
                        {"name": "juicefs", "persistentVolumeClaim": {"claimName": "juicefs"}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
                        {"name": "temporary", "emptyDir": {"sizeLimit": "32Gi"}},
                    ],
                }
            },
        },
    }


def _extract_archive(worktree: Path, target: Path) -> None:
    """Extract tracked Git files from HEAD without mixing worktree contents.

    Raises:
        subprocess.CalledProcessError: If archive creation or extraction fails.
        RuntimeError: If the Git archive stream cannot be read completely.
    """
    target.mkdir(parents=True, exist_ok=False)
    archive = subprocess.Popen(
        ["git", "-C", str(worktree), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
    )
    if archive.stdout is None:
        raise RuntimeError("Could not open Git archive stream")
    try:
        subprocess.run(["tar", "-x", "-C", str(target)], stdin=archive.stdout, check=True)
    finally:
        archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError(f"Failed to stage committed source: {worktree}")


def _stage_sources(
    *, core_root: Path, feature_root: Path, staging_root: Path,
    stage_name: str, core_sha: str, feature_sha: str,
) -> Path:
    """Publish a fresh immutable two-revision source directory on JuiceFS.

    Returns:
        Immutable directory visible to the Kubernetes PVC.

    Raises:
        FileExistsError: If that source identity was already staged.
        RuntimeError: If the requested JuiceFS mount is absent.
    """
    if not JUICEFS_ROOT.is_mount():
        raise RuntimeError("JuiceFS Prisma is not mounted on this workstation")
    stage = staging_root / stage_name
    if stage.exists():
        raise FileExistsError(f"Immutable source already staged: {stage}")
    stage.mkdir(parents=True)
    _extract_archive(core_root, stage / "core")
    _extract_archive(feature_root, stage / "feature")
    (stage / ".nexuml-recovery-sources.json").write_text(
        json.dumps({"core": core_sha, "feature": feature_sha}, sort_keys=True),
        encoding="utf-8",
    )
    return stage


def submit(
    manifest: dict[str, Any], *, core_root: Path, feature_root: Path,
    staging_root: Path, stage_name: str, core_sha: str, feature_sha: str,
) -> Path:
    """Publish committed sources and create exactly one GH200 Job.

    Returns:
        Persistent source staging path.

    Raises:
        RuntimeError: If a named Kubernetes Job is already present.
        subprocess.CalledProcessError: If Kubernetes submission fails.
    """
    job = str(manifest["metadata"]["name"])
    existing = subprocess.run(
        ["kubectl", "-n", NAMESPACE, "get", "job", job, "-o", "name"],
        text=True, capture_output=True, check=False,
    )
    if existing.returncode == 0:
        raise RuntimeError(f"Refusing to replace existing Job: {job}")
    if "NotFound" not in existing.stderr and "not found" not in existing.stderr.lower():
        raise RuntimeError(f"Cannot verify Job-name availability: {existing.stderr.strip()}")
    stage = _stage_sources(
        core_root=core_root, feature_root=feature_root,
        staging_root=staging_root, stage_name=stage_name,
        core_sha=core_sha, feature_sha=feature_sha,
    )
    subprocess.run(
        ["kubectl", "-n", NAMESPACE, "apply", "-f", "-"],
        input=json.dumps(manifest), text=True, check=True,
    )
    return stage


def main() -> None:
    """Plan or submit the two-revision FD003 recovery Job to GH200."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--core-root", type=Path, default=DEFAULT_CORE_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--job", default=DEFAULT_JOB)
    parser.add_argument("--submit", action="store_true", help="Stage sources and create the Job")
    args = parser.parse_args()
    feature_sha = _revision(
        args.feature_root,
        "feature/nex-233-flow-matching-predictive-maintenance-degradation-time-and",
    )
    core_sha = _revision(args.core_root, "main")
    stage_name = f"{feature_sha[:12]}-{core_sha[:12]}-{args.job}"
    subpath = _stage_subpath(args.staging_root, stage_name)
    manifest = build_manifest(
        job=args.job, subpath=subpath, experiment_root=args.experiment_root
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if not args.submit:
        print("[plan] Add --submit to stage the revisions and create the GH200 Job")
        return
    stage = submit(
        manifest, core_root=args.core_root, feature_root=args.feature_root,
        staging_root=args.staging_root, stage_name=stage_name,
        core_sha=core_sha, feature_sha=feature_sha,
    )
    print(f"[submitted] {args.job} (GH200 only); source: {stage}", flush=True)
    print(f"Monitor: kubectl -n {NAMESPACE} logs -f job/{args.job}", flush=True)


if __name__ == "__main__":
    main()
