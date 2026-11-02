#!/usr/bin/env python3
"""Recreate only image-pull-blocked FD003 confirmation Jobs on cached GPU nodes.

Reuse the ORIGINAL rendered manifests and immutable staged source, rather
than regenerating from a possibly changed local repository. Preview by
default; --apply deletes and recreates the five specified failed Jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from submit_rul_enhanced_cfc_fd003 import NAMESPACE, RENDERED

NODE_POOL = ("cip-gpu-4", "cip-gpu-9", "cip-gpu-14")
FAILED_JOBS = (
    "nexuml-rul-fd003-cfc-confirm-baseline-s44",
    "nexuml-rul-fd003-cfc-confirm-finetune-s43",
    "nexuml-rul-fd003-cfc-confirm-finetune-s44",
    "nexuml-rul-fd003-cfc-confirm-all-no-trend-s42",
    "nexuml-rul-fd003-cfc-confirm-all-no-trend-s43",
)
COMPLETE_RUN_MARKERS = (
    "baseline-s42.execution.json",
    "baseline-s43.execution.json",
    "finetune-s42.execution.json",
    "all-no-trend-s44.execution.json",
)
DEFAULT_RESULTS = Path(
    "/mnt/juicefs_prisma/models/nexuml/.experiments/"
    "rul/enhanced-cfc/fd003/confirmation"
)
_HOSTNAME = re.compile(
    r"(?P<indent>^\s*)- key: kubernetes\.io/hostname\n"
    r"(?P=indent)  operator: (?:NotIn|In)\n"
    r"(?P=indent)  values: \[[^\]\n]+\]",
    flags=re.MULTILINE,
)


def confirmed_source_digest(results: Path) -> str:
    """Return shared source hash from the four completed, attested runs.

    Returns:
        Exact immutable source snapshot SHA-256.

    Raises:
        ValueError: If source identity or completion evidence is inconsistent.
    """
    digests: set[str] = set()
    for name in COMPLETE_RUN_MARKERS:
        data = json.loads((results / name).read_text(encoding="utf-8"))
        digest = data.get("source_sha256") if isinstance(data, dict) else None
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or data.get("timing_includes_fit_validate_test") is not True
        ):
            raise ValueError(f"Invalid completed run attestation: {name}")
        digests.add(digest)
    if len(digests) != 1:
        raise ValueError("Completed confirmation runs used different source digests")
    return digests.pop()


def pin_original_manifest(text: str, *, job: str, digest: str) -> str:
    """Change only the hostname node-affinity allowlist.

    Args:
        text: Original immutable-source Job manifest.
        job: Expected Kubernetes Job name.
        digest: Source digest attested by the four completed runs.

    Returns:
        Rewritten manifest retaining the original source and job identity.

    Raises:
        ValueError: If the input is not exactly the expected campaign Job.
    """
    if job not in FAILED_JOBS:
        raise ValueError(f"Unexpected retry Job: {job}")
    checks = (
        f"name: {job}\n",
        "nexuml-study: rul-fd003-cfc-confirm",
        f'nexuml.ika.rub.de/source-sha256: "{digest}"',
        f"subPath: job-staging/nexuml-nex233/{digest}",
        'nvidia.com/gpu: "1"',
        "nvidia.com/cuda.runtime.major",
        "kubernetes.io/arch: amd64",
        "imagePullPolicy: IfNotPresent",
        "harbor.ika.rub.de/library/nexuml:latest",
    )
    if any(value not in text for value in checks):
        raise ValueError(f"Untrusted or inconsistent original manifest: {job}")
    matches = list(_HOSTNAME.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one hostname affinity for {job}")
    match = matches[0]
    indent = match.group("indent")
    new = (
        f"{indent}- key: kubernetes.io/hostname\n"
        f"{indent}  operator: In\n"
        f"{indent}  values: [{', '.join(NODE_POOL)}]"
    )
    return text[:match.start()] + new + text[match.end():]


def _kubectl_json(namespace: str, resource: str, name: str) -> dict[str, Any]:
    """Read a Kubernetes resource through kubectl.

    Returns:
        Kubernetes resource JSON.

    Raises:
        subprocess.CalledProcessError: If the object cannot be read.
        ValueError: If kubectl did not return a JSON object.
    """
    result = subprocess.run(
        ["kubectl", "-n", namespace, "get", resource, name, "-o", "json"],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Kubernetes response for {name}")
    return payload


def verify_image_pull_blocked(namespace: str, job: str, digest: str) -> None:
    """Refuse to delete a completed or actively training Job.

    Raises:
        ValueError: If the Job or its Pod is not image-pull-blocked.
        subprocess.CalledProcessError: If Kubernetes cannot be inspected.
    """
    obj = _kubectl_json(namespace, "job", job)
    annotations = obj.get("metadata", {}).get("annotations", {})
    if annotations.get("nexuml.ika.rub.de/source-sha256") != digest:
        raise ValueError(f"Live Job has different source snapshot: {job}")
    if obj.get("status", {}).get("succeeded", 0):
        raise ValueError(f"Refusing to restart completed Job: {job}")
    pods = subprocess.run(
        [
            "kubectl", "-n", namespace, "get", "pods",
            "-l", f"batch.kubernetes.io/job-name={job}", "-o", "json",
        ],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(pods.stdout)
    items = payload.get("items", [])
    if not items:
        raise ValueError(f"No image-pull-blocked Pod found for {job}")
    reasons = [
        status.get("state", {}).get("waiting", {}).get("reason")
        for item in items
        for status in item.get("status", {}).get("containerStatuses", [])
    ]
    if not reasons or any(
        reason not in {"ErrImagePull", "ImagePullBackOff"} for reason in reasons
    ):
        raise ValueError(f"Job is no longer exclusively image-pull-blocked: {job}")


def main() -> None:
    """Preview or safely recreate the five failed Jobs without restaging code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    digest = confirmed_source_digest(args.results)
    previews = RENDERED / "cached-pool-retries"
    previews.mkdir(parents=True, exist_ok=True)

    # Preflight all jobs before deleting even one.
    for job in FAILED_JOBS:
        original = (RENDERED / f"{job}.yaml").read_text(encoding="utf-8")
        pinned = pin_original_manifest(original, job=job, digest=digest)
        output = previews / f"{job}.yaml"
        output.write_text(pinned, encoding="utf-8")
        verify_image_pull_blocked(args.namespace, job, digest)
        subprocess.run(
            ["kubectl", "-n", args.namespace, "create",
             "--dry-run=client", "-f", str(output)],
            check=True,
        )
        print(f"[verified] {job}: {', '.join(NODE_POOL)}", flush=True)

    if not args.apply:
        print("Preview only: pass --apply to delete and recreate these five Jobs.")
        return

    for job in FAILED_JOBS:
        verify_image_pull_blocked(args.namespace, job, digest)
        subprocess.run(
            ["kubectl", "-n", args.namespace, "delete", "job", job,
             "--cascade=foreground", "--wait=true"],
            check=True,
        )
        subprocess.run(
            ["kubectl", "-n", args.namespace, "create",
             "-f", str(previews / f"{job}.yaml")],
            check=True,
        )
        print(f"[recreated] {job}", flush=True)


if __name__ == "__main__":
    main()
