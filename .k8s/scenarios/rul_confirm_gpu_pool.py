"""Discover the global GPU node pool exclusions for FD003 confirmation Jobs.

Never guess GH200 Kubernetes hostnames. Inspect node labels just before
rendering a Job so hostname node affinity does not silently include GH200s.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

_GH200 = re.compile(r"gh[\W_]*200|grace[\W_]*hopper", re.IGNORECASE)
_STUDENT_EXCLUSION = "student-worker-1"
_GPU_LABELS = (
    "nvidia.com/gpu.product",
    "nvidia.com/gpu.family",
    "nvidia.com/gpu.machine",
)


def excluded_hostnames(node_list: dict[str, Any]) -> tuple[str, ...]:
    """Find hostname labels for GH200 nodes and student-worker-1.

    Args:
        node_list: Kubernetes NodeList JSON, including metadata and labels.

    Returns:
        Sorted values for a required hostname NotIn node-affinity expression.

    Raises:
        ValueError: If GH200 nodes cannot be identified or hostnames are missing.
    """
    excluded = {_STUDENT_EXCLUSION}
    gh200_found = False
    for node in node_list.get("items", []):
        metadata = node.get("metadata", {})
        name = str(metadata.get("name", ""))
        labels = metadata.get("labels") or {}
        gpu_info = " ".join(str(labels.get(key, "")) for key in _GPU_LABELS)
        if _GH200.search(name + " " + gpu_info):
            hostname = labels.get("kubernetes.io/hostname")
            if not isinstance(hostname, str) or not hostname:
                raise ValueError(f"GH200 Node lacks hostname label: {name}")
            excluded.add(hostname)
            gh200_found = True
        if name == _STUDENT_EXCLUSION:
            hostname = labels.get("kubernetes.io/hostname")
            if isinstance(hostname, str) and hostname:
                excluded.add(hostname)
    if not gh200_found:
        raise ValueError(
            "No GH200 Node identifiable by name or NVIDIA GPU labels. "
            "Inspect kubectl get nodes -L nvidia.com/gpu.product before "
            "scheduling, so GH200 hardware is never included accidentally."
        )
    return tuple(sorted(excluded))


def discover_excluded_hostnames() -> tuple[str, ...]:
    """Read cluster nodes and derive required excluded hostnames.

    Returns:
        Verified exclusion names for the live cluster.

    Raises:
        ValueError: If GH200 nodes cannot be identified safely.
    """
    result = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "json"],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Expected Kubernetes NodeList JSON object")
    return excluded_hostnames(payload)
