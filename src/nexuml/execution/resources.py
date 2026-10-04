"""Bounded read-only Local and Ray resource probes; no implicit runtime startup."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from nexuml.execution.schemas import NodeResources, ResourceSnapshot


class DiscoveryBudget:
    """One finite deadline and result bound shared by all probes in a request."""

    def __init__(self, timeout: float = 10, limit: int = 1000):
        if not 0 < timeout <= 60 or not 0 < limit <= 1000:
            raise ValueError("Discovery requires timeout in (0, 60] and limit in [1, 1000].")
        self.deadline = time.monotonic() + timeout
        self.limit = limit

    def remaining(self) -> float:
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Discovery deadline reached.")
        return seconds


def _text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def _cgroup_directories() -> list[Path]:
    membership = _text(Path("/proc/self/cgroup")) or ""
    mounts = _text(Path("/proc/self/mountinfo")) or ""
    group = next((line[3:] for line in membership.splitlines() if line.startswith("0::")), None)
    if group is None:
        return []
    for line in mounts.splitlines():
        if " - cgroup2 " not in line:
            continue
        fields = line.split()
        root, mount = (Path(value.replace("\\040", " ")) for value in fields[3:5])
        try:
            relative = Path(group).relative_to(root)
        except ValueError:
            continue
        current = mount / relative
        if ".." in relative.parts:
            return []
        directories = []
        while current.is_relative_to(mount) and len(directories) < 64:
            directories.append(current)
            if current == mount:
                break
            current = current.parent
        return directories
    return []


def _local_limits() -> tuple[dict[str, float], dict[str, float], dict[str, float], list[str]]:
    host: dict[str, float] = {}
    effective: dict[str, float] = {}
    unallocated: dict[str, float] = {}
    diagnostics = []
    cpus = os.cpu_count()
    if cpus is not None:
        host["CPU"] = float(cpus)
        effective["CPU"] = float(cpus)
    try:
        effective["CPU"] = min(effective.get("CPU", float("inf")), len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        diagnostics.append("CPU affinity is not observable on this platform.")
    memory = _text(Path("/proc/meminfo"))
    if memory is not None:
        values = {line.split(":")[0]: line.split(":")[1].split()[0] for line in memory.splitlines()}
        if "MemTotal" in values:
            host["memory"] = effective["memory"] = float(values["MemTotal"]) * 1024
        if "MemAvailable" in values:
            unallocated["memory"] = float(values["MemAvailable"]) * 1024
    else:
        try:
            import psutil

            memory = psutil.virtual_memory()
            host["memory"] = effective["memory"] = float(memory.total)
            unallocated["memory"] = float(memory.available)
        except (ImportError, OSError):
            diagnostics.append("Host/effective memory is unknown on this platform.")
    directories = _cgroup_directories()
    if not directories:
        diagnostics.append("Container CPU/memory limits are unknown (cgroup v2 not observable).")
    for directory in directories:
        cpu = _text(directory / "cpu.max")
        if cpu and cpu.split()[0] != "max":
            quota, period = (int(part) for part in cpu.split())
            if period > 0:
                effective["CPU"] = min(effective.get("CPU", quota / period), quota / period)
        maximum = _text(directory / "memory.max")
        if maximum and maximum != "max":
            limit = float(maximum)
            effective["memory"] = min(effective.get("memory", limit), limit)
            current = _text(directory / "memory.current")
            if current is not None:
                remaining = max(0, limit - float(current))
                unallocated["memory"] = min(unallocated.get("memory", remaining), remaining)
    diagnostics.append("CPU limits and visible GPUs are not a reservation or an idle-device count.")
    return host, effective, unallocated, diagnostics


def local_resources(timeout: float = 10, limit: int = 1000, **kwargs: Any) -> ResourceSnapshot:
    """Inspect process-effective limits and visible accelerators without training.

    Returns:
        Known limits plus explicit unknown/device-probe diagnostics.
    """
    budget = DiscoveryBudget(timeout, limit)
    host, effective, unallocated, diagnostics = _local_limits()
    # The driver probe is isolated: a wedged GPU driver cannot hang Python discovery.
    script = """
import json, torch
result = {'count': 0, 'models': [], 'total': {}, 'free': {}}
if torch.cuda.is_available():
    result['count'] = torch.cuda.device_count()
    for index in range(result['count']):
        result['models'].append(torch.cuda.get_device_properties(index).name)
        free, total = torch.cuda.mem_get_info(index)
        result['total'][f'GPU/{index}/memory'] = total
        result['free'][f'GPU/{index}/memory'] = free
elif torch.backends.mps.is_available():
    result['count'] = 1
    result['models'] = ['Apple MPS (unified memory)']
print(json.dumps(result))
"""
    models = None
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=budget.remaining(),
            check=True,
        )
        gpu = json.loads(completed.stdout)
        effective["GPU"] = gpu["count"]
        effective.update(gpu["total"])
        unallocated.update(gpu["free"])
        models = "; ".join(gpu["models"]) or None
    except (subprocess.SubprocessError, TimeoutError, ValueError):
        diagnostics.append(
            "Accelerator probe failed or timed out; accelerator capacity is unknown."
        )
    return ResourceSnapshot(
        backend="local",
        source="process limits / visible devices",
        target="local",
        reachable=True,
        submission_allowed=True,
        visible_nodes=1,
        host_total=host,
        allocatable=effective,
        unallocated=unallocated or None,
        nodes=[
            NodeResources(
                name="local",
                allocatable=effective,
                unallocated=unallocated or None,
                eligible=True,
                accelerator_type=models,
            )
        ],
        units={
            **{key: "bytes" for key in effective if key.endswith("memory")},
            "CPU": "cores",
            "GPU": "visible devices",
        },
        diagnostics=diagnostics,
    )


def ray_resources(
    execution: Any, timeout: float = 10, limit: int = 1000, **kwargs: Any
) -> ResourceSnapshot:
    """Isolate Ray inspection so a wedged client cannot outlive the finite budget.

    Returns:
        A sourced snapshot or an explicit unknown diagnostic.
    """
    from nexuml.core.serialization import lower_component

    budget = DiscoveryBudget(timeout, limit)
    with tempfile.TemporaryDirectory(prefix="nexuml-ray-probe-") as temporary:
        response = Path(temporary) / "response.json"
        try:
            subprocess.run(
                [sys.executable, "-m", "nexuml.execution.resources", str(response)],
                input=json.dumps(
                    {"execution": lower_component(execution), "timeout": timeout, "limit": limit}
                ),
                text=True,
                capture_output=True,
                timeout=budget.remaining() + 1,
                check=True,
            )
            snapshot = ResourceSnapshot.model_validate_json(response.read_text())
        except (subprocess.SubprocessError, TimeoutError, OSError, ValueError):
            snapshot = ResourceSnapshot(
                backend="ray-cluster",
                source="Ray Dashboard State API",
                target=execution.target.dashboard_address,
                diagnostics=["Ray inspection failed or timed out; capacity is unknown."],
            )
    if execution.target.context:
        try:
            from nexuml.execution.definitions import KubernetesJobExecution
            from nexuml.execution.kubernetes import _bounded_native

            candidates = _bounded_native(
                "ray_candidates",
                KubernetesJobExecution(
                    context=execution.target.context, namespace=execution.target.namespace
                ),
                timeout=budget.remaining(),
                limit=limit,
            )
            snapshot.ray_clusters = candidates["ray_clusters"]
            snapshot.diagnostics.extend(candidates["diagnostics"])
        except (ImportError, TimeoutError, RuntimeError, ValueError):
            snapshot.diagnostics.append(
                "Selected-scope KubeRay candidates are unavailable; Ray observations retained."
            )
    return snapshot


def _ray_resources(
    execution: Any, timeout: float = 10, limit: int = 1000, **kwargs: Any
) -> ResourceSnapshot:
    """Inspect the selected Ray Dashboard State API without calling ray.init.

    Returns:
        Live node scheduling totals; available capacity stays unknown if not exposed.
    """
    budget = DiscoveryBudget(timeout, limit)
    address = execution.target.dashboard_address
    from nexuml.execution.definitions import RayClusterTarget

    candidates = [execution.target.address]
    configured = os.getenv("RAY_ADDRESS")
    if configured:
        try:
            candidate = RayClusterTarget(address=configured).address
            if candidate not in candidates:
                candidates.append(candidate)
        except ValueError:
            pass  # Never expose a credential-bearing environment URL as a target option.
    snapshot = ResourceSnapshot(
        backend="ray-cluster",
        source="Ray Dashboard State API",
        target=address,
        targets=candidates,
        options={"target.address": candidates},
    )
    if not address:
        snapshot.diagnostics.append(
            "Set target.dashboard_address for read-only cluster inspection. "
            "The execution address alone does not expose Dashboard capacity."
        )
        return snapshot
    if not address.startswith(("https://", "http://")):
        snapshot.diagnostics.append("Dashboard inspection requires an explicit HTTP(S) address.")
        return snapshot
    try:
        from ray.util.state import list_nodes

        rows = list_nodes(address=address, timeout=budget.remaining(), limit=limit, detail=True)
        live = [row for row in rows if row.state == "ALIVE"]
        snapshot.reachable = True
        snapshot.visible_nodes = len(live)
        snapshot.nodes = [
            NodeResources(name=row.node_id, allocatable=row.resources_total) for row in live
        ]
        snapshot.allocatable = {
            name: sum(node.allocatable.get(name, 0) for node in snapshot.nodes)
            for name in {name for node in snapshot.nodes for name in node.allocatable}
        }
        snapshot.matching_nodes = sum(
            all(
                node.allocatable.get(name, 0) >= amount
                for name, amount in execution.resources_per_worker.items()
            )
            for node in snapshot.nodes
        )
        if len(rows) >= limit:
            snapshot.diagnostics.append("Node results reached the limit; counts may be partial.")
        snapshot.diagnostics.append(
            "Ray State API reports total scheduling resources, not currently available "
            "resources. Immediate availability is unknown; "
            "these are Ray nodes, not Kubernetes nodes."
        )
    except Exception as error:  # noqa: BLE001 - independent target probe; never expose client secrets
        snapshot.reachable = False
        snapshot.diagnostics.append(f"Ray inspection failed ({type(error).__name__}).")
    return snapshot


if __name__ == "__main__":
    from nexuml.core.serialization import restore_component

    request = json.loads(sys.stdin.read())
    execution = restore_component(kind="execution_backend", value=request["execution"])
    result = _ray_resources(execution, request["timeout"], request["limit"])
    Path(sys.argv[1]).write_text(result.model_dump_json(), encoding="utf-8")
