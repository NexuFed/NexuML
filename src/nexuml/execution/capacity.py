"""Kubernetes request accounting, not a replacement for the native scheduler."""

from typing import Any

from nexuml.execution.schemas import NodeResources


def quantities(values: dict[str, Any]) -> dict[str, float]:
    """Parse native Kubernetes quantities without confusing cores, bytes and devices.

    Returns:
        Named finite nonnegative scheduling quantities.

    Raises:
        ValueError: For negative or nonfinite resource requests.
    """
    import math
    from kubernetes.utils.quantity import parse_quantity

    try:
        parsed = {name: float(parse_quantity(str(value))) for name, value in values.items()}
    except (ValueError, ArithmeticError) as error:
        raise ValueError(
            "Invalid native resource quantity; use cores, bytes or device units."
        ) from error
    if any(not math.isfinite(value) or value < 0 for value in parsed.values()):
        raise ValueError("Resource quantities must be finite and nonnegative.")
    return parsed


def _add(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {name: left.get(name, 0) + right.get(name, 0) for name in left.keys() | right.keys()}


def _maximum(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {name: max(left.get(name, 0), right.get(name, 0)) for name in left.keys() | right.keys()}


def _requests(resources: dict[str, Any]) -> dict[str, float]:
    return quantities({**resources.get("limits", {}), **resources.get("requests", {})})


def pod_requests(spec: dict[str, Any]) -> dict[str, float]:
    """Account for ordinary/init/sidecar containers, pod requests and overhead.

    Returns:
        Effective requests used for native scheduling, plus one pod slot.
    """
    ordinary: dict[str, float] = {}
    for container in spec.get("containers", []):
        ordinary = _add(ordinary, _requests(container.get("resources", {})))
    sidecars: dict[str, float] = {}
    initialization: dict[str, float] = {}
    for container in spec.get("initContainers", []):
        request = _requests(container.get("resources", {}))
        if container.get("restartPolicy") == "Always":
            sidecars = _add(sidecars, request)
            candidate = sidecars
        else:
            candidate = _add(sidecars, request)
        initialization = _maximum(initialization, candidate)
    effective = _maximum(_add(ordinary, sidecars), initialization)
    effective.update(quantities(spec.get("resources", {}).get("requests", {})))
    effective = _add(effective, quantities(spec.get("overhead", {})))
    effective["pods"] = 1
    return effective


def _requirement_matches(requirement: dict, values: dict[str, str]) -> bool:
    key = requirement["key"]
    operator = requirement["operator"]
    choices = requirement.get("values", [])
    value = values.get(key)
    if operator == "In":
        return value in choices
    if operator == "NotIn":
        return value not in choices
    if operator == "Exists":
        return key in values
    if operator == "DoesNotExist":
        return key not in values
    if operator in {"Gt", "Lt"}:
        try:
            return (
                int(value) > int(choices[0]) if operator == "Gt" else int(value) < int(choices[0])
            )
        except (ValueError, TypeError, IndexError):
            return False
    return False


def _tolerates(taint: dict, toleration: dict) -> bool:
    if toleration.get("effect") and toleration["effect"] != taint.get("effect"):
        return False
    if toleration.get("operator", "Equal") == "Exists":
        return not toleration.get("key") or toleration["key"] == taint.get("key")
    return toleration.get("key") == taint.get("key") and toleration.get("value", "") == taint.get(
        "value", ""
    )


def node_matches(node: dict, pod_spec: dict, request: dict[str, float]) -> bool:
    """Check observable node constraints and per-node allocatable fit.

    Returns:
        Whether the node matches this bounded subset of scheduler constraints.
    """
    # ponytail: observable node constraints only; native scheduler owns volumes/gang/inter-pod fit.
    metadata = node.get("metadata", {})
    labels = metadata.get("labels", {})
    node_spec = node.get("spec", {})
    status = node.get("status", {})
    if node_spec.get("unschedulable") or not any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in status.get("conditions", [])
    ):
        return False
    if pod_spec.get("nodeName") and pod_spec["nodeName"] != metadata.get("name"):
        return False
    if any(labels.get(key) != value for key, value in pod_spec.get("nodeSelector", {}).items()):
        return False
    required = (
        pod_spec.get("affinity", {})
        .get("nodeAffinity", {})
        .get("requiredDuringSchedulingIgnoredDuringExecution")
    )
    if required is not None:
        terms = required.get("nodeSelectorTerms", [])
        if not any(
            bool(term.get("matchExpressions") or term.get("matchFields"))
            and all(_requirement_matches(item, labels) for item in term.get("matchExpressions", []))
            and all(
                _requirement_matches(item, {"metadata.name": metadata.get("name", "")})
                for item in term.get("matchFields", [])
            )
            for term in terms
        ):
            return False
    if any(
        taint.get("effect") in {"NoSchedule", "NoExecute"}
        and not any(_tolerates(taint, item) for item in pod_spec.get("tolerations", []))
        for taint in node_spec.get("taints", [])
    ):
        return False
    allocatable = quantities(status.get("allocatable", {}))
    return all(allocatable.get(key, 0) >= amount for key, amount in request.items())


def node_resources(
    nodes: list[dict], pods: list[dict] | None, pod_spec: dict
) -> list[NodeResources]:
    """Keep allocatable and estimated unallocated resources separate.

    Returns:
        One safe resource row per visible node.
    """
    occupied: dict[str, dict[str, float]] = {}
    for pod in pods or []:
        if pod.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
            continue
        spec = pod.get("spec", {})
        name = spec.get("nodeName")
        if name:
            occupied[name] = _add(occupied.get(name, {}), pod_requests(spec))
    request = pod_requests(pod_spec)
    rows = []
    for node in nodes:
        name = node["metadata"]["name"]
        allocatable = quantities(node.get("status", {}).get("allocatable", {}))
        remaining = (
            None
            if pods is None
            else {
                key: max(0, amount - occupied.get(name, {}).get(key, 0))
                for key, amount in allocatable.items()
            }
        )
        labels = node.get("metadata", {}).get("labels", {})
        rows.append(
            NodeResources(
                name=name,
                allocatable=allocatable,
                unallocated=remaining,
                eligible=node_matches(node, pod_spec, request),
                accelerator_type=labels.get("nvidia.com/gpu.product")
                or labels.get("amd.com/gpu.product"),
            )
        )
    return rows
