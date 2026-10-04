"""One native Kubernetes adapter for template-owned Job, RayJob and PyTorchJob."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from nexuml.core.serialization import lower_component, restore_component
from nexuml.execution.capacity import node_resources, pod_requests, quantities
from nexuml.execution.resources import DiscoveryBudget
from nexuml.execution.schemas import NativeReference, ResourceSnapshot
from nexuml.execution.semantics import ExecutionError
from nexuml.execution.templates import authorized_path, pod_roles, template_plan


def _bounded_native(
    action: str, execution: Any, payload: dict | None = None, timeout: float = 10, limit: int = 1000
) -> dict:
    budget = DiscoveryBudget(timeout, limit)
    with tempfile.TemporaryDirectory(prefix="nexuml-probe-") as temporary:
        response = Path(temporary) / "response.json"
        request = json.dumps(
            {
                "action": action,
                "execution": lower_component(execution),
                "payload": payload or {},
                "timeout": timeout,
                "limit": limit,
            }
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "nexuml.execution.kubernetes", str(response)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=os.name != "nt",
        )
        try:
            # Allow bounded response publication after the child's network deadline.
            process.communicate(request, timeout=budget.remaining() + 1)
        except (subprocess.TimeoutExpired, TimeoutError) as error:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=5,
                )
            process.wait()
            raise TimeoutError(
                "Selected Kubernetes probe timed out, including credential helpers."
            ) from error
        if process.returncode != 0 or not response.is_file():
            raise ExecutionError("Selected Kubernetes probe could not initialize.")
        result = json.loads(response.read_text())
        if "error" in result:
            raise ExecutionError(result["error"])
        return result


def _client(execution: Any, budget: DiscoveryBudget) -> tuple[Any, str]:
    from kubernetes import config

    contexts, current = config.list_kube_config_contexts()
    selected = execution.context or (current or {}).get("name")
    if not selected or selected not in {context["name"] for context in contexts or []}:
        raise ValueError("Select an existing kubeconfig context.")
    budget.remaining()
    api = config.new_client_from_config(context=selected)
    return api, selected


def _context_names() -> list[str]:
    from kubernetes import config

    try:
        contexts, _ = config.list_kube_config_contexts()
        return [context["name"] for context in (contexts or [])][:1000]
    except config.ConfigException:
        return []


def _timeout(budget: DiscoveryBudget) -> tuple[float, float]:
    remaining = budget.remaining()
    return min(3, remaining), remaining


def _access(api: Any, execution: Any, budget: DiscoveryBudget) -> bool | None:
    from kubernetes import client

    group, version = execution.api_version.split("/")
    result = client.AuthorizationV1Api(api).create_self_subject_access_review(
        body={
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SelfSubjectAccessReview",
            "spec": {
                "resourceAttributes": {
                    "group": group,
                    "version": version,
                    "namespace": execution.namespace,
                    "resource": execution.plural,
                    "verb": "create",
                }
            },
        },
        _request_timeout=_timeout(budget),
    )
    return result.status.allowed


def _create(
    api: Any, execution: Any, manifest: dict, budget: DiscoveryBudget, *, dry_run: bool = False
) -> Any:
    from kubernetes import client

    options = {"_request_timeout": _timeout(budget)}
    if dry_run:
        options["dry_run"] = "All"
    if execution.resource_kind == "Job":
        return client.BatchV1Api(api).create_namespaced_job(
            execution.namespace, manifest, **options
        )
    group, version = execution.api_version.split("/")
    return client.CustomObjectsApi(api).create_namespaced_custom_object(
        group,
        version,
        execution.namespace,
        execution.plural,
        manifest,
        **options,
    )


def _read(api: Any, execution: Any, name: str, budget: DiscoveryBudget) -> dict:
    from kubernetes import client

    if execution.resource_kind == "Job":
        result = client.BatchV1Api(api).read_namespaced_job(
            name, execution.namespace, _request_timeout=_timeout(budget)
        )
    else:
        group, version = execution.api_version.split("/")
        result = client.CustomObjectsApi(api).get_namespaced_custom_object(
            group,
            version,
            execution.namespace,
            execution.plural,
            name,
            _request_timeout=_timeout(budget),
        )
    return api.sanitize_for_serialization(result)


def _verify_reference(resource: dict, reference: NativeReference) -> None:
    if resource["metadata"]["uid"] != reference.uid:
        raise ExecutionError("Native resource identity changed; refusing a same-named replacement.")


def _check_quota(api: Any, execution: Any, manifest: dict, budget: DiscoveryBudget) -> list[str]:
    from kubernetes import client

    try:
        result = api.sanitize_for_serialization(
            client.CoreV1Api(api).list_namespaced_resource_quota(
                execution.namespace, limit=budget.limit, _request_timeout=_timeout(budget)
            )
        )
    except client.ApiException:
        return ["Namespace quota visibility is restricted; admission remains authoritative."]
    totals: dict[str, float] = {}
    for _, replicas, pod in pod_roles(manifest, execution):
        for name, amount in pod_requests(pod).items():
            totals[name] = totals.get(name, 0) + replicas * amount
    proposed = {
        "pods": totals.get("pods", 0),
        f"count/{execution.plural}.{execution.api_version.split('/')[0]}": 1,
    }
    for name, amount in totals.items():
        if name != "pods":
            proposed["requests." + name] = amount
            if name in {"cpu", "memory"}:
                proposed[name] = amount
    diagnostics = []
    for quota in result["items"]:
        if quota.get("spec", {}).get("scopes") or quota.get("spec", {}).get("scopeSelector"):
            diagnostics.append(
                "Scoped quota feasibility is unknown; admission remains authoritative."
            )
            continue
        status = quota.get("status", {})
        hard, used = quantities(status.get("hard", {})), quantities(status.get("used", {}))
        for key, amount in proposed.items():
            if key in hard and key not in used:
                diagnostics.append("Quota usage is incomplete; remaining capacity is unknown.")
            elif key in hard and amount > max(0, hard[key] - used[key]):
                raise ExecutionError(f"Namespace quota cannot fit the reviewed {key} request.")
    if result.get("metadata", {}).get("continue"):
        diagnostics.append("Quota enumeration is truncated; admission remains authoritative.")
    return diagnostics


def _logs(api: Any, execution: Any, reference: NativeReference, budget: DiscoveryBudget) -> dict:
    from kubernetes import client

    logs = {}
    try:
        selector = (
            "job-name=" if reference.kind == "Job" else "training.kubeflow.org/job-name="
        ) + reference.name
        core = client.CoreV1Api(api)
        pods = api.sanitize_for_serialization(
            core.list_namespaced_pod(
                reference.namespace,
                label_selector=selector,
                limit=20,
                _request_timeout=_timeout(budget),
            )
        )
        for pod in pods["items"]:
            if not any(
                owner.get("uid") == reference.uid
                for owner in pod.get("metadata", {}).get("ownerReferences", [])
            ):
                continue
            for container in pod.get("spec", {}).get("containers", []):
                name = pod["metadata"]["name"]
                text = core.read_namespaced_pod_log(
                    name,
                    reference.namespace,
                    container=container["name"],
                    tail_lines=100,
                    limit_bytes=8192,
                    timestamps=True,
                    _request_timeout=_timeout(budget),
                )
                logs[name + "/" + container["name"]] = text
                if sum(len(text) for text in logs.values()) >= 32768:
                    return {"logs": logs, "diagnostics": ["Native log tails are bounded/partial."]}
        return {"logs": logs, "diagnostics": ["Native log tails, not a complete log archive."]}
    except Exception as error:  # noqa: BLE001 - preserve verified native state without client details
        return {"logs": logs, "diagnostics": [f"Native logs unavailable ({type(error).__name__})."]}


def _ray_candidates(api: Any, context: str, namespace: str, budget: DiscoveryBudget) -> dict:
    from kubernetes import client

    result = client.CustomObjectsApi(api).list_namespaced_custom_object(
        "ray.io",
        "v1",
        namespace,
        "rayclusters",
        limit=budget.limit,
        _request_timeout=_timeout(budget),
    )
    return {
        "ray_clusters": [
            {"name": item["metadata"]["name"], "context": context, "namespace": namespace}
            for item in result["items"]
        ],
        "diagnostics": ["KubeRay cluster candidates are truncated."]
        if result.get("metadata", {}).get("continue")
        else [],
    }


def native_status(resource: dict) -> str:
    """Interpret actual native workload conditions, never submitter exit.

    Returns:
        Submitted/pending/running or a verified terminal state.
    """
    status = resource.get("status", {})
    conditions = {
        item.get("type")
        for item in status.get("conditions", [])
        if str(item.get("status")).lower() == "true"
    }
    if conditions & {"Complete", "Succeeded"}:
        return "succeeded"
    if "Failed" in conditions:
        return "failed"
    if "Running" in conditions:
        return "running"
    if resource.get("kind") == "RayJob":
        state = str(status.get("jobStatus", "")).upper()
        return {
            "SUCCEEDED": "succeeded",
            "FAILED": "failed",
            "STOPPED": "cancelled",
            "RUNNING": "running",
            "PENDING": "pending",
        }.get(state, "submitted")
    if status.get("active", 0):
        return "running"
    return "pending"


def _discover(
    api: Any, execution: Any, context: str, budget: DiscoveryBudget, payload: dict
) -> dict:
    from kubernetes import client

    snapshot = ResourceSnapshot(
        backend=execution.component_name,
        source="Kubernetes allocatable / effective pod requests",
        target=context,
        context=context,
        namespace=execution.namespace,
        targets=_context_names(),
        units={"cpu": "cores", "memory": "bytes", "pods": "pod slots"},
    )
    try:
        resources = api.call_api(
            "/apis/" + execution.api_version,
            "GET",
            response_type="object",
            auth_settings=["BearerToken"],
            _return_http_data_only=True,
            _request_timeout=_timeout(budget),
        )
        snapshot.reachable = True
        snapshot.api_supported = any(
            item["name"] == execution.plural for item in resources["resources"]
        )
        if not snapshot.api_supported:
            snapshot.diagnostics.append(
                f"Selected cluster does not serve {execution.resource_kind}."
            )
    except Exception as error:  # noqa: BLE001 - retain independent probes without client secrets
        snapshot.diagnostics.append(
            f"Resource API inspection unavailable ({type(error).__name__})."
        )
    try:
        snapshot.submission_allowed = _access(api, execution, budget)
        if snapshot.submission_allowed is False:
            snapshot.diagnostics.append("Selected context/namespace denies workload creation.")
    except Exception as error:  # noqa: BLE001
        snapshot.diagnostics.append(f"Create permission is unknown ({type(error).__name__}).")
    core = client.CoreV1Api(api)
    nodes, pods = None, None
    try:
        result = api.sanitize_for_serialization(
            core.list_node(limit=budget.limit, _request_timeout=_timeout(budget))
        )
        nodes = result["items"]
        snapshot.visible_nodes = len(nodes)
        if result.get("metadata", {}).get("continue"):
            snapshot.diagnostics.append("Node enumeration is truncated; counts are partial.")
    except Exception as error:  # noqa: BLE001
        snapshot.diagnostics.append(f"Node capacity is unknown ({type(error).__name__}).")
    try:
        result = api.sanitize_for_serialization(
            core.list_pod_for_all_namespaces(limit=budget.limit, _request_timeout=_timeout(budget))
        )
        if result.get("metadata", {}).get("continue"):
            snapshot.diagnostics.append(
                "Pod accounting is truncated; unallocated capacity is unknown."
            )
        else:
            pods = result["items"]
    except Exception as error:  # noqa: BLE001
        snapshot.diagnostics.append(f"Unallocated capacity is unknown ({type(error).__name__}).")
    manifest = payload.get("manifest")
    roles = pod_roles(manifest, execution) if manifest else []
    selected_role = execution.worker_group or (roles[0][0] if len(roles) == 1 else None)
    pod_spec = next((pod for name, _, pod in roles if name == selected_role), {"containers": []})
    if nodes is not None:
        snapshot.nodes = node_resources(nodes, pods, pod_spec)
        if selected_role is None:
            snapshot.diagnostics.append(
                "No single workload role selected; per-role matching is shown, "
                "overall fit is unknown."
            )
            for node in snapshot.nodes:
                node.eligible = None
        names = {name for node in snapshot.nodes for name in node.allocatable}
        snapshot.allocatable = {
            name: sum(node.allocatable.get(name, 0) for node in snapshot.nodes) for name in names
        }
        if pods is not None:
            snapshot.unallocated = {
                name: sum(node.unallocated.get(name, 0) for node in snapshot.nodes)
                for name in names
            }
        snapshot.matching_nodes = (
            sum(bool(node.eligible) for node in snapshot.nodes) if selected_role else None
        )
        for name, replicas, pod in roles:
            matches = node_resources(nodes, None, pod)
            snapshot.roles.append(
                {
                    "name": name,
                    "replicas": replicas,
                    "requests": pod_requests(pod),
                    "matching_nodes": sum(bool(row.eligible) for row in matches),
                }
            )
            if replicas and not any(row.eligible for row in matches):
                snapshot.diagnostics.append(
                    f"No visible node fits the {name} role's observable constraints."
                )
    try:
        result = api.sanitize_for_serialization(
            core.list_namespaced_resource_quota(
                execution.namespace, limit=budget.limit, _request_timeout=_timeout(budget)
            )
        )
        remaining = {}
        for quota in result["items"]:
            if quota.get("spec", {}).get("scopes") or quota.get("spec", {}).get("scopeSelector"):
                snapshot.diagnostics.append("Scoped quota feasibility is unknown.")
                continue
            status = quota.get("status", {})
            hard, used = quantities(status.get("hard", {})), quantities(status.get("used", {}))
            for key, amount in hard.items():
                if key not in used:
                    snapshot.diagnostics.append(
                        "Quota usage is incomplete; remaining capacity is unknown."
                    )
                    continue
                remaining[key] = min(remaining.get(key, amount), max(0, amount - used[key]))
        snapshot.quota_remaining = remaining
        if result.get("metadata", {}).get("continue"):
            snapshot.diagnostics.append("Quota results are truncated; limits may be incomplete.")
    except Exception as error:  # noqa: BLE001
        snapshot.diagnostics.append(f"Namespace quota is unknown ({type(error).__name__}).")
    try:
        result = api.sanitize_for_serialization(
            core.list_namespace(limit=budget.limit, _request_timeout=_timeout(budget))
        )
        snapshot.namespaces = [item["metadata"]["name"] for item in result["items"]]
        if result.get("metadata", {}).get("continue"):
            snapshot.diagnostics.append("Namespace options are truncated.")
    except Exception:  # noqa: BLE001 - namespace enumeration isn't required for explicit selection
        snapshot.diagnostics.append(
            "Namespace options are restricted; enter an authorized namespace."
        )
    try:
        candidates = _ray_candidates(api, context, execution.namespace, budget)
        snapshot.ray_clusters = candidates["ray_clusters"]
        snapshot.diagnostics.extend(candidates["diagnostics"])
    except Exception:  # noqa: BLE001
        snapshot.diagnostics.append("KubeRay cluster candidates are unavailable in this scope.")
    snapshot.diagnostics.append(
        "Advisory observable constraints only; volumes, gang/inter-pod scheduling, controller "
        "health, device allocation and autoscaler capacity are not verified."
    )
    snapshot.options = {"context": snapshot.targets, "namespace": snapshot.namespaces}
    return snapshot.model_dump(mode="json")


def discover_resources(
    execution: Any,
    timeout: float = 10,
    limit: int = 1000,
    authorized_root: Path | None = None,
    **kwargs: Any,
) -> ResourceSnapshot:
    """Bound context authentication and selected-scope read-only capacity queries.

    Returns:
        Independent successes with unknown/forbidden/timeout diagnostics.
    """
    manifest = None
    contexts = _context_names()
    # Reject unauthorized inputs before any authentication helper or remote probe.
    if execution.template:
        authorized_path(execution.template, authorized_root)
    if execution.handoff and not execution.handoff.destination.startswith("s3://"):
        authorized_path(execution.handoff.destination, authorized_root)
    if execution.template:
        from nexuml.core.types import ScenarioSpec

        try:
            manifest = template_plan(
                kwargs.get("scenario") or ScenarioSpec(name="discovery", execution=execution),
                authorized_root=authorized_root,
            )["manifest"]
        except (ValueError, OSError) as error:
            return ResourceSnapshot(
                backend=execution.component_name, source="template review", diagnostics=[str(error)]
            )
    try:
        result = _bounded_native("discover", execution, {"manifest": manifest}, timeout, limit)
        return ResourceSnapshot.model_validate(result)
    except (TimeoutError, ExecutionError) as error:
        return ResourceSnapshot(
            backend=execution.component_name,
            source="Kubernetes",
            target=execution.context,
            targets=contexts,
            diagnostics=[str(error)],
        )


def prepare_job(
    scenario: Any,
    authorized_root: Path | None = None,
    frozen_plan: dict | None = None,
    **kwargs: Any,
) -> dict:
    """Review exact inputs, native create access and admission without submitting.

    Returns:
        A private frozen plan with safe review summary and native dry-run evidence.

    Raises:
        ExecutionError: On stale template, denied create access or admission rejection.
    """
    plan = frozen_plan or template_plan(scenario, authorized_root=authorized_root)
    from nexuml.core.config import ResolvedConfig

    execution = scenario.execution
    if plan.get("resolved_execution") and execution.context is None:
        resolved = restore_component(kind="execution_backend", value=plan["resolved_execution"])
        execution = execution.model_copy(update={"context": resolved.context})
    selected = scenario.model_copy(update={"execution": execution})
    source_revision = hashlib.sha256(
        ResolvedConfig.from_scenario(selected).to_yaml().encode()
    ).hexdigest()
    if plan["source_revision"] != source_revision:
        raise ExecutionError("Configuration changed after review; renew the launch review.")
    revision = hashlib.sha256(
        authorized_path(plan["template_path"], authorized_root).read_bytes()
    ).hexdigest()
    if revision != plan["summary"]["template_revision"]:
        raise ExecutionError("Template changed after review; renew the launch review.")
    result = _bounded_native("preflight", execution, {"manifest": plan["manifest"]}, timeout=30)
    execution = execution.model_copy(update={"context": result["context"]})
    plan["resolved_execution"] = lower_component(execution)
    plan["source_revision"] = hashlib.sha256(
        ResolvedConfig.from_scenario(scenario.model_copy(update={"execution": execution}))
        .to_yaml()
        .encode()
    ).hexdigest()
    result["diagnostics"] = plan["summary"].get("diagnostics", []) + result.get("diagnostics", [])
    plan["summary"].update(result)
    return plan


def inspect(reference: NativeReference) -> dict:
    """Inspect only the recorded native identity through its trusted context.

    Returns:
        Actual workload state and safe reference, without fabricated telemetry.
    """
    execution = restore_component(
        kind="execution_backend",
        value={
            "type": reference.backend,
            "version": reference.version,
            "params": {"context": reference.context, "namespace": reference.namespace},
        },
    )
    return _bounded_native("inspect", execution, {"reference": reference.model_dump()}, timeout=30)


def cancel(reference: NativeReference) -> dict:
    """Delete only a UID-verified owned workload and confirm foreground termination.

    Returns:
        A confirmed cancellation or an explicit unknown/error state.

    Raises:
        ExecutionError: If cancellation cannot stop the native workload safely.
    """
    execution = restore_component(
        kind="execution_backend",
        value={
            "type": reference.backend,
            "version": reference.version,
            "params": {"context": reference.context, "namespace": reference.namespace},
        },
    )
    if not execution.capabilities.get("cancellation"):
        raise ExecutionError(
            "This backend cannot confirm native cancellation; use native operator tooling."
        )
    return _bounded_native("cancel", execution, {"reference": reference.model_dump()}, timeout=60)


def run_job(
    scenario: Any,
    observer: Any = None,
    frozen_plan: dict | None = None,
    authorized_root: Path | None = None,
    **kwargs: Any,
) -> dict:
    """Submit once, publish the reference promptly, then observe the native workload.

    Returns:
        Native reference and actual terminal state.

    """
    plan = frozen_plan or prepare_job(scenario, authorized_root=authorized_root)
    if plan["local_path"] is None:
        from nexuml.storage.s3 import S3Client

        S3Client().upload_bytes(plan["worker_yaml"].encode(), plan["worker_path"], exclusive=True)
    else:
        path = Path(plan["local_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(plan["worker_yaml"])
    execution = scenario.execution.model_copy(
        update={"context": plan["summary"].get("context") or scenario.execution.context}
    )
    submitted = _bounded_native("submit", execution, {"manifest": plan["manifest"]}, timeout=30)
    reference = NativeReference.model_validate(submitted["native_reference"])
    if observer is not None:
        observer.record("native_reference", reference=reference.model_dump())
        observer.record("native_status", status="submitted")
    previous, previous_logs = None, {}
    while True:
        result = inspect(reference)
        state = result["status"]
        if observer is not None:
            if state != previous:
                observer.record("native_status", status=state)
            for name, text in result.get("logs", {}).items():
                if text != previous_logs.get(name):
                    observer.record("log", text=f"[{name}: native log tail]\n{text}")
            previous_logs = result.get("logs", {})
        previous = state
        if state in {"succeeded", "failed", "cancelled"}:
            return {
                "native_reference": reference.model_dump(),
                "status": state,
                "metrics": {},
                "artifacts": [],
                "telemetry": "native status/log tails; scalar metrics unavailable",
            }
        time.sleep(2)


def _native(action: str, execution: Any, payload: dict, budget: DiscoveryBudget) -> dict:
    from kubernetes import client

    api, context = _client(execution, budget)
    try:
        if action == "ray_candidates":
            return _ray_candidates(api, context, execution.namespace, budget)
        if action == "discover":
            return _discover(api, execution, context, budget, payload)
        if action == "preflight":
            try:
                allowed = _access(api, execution, budget)
            except client.ApiException:
                allowed = None
            if allowed is False:
                raise ExecutionError("Selected context/namespace denies workload creation.")
            diagnostics = _check_quota(api, execution, payload["manifest"], budget)
            _create(api, execution, payload["manifest"], budget, dry_run=True)
            return {"context": context, "dry_run": True, "diagnostics": diagnostics}
        if action == "submit":
            manifest = payload["manifest"]
            try:
                resource = api.sanitize_for_serialization(_create(api, execution, manifest, budget))
            except Exception as error:  # noqa: BLE001 - inspect before any retry
                if isinstance(error, client.ApiException) and error.status in {
                    400,
                    401,
                    403,
                    404,
                    422,
                }:
                    raise ExecutionError(
                        f"Native create rejected (HTTP {error.status})."
                    ) from error
                try:
                    resource = _read(api, execution, manifest["metadata"]["name"], budget)
                except Exception:  # noqa: BLE001
                    raise ExecutionError(
                        "Submission acknowledgement lost; ownership is unknown. Do not resubmit."
                    ) from error
            expected = manifest["metadata"]["labels"]["nexuml.org/invocation"]
            if (
                resource.get("metadata", {}).get("labels", {}).get("nexuml.org/invocation")
                != expected
            ):
                raise ExecutionError(
                    "Submission correlation does not match; refusing an unrelated resource."
                )
            reference = NativeReference(
                backend=execution.component_name,
                version=execution.component_version,
                context=context,
                namespace=execution.namespace,
                api_version=execution.api_version,
                kind=execution.resource_kind,
                name=resource["metadata"]["name"],
                uid=resource["metadata"]["uid"],
            )
            return {"native_reference": reference.model_dump(), "status": "submitted"}
        reference = NativeReference.model_validate(payload["reference"])
        if (
            reference.kind != execution.resource_kind
            or reference.api_version != execution.api_version
        ):
            raise ExecutionError("Native reference does not match its backend resource API.")
        resource = _read(api, execution, reference.name, budget)
        _verify_reference(resource, reference)
        if action == "inspect":
            result = {"native_reference": reference.model_dump(), "status": native_status(resource)}
            if execution.capabilities.get("logs"):
                result.update(_logs(api, execution, reference, budget))
            return result
        if action != "cancel" or not execution.capabilities.get("cancellation"):
            raise ExecutionError("Unsupported native control.")
        state = native_status(resource)
        if state in {"succeeded", "failed"}:
            return {"native_reference": reference.model_dump(), "status": state}
        options = {"preconditions": {"uid": reference.uid}, "propagationPolicy": "Foreground"}
        if execution.resource_kind == "Job":
            client.BatchV1Api(api).delete_namespaced_job(
                reference.name, execution.namespace, body=options, _request_timeout=_timeout(budget)
            )
        else:
            group, version = execution.api_version.split("/")
            client.CustomObjectsApi(api).delete_namespaced_custom_object(
                group,
                version,
                execution.namespace,
                execution.plural,
                reference.name,
                body=options,
                _request_timeout=_timeout(budget),
            )
        while True:
            try:
                resource = _read(api, execution, reference.name, budget)
                _verify_reference(resource, reference)
            except client.ApiException as error:
                if error.status == 404:
                    return {"native_reference": reference.model_dump(), "status": "cancelled"}
                raise
            time.sleep(min(0.5, budget.remaining()))
    finally:
        api.close()


def main() -> None:
    """Run a private bounded probe, never a workload submission endpoint."""
    request = json.loads(sys.stdin.read())
    try:
        execution = restore_component(kind="execution_backend", value=request["execution"])
        result = _native(
            request["action"],
            execution,
            request["payload"],
            DiscoveryBudget(request["timeout"], request["limit"]),
        )
    except Exception as error:  # noqa: BLE001 - do not serialize API headers, bodies or credentials
        message = (
            str(error)
            if isinstance(error, ExecutionError)
            else (
                f"Kubernetes probe failed ({type(error).__name__}, "
                f"status {getattr(error, 'status', 'unknown')}). Inspect native permissions/setup."
            )
        )
        result = {"error": message}
    Path(sys.argv[1]).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
