"""Read-only discovery keeps independent observations, unknowns and result bounds."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexuml.execution import resources
from nexuml.execution.definitions import LocalExecution, RayClusterExecution
from nexuml.execution.schemas import ResourceSnapshot


def test_local_affinity_and_cgroup_limits(monkeypatch):
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.setattr(
        resources, "_cgroup_directories", lambda: [Path("/bounded"), Path("/parent")]
    )
    readings = {
        "/proc/meminfo": "MemTotal: 16384 kB\nMemAvailable: 12288 kB\n",
        "/bounded/cpu.max": "200000 100000",
        "/parent/cpu.max": "150000 100000",
        "/bounded/memory.max": "8388608",
        "/bounded/memory.current": "6291456",
        "/parent/memory.max": "4194304",
        "/parent/memory.current": "3145728",
    }
    monkeypatch.setattr(resources, "_text", lambda path: readings.get(str(path)))
    host, effective, free, diagnostics = resources._local_limits()
    assert host == {"CPU": 16, "memory": 16384 * 1024}
    assert effective == {"CPU": 1.5, "memory": 4194304}
    assert free["memory"] == 1048576
    assert "not a reservation" in " ".join(diagnostics)


def test_gpu_timeout_preserves_known_cpu(monkeypatch):
    monkeypatch.setattr(resources, "_local_limits", lambda: ({"CPU": 8}, {"CPU": 2}, {}, []))

    def timeout(*args, **kwargs):
        assert 0 < kwargs["timeout"] <= 1
        raise subprocess.TimeoutExpired("probe", 1)

    monkeypatch.setattr(resources.subprocess, "run", timeout)
    value = resources.local_resources(timeout=1)
    assert value.allocatable == {"CPU": 2}
    assert "GPU" not in value.allocatable
    assert "unknown" in " ".join(value.diagnostics)


def test_visible_device_units_and_known_zero(monkeypatch):
    monkeypatch.setattr(resources, "_local_limits", lambda: ({}, {"CPU": 2}, {}, []))
    monkeypatch.setattr(
        resources.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({"count": 0, "models": [], "total": {}, "free": {}})
        ),
    )
    value = LocalExecution().discover()
    assert value.allocatable["GPU"] == 0
    assert value.units["GPU"] == "visible devices"
    unknown = ResourceSnapshot(backend="example", source="unknown")
    assert unknown.visible_nodes is None and unknown.unallocated is None
    assert value.model_dump(mode="json")["allocatable"]["GPU"] == 0


def test_budgets_reject_unbounded_and_expire(monkeypatch):
    for timeout, limit in [(0, 1), (61, 1), (10, 0), (10, 1001)]:
        with pytest.raises(ValueError):
            resources.DiscoveryBudget(timeout, limit)
    clock = iter([1, 3])
    monkeypatch.setattr(resources.time, "monotonic", lambda: next(clock))
    budget = resources.DiscoveryBudget(timeout=1)
    with pytest.raises(TimeoutError):
        budget.remaining()


def test_ray_read_only_live_nodes_are_not_kubernetes_nodes(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        list_nodes=lambda **kwargs: (
            calls.append(kwargs)
            or [
                SimpleNamespace(state="ALIVE", node_id="one", resources_total={"CPU": 4, "GPU": 1}),
                SimpleNamespace(
                    state="DEAD", node_id="old", resources_total={"CPU": 100, "GPU": 100}
                ),
                SimpleNamespace(state="ALIVE", node_id="two", resources_total={"CPU": 2}),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "ray.util.state", fake)
    execution = RayClusterExecution(
        target={"address": "ray://explicit:10001", "dashboard_address": "http://explicit:8265"},
        resources_per_worker={"CPU": 4, "GPU": 1},
    )
    value = resources._ray_resources(execution, limit=3)
    assert value.visible_nodes == 2 and value.matching_nodes == 1
    assert value.allocatable == {"CPU": 6, "GPU": 1}
    assert value.unallocated is None  # this supported API does not expose it
    assert calls[0]["address"] == "http://explicit:8265" and calls[0]["detail"]
    assert not value.complete and "partial" in " ".join(value.diagnostics)


def test_ray_without_dashboard_never_initializes_runtime(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "ray://explicit:10001")

    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery must not initialize Ray")

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(init=forbidden))
    value = resources._ray_resources(RayClusterExecution())
    assert value.visible_nodes is None
    assert "ray://explicit:10001" in value.targets
    assert "dashboard_address" in " ".join(value.diagnostics)


def test_ray_probe_timeout_is_bounded_and_credentials_are_not_options(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "http://user:secret@host:8265")
    value = resources._ray_resources(RayClusterExecution())
    assert "secret" not in value.model_dump_json()

    def timeout(*args, **kwargs):
        assert 0 < kwargs["timeout"] <= 2
        raise subprocess.TimeoutExpired("probe", 1)

    monkeypatch.setattr(resources.subprocess, "run", timeout)
    assert "unknown" in " ".join(
        resources.ray_resources(RayClusterExecution(), timeout=1).diagnostics
    )


def test_unobservable_memory_and_cgroup_limits_stay_diagnosed(monkeypatch):
    monkeypatch.setattr(resources, "_text", lambda path: None)
    monkeypatch.setattr(resources, "_cgroup_directories", lambda: [])
    monkeypatch.setitem(sys.modules, "psutil", None)
    _, effective, _, diagnostics = resources._local_limits()
    assert "memory" not in effective
    assert any("memory is unknown" in diagnostic for diagnostic in diagnostics)
    assert any("limits are unknown" in diagnostic for diagnostic in diagnostics)


def test_kuberay_candidates_use_only_selected_scope_and_remain_identities(monkeypatch):
    pytest.importorskip("kubernetes")
    from kubernetes import client
    from nexuml.execution import kubernetes as native

    calls = []
    monkeypatch.setattr(
        client,
        "CustomObjectsApi",
        lambda _: SimpleNamespace(
            list_namespaced_custom_object=lambda *args, **kwargs: (
                calls.append((args, kwargs))
                or {
                    "items": [
                        {"metadata": {"name": "selected-ray", "annotations": {"token": "private"}}}
                    ],
                    "metadata": {"continue": "opaque-page"},
                }
            )
        ),
    )
    result = native._ray_candidates(
        object(), "selected-context", "selected-namespace", resources.DiscoveryBudget(limit=2)
    )
    assert calls[0][0] == ("ray.io", "v1", "selected-namespace", "rayclusters")
    assert calls[0][1]["limit"] == 2
    assert result["ray_clusters"] == [
        {"name": "selected-ray", "context": "selected-context", "namespace": "selected-namespace"}
    ]
    assert "truncated" in " ".join(result["diagnostics"])
    assert "private" not in json.dumps(result)


def test_selected_kubernetes_context_and_forbidden_reads(monkeypatch):
    pytest.importorskip("kubernetes")
    from kubernetes import client, config
    from nexuml.execution import kubernetes as native
    from nexuml.execution.definitions import PyTorchJobExecution

    calls = []
    monkeypatch.setattr(
        config,
        "list_kube_config_contexts",
        lambda: ([{"name": "one"}, {"name": "two"}], {"name": "one"}),
    )
    monkeypatch.setattr(
        config, "new_client_from_config", lambda **kwargs: calls.append(kwargs) or "client"
    )
    assert native._client(PyTorchJobExecution(context="two"), resources.DiscoveryBudget()) == (
        "client",
        "two",
    )
    assert calls == [{"context": "two"}]

    def forbidden(*args, **kwargs):
        raise client.ApiException(status=403, reason="private-header-token")

    core = SimpleNamespace(
        list_node=forbidden,
        list_pod_for_all_namespaces=forbidden,
        list_namespaced_resource_quota=forbidden,
        list_namespace=forbidden,
    )
    monkeypatch.setattr(client, "CoreV1Api", lambda _: core)
    monkeypatch.setattr(
        client,
        "CustomObjectsApi",
        lambda _: SimpleNamespace(list_namespaced_custom_object=forbidden),
    )
    monkeypatch.setattr(native, "_access", lambda *args: True)
    api = SimpleNamespace(
        call_api=lambda *args, **kwargs: {"resources": [{"name": "trainjobs"}]},
        sanitize_for_serialization=lambda value: value,
    )
    snapshot = ResourceSnapshot.model_validate(
        native._discover(
            api, PyTorchJobExecution(context="two"), "two", resources.DiscoveryBudget(), {}
        )
    )
    assert snapshot.api_supported is False and snapshot.submission_allowed is True
    assert snapshot.visible_nodes is None and snapshot.allocatable is None
    assert snapshot.controller_healthy is None
    assert "private-header-token" not in snapshot.model_dump_json()
    assert "PyTorchJob" in " ".join(snapshot.diagnostics)


def test_unselected_multi_role_fit_stays_unknown_but_role_fits_are_reported(monkeypatch):
    pytest.importorskip("kubernetes")
    from kubernetes import client
    from nexuml.execution import kubernetes as native
    from nexuml.execution.definitions import RayJobExecution

    pod = {"containers": [{"resources": {"requests": {"cpu": "1"}}}]}
    manifest = {
        "spec": {
            "rayClusterSpec": {
                "headGroupSpec": {"template": {"spec": pod}},
                "workerGroupSpecs": [
                    {"groupName": "training", "replicas": 2, "template": {"spec": pod}}
                ],
            }
        }
    }
    node = {
        "metadata": {"name": "one"},
        "status": {
            "allocatable": {"cpu": "8", "pods": "10"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }
    monkeypatch.setattr(
        client,
        "CoreV1Api",
        lambda _: SimpleNamespace(
            list_node=lambda **_: {"items": [node]},
            list_pod_for_all_namespaces=lambda **_: {"items": []},
            list_namespaced_resource_quota=lambda *_, **__: {"items": []},
            list_namespace=lambda **_: {"items": []},
        ),
    )
    monkeypatch.setattr(native, "_access", lambda *args: True)
    monkeypatch.setattr(native, "_context_names", lambda: ["selected"])
    monkeypatch.setattr(
        native, "_ray_candidates", lambda *args: {"ray_clusters": [], "diagnostics": []}
    )
    api = SimpleNamespace(
        call_api=lambda *_, **__: {"resources": [{"name": "rayjobs"}]},
        sanitize_for_serialization=lambda value: value,
    )
    result = native._discover(
        api,
        RayJobExecution(context="selected"),
        "selected",
        resources.DiscoveryBudget(),
        {"manifest": manifest},
    )
    assert result["allocatable"]["cpu"] == 8
    assert result["matching_nodes"] is None and result["nodes"][0]["eligible"] is None
    assert [role["matching_nodes"] for role in result["roles"]] == [1, 1]
    assert "overall fit is unknown" in " ".join(result["diagnostics"])
