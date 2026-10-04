import pytest

pytest.importorskip("kubernetes")

from nexuml.execution.capacity import node_matches, node_resources, pod_requests, quantities


def _node(
    *,
    name="node-a",
    labels=None,
    spec=None,
    conditions=None,
    allocatable=None,
):
    return {
        "metadata": {"name": name, "labels": labels or {"pool": "worker"}},
        "spec": spec or {},
        "status": {
            "conditions": conditions or [{"type": "Ready", "status": "True"}],
            "allocatable": allocatable
            or {"cpu": "8", "memory": "16Gi", "nvidia.com/gpu": "2", "pods": "10"},
        },
    }


def test_quantities_keep_cpu_memory_and_accelerator_keys_distinct():
    assert quantities(
        {
            "cpu": "500m",
            "memory": "2Gi",
            "nvidia.com/gpu": "1",
            "nvidia.com/mig-1g.5gb": "2",
            "example.com/shared-gpu": "3",
        }
    ) == {
        "cpu": 0.5,
        "memory": 2 * 1024**3,
        "nvidia.com/gpu": 1,
        "nvidia.com/mig-1g.5gb": 2,
        "example.com/shared-gpu": 3,
    }


def test_pod_requests_include_restartable_init_sidecar_and_overhead():
    assert pod_requests(
        {
            "containers": [{"resources": {"requests": {"cpu": "2"}}}],
            "initContainers": [
                {"restartPolicy": "Always", "resources": {"requests": {"cpu": "1"}}},
                {"resources": {"requests": {"cpu": "5"}}},
            ],
            "overhead": {"cpu": "250m"},
        }
    ) == {"cpu": 6.25, "pods": 1}


def test_later_ordinary_init_container_uses_max_not_sum():
    assert pod_requests(
        {
            "containers": [{"resources": {"requests": {"cpu": "2"}}}],
            "initContainers": [
                {"resources": {"requests": {"cpu": "4"}}},
                {"resources": {"requests": {"cpu": "5"}}},
            ],
        }
    ) == {"cpu": 5, "pods": 1}


def test_pod_level_cpu_request_overrides_container_cpu_only():
    assert pod_requests(
        {
            "containers": [{"resources": {"requests": {"cpu": "2", "nvidia.com/gpu": "1"}}}],
            "resources": {"requests": {"cpu": "4"}},
        }
    ) == {"cpu": 4, "nvidia.com/gpu": 1, "pods": 1}


def test_node_resources_subtract_bound_nonterminal_pods_but_ignore_terminal_pods():
    pods = [
        {
            "spec": {
                "nodeName": "node-a",
                "containers": [{"resources": {"requests": {"cpu": "2", "nvidia.com/gpu": "2"}}}],
            },
            "status": {"phase": "Running"},
        },
        {
            "spec": {
                "nodeName": "node-a",
                "containers": [{"resources": {"requests": {"cpu": "1"}}}],
            },
            "status": {"phase": "Succeeded"},
        },
        {
            "spec": {
                "nodeName": "node-a",
                "containers": [{"resources": {"requests": {"cpu": "1"}}}],
            },
            "status": {"phase": "Failed"},
        },
    ]

    row = node_resources([_node()], pods, {"containers": []})[0]

    assert row.allocatable == {
        "cpu": 8,
        "memory": 16 * 1024**3,
        "nvidia.com/gpu": 2,
        "pods": 10,
    }
    assert row.unallocated == {
        "cpu": 6,
        "memory": 16 * 1024**3,
        "nvidia.com/gpu": 0,
        "pods": 9,
    }
    assert row.eligible is True


def test_node_resources_leave_unallocated_unknown_when_pod_visibility_is_forbidden():
    row = node_resources([_node()], None, {"containers": []})[0]

    assert row.unallocated is None


@pytest.mark.parametrize(
    ("node", "pod_spec", "resources"),
    [
        (_node(spec={"unschedulable": True}), {}, {}),
        (_node(conditions=[{"type": "Ready", "status": "False"}]), {}, {}),
        (_node(), {"nodeSelector": {"pool": "gpu"}}, {}),
        (
            _node(labels={"pool": "worker", "disk": "ssd"}),
            {
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {"key": "disk", "operator": "In", "values": ["hdd"]}
                                    ]
                                }
                            ]
                        }
                    }
                }
            },
            {},
        ),
        (
            _node(
                spec={"taints": [{"key": "dedicated", "value": "training", "effect": "NoSchedule"}]}
            ),
            {},
            {},
        ),
        (_node(), {}, {"nvidia.com/gpu": 3}),
    ],
    ids=[
        "cordoned",
        "not-ready",
        "selector-mismatch",
        "affinity-mismatch",
        "untolerated-taint",
        "capacity",
    ],
)
def test_node_matches_rejects_ineligible_nodes(node, pod_spec, resources):
    assert node_matches(node, pod_spec, resources) is False


def test_node_matches_explicit_labels_affinity_and_toleration():
    node = _node(
        labels={"pool": "worker", "disk": "nvme"},
        spec={"taints": [{"key": "dedicated", "value": "training", "effect": "NoSchedule"}]},
    )
    pod_spec = {
        "nodeSelector": {"pool": "worker"},
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {"key": "disk", "operator": "In", "values": ["nvme"]}
                            ],
                            "matchFields": [
                                {
                                    "key": "metadata.name",
                                    "operator": "In",
                                    "values": ["node-a"],
                                }
                            ],
                        }
                    ]
                }
            }
        },
        "tolerations": [
            {"key": "dedicated", "operator": "Equal", "value": "training", "effect": "NoSchedule"}
        ],
    }

    assert node_matches(node, pod_spec, {"cpu": 2, "nvidia.com/gpu": 1}) is True


@pytest.mark.parametrize(
    ("label", "operator", "bound"),
    [("10", "Gt", "2"), ("9", "Lt", "10")],
)
def test_required_affinity_gt_and_lt_compare_integer_labels(label, operator, bound):
    node = _node(labels={"pool": "worker", "nexuml.io/size": label})
    pod_spec = {
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {"key": "nexuml.io/size", "operator": operator, "values": [bound]}
                            ]
                        }
                    ]
                }
            }
        }
    }

    assert node_matches(node, pod_spec, {}) is True


def test_disjoint_node_gpu_capacity_does_not_combine_for_one_pod():
    nodes = [
        _node(name="gpu-a", labels={"group": "a"}, allocatable={"nvidia.com/gpu": "2"}),
        _node(name="gpu-b", labels={"group": "b"}, allocatable={"nvidia.com/gpu": "2"}),
    ]
    pod_spec = {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "3"}}}]}

    rows = node_resources(nodes, [], pod_spec)

    assert sum(row.allocatable["nvidia.com/gpu"] for row in rows) == 4
    assert [row.eligible for row in rows] == [False, False]
