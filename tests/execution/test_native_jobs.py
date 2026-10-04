"""Frozen native jobs and ownership are tested without creating cluster resources."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("kubernetes")

from kubernetes import client

from nexuml.core.config import ResolvedConfig
from nexuml.core.types import ScenarioSpec
from nexuml.execution import kubernetes as native
from nexuml.execution.definitions import (
    ConfigHandoff,
    KubernetesJobExecution,
    PyTorchJobExecution,
    RayJobExecution,
)
from nexuml.execution.resources import DiscoveryBudget
from nexuml.execution.schemas import NativeReference
from nexuml.execution.semantics import ExecutionError
from nexuml.execution.templates import CONFIG_TOKEN, template_plan
from nexuml.execution.worker import launch_options


def pod(container="training"):
    return {
        "containers": [
            {
                "name": container,
                "image": "infra-owned:test",
                "command": ["python", "-m", "nexuml.execution.worker"],
                "args": ["--config", CONFIG_TOKEN],
                "resources": {"requests": {"cpu": "1", "memory": "1Gi"}},
                "volumeMounts": [{"name": "config", "mountPath": "/shared"}],
            }
        ],
        "restartPolicy": "Never",
        "serviceAccountName": "training-owned",
        "volumes": [{"name": "config", "persistentVolumeClaim": {"claimName": "infra-owned"}}],
    }


def manifest(mode="Job"):
    if mode == "Job":
        return {
            "apiVersion": "batch/v1",
            "kind": mode,
            "metadata": {"name": "example"},
            "spec": {"template": {"spec": pod()}},
        }
    if mode == "RayJob":
        return {
            "apiVersion": "ray.io/v1",
            "kind": mode,
            "metadata": {},
            "spec": {
                "entrypoint": "python -m nexuml.execution.worker --config " + CONFIG_TOKEN,
                "shutdownAfterJobFinishes": True,
                "rayClusterSpec": {
                    "headGroupSpec": {"template": {"spec": pod("ray-head")}},
                    "workerGroupSpecs": [
                        {
                            "groupName": "workers",
                            "replicas": 2,
                            "template": {"spec": pod("ray-worker")},
                        }
                    ],
                },
                "submitterPodTemplate": {"spec": pod("submitter")},
            },
        }
    master = pod("pytorch")
    master["containers"][0]["command"] = ["torchrun"]
    master["containers"][0]["args"] = [
        "--nnodes=$(WORLD_SIZE)",
        "--node-rank=$(RANK)",
        "--master-addr=$(MASTER_ADDR)",
        "--master-port=$(MASTER_PORT)",
        "--nproc-per-node=1",
        "-m",
        "nexuml.execution.worker",
        "--config",
        CONFIG_TOKEN,
    ]
    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": mode,
        "metadata": {},
        "spec": {
            "pytorchReplicaSpecs": {
                name: {"replicas": 1, "template": {"spec": copy.deepcopy(master)}}
                for name in ("Master", "Worker")
            }
        },
    }


def scenario(tmp_path, mode="Job", value=None, **params):
    path = tmp_path / "job.yaml"
    path.write_text(json.dumps(value or manifest(mode)))
    definition = {
        "Job": KubernetesJobExecution,
        "RayJob": RayJobExecution,
        "PyTorchJob": PyTorchJobExecution,
    }[mode]
    return ScenarioSpec(
        name="native-test",
        execution=definition(
            context="selected",
            template=str(path),
            handoff=ConfigHandoff(
                destination=str(tmp_path / "configs"), worker_directory="/shared"
            ),
            **params,
        ),
    )


@pytest.mark.parametrize(
    "mode,worker", [("Job", "local"), ("RayJob", "ray-cluster"), ("PyTorchJob", "local")]
)
def test_exact_template_and_worker_derivation(tmp_path, mode, worker):
    value = scenario(tmp_path, mode)
    before = copy.deepcopy(value)
    plan = template_plan(value, authorized_root=tmp_path, invocation_id="example123")
    restored = ResolvedConfig.from_yaml(plan["worker_yaml"])
    assert restored.execution.component_name == worker
    assert restored.training == value.training and restored.data == value.data
    assert value == before
    assert plan["manifest"]["metadata"]["labels"]["nexuml.org/invocation"] == "example123"
    assert plan["summary"]["worker_config"] == "/shared/nexuml-example123.yaml"
    assert not (tmp_path / "configs").exists()  # review never publishes configs
    if mode == "RayJob":
        assert [role["name"] for role in plan["summary"]["roles"]] == [
            "head",
            "workers",
            "submitter",
        ]
        assert plan["manifest"]["spec"]["shutdownAfterJobFinishes"] is True
    else:
        selected = (
            plan["manifest"]["spec"]["template"]["spec"]
            if mode == "Job"
            else plan["manifest"]["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]
        )
        assert selected["serviceAccountName"] == "training-owned"
        assert selected["volumes"][0]["persistentVolumeClaim"]["claimName"] == "infra-owned"


def test_named_binding_preserves_unrelated_policy(tmp_path):
    value = scenario(
        tmp_path,
        "RayJob",
        worker_group="workers",
        container="ray-worker",
        replicas=3,
        resources={"cpu": "2", "nvidia.com/gpu": "1"},
    )
    plan = template_plan(value)
    cluster = plan["manifest"]["spec"]["rayClusterSpec"]
    assert cluster["workerGroupSpecs"][0]["replicas"] == 3
    assert (
        cluster["headGroupSpec"]["template"]["spec"]["containers"][0]["resources"]["requests"][
            "cpu"
        ]
        == "1"
    )
    requests = cluster["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]["resources"][
        "requests"
    ]
    assert requests == {"cpu": "2", "memory": "1Gi", "nvidia.com/gpu": "1"}


@pytest.mark.parametrize("parameter", ["parallelism", "completions"])
def test_parallel_jobs_are_rejected_before_native_calls(tmp_path, parameter):
    template = manifest()
    template["spec"][parameter] = 2
    with pytest.raises(ValueError, match="one non-indexed"):
        template_plan(scenario(tmp_path, value=template))


def test_ambiguous_binding_and_master_replica_override_fail(tmp_path):
    with pytest.raises(ValueError, match="explicit worker_group"):
        template_plan(scenario(tmp_path, "PyTorchJob", resources={"cpu": "2"}))
    with pytest.raises(ValueError, match="exactly one Master"):
        template_plan(scenario(tmp_path, "PyTorchJob", worker_group="Master", replicas=2))


def test_authorized_paths_and_secret_references(tmp_path):
    root = tmp_path / "authorized"
    root.mkdir()
    value = scenario(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        template_plan(value, authorized_root=root)
    link = root / "escape.yaml"
    link.symlink_to(tmp_path / "job.yaml")
    value.execution = value.execution.model_copy(update={"template": str(link)})
    with pytest.raises(ValueError, match="outside"):
        template_plan(value, authorized_root=root)
    template = manifest()
    template["spec"]["template"]["spec"]["containers"][0]["env"] = [
        {"name": "API_TOKEN", "value": "private"}
    ]
    with pytest.raises(ValueError, match="reference credential secrets"):
        template_plan(scenario(tmp_path, value=template))


def test_existing_ray_cluster_policy_is_untouched(tmp_path):
    template = manifest("RayJob")
    template["spec"].pop("rayClusterSpec")
    template["spec"].update(
        clusterSelector={"ray.io/cluster": "shared"}, shutdownAfterJobFinishes=False
    )
    plan = template_plan(scenario(tmp_path, "RayJob", template))
    assert plan["manifest"]["spec"]["clusterSelector"] == {"ray.io/cluster": "shared"}
    assert plan["manifest"]["spec"]["shutdownAfterJobFinishes"] is False


def fake_api(monkeypatch):
    api = SimpleNamespace(sanitize_for_serialization=lambda value: value, close=lambda: None)
    monkeypatch.setattr(native, "_client", lambda execution, budget: (api, "selected"))
    return api


def test_preflight_is_dry_run_and_template_revision_is_checked(tmp_path, monkeypatch):
    fake_api(monkeypatch)
    value = scenario(tmp_path)
    monkeypatch.setattr(native, "_access", lambda *args: True)
    monkeypatch.setattr(native, "_check_quota", lambda *args: [])
    calls = []
    monkeypatch.setattr(native, "_create", lambda *args, **kwargs: calls.append(kwargs))
    plan = template_plan(value)
    result = native._native(
        "preflight", value.execution, {"manifest": plan["manifest"]}, DiscoveryBudget()
    )
    assert calls == [{"dry_run": True}]
    assert result["dry_run"]
    (tmp_path / "job.yaml").write_text("changed after review")
    with pytest.raises(ExecutionError, match="Template changed"):
        native.prepare_job(value, frozen_plan=plan)


@pytest.mark.parametrize("ack_lost", [False, True])
def test_submit_once_and_inspect_ack_loss(tmp_path, monkeypatch, ack_lost):
    fake_api(monkeypatch)
    value = scenario(tmp_path)
    plan = template_plan(value)
    resource = {
        **plan["manifest"],
        "metadata": {**plan["manifest"]["metadata"], "uid": "original-uid"},
    }
    calls = []

    def create(*args, **kwargs):
        calls.append("create")
        if ack_lost:
            raise TimeoutError("private-header-secret")
        return resource

    monkeypatch.setattr(native, "_create", create)
    monkeypatch.setattr(native, "_read", lambda *args: calls.append("read") or resource)
    result = native._native(
        "submit", value.execution, {"manifest": plan["manifest"]}, DiscoveryBudget()
    )
    assert calls == (["create", "read"] if ack_lost else ["create"])
    assert result["native_reference"]["uid"] == "original-uid"
    assert result["status"] == "submitted"


def test_native_status_and_uid_replacement():
    reference = NativeReference(
        backend="kubernetes-job",
        context="selected",
        namespace="default",
        api_version="batch/v1",
        kind="Job",
        name="named",
        uid="original",
    )
    with pytest.raises(ExecutionError, match="same-named replacement"):
        native._verify_reference({"metadata": {"uid": "replacement"}}, reference)
    assert native.native_status({"kind": "Job", "status": {}}) == "pending"
    assert native.native_status({"kind": "Job", "status": {"active": 1}}) == "running"
    for mode, condition in [("Job", "Complete"), ("PyTorchJob", "Succeeded")]:
        assert (
            native.native_status(
                {
                    "kind": mode,
                    "status": {
                        "conditions": [
                            {"type": "Running", "status": "True"},
                            {"type": condition, "status": "True"},
                        ]
                    },
                }
            )
            == "succeeded"
        )
    assert (
        native.native_status(
            {
                "kind": "RayJob",
                "status": {"jobStatus": "RUNNING", "jobDeploymentStatus": "Complete"},
            }
        )
        == "running"
    )


def test_external_launcher_mapping_and_guards():
    value = ScenarioSpec(name="torchrun")
    env = {"LOCAL_WORLD_SIZE": "2", "WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "0"}
    assert launch_options(value, env) == {"devices": 2, "num_nodes": 2, "enable_loggers": False}
    value.training.devices = 1
    with pytest.raises(ValueError, match="differs"):
        launch_options(value, env)
    with pytest.raises(ValueError, match="torchrun"):
        launch_options(value, {"WORLD_SIZE": "2"})
    with pytest.raises(ValueError, match="Inconsistent"):
        launch_options(value, {**env, "WORLD_SIZE": "3"})


def test_known_quota_violation_blocks_crd_preflight(tmp_path, monkeypatch):
    fake_api(monkeypatch)
    value = scenario(tmp_path, "RayJob")
    quota = {"items": [{"status": {"hard": {"requests.cpu": "2"}, "used": {"requests.cpu": "1"}}}]}
    monkeypatch.setattr(
        client,
        "CoreV1Api",
        lambda api: SimpleNamespace(list_namespaced_resource_quota=lambda *args, **kwargs: quota),
    )
    with pytest.raises(ExecutionError, match="requests.cpu"):
        native._check_quota(
            SimpleNamespace(sanitize_for_serialization=lambda value: value),
            value.execution,
            template_plan(value)["manifest"],
            DiscoveryBudget(),
        )


def test_handoff_is_unique_and_observation_waits_for_native_terminal(tmp_path, monkeypatch):
    value = scenario(tmp_path)
    plan = template_plan(value)
    plan["summary"]["context"] = "frozen-selected"
    calls, events = [], []
    reference = NativeReference(
        backend="kubernetes-job",
        context="frozen-selected",
        namespace="default",
        api_version="batch/v1",
        kind="Job",
        name="native",
        uid="owned",
    )

    def submit(action, execution, payload, **kwargs):
        assert action == "submit" and execution.context == "frozen-selected"
        calls.append(payload["manifest"])
        return {"native_reference": reference.model_dump()}

    states = iter(["pending", "running", "failed"])
    monkeypatch.setattr(native, "_bounded_native", submit)
    monkeypatch.setattr(
        native,
        "inspect",
        lambda ref: {"status": next(states), "logs": {"owned/train": "native output"}},
    )
    monkeypatch.setattr(native.time, "sleep", lambda _: None)
    observer = SimpleNamespace(record=lambda kind, **payload: events.append((kind, payload)))
    result = native.run_job(value, frozen_plan=plan, observer=observer)
    assert result["status"] == "failed" and len(calls) == 1
    assert [payload["status"] for kind, payload in events if kind == "native_status"] == [
        "submitted",
        "pending",
        "running",
        "failed",
    ]
    assert len([kind for kind, _ in events if kind == "log"]) == 1
    assert Path(plan["local_path"]).read_text() == plan["worker_yaml"]
    with pytest.raises(FileExistsError):
        native.run_job(value, frozen_plan=plan)
    assert len(calls) == 1  # no overwriting or duplicate native create


def test_uid_checked_cancel_waits_for_foreground_deletion(tmp_path, monkeypatch):
    fake_api(monkeypatch)
    value = scenario(tmp_path)
    reference = NativeReference(
        backend="kubernetes-job",
        context="selected",
        namespace="default",
        api_version="batch/v1",
        kind="Job",
        name="native",
        uid="owned",
    )
    reads, deletes = [], []

    def read(*args):
        reads.append("read")
        if len(reads) == 3:
            raise client.ApiException(status=404)
        return {"metadata": {"uid": "owned"}}

    monkeypatch.setattr(native, "_read", read)
    monkeypatch.setattr(
        client,
        "BatchV1Api",
        lambda api: SimpleNamespace(
            delete_namespaced_job=lambda *args, **kwargs: deletes.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(native.time, "sleep", lambda _: None)
    result = native._native(
        "cancel", value.execution, {"reference": reference.model_dump()}, DiscoveryBudget()
    )
    assert result["status"] == "cancelled" and len(reads) == 3
    assert len(deletes) == 1 and deletes[0][1]["body"] == {
        "preconditions": {"uid": "owned"},
        "propagationPolicy": "Foreground",
    }
    monkeypatch.setattr(native, "_read", lambda *args: {"metadata": {"uid": "replacement"}})
    with pytest.raises(ExecutionError, match="replacement"):
        native._native(
            "cancel", value.execution, {"reference": reference.model_dump()}, DiscoveryBudget()
        )
    assert len(deletes) == 1


def test_s3_handoff_uses_conditional_put(tmp_path, monkeypatch):
    from nexuml.storage.s3 import S3Client

    writes = []
    storage = S3Client()
    monkeypatch.setattr(
        storage,
        "_get_client",
        lambda: SimpleNamespace(put_object=lambda **kwargs: writes.append(kwargs)),
    )
    storage.upload_bytes(b"config", "s3://owned/config.yaml", exclusive=True)
    assert writes == [
        {"Bucket": "owned", "Key": "config.yaml", "Body": b"config", "IfNoneMatch": "*"}
    ]


@pytest.mark.parametrize(
    "mode,filename",
    [("Job", "job.yaml"), ("RayJob", "ray-job.yaml"), ("PyTorchJob", "pytorch-job.yaml")],
)
def test_documented_template_api_handoff_and_exact_dry_run(tmp_path, monkeypatch, mode, filename):
    from ruamel.yaml import YAML

    path = (
        Path(__file__).resolve().parents[2] / "docs/how-to/training-backends/templates" / filename
    )
    value = scenario(tmp_path, mode, YAML(typ="safe").load(path))
    plan = template_plan(value)
    assert plan["manifest"]["apiVersion"] == value.execution.api_version
    fake_api(monkeypatch)
    monkeypatch.setattr(native, "_access", lambda *args: True)
    monkeypatch.setattr(native, "_check_quota", lambda *args: [])
    calls = []
    monkeypatch.setattr(
        native,
        "_create",
        lambda api, execution, manifest, budget, **options: calls.append((manifest, options)),
    )
    native._native("preflight", value.execution, {"manifest": plan["manifest"]}, DiscoveryBudget())
    assert calls == [(plan["manifest"], {"dry_run": True})]


def test_denied_create_and_unsafe_entrypoint_do_not_submit(tmp_path, monkeypatch):
    fake_api(monkeypatch)
    value = scenario(tmp_path)
    monkeypatch.setattr(native, "_access", lambda *args: False)
    monkeypatch.setattr(
        native, "_create", lambda *args, **kwargs: pytest.fail("Access denied must not create")
    )
    with pytest.raises(ExecutionError, match="denies"):
        native._native(
            "preflight",
            value.execution,
            {"manifest": template_plan(value)["manifest"]},
            DiscoveryBudget(),
        )
    template = manifest()
    template["spec"]["template"]["spec"]["containers"][0]["command"] = ["echo"]
    with pytest.raises(ValueError, match="training container"):
        template_plan(scenario(tmp_path, value=template))


def test_current_context_is_pinned_for_review_and_frozen_revalidation(tmp_path, monkeypatch):
    value = scenario(tmp_path)
    value.execution = value.execution.model_copy(update={"context": None})
    current = "first-current"
    calls = []

    def preflight(action, execution, payload, **kwargs):
        assert action == "preflight"
        calls.append(execution.context)
        return {"context": execution.context or current, "dry_run": True, "diagnostics": []}

    monkeypatch.setattr(native, "_bounded_native", preflight)
    plan = native.prepare_job(value, authorized_root=tmp_path)
    assert plan["resolved_execution"]["params"]["context"] == "first-current"
    assert "template-owned assumptions" in " ".join(plan["summary"]["diagnostics"])
    current = "unrelated-current"
    native.prepare_job(value, authorized_root=tmp_path, frozen_plan=plan)
    assert calls == [None, "first-current"]
    assert plan["summary"]["context"] == "first-current"
    changed = value.model_copy(
        update={"execution": value.execution.model_copy(update={"resources": {"cpu": "2"}})}
    )
    with pytest.raises(ExecutionError, match="Configuration changed"):
        native.prepare_job(changed, authorized_root=tmp_path, frozen_plan=plan)
    assert calls == [None, "first-current"]
    from nexuml.api.worker import dispatch
    from nexuml.core.serialization import lower_model

    monkeypatch.chdir(tmp_path)
    prepared = dispatch("prepare_train", {"data": lower_model(value)})
    assert "error" not in prepared, prepared
    assert prepared["data"]["execution"]["params"]["context"] == "unrelated-current"
    assert prepared["launch_plan"]["resolved_execution"] == prepared["data"]["execution"]


def test_rayjob_rejects_unsupported_strategy_before_native_probe(tmp_path, monkeypatch):
    value = scenario(tmp_path, "RayJob")
    value.training.strategy = "ddp_spawn"
    monkeypatch.setattr(native, "_bounded_native", lambda *_, **__: pytest.fail("No cluster call"))
    with pytest.raises(ValueError, match="Ray supports training.strategy"):
        native.prepare_job(value)
