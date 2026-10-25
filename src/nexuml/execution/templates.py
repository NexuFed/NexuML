"""Review and bind infrastructure-owned templates without synthesizing cluster policy."""

import copy
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from nexuml.core.config import ResolvedConfig
from nexuml.execution.capacity import pod_requests, quantities
from nexuml.execution.definitions import LocalExecution, RayJobExecution, PyTorchJobExecution
from nexuml.execution.semantics import ensure_distributed_semantics

CONFIG_TOKEN = "{{NEXUML_CONFIG}}"


def authorized_path(value: str, root: Path | None = None) -> Path:
    """Resolve an explicitly selected file without escaping an API's allowed root.

    Returns:
        A canonical authorized path.

    Raises:
        ValueError: On traversal or symlink escape.
    """
    candidate = Path(value).expanduser()
    if root is not None:
        root = root.resolve()
        if ".." in candidate.parts:
            raise ValueError("Path traversal is not allowed.")
        candidate = root / candidate
        if not candidate.resolve().is_relative_to(root):
            raise ValueError("Template/handoff path is outside the authorized working directory.")
    return candidate.resolve()


def pod_roles(manifest: dict, execution: Any) -> list[tuple[str, int, dict]]:
    """Read every workload role without inventing an unseen controller footprint.

    Returns:
        Named replica counts and existing pod specifications.
    """
    spec = manifest["spec"]
    if execution.resource_kind == "Job":
        return [("training", 1, spec["template"]["spec"])]
    if execution.resource_kind == "PyTorchJob":
        return [
            (name, int(role.get("replicas", 1)), role["template"]["spec"])
            for name, role in spec["pytorchReplicaSpecs"].items()
        ]
    roles = []
    cluster = spec.get("rayClusterSpec")
    if cluster is not None:
        roles.append(("head", 1, cluster["headGroupSpec"]["template"]["spec"]))
        roles.extend(
            (group["groupName"], int(group.get("replicas", 1)), group["template"]["spec"])
            for group in cluster.get("workerGroupSpecs", [])
        )
    if spec.get("submitterPodTemplate"):
        roles.append(("submitter", 1, spec["submitterPodTemplate"]["spec"]))
    return roles


def _bind(manifest: dict, execution: Any) -> None:
    roles = pod_roles(manifest, execution)
    if execution.resource_kind == "RayJob":
        groups = manifest["spec"].get("rayClusterSpec", {}).get("workerGroupSpecs", [])
        names = [group["groupName"] for group in groups]
    elif execution.resource_kind == "PyTorchJob":
        names = list(manifest["spec"]["pytorchReplicaSpecs"])
    else:
        names = ["training"]
    if not execution.resources and execution.replicas is None:
        return
    selected = execution.worker_group
    if selected is None:
        if len(names) != 1:
            raise ValueError(
                "Select an explicit worker_group for ambiguous replica/resource bindings."
            )
        selected = names[0]
    if selected not in names:
        raise ValueError("The selected worker_group is absent from the template.")
    pod_spec = next(pod for name, _, pod in roles if name == selected)
    containers = pod_spec.get("containers", [])
    container_name = execution.container
    if container_name is None:
        if len(containers) != 1:
            raise ValueError("Select an explicit container for ambiguous resource bindings.")
        container_name = containers[0]["name"]
    matches = [container for container in containers if container["name"] == container_name]
    if len(matches) != 1:
        raise ValueError("The selected training container is absent or ambiguous.")
    if execution.resources:
        quantities(execution.resources)
        resources = matches[0].setdefault("resources", {})
        resources.setdefault("requests", {}).update(execution.resources)
        # Overrides bind both requests and limits; unrelated resources stay intact.
        resources.setdefault("limits", {}).update(execution.resources)
    if execution.replicas is not None:
        if execution.resource_kind == "Job":
            if execution.replicas != 1:
                raise ValueError("Kubernetes Job supports one training pod, not parallel learners.")
        elif execution.resource_kind == "RayJob":
            next(group for group in groups if group["groupName"] == selected)["replicas"] = (
                execution.replicas
            )
        else:
            manifest["spec"]["pytorchReplicaSpecs"][selected]["replicas"] = execution.replicas


def template_plan(
    scenario: Any, *, authorized_root: Path | None = None, invocation_id: str | None = None
) -> dict[str, Any]:
    """Capture the exact reviewed template and derived worker configuration.

    Returns:
        A private frozen plan and a credential-free summary.

    Raises:
        ValueError: For missing handoff, unsafe inputs or unsupported launch topology.
    """
    execution = scenario.execution
    if not execution.template or execution.handoff is None:
        raise ValueError(
            "Select an infrastructure-owned template and explicit configuration handoff."
        )
    path = authorized_path(execution.template, authorized_root)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Template exceeds the 2 MiB review limit.")
    raw = path.read_bytes()
    try:
        manifest = YAML(typ="safe").load(raw)
    except YAMLError as error:
        raise ValueError(
            "Template YAML is invalid; review it locally (content is not echoed)."
        ) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("spec"), dict):
        raise ValueError("Template must contain one native workload object.")
    if (
        manifest.get("apiVersion") != execution.api_version
        or manifest.get("kind") != execution.resource_kind
    ):
        raise ValueError("Template API/kind does not match the selected execution mode.")
    if any(
        key in manifest.get("metadata", {}) for key in ("uid", "resourceVersion", "generateName")
    ):
        raise ValueError("Use a workload template, not a live resource identity or generateName.")
    if manifest.get("metadata", {}).get("namespace", execution.namespace) != execution.namespace:
        raise ValueError(
            "Template namespace differs from the selected namespace; review the template."
        )
    encoded = json.dumps(manifest)
    if CONFIG_TOKEN not in encoded:
        raise ValueError(f"Template must reference {CONFIG_TOKEN} in its remote entrypoint.")
    for _, _, pod in pod_roles(manifest, execution):
        for container in [*pod.get("containers", []), *pod.get("initContainers", [])]:
            for variable in container.get("env", []):
                name = variable.get("name", "").upper()
                if "value" in variable and any(
                    part in name for part in ("TOKEN", "PASSWORD", "SECRET", "ACCESS_KEY")
                ):
                    raise ValueError(
                        "Templates must reference credential secrets, not embed their values."
                    )
    if isinstance(execution, PyTorchJobExecution):
        ensure_distributed_semantics(scenario)
        if not isinstance(scenario.training.strategy, str) or scenario.training.strategy not in {
            "auto",
            "ddp",
        }:
            raise ValueError("PyTorchJob currently supports tested external torchrun DDP only.")
        roles = manifest["spec"]["pytorchReplicaSpecs"]
        if set(roles) - {"Master", "Worker"} or roles.get("Master", {}).get("replicas", 1) != 1:
            raise ValueError("PyTorchJob requires one Master and optional Worker replicas.")
        if "Master" not in roles:
            raise ValueError("PyTorchJob requires a Master replica.")
        process_counts = set()
        for _, replicas, pod in pod_roles(manifest, execution):
            if replicas < 0:
                raise ValueError("PyTorchJob replica counts must be nonnegative.")
            containers = pod.get("containers", [])
            training = [container for container in containers if container.get("name") == "pytorch"]
            if len(training) != 1 or training[0].get("command") != ["torchrun"]:
                raise ValueError(
                    "PyTorchJob requires a named pytorch training container using torchrun."
                )
            args = training[0].get("args", [])
            required = {
                "--nnodes=$(WORLD_SIZE)",
                "--node-rank=$(RANK)",
                "--master-addr=$(MASTER_ADDR)",
                "--master-port=$(MASTER_PORT)",
            }
            if not required.issubset(args) or "--standalone" in args:
                raise ValueError(
                    "PyTorchJob torchrun must use the operator's "
                    "WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT, not independent launchers."
                )
            processes = [
                arg.split("=", 1)[1] for arg in args if arg.startswith("--nproc-per-node=")
            ]
            if len(processes) != 1 or not processes[0].isdigit() or int(processes[0]) < 1:
                raise ValueError(
                    "PyTorchJob requires one explicit positive --nproc-per-node value."
                )
            process_counts.add(int(processes[0]))
            if scenario.training.devices not in ("auto", int(processes[0])):
                raise ValueError(
                    "PyTorchJob launcher processes differ from reviewed training.devices."
                )
            if "-m" not in args or args[args.index("-m") + 1 : args.index("-m") + 2] != [
                "nexuml.execution.worker"
            ]:
                raise ValueError("PyTorchJob must launch -m nexuml.execution.worker.")
        if len(process_counts) != 1:
            raise ValueError("PyTorchJob requires the same local process count for every replica.")
    elif execution.resource_kind == "Job":
        spec = manifest["spec"]
        if (
            spec.get("parallelism", 1) != 1
            or spec.get("completions", 1) != 1
            or spec.get("completionMode", "NonIndexed") != "NonIndexed"
        ):
            raise ValueError("Kubernetes Job supports one non-indexed training pod.")
        containers = spec["template"]["spec"].get("containers", [])
        training = [
            container
            for container in containers
            if CONFIG_TOKEN in json.dumps(container.get("args", []))
            and (execution.container is None or container.get("name") == execution.container)
        ]
        command = training[0].get("command", []) if len(training) == 1 else []
        if (
            len(command) != 3
            or Path(command[0]).name not in {"python", "python3"}
            or command[1:] != ["-m", "nexuml.execution.worker"]
        ):
            raise ValueError(
                "Job requires one selected training container running "
                "python -m nexuml.execution.worker."
            )
    elif isinstance(execution, RayJobExecution):
        execution.ray.preflight(scenario)
        spec = manifest["spec"]
        if bool(spec.get("rayClusterSpec")) == bool(spec.get("clusterSelector")):
            raise ValueError(
                "RayJob must select either rayClusterSpec or an existing clusterSelector."
            )
        if "-m nexuml.execution.worker" not in spec.get(
            "entrypoint", ""
        ) or CONFIG_TOKEN not in spec.get("entrypoint", ""):
            raise ValueError(
                "RayJob entrypoint must run python -m nexuml.execution.worker "
                "with the config placeholder."
            )
    _bind(manifest, execution)
    if isinstance(execution, PyTorchJobExecution):
        roles = manifest["spec"]["pytorchReplicaSpecs"]
        if roles["Master"].get("replicas", 1) != 1:
            raise ValueError("PyTorchJob requires exactly one Master after resource bindings.")
    identifier = invocation_id or uuid4().hex
    if not identifier.isalnum() or len(identifier) > 40:
        raise ValueError("Invocation identity must be a bounded alphanumeric identifier.")
    name = "nexuml-" + identifier.lower()
    metadata = manifest.setdefault("metadata", {})
    metadata.update(name=name, namespace=execution.namespace)
    metadata.setdefault("labels", {})["nexuml.org/invocation"] = identifier.lower()
    worker = copy.deepcopy(scenario)
    worker.execution = execution.ray if isinstance(execution, RayJobExecution) else LocalExecution()
    if isinstance(execution, RayJobExecution):
        execution.ray.preflight(worker)
    config_yaml = ResolvedConfig.from_scenario(worker).to_yaml()
    destination = execution.handoff.destination
    if destination.startswith("s3://"):
        parsed = urlsplit(destination)
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.hostname
        ):
            raise ValueError("Use an S3 handoff URI without embedded credentials/query/fragment.")
        local_path = None
        worker_path = destination.rstrip("/") + "/" + name + ".yaml"
    else:
        directory = authorized_path(destination, authorized_root)
        local_path = str(directory / (name + ".yaml"))
        if not execution.handoff.worker_directory:
            raise ValueError(
                "Shared filesystem handoff needs the explicit worker_directory mapping."
            )
        worker_path = execution.handoff.worker_directory.rstrip("/") + "/" + name + ".yaml"

    # Replace only declared launch placeholders, not arbitrary strings or manifest policy.
    def substitute(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: substitute(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [substitute(item, key) for item in value]
        if isinstance(value, str) and CONFIG_TOKEN in value:
            if key not in {"args", "command", "entrypoint"}:
                raise ValueError(
                    "Configuration placeholders are allowed only in launch commands/args."
                )
            replacement = shlex.quote(worker_path) if key == "entrypoint" else worker_path
            return value.replace(CONFIG_TOKEN, replacement)
        return value

    manifest = substitute(manifest)
    roles = pod_roles(manifest, execution)
    summary = {
        "kind": execution.resource_kind,
        "api_version": execution.api_version,
        "name": name,
        "namespace": execution.namespace,
        "worker_config": worker_path,
        "template_revision": hashlib.sha256(raw).hexdigest(),
        "roles": [
            {"name": role, "replicas": replicas, "requests": pod_requests(pod)}
            for role, replicas, pod in roles
        ],
        "diagnostics": [
            "Remote images, code, data, mounts and output access remain template-owned assumptions."
        ],
    }
    if execution.resource_kind == "RayJob" and "submitterPodTemplate" not in manifest["spec"]:
        summary["diagnostics"].append(
            "KubeRay's default submitter footprint is not defined by this template."
        )
    return {
        "manifest": manifest,
        "worker_yaml": config_yaml,
        "local_path": local_path,
        "worker_path": worker_path,
        "template_path": str(path),
        "summary": summary,
        "source_revision": hashlib.sha256(
            ResolvedConfig.from_scenario(scenario).to_yaml().encode()
        ).hexdigest(),
    }
