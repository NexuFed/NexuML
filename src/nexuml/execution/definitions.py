"""Core execution definitions; importing settings never imports cluster runtimes."""

import shlex
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any, ClassVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nexuml.core.components import ExecutionBackendDefinition
from nexuml.core.discovery import execution_backend


@execution_backend("local")
class LocalExecution(ExecutionBackendDefinition):
    """Run the existing NexuML session in the current process."""

    label = "Local"
    capabilities = {
        "status": True,
        "evaluation": True,
        "logs": True,
        "metrics": True,
        "cancellation": True,
        "resume": True,
        "artifacts": True,
        "process_ownership": True,
    }

    def run(self, scenario: Any, **kwargs: Any) -> Any:
        operation = kwargs.pop("operation", "train")
        if operation == "evaluate":
            from nexuml.cli.main import _evaluate_local

            return _evaluate_local(
                Path(kwargs["model_package"]),
                supplied=scenario if kwargs.get("evaluation_inputs", True) else None,
                scenario_file=kwargs.get("evaluation_scenario_file"),
                device=kwargs.get("device"),
                override=list(kwargs.get("overrides", ())),
            )
        if operation == "export":
            from nexuml.cli.main import _export_local

            checkpoint = kwargs.get("checkpoint")
            return _export_local(
                scenario,
                output=Path(
                    kwargs["output"] if kwargs.get("output") is not None else "exported_model"
                ),
                checkpoint=Path(checkpoint) if checkpoint is not None else None,
                export_kind=kwargs.get("export_kind", "train_package"),
            )
        if operation != "train":
            raise ValueError("Local execution operation must be train, evaluate or export")
        from nexuml.training.lightning import NexuSession

        observer = kwargs.pop("observer", None)
        on_trained = kwargs.pop("on_trained", None)
        kwargs.pop("review", None)
        kwargs.pop("authorized_root", None)
        kwargs.pop("frozen_plan", None)
        session = NexuSession(scenario=scenario, **kwargs)
        with ExitStack() as job:
            if session.scenario.interaction is not None:
                job.enter_context(session.scenario.interaction.environment.job_scope())
            try:
                if observer is not None:
                    # Lightning's callback connector creates this attribute at runtime.
                    getattr(session.trainer, "callbacks").append(observer)
                result = session.run()
                if on_trained is not None:
                    on_trained(result)
                if session.scenario.exports:
                    from nexuml.execution.artifacts import export_requests

                    result.artifacts.extend(export_requests(result, session.scenario.exports))
                return result
            except BaseException as primary:
                try:
                    session.finalize_loggers("failed")
                except BaseException as error:
                    primary.add_note(f"Logger finalization also failed: {error!r}")
                    for note in tuple(getattr(error, "__notes__", ())):
                        primary.add_note(note)
                raise

    def discover(self, **kwargs: Any) -> Any:
        from nexuml.execution.resources import local_resources

        return local_resources(**kwargs)


@execution_backend("ray-jobs-sdk")
class RayJobsSDKExecution(ExecutionBackendDefinition):
    """Submit the normal CLI with locked uv setup and one GPU by default.

    Set ``setup_command=None`` for a prepared environment. Environment/Python
    ``{run_id}`` placeholders are resolved separately for every submission.
    """

    label = "Ray Jobs SDK"
    dependencies = ("ray",)
    capabilities = {
        "status": False,
        "evaluation": True,
        "logs": False,
        "metrics": False,
        "cancellation": False,
        "resume": True,
        "artifacts": True,
        "process_ownership": False,
    }
    address: str | None = None
    verify: Annotated[str, Field(min_length=1)] | bool = True
    setup_command: list[str] | None = Field(
        default_factory=lambda: ["uv", "sync", "--locked"], min_length=1
    )
    # Schema-driven callers must omit this conditional default rather than force base Python.
    py_executable: str = Field(default_factory=lambda: "python", min_length=1)
    runtime_env: dict[str, Any] = Field(default_factory=lambda: {"working_dir": "."})
    entrypoint_num_cpus: float = Field(default=0, ge=0, allow_inf_nan=False)
    entrypoint_num_gpus: float = Field(default=1, ge=0, allow_inf_nan=False)
    entrypoint_memory: int | None = Field(default=None, ge=0)
    entrypoint_resources: dict[str, Annotated[float, Field(gt=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def default_setup_environment(cls, value: Any) -> Any:
        """Fill matching uv/interpreter defaults before freezing, without changing input.

        Returns:
            Copied settings preserving explicit interpreter and environment precedence.

        Raises:
            ValueError: If environment settings cannot select a valid interpreter.
        """
        if not isinstance(value, dict):
            return value
        values = dict(value)
        runtime_env = values.get("runtime_env", {"working_dir": "."})
        if not isinstance(runtime_env, dict):
            return values  # Field validation owns invalid runtime environment types.
        env = runtime_env.get("env_vars", {})
        if not isinstance(env, dict):
            raise ValueError("ray-jobs-sdk runtime_env.env_vars must be a mapping")
        if values.get("setup_command", ["uv", "sync", "--locked"]) is None:
            return values
        env = {
            "UV_PROJECT_ENVIRONMENT": "/tmp/ray/nexuml-envs/{run_id}",
            "UV_CACHE_DIR": "/tmp/ray/uv-cache",
            "UV_LINK_MODE": "hardlink",
            "UV_HTTP_TIMEOUT": "600",
            "UV_CONCURRENT_DOWNLOADS": "4",
            **env,
        }
        if "py_executable" not in values:
            venv = env["UV_PROJECT_ENVIRONMENT"]
            if not isinstance(venv, str) or not venv.strip():
                raise ValueError("ray-jobs-sdk UV_PROJECT_ENVIRONMENT must be a nonempty path")
            values["py_executable"] = shlex.join([f"{venv}/bin/python", "-u", "-X", "faulthandler"])
        values["runtime_env"] = {**runtime_env, "env_vars": env}
        return values

    @field_validator("setup_command")
    @classmethod
    def validate_setup_command(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (
            not value[0].strip() or any("\x00" in argument for argument in value)
        ):
            raise ValueError("setup_command requires an executable and arguments without NUL bytes")
        return value

    @field_validator("entrypoint_resources")
    @classmethod
    def validate_resource_names(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not name.strip() for name in value):
            raise ValueError("Ray Job resource names must not be empty")
        return value

    def preflight(self, scenario: Any, **kwargs: Any) -> dict[str, Any]:
        from nexuml.execution.ray_job import _ray_job_argv

        argv = _ray_job_argv("<remote-config>", **kwargs)
        if not shlex.split(self.py_executable):
            raise ValueError("ray-jobs-sdk py_executable must contain a Python command")
        super().preflight(scenario, **kwargs)
        return {"argv": argv}

    def run(self, scenario: Any, **kwargs: Any) -> str:
        from nexuml.execution.ray_job import _submit_ray_job

        return _submit_ray_job(scenario, **kwargs)


class RayClusterTarget(BaseModel):
    """An explicit existing Ray cluster and its native runtime environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    address: str = "auto"
    dashboard_address: str | None = None
    working_dir: str | None = "."
    py_executable: str | None = None
    context: str | None = None
    namespace: str = "default"

    @field_validator("address", "dashboard_address")
    @classmethod
    def validate_address(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if (
                not value.strip()
                or "@" in value
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "Use a target address without embedded credentials, query or fragment."
                )
            if value == "local":
                raise ValueError(
                    "RayCluster must connect to an existing cluster, not start Local Ray."
                )
            if any(char.isspace() for char in value) or (
                "://" in value and parsed.scheme not in {"ray", "http", "https"}
            ):
                raise ValueError(
                    "Use a Ray host address or an HTTP(S) dashboard, not another URL scheme."
                )
        return value


@execution_backend("ray-cluster")
class RayClusterExecution(ExecutionBackendDefinition):
    """Ray placement configuration; training semantics stay in TrainingSpec."""

    label = "RayCluster"
    dependencies = ("ray",)
    presentation = {
        "target": ["target"],
        "resources": ["workers", "resources_per_worker"],
        "expert": ["storage_path"],
    }
    capabilities = {
        "status": True,
        "logs": True,
        "metrics": True,
        "cancellation": False,
        "resume": False,
        "artifacts": False,
        "process_ownership": False,
    }
    target: RayClusterTarget = Field(default_factory=RayClusterTarget)
    workers: int | tuple[int, int] = 1
    resources_per_worker: dict[str, float] = Field(default_factory=lambda: {"CPU": 1.0})
    storage_path: str | None = None

    @field_validator("storage_path")
    @classmethod
    def validate_storage(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    "Storage paths cannot contain embedded credentials/query/fragment."
                )
        return value

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, value: int | tuple[int, int]) -> int | tuple[int, int]:
        if isinstance(value, int):
            if value < 1:
                raise ValueError("execution.workers must be positive")
        else:
            minimum, maximum = value
            if minimum < 1 or maximum < minimum:
                raise ValueError("execution.workers range must satisfy 1 <= min <= max")
        return value

    @field_validator("resources_per_worker")
    @classmethod
    def validate_resources(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(not key.strip() or amount <= 0 for key, amount in value.items()):
            raise ValueError("execution resources must have names and positive amounts")
        return value

    def preflight(self, scenario: Any, **kwargs: Any) -> dict[str, Any]:
        from nexuml.execution.semantics import ensure_distributed_semantics
        from nexuml.core.types import StrategySpec

        ensure_distributed_semantics(scenario)
        if isinstance(scenario.training.strategy, str) and scenario.training.strategy not in {
            "auto",
            "ddp",
            "fsdp",
            "deepspeed",
        }:
            raise ValueError("Ray supports training.strategy auto, ddp, fsdp or deepspeed.")
        if isinstance(scenario.training.strategy, StrategySpec):
            from ray.train.lightning import RayDDPStrategy, RayFSDPStrategy, RayDeepSpeedStrategy

            factory = scenario.training.strategy.resolve()
            if not isinstance(factory, type) or not issubclass(
                factory, (RayDDPStrategy, RayFSDPStrategy, RayDeepSpeedStrategy)
            ):
                raise ValueError(
                    "Typed Ray training.strategy must be an official Ray Lightning strategy "
                    "or subclass."
                )
        return super().preflight(scenario, **kwargs)

    def run(self, scenario: Any, **kwargs: Any) -> Any:
        from nexuml.execution import run_ray

        return run_ray(scenario)

    def discover(self, **kwargs: Any) -> Any:
        from nexuml.execution.resources import ray_resources

        return ray_resources(self, **kwargs)


class ConfigHandoff(BaseModel):
    """Explicit shared filesystem or S3 destination for one frozen worker config."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    destination: str
    worker_directory: str | None = None

    @field_validator("destination", "worker_directory")
    @classmethod
    def validate_destination(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if (
                not value.strip()
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "Handoff must be a plain path or S3 URI without embedded credentials."
                )
            if "://" in value and parsed.scheme != "s3":
                raise ValueError(
                    "Only shared-filesystem and S3 configuration handoff are supported."
                )
        return value


class KubernetesExecution(ExecutionBackendDefinition):
    """Infrastructure-owned template and named training resource bindings."""

    dependencies = ("kubernetes",)
    presentation = {
        "target": ["context", "namespace"],
        "setup": ["template", "handoff"],
        "resources": ["worker_group", "container", "replicas", "resources"],
        "expert": ["ray"],
    }
    api_version: ClassVar[str]
    resource_kind: ClassVar[str]
    plural: ClassVar[str]
    capabilities = {
        "status": True,
        "logs": True,
        "metrics": False,
        "cancellation": True,
        "resume": False,
        "artifacts": False,
        "process_ownership": False,
        "native_reference": True,
    }
    context: str | None = None
    namespace: str = Field(
        default="default", min_length=1, max_length=63, pattern=r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$"
    )
    template: str | None = None
    handoff: ConfigHandoff | None = None
    container: str | None = None
    worker_group: str | None = None
    replicas: int | None = Field(default=None, ge=1)
    resources: dict[str, str] = Field(default_factory=dict)

    def preflight(self, scenario: Any, **kwargs: Any) -> dict[str, Any]:
        super().preflight(scenario, **kwargs)
        from nexuml.execution.kubernetes import prepare_job

        return prepare_job(scenario, **kwargs)

    def run(self, scenario: Any, **kwargs: Any) -> Any:
        from nexuml.execution.kubernetes import run_job

        kwargs["frozen_plan"] = kwargs.pop("review", None) or kwargs.get("frozen_plan")
        return run_job(scenario, **kwargs)

    def discover(self, **kwargs: Any) -> Any:
        from nexuml.execution.kubernetes import discover_resources

        return discover_resources(self, **kwargs)

    def inspect(self, reference: Any) -> dict[str, Any]:
        from nexuml.execution.kubernetes import inspect

        return inspect(reference)

    def cancel(self, reference: Any) -> dict[str, Any]:
        from nexuml.execution.kubernetes import cancel

        return cancel(reference)


@execution_backend("ray-job")
class RayJobExecution(KubernetesExecution):
    """Submit a KubeRay RayJob; its template owns the cluster and cleanup policy."""

    label = "RayJob"
    api_version = "ray.io/v1"
    resource_kind = "RayJob"
    plural = "rayjobs"
    capabilities = {**KubernetesExecution.capabilities, "cancellation": False, "logs": False}
    ray: RayClusterExecution = Field(
        default_factory=lambda: RayClusterExecution(target=RayClusterTarget(working_dir=None))
    )


@execution_backend("kubernetes-job")
class KubernetesJobExecution(KubernetesExecution):
    """Submit one Kubernetes training pod."""

    label = "Kubernetes Job"
    api_version = "batch/v1"
    resource_kind = "Job"
    plural = "jobs"


@execution_backend("kubernetes-pytorchjob")
class PyTorchJobExecution(KubernetesExecution):
    """Submit the served Kubeflow Training Operator PyTorchJob resource."""

    label = "Kubernetes PyTorchJob"
    api_version = "kubeflow.org/v1"
    resource_kind = "PyTorchJob"
    plural = "pytorchjobs"
