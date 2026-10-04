"""Call existing framework APIs in a fresh process, never parse terminal output."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml.error import YAMLError

from nexuml.core.config import ResolvedConfig
from nexuml.core.serialization import lower_model, restore_model_data


def configuration(payload: dict) -> ResolvedConfig:
    """Restore portable typed configuration, preserving explicit stage order.

    Returns:
        Existing validated configuration model, without compiling it.

    Raises:
        ValueError: If the representation or ordered stage view is invalid.
    """
    if (payload.get("yaml") is None) == (payload.get("data") is None):
        raise ValueError("Provide exactly one of yaml or data.")
    if payload.get("yaml") is not None:
        return ResolvedConfig.from_yaml(payload["yaml"])
    data = payload["data"]
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping.")
    order = payload.get("stage_order")
    if order is not None:
        stages = data.get("pipeline", {}).get("stages", {})
        if len(order) != len(set(order)) or set(order) != set(stages):
            raise ValueError("stage_order must contain each configured stage exactly once.")
        data = {
            **data,
            "pipeline": {
                **data.get("pipeline", {}),
                "stages": {name: stages[name] for name in order},
            },
        }
    return ResolvedConfig.model_validate(restore_model_data(data, ResolvedConfig))


def document(config: ResolvedConfig) -> dict:
    """Return editable existing configuration plus representation metadata.

    Returns:
        Portable config, ordered stage names, ordinary YAML and semantic revision.
    """
    text = config.to_yaml()
    return {
        "data": lower_model(config),
        "execution_capabilities": dict(config.execution.capabilities),
        "stage_order": list(config.pipeline.stages),
        "yaml": text,
        "semantic_revision": hashlib.sha256(text.encode()).hexdigest(),
    }


def catalog() -> dict:
    """Read authoritative registries/schemas and existing discovery diagnostics.

    Returns:
        Existing library/component/scenario metadata and runtime dependency availability.
    """
    from nexuml.core.backends import backend_rows
    from nexuml.core.discovery import LibraryConfig, discover_library_packages
    from nexuml.core.registry import get_component_registry
    from nexuml.core.scenario_registry import get_scenario_registry
    from nexuml.execution import catalog as execution_catalog

    packages = discover_library_packages()
    components = get_component_registry()
    scenarios = get_scenario_registry()
    entries = [
        {
            "kind": e.kind,
            "name": e.name,
            "version": e.version,
            "import_target": e.import_target,
            "schema": e.definition_type.model_json_schema(),
        }
        for e in components.entries()
    ]
    recipes = [
        {"name": name, "import_target": f"{fn.__module__}.{fn.__qualname__}"}
        for name, fn in sorted(scenarios.list().items())
    ]
    errors = [
        {
            "module": e.module,
            "phase": e.phase,
            "type": e.error_type,
            "message": e.message,
            "key": e.key,
        }
        for e in [*components.errors, *scenarios.errors]
    ]
    return {
        "libraries": {"packages": packages, "roots": LibraryConfig.load().roots},
        "components": entries,
        "scenarios": recipes,
        "errors": errors,
        "schema": ResolvedConfig.model_json_schema(),
        "backend_descriptions": [
            {"category": category, "name": name, "import_target": target}
            for category, name, target in backend_rows()
        ],
        "execution_backends": execution_catalog(),
        "backends": {
            row["type"]: {"available": row["available"], **row["capabilities"]}
            for row in execution_catalog()
        },
    }


def dispatch(action: str, payload: dict) -> dict:
    """Route transport calls to existing NexuML APIs.

    Returns:
        Portable worker result.

    Raises:
        ValueError: For unknown operations (there is no arbitrary code endpoint).
    """
    if action == "catalog":
        return catalog()
    if action == "validate":
        return document(configuration(payload))
    if action == "prepare_train":
        from nexuml.training.lightning import load_scenario_from_trainer_checkpoint
        from nexuml.execution import preflight
        from nexuml.execution.semantics import ExecutionError

        config = configuration(payload)
        if not config.execution.capabilities.get("resume") and payload.get("trainer_checkpoint"):
            raise ValueError("Trainer checkpoint resume is local-only; Ray owns its recovery.")
        if payload.get("trainer_checkpoint"):
            config = ResolvedConfig.from_scenario(
                load_scenario_from_trainer_checkpoint(
                    payload["trainer_checkpoint"],
                    fallback=config.to_scenario(),
                )
            )
        result = document(config)
        try:
            plan = preflight(
                config.to_scenario(),
                trainer_checkpoint=payload.get("trainer_checkpoint"),
                authorized_root=Path.cwd(),
            )
        except (ValueError, ExecutionError) as error:
            # The definition owns validation; transport adds a navigable selection location.
            return {
                "error": {
                    "status": 422,
                    "code": "validation",
                    "message": str(error),
                    "fields": [
                        {"loc": ["execution"], "message": str(error), "type": "execution_preflight"}
                    ],
                }
            }
        if plan:
            if plan.get("resolved_execution"):
                config = config.model_copy(
                    update={
                        "execution": restore_model_data(
                            {"execution": plan["resolved_execution"]}, ResolvedConfig
                        )["execution"]
                    }
                )
                result = document(config)
            expected = payload.get("template_revision")
            if expected and expected != plan.get("summary", {}).get("template_revision"):
                raise ValueError("Template changed after review; renew the launch review.")
            result["launch_review"] = plan.get("summary", plan)
            result["launch_plan"] = plan
        return result
    if action == "discover_execution":
        from nexuml.execution import discover

        config = configuration(payload)
        return discover(
            config.execution,
            timeout=payload.get("timeout", 10),
            authorized_root=Path.cwd(),
            scenario=config.to_scenario(),
        ).model_dump(mode="json")
    if action in {"inspect_execution", "cancel_execution"}:
        from nexuml.execution import inspect, cancel
        from nexuml.execution.schemas import NativeReference

        reference = NativeReference.model_validate(payload["reference"])
        return inspect(reference) if action == "inspect_execution" else cancel(reference)
    if action == "build":
        from nexuml.core.compiler import compile

        config = configuration(payload)
        pipeline = compile(config.to_scenario())
        return {
            "shapes": pipeline.input_sizes,
            "stages": list(pipeline.stages),
            "semantic_revision": document(config)["semantic_revision"],
        }
    if action in {"train", "export"}:
        return execute(action, payload)
    if action == "resolve":
        from nexuml.core.scenario_registry import get_scenario_registry

        scenario = get_scenario_registry().get(payload["name"])()
        return document(ResolvedConfig.from_scenario(scenario))
    if action == "resolve_file":
        from nexuml.core.scenario_loader import load_scenario_file

        loaded = load_scenario_file(Path(payload["path"]))
        return document(ResolvedConfig.from_scenario(loaded.scenario))
    raise ValueError("Unknown API operation.")


def execute(action: str, payload: dict) -> dict:
    """Delegate execution and exports to the existing session/backend lifecycle.

    Returns:
        Actual results and existing-format artifact references.

    Raises:
        ValueError: When an unsupported backend/export combination is requested.
    """
    from nexuml.api.observation import Observation
    from nexuml.execution import run
    from nexuml.execution.artifacts import export_requests

    config = configuration(payload)
    observer = Observation(Path(payload["observations"])) if action == "train" else None
    if observer is not None:
        observer.record("progress", phase="preparation / data setup")
    if action == "train":
        result = run(
            config.to_scenario(),
            trainer_checkpoint=payload.get("trainer_checkpoint"),
            observer=observer,
            enable_progress_bar=False,
            authorized_root=Path.cwd(),
            frozen_plan=payload.get("launch_plan"),
        )
        if isinstance(result, dict):
            return result
        artifacts = list(getattr(result, "artifacts", []))
        callback = getattr(getattr(result, "trainer", None), "checkpoint_callback", None)
        for source in (
            getattr(callback, "best_model_path", ""),
            getattr(callback, "last_model_path", ""),
        ):
            if source and Path(source).is_file():
                artifacts.append({"path": str(Path(source).resolve()), "kind": "checkpoint"})
        return {
            "status": "succeeded",
            "metrics": getattr(result, "metrics", {}),
            "validation_results": getattr(result, "validation_results", []),
            "test_results": getattr(result, "test_results", []),
            "evaluation_results": getattr(result, "eval_algorithm_results", {}),
            "artifacts": artifacts,
        }
    if not config.execution.capabilities.get("artifacts") or not config.execution.capabilities.get(
        "resume"
    ):
        raise ValueError("Model export from a Trainer checkpoint is local-only.")
    from nexuml.training.lightning import NexuSession
    from nexuml.core.types import ExportSpec

    session = NexuSession(
        config.to_scenario(),
        trainer_checkpoint=payload.get("trainer_checkpoint"),
        enable_progress_bar=False,
    )
    session.setup()
    requests = [ExportSpec(kind=payload["export_kind"], output=payload["output"])]
    return {"artifacts": export_requests(session, requests, payload.get("trainer_checkpoint"))}


def main() -> None:
    """Read a private request and write a plain response, not terminal logs."""
    action, request, response = sys.argv[1:]
    try:
        result = dispatch(action, json.loads(Path(request).read_text(encoding="utf-8")))
    except ValidationError as exc:
        fields = [
            {"loc": list(e["loc"]), "message": e["msg"], "type": e["type"]}
            for e in exc.errors(include_input=False, include_context=False)
        ]
        result = {
            "error": {
                "status": 422,
                "code": "validation",
                "message": "NexuML rejected the configuration.",
                "fields": fields,
            }
        }
    except (ValueError, TypeError, KeyError, YAMLError) as exc:
        result = {"error": {"status": 422, "code": "validation", "message": str(exc)}}
    except ImportError as exc:
        result = {"error": {"status": 422, "code": "missing_dependency", "message": str(exc)}}
    except Exception as exc:
        result = {"error": {"status": 500, "code": "execution", "message": str(exc)}}
    from nexuml.core.export import _make_json_safe

    try:
        encoded = json.dumps(_make_json_safe(result), allow_nan=False)
    except (TypeError, ValueError):
        encoded = json.dumps(
            {
                "error": {
                    "status": 500,
                    "code": "serialization",
                    "message": (
                        "NexuML returned nonfinite or unsupported results; "
                        "inspect domain logs/artifacts."
                    ),
                }
            }
        )
    Path(response).write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
