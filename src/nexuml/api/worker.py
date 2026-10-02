"""Call existing framework APIs in a fresh process, never parse terminal output."""

from __future__ import annotations

import hashlib
import importlib.util
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
        "backends": {
            "local": {"available": True, "cancellation": True},
            "ray": {
                "available": importlib.util.find_spec("ray") is not None,
                "cancellation": False,
                "telemetry": "driver logs and final results",
            },
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

        config = configuration(payload)
        if config.execution.kind == "ray" and payload.get("trainer_checkpoint"):
            raise ValueError("Trainer checkpoint resume is local-only; Ray owns its recovery.")
        if payload.get("trainer_checkpoint"):
            config = ResolvedConfig.from_scenario(
                load_scenario_from_trainer_checkpoint(
                    payload["trainer_checkpoint"],
                    fallback=config.to_scenario(),
                )
            )
        return document(config)
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
    from nexuml.core.export import export_onnx, export_package, export_safetensors
    from nexuml.training.lightning import NexuSession

    config = configuration(payload)
    if config.execution.kind == "ray":
        if action == "export" or payload.get("trainer_checkpoint"):
            raise ValueError("Model export and Trainer checkpoint resume are local-only.")
        from nexuml.execution import run_ray

        result = run_ray(config.to_scenario())
        return {
            "metrics": getattr(result, "metrics", {}),
            "artifacts": [],
            "telemetry": "driver logs and final results; remote cancellation unsupported",
        }
    session = NexuSession(
        config.to_scenario(),
        trainer_checkpoint=payload.get("trainer_checkpoint"),
        enable_progress_bar=False,
    )
    artifacts = []
    if action == "train":
        # Append through the existing public Trainer seam; no new session callback API.
        session.trainer.callbacks.append(Observation(Path(payload["observations"])))
        result = session.run()
        callback = session.trainer.checkpoint_callback
        for source in (
            getattr(callback, "best_model_path", ""),
            getattr(callback, "last_model_path", ""),
        ):
            if source and Path(source).is_file():
                artifacts.append({"path": str(Path(source).resolve()), "kind": "checkpoint"})
        requests = config.exports
    else:
        # Export only a confirmed completed operation's actual Trainer checkpoint.
        session.setup()
        result = None
        from nexuml.core.types import ExportSpec

        requests = [ExportSpec(kind=payload["export_kind"], output=payload["output"])]
    for request in requests:
        output = Path(request.output or "exported_model")
        if request.kind == "train_package":
            path = export_package(
                session.pipeline,
                output,
                lightning_module=session.lightning_module,
                trainer=session.trainer if action == "train" else None,
                checkpoint_path=payload.get("trainer_checkpoint") if action == "export" else None,
            )
            artifacts.extend(
                {"path": str(p.resolve()), "kind": "train_package"}
                for p in path.rglob("*")
                if p.is_file()
            )
        elif request.kind == "safetensors":
            path = export_safetensors(
                session.pipeline, output, include=request.include, exclude=request.exclude
            )
            artifacts.extend(
                {"path": str(p.resolve()), "kind": "safetensors"}
                for p in (path, path.with_suffix(".json"))
            )
        else:
            path = export_onnx(session.pipeline, output)
            artifacts.append({"path": str(path.resolve()), "kind": "onnx"})
    if result is None:
        return {"artifacts": artifacts}
    return {
        "validation_results": result.validation_results,
        "test_results": result.test_results,
        "evaluation_results": result.eval_algorithm_results,
        "artifacts": artifacts,
    }


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
