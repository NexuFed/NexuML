"""Export, reload, and selective checkpoint loading for trained pipelines."""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import importlib
import importlib.metadata
import json
import logging
import pkgutil
import sys
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tensordict import TensorDict
from torch.package.package_exporter import PackageExporter
from torch.package.package_importer import PackageImporter

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.registry import LayerRegistry, get_registry
from nexuml.core.types import CheckpointLoadSpec, ScenarioSpec

logger = logging.getLogger(__name__)

PACKAGE_FILENAME = "pipeline.package"
PACKAGE_PICKLE_PACKAGE = "nexuml_export"
PACKAGE_PICKLE_NAME = "artifact.pkl"
LEGACY_PACKAGE_PICKLE_PACKAGE = "model"
LEGACY_PACKAGE_PICKLE_NAME = "pipeline.pkl"
REQUIREMENTS_FILENAME = "requirements.txt"
CHECKPOINT_SIDECAR = "lightning.ckpt"

# Modules that are expected to live in the target runtime environment.
_RUNTIME_EXTERN_PATTERNS = [
    # stdlib fragments that torch.package may otherwise try to intern
    "builtins",
    "collections",
    "copy",
    "datetime",
    "fnmatch",
    "importlib",
    "inspect",
    "io",
    "json",
    "logging",
    "pathlib",
    "sys",
    "typing",
    "typing_extensions",
    # heavy third-party runtime dependencies
    "torch.**",
    "torchaudio.**",
    "torchvision.**",
    "tensordict.**",
    "timm.**",
    "numpy.**",
    "librosa.**",
    "torchmetrics.**",
    "torchrl.**",
    "lightning.**",
    "pydantic.**",
    "ruamel.**",
    "safetensors.**",
    "pandas.**",
    "yaml.**",
    "soundfile.**",
    "PIL.**",
    "tqdm.**",
    "sklearn.**",
    "matplotlib.**",
    "umap.**",
    "transformers.**",
    "nvidia.**",
    "onnxscript.**",
    "optuna.**",
    "mlflow.**",
    "dagshub.**",
    "dvclive.**",
    "typer.**",
    "rich.**",
]

# Top-level names covered by the extern policy.
_RUNTIME_TOP_LEVELS = {
    pat.rstrip(".*") for pat in _RUNTIME_EXTERN_PATTERNS if "**" in pat
} | {pat for pat in _RUNTIME_EXTERN_PATTERNS if "**" not in pat}

# Override module -> distribution name mapping for packages whose PyPI name
# differs from the import name.
_MODULE_DISTRIBUTION_OVERRIDES: dict[str, str] = {
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "yaml": "PyYAML",
    "ruamel": "ruamel.yaml",
}

_STDLIB_MODULES = sys.stdlib_module_names


@dataclass
class LoadReport:
    """Selective load result for package/state reuse."""

    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    shape_mismatched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "matched": self.matched,
            "missing": self.missing,
            "unexpected": self.unexpected,
            "excluded": self.excluded,
            "shape_mismatched": self.shape_mismatched,
        }


@dataclass
class TrainingReload:
    """Reloaded package prepared for current-codebase training."""

    pipeline: CompiledPipeline
    lightning_module: Any
    scenario: ScenarioSpec
    metadata: dict[str, Any]
    report: LoadReport


def _config_hash(config: ResolvedConfig) -> str:
    yaml_str = config.to_yaml()
    return hashlib.sha256(yaml_str.encode()).hexdigest()[:16]


def _make_json_safe(value: Any) -> Any:
    """Recursively convert common non-JSON types into plain JSON-safe values.

    Returns:
        A JSON-serializable representation of *value*.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    # Pydantic v2 models
    if hasattr(value, "model_dump"):
        return _make_json_safe(value.model_dump(mode="json"))
    # dataclasses
    if is_dataclass(value) and not isinstance(value, type):
        from dataclasses import asdict

        return _make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    return str(value)


def _artifact_metadata(
    pipeline: CompiledPipeline,
    metadata: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x_keys, y_keys = _infer_io_keys(pipeline.resolved_config)
    meta = {
        "schema_version": 2,
        "config_hash": _config_hash(pipeline.resolved_config),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "loss_keys": pipeline.loss_keys,
        "metric_keys": pipeline.metric_keys,
        "x_keys": x_keys,
        "y_keys": y_keys,
        "input_sizes": {k: list(v) for k, v in pipeline.input_sizes.items()},
        "optimizer_spec": _flatten_optimizer_spec(pipeline._optimizer_spec),
        "optimizer_spec_raw": pipeline._optimizer_spec,
        "scheduler_spec": pipeline._scheduler_spec,
        "training_state_available": bool(training_state),
    }
    if metadata:
        meta.update(metadata)
    return _make_json_safe(meta)


def _package_payload(
    pipeline: CompiledPipeline,
    metadata: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packaged_pipeline = copy.deepcopy(pipeline).cpu().eval()
    return {
        "pipeline": packaged_pipeline,
        "resolved_config": pipeline.resolved_config.model_dump(mode="json"),
        "metadata": _make_json_safe(metadata),
        "state_dict": {k: v.detach().cpu() for k, v in pipeline.state_dict().items()},
        "training_state": training_state or {},
    }


def _is_stdlib_module(module_name: str) -> bool:
    top = module_name.split(".")[0]
    return top in _STDLIB_MODULES


def _is_runtime_module(module_name: str) -> bool:
    top = module_name.split(".")[0]
    if top in _RUNTIME_TOP_LEVELS:
        return True
    if top in _STDLIB_MODULES:
        return True
    for pat in _RUNTIME_EXTERN_PATTERNS:
        if fnmatch.fnmatch(module_name, pat) or fnmatch.fnmatch(top, pat):
            return True
    return False


def _discover_pipeline_module_packages(pipeline: CompiledPipeline) -> set[str]:
    """Find top-level source packages referenced by concrete pipeline layers.

    Returns:
        Set of top-level package names that are not NexuML-owned or runtime deps.
    """
    packages: set[str] = set()
    for _stage, _name, layer in pipeline.iter_layers():
        module_name = getattr(layer.__class__, "__module__", None)
        if not module_name:
            continue
        top = module_name.split(".")[0]
        if top in ("nexuml", "nexuml_library") or _is_runtime_module(top):
            continue
        packages.add(top)
    return packages


def _apply_package_policy(
    exporter: PackageExporter,
    pipeline: CompiledPipeline,
    include_modules: list[str] | None = None,
) -> set[str]:
    """Apply the export packaging policy and return the set of externed modules.

    Order: externalize runtime deps, intern NexuML + library + discovered custom
    modules, then apply explicit includes. Deny known dev/test modules so they
    cannot be silently packaged.

    Returns:
        Set of discovered custom top-level package names.
    """
    # 1. Runtime / third-party modules stay external.
    for pattern in _RUNTIME_EXTERN_PATTERNS:
        exporter.extern(pattern)

    # 2. NexuML-owned and built-in library source modules are hermetic.
    exporter.intern("nexuml.**")
    exporter.intern("nexuml_library.**")

    # 3. Best-effort interning of concrete pipeline layer source packages.
    custom_packages = _discover_pipeline_module_packages(pipeline)
    for package in sorted(custom_packages):
        exporter.intern(f"{package}.**")

    # 4. Explicit user-provided include patterns for dynamic custom code.
    for pattern in include_modules or []:
        explicit_package = pattern.removesuffix(".**").removesuffix(".*")
        if explicit_package in custom_packages:
            continue
        if not _save_explicit_source_modules(exporter, pattern):
            exporter.intern(pattern)

    # 5. Deny accidental packaging of test/dev modules.
    exporter.deny("tests.**")
    exporter.deny("test_*.**")

    return custom_packages


def _save_explicit_source_modules(exporter: PackageExporter, pattern: str) -> bool:
    """Force-save source modules for an explicit include pattern.

    ``PackageExporter.intern()`` only affects modules discovered through pickle
    globals/source imports. Dynamic imports can stay invisible, so explicit
    includes also save importable ``.py`` modules under the requested package.

    Returns:
        True when at least one source file was saved.
    """
    package_name = pattern.removesuffix(".**").removesuffix(".*")
    if not package_name or any(ch in package_name for ch in "*?[]"):
        return False
    try:
        package = importlib.import_module(package_name)
    except Exception as exc:
        logger.warning("Could not import explicit include package %s: %s", package_name, exc)
        return False

    modules = [package]
    if hasattr(package, "__path__"):
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception as exc:
                logger.warning("Could not import explicit include module %s: %s", info.name, exc)

    saved = False
    for module in modules:
        source = getattr(module, "__file__", None)
        if source and source.endswith(".py"):
            exporter.save_source_file(module.__name__, source, dependencies=False)
            saved = True
    return saved


def _validate_package_policy(
    exporter: PackageExporter,
    custom_packages: set[str],
) -> None:
    """Ensure no NexuML-owned module leaked external and custom sources were interned.

    Raises:
        RuntimeError: If a NexuML-owned module is externalized or a custom
            source package could not be interned.
    """
    externed = set(exporter.externed_modules())
    interned = set(exporter.interned_modules())

    leaked = sorted(
        m
        for m in externed
        if m.startswith("nexuml.") or m.startswith("nexuml_library.")
    )
    if leaked:
        raise RuntimeError(
            f"NexuML-owned modules were externalized by torch.package: {leaked}. "
            "This usually means a non-source module was referenced; "
            "ensure all runtime dependencies are listed in the extern policy."
        )

    for package in sorted(custom_packages):
        package_interned = any(
            m == package or m.startswith(f"{package}.") for m in interned
        )
        package_externed = any(
            m == package or m.startswith(f"{package}.") for m in externed
        )
        if not package_interned and package_externed:
            raise RuntimeError(
                f"Custom source package '{package}' could not be interned. "
                "It may be a non-source (extension/bytecode) module or may not "
                "be importable from the filesystem. Add an explicit include "
                "pattern to export_package(include_modules=...) if the module "
                "is loaded dynamically."
            )


def _normalize_module_name(module_name: str) -> str:
    """Map a referenced module to its top-level import name.

    Returns:
        The top-level import name of *module_name*.
    """
    return module_name.split(".")[0]


def _resolve_distribution(module_name: str) -> tuple[str, str | None]:
    """Return (distribution_name, version) for an external module.

    Returns:
        Tuple of distribution name and installed version (or None if unknown).
    """
    top = _normalize_module_name(module_name)
    dist = _MODULE_DISTRIBUTION_OVERRIDES.get(top, top)
    try:
        version = importlib.metadata.version(dist)
    except Exception:
        version = None
    return dist, version


def _collect_external_dependencies(
    exporter: PackageExporter,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect actual external dependencies referenced by the packaged payload.

    Returns:
        Tuple of (structured dependency entries, requirements.txt lines).
    """
    externed = sorted(set(exporter.externed_modules()))
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    seen: set[str] = set()

    for module_name in externed:
        if _is_stdlib_module(module_name):
            continue
        top = _normalize_module_name(module_name)
        if top in ("nexuml", "nexuml_library"):
            continue
        dist, version = _resolve_distribution(module_name)
        if dist in seen:
            continue
        seen.add(dist)
        entries.append(
            {
                "module": top,
                "distribution": dist,
                "version": version,
                "specifier": f"{dist}=={version}" if version else dist,
                "reason": "extern",
            }
        )
        lines.append(f"{dist}=={version}" if version else dist)

    return entries, lines


def _load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Load and normalize Lightning checkpoint provenance for package metadata.

    Returns:
        JSON-safe dictionary of checkpoint-derived metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        return {"source": str(checkpoint_path)}

    callbacks = raw.get("callbacks") or {}
    model_checkpoint_state: dict[str, Any] | None = None
    for key, value in callbacks.items():
        if "ModelCheckpoint" in key and isinstance(value, dict):
            model_checkpoint_state = value
            break

    checkpoint_meta: dict[str, Any] = {
        "source": str(checkpoint_path),
        "epoch": raw.get("epoch"),
        "global_step": raw.get("global_step"),
        "hyper_parameters": _make_json_safe(raw.get("hyper_parameters") or {}),
        "callbacks": _make_json_safe(callbacks),
        "loops": _make_json_safe(raw.get("loops") or {}),
        "optimizer_states_count": len(raw.get("optimizer_states", []) or []),
        "scheduler_states_count": len(raw.get("lr_schedulers", []) or []),
    }

    if isinstance(model_checkpoint_state, dict):
        checkpoint_meta["best_model_path"] = model_checkpoint_state.get("best_model_path")
        checkpoint_meta["best_model_score"] = _make_json_safe(
            model_checkpoint_state.get("best_model_score")
        )
        checkpoint_meta["monitor"] = model_checkpoint_state.get("monitor")
        checkpoint_meta["mode"] = model_checkpoint_state.get("mode")

    logged_metrics = raw.get("logged_metrics") or {}
    if logged_metrics:
        checkpoint_meta["validation_metrics"] = _make_json_safe(
            {k: v for k, v in logged_metrics.items() if k.startswith("val/")}
        )

    checkpoint_meta["nexuml_eval"] = _make_json_safe(raw.get("nexuml_eval") or {})
    checkpoint_meta["nexuml_post_train"] = _make_json_safe(raw.get("nexuml_post_train") or {})

    return checkpoint_meta


def _training_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "optimizers": checkpoint.get("optimizer_states", []),
        "schedulers": checkpoint.get("lr_schedulers", []),
        "callbacks": checkpoint.get("callbacks", {}),
        "loops": checkpoint.get("loops", {}),
    }


def _extern_runtime_dependencies(exporter: PackageExporter) -> None:
    """Legacy helper: keep runtime modules external.

    Prefer :func:`_apply_package_policy` for full policy handling.
    """
    for pattern in _RUNTIME_EXTERN_PATTERNS:
        exporter.extern(pattern)


def export_package(
    pipeline: CompiledPipeline,
    path: Path,
    metadata: dict[str, Any] | None = None,
    lightning_module: Any | None = None,
    trainer: Any | None = None,
    checkpoint_path: str | Path | None = None,
    include_modules: list[str] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a trained pipeline as a rich package-backed artifact directory.

    Args:
        pipeline: Compiled pipeline to export.
        path: Destination directory.
        metadata: Optional provenance metadata merged into the artifact.
        lightning_module: Optional Lightning module for checkpoint sidecars.
        trainer: Optional Lightning trainer for training-state sidecars.
        checkpoint_path: Optional source Lightning checkpoint to preserve.
        include_modules: Optional glob patterns for additional source modules to
            intern (useful for dynamic imports invisible to torch.package).
        source_metadata: Optional metadata describing the export source (e.g.
            CLI checkpoint path). Merged into *metadata*.

    Returns:
        Path to the created export directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    metadata = {**(metadata or {}), **(source_metadata or {})}

    # Gather training state from the live trainer when available.
    training_state: dict[str, Any] = {}
    if trainer is not None:
        optimizers = trainer.optimizers if hasattr(trainer, "optimizers") else []
        lr_schedulers = (
            trainer.lr_scheduler_configs if hasattr(trainer, "lr_scheduler_configs") else []
        )
        training_state = {
            "trainer": {
                "current_epoch": int(getattr(trainer, "current_epoch", 0)),
                "global_step": int(getattr(trainer, "global_step", 0)),
            },
            "optimizers": [opt.state_dict() for opt in optimizers],
            "schedulers": [
                cfg.scheduler.state_dict() if hasattr(cfg, "scheduler") else {}
                for cfg in lr_schedulers
            ],
        }

    checkpoint_raw: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint_raw, dict):
            metadata["checkpoint"] = _load_checkpoint_metadata(checkpoint_path)
            training_state = _training_state_from_checkpoint(checkpoint_raw)
        # Preserve the original checkpoint as a sidecar.
        import shutil

        shutil.copy2(checkpoint_path, path / CHECKPOINT_SIDECAR)

    meta = _artifact_metadata(pipeline, metadata=metadata, training_state=training_state)

    torch.save(
        {k: v.detach().cpu() for k, v in pipeline.state_dict().items()}, path / "state_dict.pt"
    )
    pipeline.resolved_config.save(path / "resolved_config.yaml")
    if training_state:
        torch.save(training_state, path / "training_state.pt")

    package_path = path / PACKAGE_FILENAME
    # Save the legacy pipeline entry first so torch.package resolves the full
    # dependency graph; we then read the actual externed modules and write the
    # primary payload with the dependency manifest included.
    with PackageExporter(str(package_path)) as exporter:
        custom_packages = _apply_package_policy(exporter, pipeline, include_modules=include_modules)

        legacy_pipeline = copy.deepcopy(pipeline).cpu().eval()
        exporter.save_pickle(
            LEGACY_PACKAGE_PICKLE_PACKAGE,
            LEGACY_PACKAGE_PICKLE_NAME,
            legacy_pipeline,
        )

        external_deps, requirements_lines = _collect_external_dependencies(exporter)
        meta["external_dependencies"] = external_deps

        payload = _package_payload(pipeline, meta, training_state=training_state)
        exporter.save_pickle(PACKAGE_PICKLE_PACKAGE, PACKAGE_PICKLE_NAME, payload)

        _validate_package_policy(exporter, custom_packages)

    if requirements_lines:
        (path / REQUIREMENTS_FILENAME).write_text("\n".join(requirements_lines) + "\n")
    else:
        (path / REQUIREMENTS_FILENAME).write_text("")
    (path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    # Generate or preserve a Lightning checkpoint sidecar.
    sidecar_path = path / CHECKPOINT_SIDECAR
    if not sidecar_path.exists():
        if checkpoint_path is not None:
            # Already copied above.
            pass
        elif trainer is not None and getattr(trainer, "model", None) is not None:
            trainer.save_checkpoint(str(sidecar_path))
        elif lightning_module is not None:
            torch.save(
                {
                    "state_dict": lightning_module.state_dict(),
                    "epoch": 0,
                    "global_step": 0,
                    "hyper_parameters": _make_json_safe(
                        getattr(lightning_module, "hparams", {}) or {}
                    ),
                },
                sidecar_path,
            )

    logger.info("Exported pipeline to %s", path)
    return path


def _load_packaged_payload(package_path: Path) -> dict[str, Any]:
    importer = PackageImporter(str(package_path))
    try:
        return importer.load_pickle(PACKAGE_PICKLE_PACKAGE, PACKAGE_PICKLE_NAME)
    except Exception:
        # Legacy fallback used by older NexuFL adapters.
        legacy_pipeline = importer.load_pickle(
            LEGACY_PACKAGE_PICKLE_PACKAGE,
            LEGACY_PACKAGE_PICKLE_NAME,
        )
        resolved_config = _normalize_config(getattr(legacy_pipeline, "resolved_config", None))
        if resolved_config is not None:
            x_keys, y_keys = _infer_io_keys(resolved_config)
        else:
            x_keys, y_keys = [], []
        return {
            "pipeline": legacy_pipeline,
            "resolved_config": resolved_config,
            "metadata": {
                "loss_keys": dict(getattr(legacy_pipeline, "loss_keys", {})),
                "x_keys": x_keys,
                "y_keys": y_keys,
                "input_sizes": {
                    key: list(value)
                    for key, value in dict(getattr(legacy_pipeline, "input_sizes", {})).items()
                },
                "optimizer_spec": _flatten_optimizer_spec(
                    dict(getattr(legacy_pipeline, "_optimizer_spec", {}))
                ),
            },
            "state_dict": {k: v.detach().cpu() for k, v in legacy_pipeline.state_dict().items()},
            "training_state": {},
        }


def _infer_io_keys(config: ResolvedConfig) -> tuple[list[str], list[str]]:
    x_keys = list(config.data.input_shapes.keys()) or [config.data.feature_key]

    y_candidates: list[str] = []
    y_candidates.extend([target.key for target in config.data.targets])
    if config.data.merge_labels:
        y_candidates.extend(config.data.merge_labels.keys())

    for stage_layers in config.pipeline.stages.values():
        for layer_spec in stage_layers:
            label_key = layer_spec.params.get("label_key")
            if isinstance(label_key, str):
                y_candidates.append(label_key)

    y_keys = list(dict.fromkeys(y_candidates))
    if not y_keys:
        # NexuFL TrainerV1 adapters expect at least one target key.
        y_keys = ["target"]
    return x_keys, y_keys


def _flatten_optimizer_spec(optimizer_spec: dict[str, Any]) -> dict[str, Any]:
    if not optimizer_spec:
        return {"type": "adam", "lr": 1e-3}

    opt_type = str(optimizer_spec.get("type", "adam")).rsplit(".", 1)[-1].lower()
    params = dict(optimizer_spec.get("params", {}))
    flattened = {"type": opt_type}

    if "lr" in params:
        flattened["lr"] = params["lr"]
    if "weight_decay" in params:
        flattened["weight_decay"] = params["weight_decay"]
    if "momentum" in params:
        flattened["momentum"] = params["momentum"]

    return flattened


def _normalize_config(config: Any) -> ResolvedConfig | None:
    if config is None:
        return None
    if isinstance(config, ResolvedConfig):
        return config
    if hasattr(config, "model_dump"):
        return ResolvedConfig.model_validate(config.model_dump(mode="json"))
    if isinstance(config, dict):
        return ResolvedConfig.model_validate(config)
    raise TypeError(f"Unsupported packaged config type: {type(config)!r}")


def _load_artifact(
    source: Path,
) -> tuple[dict[str, torch.Tensor], ResolvedConfig | None, dict[str, Any]]:
    source = Path(source)
    if source.is_dir():
        package_path = source / PACKAGE_FILENAME
        if package_path.exists():
            payload = _load_packaged_payload(package_path)
            config = _normalize_config(payload.get("resolved_config"))
            metadata = dict(payload.get("metadata", {}))
            return payload["state_dict"], config, metadata
        state_dict = torch.load(source / "state_dict.pt", map_location="cpu", weights_only=True)
        config = ResolvedConfig.load(source / "resolved_config.yaml")
        metadata = json.loads((source / "metadata.json").read_text())
        return state_dict, config, metadata

    if source.suffix == ".safetensors":
        state_dict = load_safetensors_file(str(source))
        manifest_path = source.with_suffix(".json")
        metadata = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        config = None
        return state_dict, config, metadata

    if source.suffix == ".package":
        payload = _load_packaged_payload(source)
        config = _normalize_config(payload.get("resolved_config"))
        metadata = dict(payload.get("metadata", {}))
        return payload["state_dict"], config, metadata

    raw = torch.load(source, map_location="cpu", weights_only=True)
    # Lightning .ckpt files wrap model weights: {"state_dict": {...}, "epoch": N, ...}
    # The pipeline is stored as self.pipeline, so keys carry a "pipeline." prefix.
    # Strip it so the keys match the exported-package convention.
    if isinstance(raw, dict) and "state_dict" in raw and "epoch" in raw:
        inner = raw["state_dict"]
        prefix = "pipeline."
        state_dict = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in inner.items()
        }
        return state_dict, None, {}
    return raw, None, {}


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def load_weights(
    pipeline: CompiledPipeline,
    source: str | Path,
    checkpoint: CheckpointLoadSpec | None = None,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    allow_missing: bool | None = None,
    allow_shape_mismatch: bool | None = None,
    freeze_loaded: bool | None = None,
) -> LoadReport:
    """Selectively load weights into an already-compiled pipeline.

    Returns:
        LoadReport summarising matched, missing, and excluded keys.

    Raises:
        ValueError: On shape mismatch or unexpected missing keys when not allowed.
    """
    checkpoint = checkpoint or CheckpointLoadSpec()
    include = include if include is not None else list(checkpoint.include)
    exclude = exclude if exclude is not None else list(checkpoint.exclude)
    allow_missing = checkpoint.allow_missing if allow_missing is None else allow_missing
    allow_shape_mismatch = (
        checkpoint.allow_shape_mismatch if allow_shape_mismatch is None else allow_shape_mismatch
    )
    freeze_loaded = checkpoint.freeze_loaded if freeze_loaded is None else freeze_loaded

    state_dict, _, _ = _load_artifact(Path(source))
    target_state = pipeline.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    report = LoadReport()

    for key, value in state_dict.items():
        if include and not _matches_any(key, include):
            report.excluded.append(key)
            continue
        if exclude and _matches_any(key, exclude):
            report.excluded.append(key)
            continue
        if key not in target_state:
            report.unexpected.append(key)
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            report.shape_mismatched.append(key)
            if allow_shape_mismatch:
                continue
            raise ValueError(
                f"Shape mismatch for {key}: source={tuple(value.shape)} "
                f"target={tuple(target_state[key].shape)}"
            )
        filtered_state[key] = value
        report.matched.append(key)

    report.missing = [key for key in target_state.keys() if key not in filtered_state]
    if report.missing and not allow_missing:
        raise ValueError(f"Missing keys after selective load: {report.missing}")

    pipeline.load_state_dict(filtered_state, strict=False)

    if freeze_loaded:
        matched_prefixes = tuple(report.matched)
        for name, param in pipeline.named_parameters():
            if name in matched_prefixes:
                param.requires_grad_(False)

    return report


def load_package(
    path: Path,
    registry: LayerRegistry | None = None,
) -> tuple[CompiledPipeline, ResolvedConfig, dict[str, Any]]:
    """Reload an exported pipeline into the current codebase.

    Returns:
        Tuple of ``(pipeline, resolved_config, metadata)``.

    Raises:
        ValueError: If no scenario config is found in the artifact.
    """
    if registry is None:
        registry = get_registry()

    state_dict, config, metadata = _load_artifact(Path(path))
    if config is None:
        raise ValueError(f"Cannot reconstruct pipeline from {path}: no scenario config found.")
    current_hash = _config_hash(config)
    saved_hash = metadata.get("config_hash", "")
    if current_hash != saved_hash:
        logger.warning(
            "Config hash mismatch during load: saved=%s current=%s",
            saved_hash,
            current_hash,
        )

    scenario = config.to_scenario()
    pipeline = compile_pipeline(scenario, registry)
    pipeline.load_state_dict(state_dict, strict=False)
    return pipeline, config, metadata


def load_inference_package(path: Path) -> tuple[CompiledPipeline, ResolvedConfig, dict[str, Any]]:
    """Load the packaged pipeline object directly from the torch.package artifact.

    Returns:
        Tuple of ``(pipeline, resolved_config, metadata)``.

    Raises:
        FileNotFoundError: If no package artifact exists at *path*.
    """
    path = Path(path)
    package_path = path / PACKAGE_FILENAME if path.is_dir() else path
    if not package_path.exists():
        raise FileNotFoundError(f"No package artifact found at {package_path}")

    payload = _load_packaged_payload(package_path)
    pipeline = payload["pipeline"]
    config = _normalize_config(payload["resolved_config"])
    pipeline.resolved_config = config
    metadata = dict(payload.get("metadata", {}))
    assert config is not None, "Config must not be None for inference package"
    return pipeline, config, metadata


def load_package_for_training(
    path: Path,
    scenario: ScenarioSpec | None = None,
    registry: LayerRegistry | None = None,
    checkpoint: CheckpointLoadSpec | None = None,
) -> TrainingReload:
    """Reload a package into the current codebase for resume or fine-tuning.

    Returns:
        TrainingReload with pipeline, lightning module, scenario, and load report.

    Raises:
        ValueError: If no scenario is provided and the artifact has no packaged config.
    """
    if registry is None:
        registry = get_registry()

    state_dict, config, metadata = _load_artifact(Path(path))
    scenario = scenario or (config.to_scenario() if config is not None else None)
    if scenario is None:
        raise ValueError("A scenario must be provided when the artifact has no packaged config.")

    pipeline = compile_pipeline(scenario, registry)
    report = load_weights(
        pipeline,
        path,
        checkpoint=checkpoint,
    )

    from nexuml.training.lightning import NexuLightningModule

    lightning_module = NexuLightningModule(pipeline)
    return TrainingReload(
        pipeline=pipeline,
        lightning_module=lightning_module,
        scenario=scenario,
        metadata=metadata,
        report=report,
    )


def export_safetensors(
    pipeline: CompiledPipeline,
    path: Path,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export pipeline weights as SafeTensors plus a JSON manifest.

    Returns:
        Path to the created ``.safetensors`` file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    include = include or []
    exclude = exclude or []

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in pipeline.state_dict().items():
        if include and not _matches_any(key, include):
            continue
        if exclude and _matches_any(key, exclude):
            continue
        state_dict[key] = value.detach().cpu()

    save_safetensors_file(state_dict, str(path))
    manifest = _artifact_metadata(pipeline, metadata=metadata)
    manifest["included_keys"] = sorted(state_dict.keys())
    manifest["format"] = "safetensors"
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, default=str))
    return path


class _OnnxWrapper(torch.nn.Module):
    def __init__(self, pipeline: CompiledPipeline, input_key: str, output_key: str):
        super().__init__()
        self.pipeline = copy.deepcopy(pipeline).cpu().eval()
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        td = TensorDict({self.input_key: x}, batch_size=[x.shape[0]])
        x_out, _ = self.pipeline(td, None)
        return x_out[self.output_key]


def export_onnx(
    pipeline: CompiledPipeline,
    path: Path,
    input_key: str | None = None,
    output_key: str = "reconstructed",
    opset_version: int = 18,
) -> Path:
    """Export an inference-only ONNX graph for single-input pipelines.

    Returns:
        Path to the created ``.onnx`` file.

    Raises:
        ImportError: If the ``onnxscript`` package is not installed.
    """
    try:
        import onnxscript  # noqa: F401
    except ImportError as exc:
        raise ImportError("ONNX export requires the 'onnxscript' package.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = pipeline.resolved_config
    input_key = input_key or config.data.feature_key
    input_shape = tuple(
        config.data.input_shapes.get(input_key) or config.data.params.get("feature_shape", (128,))
    )
    dummy = torch.randn(1, *input_shape)
    wrapper = _OnnxWrapper(pipeline, input_key=input_key, output_key=output_key)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=[input_key],
        output_names=[output_key],
        opset_version=opset_version,
        dynamic_axes={input_key: {0: "batch"}, output_key: {0: "batch"}},
    )
    return path


def infer(
    pipeline: CompiledPipeline,
    x: TensorDict,
    y: TensorDict | None = None,
) -> TensorDict:
    """Run inference on a pipeline (eval mode, no grad).

    Returns:
        Output TensorDict from the pipeline.
    """
    pipeline.eval()
    with torch.no_grad():
        x_out, _ = pipeline(x, y)
    return x_out
