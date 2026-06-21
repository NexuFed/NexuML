"""Export, reload, and selective checkpoint loading for trained pipelines."""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tensordict import TensorDict
from torch.package import PackageExporter, PackageImporter

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
    return meta


def _package_payload(
    pipeline: CompiledPipeline,
    metadata: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packaged_pipeline = copy.deepcopy(pipeline).cpu().eval()
    return {
        "pipeline": packaged_pipeline,
        "resolved_config": pipeline.resolved_config.model_dump(mode="json"),
        "metadata": metadata,
        "state_dict": {k: v.detach().cpu() for k, v in pipeline.state_dict().items()},
        "training_state": training_state or {},
    }


def _extern_runtime_dependencies(exporter: PackageExporter) -> None:
    patterns = [
        "io",
        "sys",
        "json",
        "pathlib",
        "datetime",
        "collections",
        "typing",
        "fnmatch",
        "torch.**",
        "tensordict.**",
        "timm.**",
        "numpy.**",
        "librosa.**",
        "torchaudio.**",
        "torchmetrics.**",
        "torchvision.**",
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
    for pattern in patterns:
        exporter.extern(pattern)


def export_package(
    pipeline: CompiledPipeline,
    path: Path,
    metadata: dict[str, Any] | None = None,
    lightning_module: Any | None = None,
    trainer: Any | None = None,
) -> Path:
    """Export a trained pipeline as a rich package-backed artifact directory.

    Returns:
        Path to the created export directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

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

    meta = _artifact_metadata(pipeline, metadata=metadata, training_state=training_state)

    torch.save(
        {k: v.detach().cpu() for k, v in pipeline.state_dict().items()}, path / "state_dict.pt"
    )
    pipeline.resolved_config.save(path / "resolved_config.yaml")
    (path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    if training_state:
        torch.save(training_state, path / "training_state.pt")

    package_path = path / PACKAGE_FILENAME
    payload = _package_payload(pipeline, meta, training_state=training_state)
    with PackageExporter(str(package_path)) as exporter:
        exporter.intern("nexuml.**")
        exporter.extern("nexuml_library.**")
        _extern_runtime_dependencies(exporter)
        exporter.save_pickle(PACKAGE_PICKLE_PACKAGE, PACKAGE_PICKLE_NAME, payload)
        # Backward-compatible entrypoint expected by NexuFL package loader.
        exporter.save_pickle(
            LEGACY_PACKAGE_PICKLE_PACKAGE,
            LEGACY_PACKAGE_PICKLE_NAME,
            payload["pipeline"],
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
