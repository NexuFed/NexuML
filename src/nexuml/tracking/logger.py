"""Logger factory for NexuML experiment tracking."""

from __future__ import annotations

import logging
import importlib
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from nexuml.core.types import LoggingSpec

from nexuml.core.log_paths import resolve_logs_file_uri, resolve_logs_root, resolve_logs_root_str

logger = logging.getLogger(__name__)
_UNSUPPORTED_ARTIFACT_LOGGERS: set[str] = set()

# ANSI colour helpers (disabled when not a TTY or when NO_COLOR is set)
_USE_COLOR = os.isatty(2) and not os.environ.get("NO_COLOR")


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _USE_COLOR else text


def _cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m" if _USE_COLOR else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _USE_COLOR else text


def get_temp_artifact_root() -> Path:
    """Return the directory used for temporary artifact staging."""
    root = resolve_logs_root(os.environ.get("NEXUML_TEMP_ARTIFACT_DIR", "/tmp/mlruns"))
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def staged_artifact_path(
    artifact_name: str,
    *,
    prefix: str = "nexuml_artifact_",
) -> Iterator[Path]:
    """Yield a temporary filesystem path for building an artifact before logging it."""
    with tempfile.TemporaryDirectory(
        prefix=prefix,
        dir=get_temp_artifact_root(),
    ) as tmp_dir:
        yield Path(tmp_dir) / Path(artifact_name).name


def _normalize_mlflow_tracking_uri(tracking_uri: str) -> str:
    """Normalize local MLflow file-store URIs.

    Returns:
        Normalized tracking URI string with relative paths resolved to absolute.
    """
    tracking_uri = resolve_logs_file_uri(tracking_uri)
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_path = tracking_uri[len("sqlite:///") :]
        if db_path and not db_path.startswith("/"):
            resolved = resolve_logs_root(db_path)
            if resolved.suffix == "":
                resolved = resolved.with_suffix(".db")
            return f"sqlite:///{resolved.resolve()}"
    if tracking_uri.startswith("file://") and not tracking_uri.startswith("file:///"):
        local_path = tracking_uri[len("file://") :]
        if local_path and not local_path.startswith("/"):
            return resolve_logs_root(local_path).resolve().as_uri()
    if tracking_uri.startswith("sqlite://") and not tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri[len("sqlite://") :]
        if db_path and not db_path.startswith("/"):
            resolved = resolve_logs_root(db_path)
            if resolved.suffix == "":
                resolved = resolved.with_suffix(".db")
            return f"sqlite:///{resolved.resolve()}"
    if tracking_uri.startswith("sqlite:") and not tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri[len("sqlite:") :]
        if db_path.startswith("//"):
            return tracking_uri
        if db_path:
            resolved = resolve_logs_root(db_path)
            if resolved.suffix == "":
                resolved = resolved.with_suffix(".db")
            return f"sqlite:///{resolved.resolve()}"
    return tracking_uri


def _configure_mlflow_tracking_uri(tracking_uri: str) -> str:
    """Configure MLflow tracking for local, remote, or Dagshub URIs.

    Returns:
        The normalized tracking URI that was configured with MLflow.
    """
    normalized_tracking_uri = _normalize_mlflow_tracking_uri(tracking_uri)

    import mlflow

    if normalized_tracking_uri.startswith("https://dagshub.com/"):
        dagshub = importlib.import_module("dagshub")

        dagshub_owner, dagshub_repo = normalized_tracking_uri[len("https://dagshub.com/") :].split(
            "/"
        )[:2]
        dagshub.init(
            repo_owner=dagshub_owner,
            repo_name=dagshub_repo,
            mlflow=True,
            dvc=False,
        )
        return normalized_tracking_uri

    mlflow.set_tracking_uri(normalized_tracking_uri)
    return normalized_tracking_uri


def _resolve_mlflow_artifact_location(
    tracking_uri: str,
    artifact_location: str | None,
) -> str | None:
    """Resolve the final MLflow artifact store location.

    Returns:
        Resolved artifact store URI, or ``None`` if no artifact location can
        be determined.
    """
    if artifact_location:
        if artifact_location.startswith("file:"):
            return resolve_logs_file_uri(artifact_location)
        if "://" in artifact_location:
            return artifact_location
        return resolve_logs_root(artifact_location).resolve().as_uri()

    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri[len("sqlite:///") :]).expanduser().resolve()
        artifact_root = db_path.parent / "mlflow_artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        return artifact_root.as_uri()

    if tracking_uri.startswith("file:///"):
        store_path = Path(tracking_uri[len("file:///") :]).expanduser().resolve()
        artifact_root = store_path / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        return artifact_root.as_uri()

    return None


def _get_or_create_mlflow_experiment_id(
    client: Any,
    experiment_name: str,
    artifact_location: str | None,
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None and experiment.lifecycle_stage != "deleted":
        if artifact_location is not None:
            existing = (experiment.artifact_location or "").rstrip("/")
            expected = artifact_location.rstrip("/")
            if existing != expected:
                logger.warning(
                    "MLflow experiment '%s' has artifact_location '%s' but expected '%s'. "
                    "New runs will land in the old location. To fix, delete the experiment "
                    "from the database (or remove the .db file) and re-run.",
                    experiment_name,
                    existing,
                    expected,
                )
        return experiment.experiment_id
    if artifact_location is not None:
        return client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    return client.create_experiment(experiment_name)


def _prepare_mlflow_run(
    tracking_uri: str,
    artifact_location: str | None,
    experiment_name: str,
    run_name: str,
    run_id: str | None,
    parent_run_id: str | None,
    auto_nested_runs: bool,
    tags: dict[str, Any] | None,
) -> tuple[str, str]:
    """Create or reuse an MLflow run and return the run id for Lightning.

    Returns:
        Tuple of ``(run_id, run_name)`` for the created or reused MLflow run.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

    client = MlflowClient(tracking_uri=tracking_uri)

    if run_id is not None:
        return run_id, run_name

    experiment_id = _get_or_create_mlflow_experiment_id(client, experiment_name, artifact_location)
    resolved_tags = dict(tags or {})
    resolved_run_name = run_name
    if parent_run_id is None and auto_nested_runs:
        active_run = mlflow.active_run()
        if active_run is not None:
            parent_run_id = active_run.info.run_id
        else:
            parent_run_id = _find_or_create_auto_parent_run(
                client=client,
                experiment_id=experiment_id,
                parent_name=run_name,
                tags=resolved_tags,
            )
            resolved_run_name = (
                f"{run_name}_v{_next_child_version(client, experiment_id, parent_run_id, run_name)}"
            )
    resolved_tags.setdefault(MLFLOW_RUN_NAME, resolved_run_name)
    if parent_run_id is not None:
        resolved_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id

    run = client.create_run(
        experiment_id=experiment_id,
        tags={key: str(value) for key, value in resolved_tags.items()},
    )
    return run.info.run_id, resolved_run_name


def create_loggers(
    logging_spec: "LoggingSpec | None",
    run_name: str | None = None,
) -> list[Any]:
    """Build a list of Lightning loggers from a LoggingSpec.

    Returns an empty list (disables logging) if spec is None.
    Each backend is imported lazily so missing optional deps don't crash the import.

    Args:
        logging_spec: LoggingSpec instance or None.
        run_name: Override run name; falls back to logging_spec.run_name.

    Returns:
        List of Lightning logger instances ready for Trainer(logger=...).
    """
    if logging_spec is None:
        return []

    os.environ["NEXUML_TEMP_ARTIFACT_DIR"] = resolve_logs_root_str(logging_spec.temp_artifact_dir)
    run_name = run_name or logging_spec.run_name or "run"
    experiment_name = logging_spec.experiment_name
    loggers: list[Any] = []

    # TensorBoard
    if logging_spec.tensorboard is not None:
        tb = _create_tensorboard_logger(
            log_dir=logging_spec.tensorboard.log_dir,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        if tb is not None:
            loggers.append(tb)

    # DVCLive
    if logging_spec.dvclive is not None:
        dvc = _create_dvclive_logger(logging_spec.dvclive.dir)
        if dvc is not None:
            loggers.append(dvc)

    # MLflow
    if logging_spec.mlflow is not None:
        ml = _create_mlflow_logger(
            tracking_uri=logging_spec.mlflow.tracking_uri,
            artifact_location=logging_spec.mlflow.artifact_location,
            experiment_name=logging_spec.mlflow.experiment_name or experiment_name,
            run_name=run_name,
            log_model=logging_spec.mlflow.log_model,
            log_system_metrics=logging_spec.log_system_metrics,
            run_id=logging_spec.mlflow.run_id,
            parent_run_id=logging_spec.mlflow.parent_run_id,
            auto_nested_runs=logging_spec.mlflow.auto_nested_runs,
            tags=logging_spec.mlflow.tags,
            synchronous=logging_spec.mlflow.synchronous,
        )
        if ml is not None:
            loggers.append(ml)

    return loggers


def _create_tensorboard_logger(
    log_dir: str,
    experiment_name: str,
    run_name: str,
) -> Any | None:
    try:
        import lightning as L
        import lightning.pytorch.loggers  # noqa: F401

        log_dir_path = resolve_logs_root(log_dir)
        version = _next_version(log_dir_path / experiment_name, run_name)
        tb_logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=str(log_dir_path),
            name=experiment_name,
            version=f"{run_name}_v{version}",
            default_hp_metric=False,
        )
        logger.info(
            f"TensorBoard: {tb_logger.log_dir}\n"
            f"  Run: tensorboard --logdir {log_dir_path / experiment_name} --port 6007"
        )
        return tb_logger
    except Exception as e:
        logger.warning(f"Could not create TensorBoard logger: {e}")
        return None


def _create_dvclive_logger(log_dir: str) -> Any | None:
    try:
        DVCLiveLogger = importlib.import_module("dvclive.lightning").DVCLiveLogger

        resolved_log_dir = resolve_logs_root_str(log_dir)
        dvc_logger = DVCLiveLogger(dir=resolved_log_dir)
        logger.info(f"DVCLive: {resolved_log_dir}")
        return dvc_logger
    except ImportError:
        logger.warning("dvclive not installed. Skipping DVCLive logger.")
        return None
    except Exception as e:
        logger.warning(f"Could not create DVCLive logger: {e}")
        return None


def _create_mlflow_logger(
    tracking_uri: str,
    artifact_location: str | None,
    experiment_name: str,
    run_name: str,
    log_model: bool,
    log_system_metrics: bool,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    auto_nested_runs: bool = True,
    tags: dict[str, Any] | None = None,
    synchronous: bool | None = None,
) -> Any | None:
    try:
        import lightning as L
        import lightning.pytorch.loggers  # noqa: F401
        import mlflow

        tracking_uri = _configure_mlflow_tracking_uri(tracking_uri)
        artifact_location = _resolve_mlflow_artifact_location(tracking_uri, artifact_location)
        resolved_tags = _augment_mlflow_tags(tags)

        if log_system_metrics:
            try:
                getattr(mlflow, "enable_system_metrics_logging")()
            except Exception:
                pass

        resolved_run_id, resolved_run_name = _prepare_mlflow_run(
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            experiment_name=experiment_name,
            run_name=run_name,
            run_id=run_id,
            parent_run_id=parent_run_id,
            auto_nested_runs=auto_nested_runs,
            tags=resolved_tags,
        )

        ml_logger = L.pytorch.loggers.MLFlowLogger(
            experiment_name=experiment_name,
            run_name=resolved_run_name,
            tracking_uri=tracking_uri,
            log_model=log_model,
            tags=resolved_tags,
            run_id=resolved_run_id,
            synchronous=synchronous,
        )
        logger.info(
            f"MLflow: {tracking_uri} | experiment={experiment_name} | run={resolved_run_name}"
        )
        return ml_logger
    except ImportError:
        logger.warning("mlflow not installed. Skipping MLflow logger.")
        return None
    except Exception as e:
        logger.warning(f"Could not create MLflow logger: {e}")
        if "mlflow db upgrade" in str(e):
            logger.warning(f"Upgrade MLflow with: mlflow db upgrade {tracking_uri}")
        return None


def _next_version(log_dir: Path, run_name: str) -> int:
    """Return the next unused version number for a run name prefix."""
    if not log_dir.exists():
        return 0
    existing = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)]
    if not existing:
        return 0
    versions = []
    for d in existing:
        parts = d.name.rsplit("_v", 1)
        if len(parts) == 2:
            try:
                versions.append(int(parts[1].split("_")[0]))
            except ValueError:
                pass
    return max(versions, default=-1) + 1


def iter_loggers(logger_obj: Any) -> list[Any]:
    """Normalize Lightning logger containers into a flat list.

    Returns:
        Flat list of non-None logger instances.
    """
    if logger_obj is None or logger_obj is False:
        return []
    if isinstance(logger_obj, (list, tuple, set)):
        return [logger for logger in logger_obj if logger is not None]

    nested = getattr(logger_obj, "loggers", None)
    if nested is not None:
        return [logger for logger in nested if logger is not None]

    return [logger_obj]


def log_artifact(
    logger_obj: Any,
    source_path: str | Path,
    artifact_path: str | None = None,
) -> None:
    """Log or copy a file artifact to all configured logger backends.

    Raises:
        FileNotFoundError: If *source_path* does not exist.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Artifact source does not exist: {source}")

    for backend in iter_loggers(logger_obj):
        experiment = getattr(backend, "experiment", None)
        run_id = getattr(backend, "run_id", None)
        if experiment is not None and run_id is not None and hasattr(experiment, "log_artifact"):
            experiment.log_artifact(run_id, str(source), artifact_path)
            continue

        destination_root = _local_artifact_root(backend)
        if destination_root is not None:
            destination = destination_root
            if artifact_path:
                destination = destination / artifact_path
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / source.name)
            continue

        backend_name = type(backend).__name__
        if backend_name not in _UNSUPPORTED_ARTIFACT_LOGGERS:
            logger.info(
                "Skipping artifact logging for unsupported logger backend '%s' (artifact=%s).",
                backend_name,
                source.name,
            )
            _UNSUPPORTED_ARTIFACT_LOGGERS.add(backend_name)


def log_text_artifact(
    logger_obj: Any,
    text: str,
    artifact_name: str,
    artifact_path: str | None = None,
) -> None:
    """Write text content to a temporary file and log it as an artifact."""
    target_name = str(Path(artifact_name).with_suffix(Path(artifact_name).suffix or ".txt"))
    with staged_artifact_path(target_name, prefix="nexuml_text_") as final_path:
        final_path.write_text(text, encoding="utf-8")
        log_artifact(logger_obj, final_path, artifact_path=artifact_path)


def _to_hwc_uint8(image: Any) -> Any:
    """Convert an image in any common format to a HWC uint8 numpy array.

    Returns:
        HWC uint8 numpy array.
    """
    import numpy as np

    # torch tensor: assume CHW layout, convert to HWC
    if hasattr(image, "detach"):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr.transpose(1, 2, 0)  # CHW → HWC
    elif hasattr(image, "convert"):
        # PIL Image
        arr = np.asarray(image)
    else:
        arr = np.asarray(image)

    if arr.dtype != np.uint8:
        # float image: scale [0, 1] → [0, 255]
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    return arr


def log_image(
    logger_obj: Any,
    tag: str,
    image: Any,
    step: int | None = None,
    artifact_path: str | None = None,
) -> None:
    """Log an image to all configured logger backends.

    Args:
        logger_obj: Lightning logger or list of loggers.
        tag: Name/tag for the image (e.g. ``"val/reconstruction"``).
            Slashes are used as subdirectory separators for file-based backends.
        image: Image data — numpy array (HWC, uint8 or float [0,1]),
            PIL Image, or torch tensor (CHW, uint8 or float [0,1]).
        step: Global step for time-series backends (TensorBoard).
        artifact_path: Optional subdirectory within the run's artifact store
            (MLflow / file-based backends only).
    """
    _UNSUPPORTED_IMAGE_LOGGERS: set[str] = set()
    file_backends: list[Any] = []

    for backend in iter_loggers(logger_obj):
        experiment = getattr(backend, "experiment", None)
        run_id = getattr(backend, "run_id", None)

        # TensorBoard — native add_image
        if experiment is not None and hasattr(experiment, "add_image"):
            hwc = _to_hwc_uint8(image)
            experiment.add_image(tag, hwc, global_step=step, dataformats="HWC")
            continue

        # MLflow — log_artifact via temp PNG
        if experiment is not None and run_id is not None and hasattr(experiment, "log_artifact"):
            artifact_name = f"{tag}.png".replace("/", "_")
            with staged_artifact_path(artifact_name, prefix="nexuml_img_") as tmp_path:
                _save_image_to_path(_to_hwc_uint8(image), tmp_path)
                experiment.log_artifact(run_id, str(tmp_path), artifact_path)
            continue

        file_backends.append(backend)

    # File-copy fallback (DVCLive, etc.)
    if file_backends:
        artifact_name = f"{tag}.png".replace("/", "_")
        artifact_dir = artifact_path or (str(Path(tag).parent) if "/" in tag else None)
        with staged_artifact_path(artifact_name, prefix="nexuml_img_") as tmp_path:
            _save_image_to_path(_to_hwc_uint8(image), tmp_path)
            for backend in file_backends:
                backend_name = type(backend).__name__
                destination_root = _local_artifact_root(backend)
                if destination_root is not None:
                    dest = destination_root / artifact_dir if artifact_dir else destination_root
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_path, dest / tmp_path.name)
                elif backend_name not in _UNSUPPORTED_IMAGE_LOGGERS:
                    logger.info(
                        "Skipping image logging for unsupported logger backend '%s' (tag=%s).",
                        backend_name,
                        tag,
                    )
                    _UNSUPPORTED_IMAGE_LOGGERS.add(backend_name)


def _save_image_to_path(hwc_uint8: Any, path: Path) -> None:
    """Save a HWC uint8 numpy array as a PNG file."""
    try:
        from PIL import Image as PILImage

        PILImage.fromarray(hwc_uint8).save(path)
        return
    except ImportError:
        pass
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(hwc_uint8.shape[1] / 100, hwc_uint8.shape[0] / 100), dpi=100
        )
        ax.imshow(hwc_uint8)
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not save image to %s: %s", path, exc)


def _local_artifact_root(logger_obj: Any) -> Path | None:
    backend_name = type(logger_obj).__name__
    if "TensorBoard" in backend_name:
        log_dir = getattr(logger_obj, "log_dir", None)
        return Path(log_dir) / "artifacts" if log_dir is not None else None
    if "DVCLive" in backend_name:
        base_dir = (
            getattr(logger_obj, "log_dir", None)
            or getattr(logger_obj, "dir", None)
            or getattr(logger_obj, "_path", None)
        )
        return Path(base_dir) / "artifacts" if base_dir is not None else None
    return None


def _augment_mlflow_tags(tags: dict[str, Any] | None) -> dict[str, Any]:
    resolved = dict(tags or {})
    for key, value in _collect_git_metadata().items():
        resolved.setdefault(key, value)
    return resolved


def _collect_git_metadata() -> dict[str, str]:
    """Return git commit metadata when the current workspace is a git repo."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return {}
    return {
        "git.commit": commit,
        "git.dirty": "true" if bool(dirty) else "false",
    }


def _find_or_create_auto_parent_run(
    client: Any,
    experiment_id: str,
    parent_name: str,
    tags: dict[str, Any],
) -> str:
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

    for run in client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    ):
        if (
            run.data.tags.get("nexuml.auto_parent") == "true"
            and run.data.tags.get(MLFLOW_RUN_NAME) == parent_name
        ):
            return run.info.run_id

    parent_tags = {key: str(value) for key, value in tags.items()}
    parent_tags["nexuml.auto_parent"] = "true"
    parent_tags[MLFLOW_RUN_NAME] = parent_name
    run = client.create_run(experiment_id=experiment_id, tags=parent_tags)
    client.set_terminated(run.info.run_id, status="FINISHED")
    return run.info.run_id


def _next_child_version(
    client: Any,
    experiment_id: str,
    parent_run_id: str,
    base_run_name: str,
) -> int:
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

    pattern = re.compile(rf"^{re.escape(base_run_name)}_v(\d+)$")
    versions: list[int] = []
    for run in client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    ):
        if run.data.tags.get(MLFLOW_PARENT_RUN_ID) != parent_run_id:
            continue
        match = pattern.match(run.data.tags.get(MLFLOW_RUN_NAME, ""))
        if match is not None:
            versions.append(int(match.group(1)))
    return max(versions, default=-1) + 1


# ---------------------------------------------------------------------------
# User-facing service info banner
# ---------------------------------------------------------------------------


def print_service_info(
    trainer_loggers: list[Any],
    scenario_name: str,
    log_dir: str | Path = ".experiments",
    logging_spec: Any | None = None,
    tuning_spec: Any | None = None,
    data_backend: str | None = None,
    training_backend: str | None = None,
) -> None:
    """Print actionable commands for all active monitoring services.

    Called once at the start of training so the user knows how to
    inspect metrics, tune hyperparameters, etc.
    """
    lines: list[str] = []
    sep = _dim("-" * 60)

    lines.append("")
    lines.append(sep)
    lines.append(_bold(f"  NexuML  |  {scenario_name}"))
    lines.append(sep)

    logs_root = os.environ.get("NEXUML_LOGS_ROOT")
    if logs_root:
        lines.append(f"  Logs root    {_dim(str(Path(logs_root).expanduser()))}")
        lines.append(sep)

    # -- Backends --
    if data_backend is not None or training_backend is not None:
        backend_parts = []
        if training_backend is not None:
            backend_parts.append(f"train: {training_backend}")
        if data_backend is not None:
            backend_parts.append(f"data: {data_backend}")
        lines.append(f"  Backends     {_dim(', '.join(backend_parts))}")
        lines.append(sep)

    has_any = False

    # -- TensorBoard --
    for lg in trainer_loggers:
        cls_name = type(lg).__name__
        if "TensorBoard" in cls_name:
            has_any = True
            log_path = getattr(lg, "log_dir", None) or str(log_dir)
            parent = str(Path(log_path).parent)
            lines.append(f"  TensorBoard  {_dim(log_path)}")
            lines.append(
                f"    {_cyan(f'tensorboard --logdir {parent} --port 6007 --host 0.0.0.0')}"
            )

    # -- MLflow --
    for lg in trainer_loggers:
        cls_name = type(lg).__name__
        if "MLFlow" in cls_name or "MLflow" in cls_name:
            has_any = True
            tracking_uri = getattr(lg, "_tracking_uri", None) or "mlruns"
            lines.append(f"  MLflow       {_dim(tracking_uri)}")
            mlflow_cmd = f"mlflow ui --backend-store-uri {tracking_uri} --port 8080 --host 0.0.0.0"
            lines.append(f"    {_cyan(mlflow_cmd)}")

    # -- DVCLive --
    for lg in trainer_loggers:
        cls_name = type(lg).__name__
        if "DVCLive" in cls_name:
            has_any = True
            dvc_dir = getattr(lg, "dir", None) or getattr(lg, "_path", None) or "dvclive"
            lines.append(f"  DVCLive     {_dim(str(dvc_dir))}")
            lines.append(f"    {_cyan('dvc exp show --no-pager')}")
            lines.append(f"    {_cyan('dvc plots show --open')}")
            lines.append(f"    {_cyan('dvc plots diff')}")
            lines.append(
                f"    {_cyan('dvc studio login')} {_dim('# optional web UI via DVC Studio')}"
            )
            lines.append(
                f"    {_cyan('dvc exp push origin')}"
                f" {_dim('# share experiments to Studio / remote')}"
            )

    if (
        logging_spec is not None
        and getattr(logging_spec, "dvclive", None) is not None
        and not any("DVCLive" in type(lg).__name__ for lg in trainer_loggers)
    ):
        has_any = True
        lines.append(
            f"  DVCLive     {_dim(resolve_logs_root_str(logging_spec.dvclive.dir))}"
            f" {_dim('(configured but unavailable)')}"
        )
        lines.append(f"    {_dim('Install dvclive to enable local DVC experiment logging.')}")
        lines.append(
            f"    {_dim('Once enabled, inspect runs with dvc exp show and dvc plots show --open.')}"
        )

    # -- Optuna --
    if tuning_spec is not None:
        has_any = True
        storage_path = resolve_logs_root(tuning_spec.storage).with_suffix(".db")
        storage_url = f"sqlite:///{storage_path}"
        lines.append(f"  Optuna       {_dim(str(storage_path))}")
        lines.append(f"    {_cyan(f'optuna-dashboard {storage_url}')}")

    if not has_any:
        lines.append(
            _dim(
                "  No logging backends configured. "
                "Set scenario.logging to enable TensorBoard/MLflow."
            )
        )

    lines.append(sep)
    lines.append("")

    msg = "\n".join(lines)
    # Log at WARNING level so it's visible regardless of log config
    logger.warning(msg)
