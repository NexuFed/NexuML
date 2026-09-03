"""Default training scenario fragments."""

from __future__ import annotations

from collections.abc import Callable

import torch
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from nexuml import callback, optimizer, scheduler
from nexuml.core.types import (
    AutoBatchSizeSpec,
    BatchSizeSpec,
    CallbackSpec,
    CheckpointLoadSpec,
    DVCLiveSpec,
    ExportSpec,
    LoggingSpec,
    MLflowSpec,
    TensorBoardSpec,
    TrainingSpec,
    TuningSpec,
)

DEFAULT_AUTO_BATCH_SIZE = AutoBatchSizeSpec(
    min=8,
    max=128,
    candidates="power_of_two",
    safety="margin",
    margin=0.8,
)


def default_training(
    lr: float = 1e-3,
    batch_size: BatchSizeSpec | None = 64,
    max_epochs: int = 10,
    loss_keys: dict[str, float] | None = None,
    metric_keys: list[str] | None = None,
    optimizer_factory: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
) -> TrainingSpec:
    """Create a default TrainingSpec.

    Returns:
        TrainingSpec: Default training configuration with optimizer, scheduler
            and loss keys.
    """
    resolved_batch_size = batch_size if batch_size is not None else DEFAULT_AUTO_BATCH_SIZE
    return TrainingSpec(
        optimizer=optimizer(optimizer_factory, lr=lr),
        scheduler=scheduler(torch.optim.lr_scheduler.ConstantLR, factor=1.0, total_iters=0),
        loss_keys=loss_keys or {"reconstruction_loss": 1.0},
        metric_keys=metric_keys or [],
        max_epochs=max_epochs,
        batch_size=resolved_batch_size,
        lr=lr,
    )


def default_logging(
    experiment_name: str = "NexuML",
    run_name: str | None = None,
    log_system_metrics: bool = False,
    use_tensorboard: bool = True,
    use_mlflow: bool = True,
    use_dvclive: bool = False,
) -> LoggingSpec:
    """Create a default LoggingSpec.

    Returns:
        LoggingSpec: Default logging configuration with optional TensorBoard,
            MLflow and DVC Live backends.
    """
    return LoggingSpec(
        tensorboard=TensorBoardSpec(log_dir=".experiments/tensorboard")
        if use_tensorboard
        else None,
        mlflow=MLflowSpec(
            tracking_uri="sqlite:///./.experiments/mlflow.db",
            # tracking_uri="file:./.experiments/mlflow",
            # tracking_uri="http://ml-flow.ika.rub.de"
            # tracking_uri="https://dagshub.com/<USER>/<REPO>.mlflow"
            experiment_name=experiment_name,
            log_model=False,
        )
        if use_mlflow
        else None,
        dvclive=DVCLiveSpec(
            dir=".experiments/dvclive",
        )
        if use_dvclive
        else None,
        experiment_name=experiment_name,
        run_name=run_name,
        log_system_metrics=log_system_metrics,
    )


def default_callbacks() -> list[CallbackSpec]:
    """Create a default list of CallbackSpec.

    Returns:
        list[CallbackSpec]: Default callbacks for training.
    """
    return [
        callback(EarlyStopping, monitor="val/loss", patience=5),
        callback(LearningRateMonitor),
        callback(
            ModelCheckpoint,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="{epoch:02d}-{val_loss:.4f}",
            save_last=True,
        ),
        callback(RichProgressBar),
        callback(DeviceStatsMonitor),
    ]


def default_checkpoint(
    path: str | None = None,
) -> CheckpointLoadSpec:
    """Create a default CheckpointLoadSpec.

    Returns:
        CheckpointLoadSpec: Default checkpoint loading configuration.
    """
    return CheckpointLoadSpec(
        source=path, allow_missing=True, allow_shape_mismatch=True, freeze_loaded=False
    )


def default_tuning() -> TuningSpec:
    """Create a default tuning configuration.

    Returns:
        TuningSpec: Default tuning configuration.
    """
    return TuningSpec(
        n_trials=2,
        directions=["minimize"],
        metric_key="val/loss",
        storage=".experiments/optuna/optuna.log",
        prune=False,
    )


def default_exports(
    path: str | None = None,
) -> list[ExportSpec]:
    """Create a default exports configuration.

    Returns:
        list[ExportSpec]: Default exports configuration.
    """
    return [ExportSpec(kind="train_package", output=path)]
