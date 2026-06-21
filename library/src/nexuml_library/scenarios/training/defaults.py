"""Default training scenario fragments."""

from __future__ import annotations

from nexuml.core.types import (
    AutoBatchSizeSpec,
    BatchSizeSpec,
    DVCLiveSpec,
    LoggingSpec,
    MLflowSpec,
    OptimizerSpec,
    SchedulerSpec,
    TensorBoardSpec,
    TrainingSpec,
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
    optimizer_type: str = "torch.optim.Adam",
) -> TrainingSpec:
    """Create a default TrainingSpec.

    Returns:
        TrainingSpec: Default training configuration with optimizer, scheduler
            and loss keys.
    """
    resolved_batch_size = batch_size if batch_size is not None else DEFAULT_AUTO_BATCH_SIZE
    return TrainingSpec(
        optimizer=OptimizerSpec(type=optimizer_type, params={"lr": lr}),
        scheduler=SchedulerSpec(
            type="torch.optim.lr_scheduler.ConstantLR",
            params={"factor": 1.0, "total_iters": 0},
        ),
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
