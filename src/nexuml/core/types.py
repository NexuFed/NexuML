"""Core type definitions for NexuML pipeline specifications."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AxisKeySpec(BaseModel):
    """Typed axis-key declaration with source provenance.

    ``source`` indicates where the axis value is resolved from at runtime:

    - ``"x"``: from the transformed pipeline output TensorDict (e.g. a
      reconstructed feature or anomaly score produced by a pipeline layer).
    - ``"y"``: from the label TensorDict — most common for grouping axes such
      as ``machine``, ``section``, or ``target``.
    - ``"metadata"``: from the dataset metadata DataFrame attached to the batch
      by ``_attach_eval_metadata``.  Use this source for non-tensor columns that
      are not part of the label TensorDict but are needed for grouped evaluation
      (e.g. ``basename``, ``domain``).  Only available during eval/test phases.
    """

    key: str
    source: Literal["x", "y", "metadata"] = "y"


class SpecModel(BaseModel):
    """Shared base model with compatibility helpers for older call sites/tests."""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpecModel":
        return cls.model_validate(data)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.model_dump(mode="json") == other
        return super().__eq__(other)


class LayerSpec(SpecModel):
    """Specification for a single layer in the pipeline."""

    type_key: str
    keys_in: list[str] | dict[str, str]
    keys_out: list[str]
    params: dict[str, Any] = Field(default_factory=dict)
    meta_in: dict[str, str] | None = None
    meta_out: dict[str, str] | None = None


class PipelineSpec(SpecModel):
    """Ordered specification of pipeline stages, each containing layer specs."""

    stages: dict[str, list[LayerSpec]] = Field(default_factory=dict)


class OptimizerSpec(SpecModel):
    """Specification for an optimizer."""

    type: str = "torch.optim.Adam"
    params: dict[str, Any] = Field(default_factory=lambda: {"lr": 1e-3})


class SchedulerSpec(SpecModel):
    """Specification for a learning rate scheduler."""

    type: str = "torch.optim.lr_scheduler.ConstantLR"
    params: dict[str, Any] = Field(default_factory=lambda: {"factor": 1.0, "total_iters": 0})
    warmup: str | int | None = None

    def resolve_warmup(self, max_epochs: int) -> int:
        """Resolve the warmup spec to an integer number of epochs.

        Accepts an integer (used directly), a percentage string like ``"5%"``
        (computed as a fraction of *max_epochs*), or ``None`` (returns 0).
        The result is always at least 1 when a non-zero warmup is specified.

        Returns:
            Resolved warmup epoch count.
        """
        if self.warmup is None:
            return 0
        if isinstance(self.warmup, int):
            return max(1, self.warmup)
        w = str(self.warmup).strip()
        if w.endswith("%"):
            pct = float(w[:-1]) / 100.0
            return max(1, round(pct * max_epochs))
        return max(1, int(w))


class AutoBatchSizeSpec(SpecModel):
    """Structured config for runtime automatic training batch-size resolution."""

    mode: Literal["auto"] = "auto"
    min: int = 1
    max: int = 128
    candidates: Literal["power_of_two"] = "power_of_two"
    safety: Literal["largest", "previous_power_of_two", "margin"] = "previous_power_of_two"
    margin: float = 0.8

    @model_validator(mode="after")
    def validate_bounds(self) -> "AutoBatchSizeSpec":
        if self.min <= 0:
            raise ValueError("auto batch size min must be positive")
        if self.max <= 0:
            raise ValueError("auto batch size max must be positive")
        if self.min > self.max:
            raise ValueError("auto batch size min must be <= max")
        if self.safety == "margin" and not (0 < self.margin <= 1):
            raise ValueError("auto batch size margin must be in (0, 1]")
        return self


BatchSizeSpec = int | AutoBatchSizeSpec


class TrainingSpec(SpecModel):
    """Specification for training configuration."""

    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)
    loss_keys: dict[str, float] = Field(default_factory=dict)
    metric_keys: list[str] = Field(default_factory=list)
    progress_bar_keys: list[str] | Literal["all", "none"] = Field(default_factory=lambda: ["loss"])
    max_epochs: int = 10
    batch_size: BatchSizeSpec = 64
    lr: float = 1e-3
    accelerator: str = "auto"
    devices: str | int = "auto"
    strategy: str = "auto"
    precision: str | int = "32-true"

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: BatchSizeSpec) -> BatchSizeSpec:
        if isinstance(value, int) and value <= 0:
            raise ValueError("training.batch_size must be positive")
        return value


class TargetSpec(SpecModel):
    """Specification for a synthetic target."""

    type: str  # "multiclass", "multilabel", "regression"
    key: str  # TensorDict key for this target
    num_classes: int | None = None
    num_outputs: int | None = None
    positive_fraction: float | None = None


class DatasetSpec(SpecModel):
    """Specification for a single dataset in the pipeline."""

    type_key: str
    params: dict[str, Any] = Field(default_factory=dict)
    modality: str = "audio"
    max_samples: int | None = None
    split_type: str = "fit"  # "fit", "all", "keep"


class LoaderSpec(SpecModel):
    """Specification for the data loader."""

    backend: str = "dali"  # e.g. "torch", "dali", or a registered custom backend
    # None defers to TrainingSpec.batch_size; explicit loader values take precedence.
    batch_size: int | None = None
    num_workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    weighted_sampling: bool = False
    shuffle_train: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("data.loader.batch_size must be positive")
        return value


class PreprocessingSpec(SpecModel):
    """Optional preprocessing stage contract."""

    enabled: bool = False
    source_view: str = "raw"
    target_view: str = "prepared"
    path: str | None = None
    until_x_keys: list[str] = Field(default_factory=list)
    until_y_keys: list[str] = Field(default_factory=list)
    x_keys: list[str] | None = None
    y_keys: list[str] | None = None
    include_labels: bool = True
    label_prefix: str = "label__"
    writer: Literal[
        "webdataset", "tensordict_memmap", "numpy", "numpy_mmap", "torch", "tensor_shards"
    ] = "numpy"
    writer_params: dict[str, Any] = Field(default_factory=dict)
    overwrite: bool = False


class DataSpec(SpecModel):
    """Specification for data configuration."""

    source_type: str = "synthetic"
    params: dict[str, Any] = Field(default_factory=dict)
    targets: list[TargetSpec] = Field(default_factory=list)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    datasets: list[DatasetSpec] = Field(default_factory=list)
    loader: LoaderSpec = Field(default_factory=LoaderSpec)
    input_shapes: dict[str, list[int]] = Field(default_factory=dict)
    num_classes: int | None = None
    merge_labels: dict[str, dict] | None = None
    feature_key: str = "features"
    skip_pipeline_stages: list[str] = Field(default_factory=list)
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)

    @model_validator(mode="after")
    def validate_splits(self) -> "DataSpec":
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must sum to 1.0")
        return self


class DistanceEstimatorSpec(SpecModel):
    """Specification for a distance estimator used in anomaly detection."""

    type: str = "mahalanobis"
    params: dict[str, Any] = Field(default_factory=dict)
    group_label_keys: list[str] = Field(default_factory=list)
    missing_label_policy: Literal["error", "skip", "unknown"] = "error"
    fallback_policy: Literal["error", "parent", "global", "parent_or_global"] = "error"
    storage_backend: Literal["ram", "memmap"] = "memmap"
    storage_path: str | None = None
    max_samples: int | None = None
    retain_storage: bool = False


class EvalAlgorithmSpec(SpecModel):
    """Specification for a post-training evaluation algorithm."""

    type: str
    name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    axis_keys: list[AxisKeySpec | str] = Field(default_factory=list)
    feature_key: str | None = None
    label_key: str | None = None

    def contract_summary(self) -> dict[str, Any]:
        """Return a summary of the key contract for diagnostics."""
        return {
            "type": self.type,
            "name": self.name,
            "feature_key": self.feature_key,
            "label_key": self.label_key,
            "axis_keys": [
                a if isinstance(a, str) else {"key": a.key, "source": a.source}
                for a in self.axis_keys
            ],
        }


class EvaluationSpec(SpecModel):
    """Specification for evaluation configuration."""

    metrics: list[str] = Field(default_factory=lambda: ["mse", "mae"])
    algorithms: list[EvalAlgorithmSpec] = Field(default_factory=list)
    test_result_metrics: list[str] | Literal["all", "none"] = "none"


class TensorBoardSpec(SpecModel):
    """TensorBoard logger configuration."""

    log_dir: str = ".experiments/tensorboard"


class DVCLiveSpec(SpecModel):
    """DVCLive logger configuration."""

    dir: str = ".experiments/dvclive"


class MLflowSpec(SpecModel):
    """MLflow logger configuration."""

    tracking_uri: str = "file:./.experiments/mlflow"
    artifact_location: str | None = None
    experiment_name: str | None = None
    log_model: bool = False
    run_id: str | None = None
    parent_run_id: str | None = None
    auto_nested_runs: bool = True
    tags: dict[str, Any] = Field(default_factory=dict)
    synchronous: bool | None = None


class DiagramSpec(SpecModel):
    """Mermaid diagram export configuration."""

    enabled: bool = True
    depth: int = 2
    direction: str = "TB"
    show_params: bool = True
    show_shapes: bool = True
    show_metrics: bool = True
    output_dir: str = ".experiments/diagrams"


class LoggingSpec(SpecModel):
    """Experiment logging configuration."""

    tensorboard: TensorBoardSpec | None = None
    dvclive: DVCLiveSpec | None = None
    mlflow: MLflowSpec | None = None
    diagram: DiagramSpec | None = Field(default_factory=DiagramSpec)
    temp_artifact_dir: str = "/tmp/mlruns"
    experiment_name: str = "NexuML"
    run_name: str | None = None
    log_system_metrics: bool = False


class CallbackSpec(SpecModel):
    """Specification for a training callback."""

    type: str  # dotted path or known alias: "checkpoint", "lr_monitor", etc.
    params: dict[str, Any] = Field(default_factory=dict)


class TuningSpec(SpecModel):
    """Specification for hyperparameter tuning with Optuna."""

    n_trials: int = 50
    directions: list[str] = Field(default_factory=lambda: ["minimize"])
    metric_key: str = "val/loss"
    storage: str = ".experiments/optuna/optuna.log"
    prune: bool = False


class CheckpointLoadSpec(SpecModel):
    """Selective weight loading policy for resume/fine-tune flows."""

    source: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    allow_missing: bool = True
    allow_shape_mismatch: bool = True
    freeze_loaded: bool = False


class ExportSpec(SpecModel):
    """Requested export artifact."""

    kind: Literal["train_package", "onnx", "safetensors"] = "train_package"
    output: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class ScenarioSpec(SpecModel):
    """Complete scenario specification combining all components."""

    name: str
    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    logging: LoggingSpec | None = None
    callbacks: list[CallbackSpec] = Field(default_factory=list)
    tuning: TuningSpec | None = None
    checkpoint: CheckpointLoadSpec | None = None
    exports: list[ExportSpec] = Field(default_factory=list)
