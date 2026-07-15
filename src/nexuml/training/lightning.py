"""Lightning-based training backend for NexuML."""

from __future__ import annotations

import builtins
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
import yaml
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.log_paths import resolve_logs_root
from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.registry import LayerRegistry, get_registry
from nexuml.core.types import AutoBatchSizeSpec, EvalAlgorithmSpec, ScenarioSpec, TrainingSpec
from nexuml.data.auto_batch import resolve_with_probe
from nexuml.data.dataset import NexuDataset
from nexuml.data.export import export_data_module
from nexuml.data.exported import ExportedDataset
from nexuml.data.module import NexuDataModule
from nexuml.evaluation.algorithm import EvalAlgorithm
from nexuml.evaluation.registry import create_algorithm

logger = logging.getLogger("lightning.pytorch.nexuml.training")

# Enable Tensor Cores on Ampere+ GPUs (suppresses the "trade-off precision for performance" tip)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


class NexuLightningModule(L.LightningModule):
    """Lightning module wrapping a CompiledPipeline."""

    def __init__(
        self,
        pipeline: CompiledPipeline | None = None,
        scenario: ScenarioSpec | dict[str, Any] | None = None,
        runtime_metadata: dict[str, Any] | None = None,
        registry: LayerRegistry | None = None,
    ):
        super().__init__()
        if scenario is not None and not isinstance(scenario, ScenarioSpec):
            scenario = ScenarioSpec.model_validate(scenario)
        if pipeline is None:
            if scenario is None:
                raise ValueError("Either 'pipeline' or 'scenario' must be provided.")
            pipeline = compile(scenario, registry or get_registry())

        self.pipeline = pipeline
        self.scenario = scenario
        self._eval_results: dict[str, float] = {}
        self._stage_metric_results: dict[str, dict[str, float]] = {"val": {}, "test": {}}
        self._evaluation_algorithms = self._instantiate_evaluation_algorithms(
            scenario.evaluation.algorithms if scenario is not None else []
        )
        self.save_hyperparameters(
            {
                "scenario": scenario.model_dump(mode="json") if scenario is not None else None,
                "runtime_metadata": runtime_metadata or {},
            }
        )

    def forward(
        self, x: TensorDict, y: TensorDict | None = None
    ) -> tuple[TensorDict, TensorDict | None]:
        return self.pipeline(x, y)

    def forward_until(
        self,
        x: TensorDict,
        y: TensorDict | None = None,
        *,
        x_keys: Sequence[str] | None = None,
        y_keys: Sequence[str] | None = None,
    ) -> tuple[TensorDict, TensorDict | None]:
        return self.pipeline.forward_until(x, y, x_keys=x_keys, y_keys=y_keys)

    def _compute_loss(self, x: TensorDict) -> tuple[torch.Tensor, dict[str, float]]:
        """Aggregate weighted losses from pipeline output.

        Returns:
            Tuple of ``(loss, loss_dict)`` where *loss* is the weighted sum
            tensor and *loss_dict* maps loss names to their scalar values.
        """
        losses: list[torch.Tensor] = []
        loss_dict: dict[str, float] = {}

        for loss_key, weight in self.pipeline.loss_keys.items():
            if loss_key in x.keys():
                val_tensor = cast(torch.Tensor, x[loss_key])
                if val_tensor.dim() == 0:
                    losses.append(weight * val_tensor)
                else:
                    losses.append(weight * val_tensor.mean())
                loss_dict[loss_key] = (
                    val_tensor.item() if val_tensor.dim() == 0 else val_tensor.mean().item()
                )

        if not losses:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss = torch.stack(losses).sum()

        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_out, y_out = self.pipeline(x, y)
        loss, loss_dict = self._compute_loss(x_out)

        for name, val in loss_dict.items():
            self.log(
                f"train/{name}",
                val,
                prog_bar=self._should_show_in_progress_bar("train", name),
            )

        # Log pipeline metrics (e.g. adaptive_scale) to tensorboard/logger
        for metric_name in self.pipeline.metric_keys:
            if metric_name not in x_out.keys():
                continue
            value = x_out[metric_name]
            scalar = value.mean() if isinstance(value, torch.Tensor) and value.dim() > 0 else value
            self.log(
                f"train/{metric_name}",
                scalar,
                on_step=True,
                on_epoch=False,
                prog_bar=self._should_show_in_progress_bar("train", metric_name),
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_out, y_out = self.pipeline(x, y)
        loss, loss_dict = self._compute_loss(x_out)

        for name, val in loss_dict.items():
            self.log(
                f"val/{name}",
                val,
                prog_bar=self._should_show_in_progress_bar("val", name),
            )
        self._log_running_pipeline_metrics("val", x_out)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_out, y_out = self.pipeline(x, y)
        # TODO: Why does it need _attach_eval_metadata in test_step?
        x_out = self._attach_eval_metadata(x_out, x, dataloader_idx=0)
        self._run_eval_batch(x_out, y_out)
        loss, loss_dict = self._compute_loss(x_out)

        for name, val in loss_dict.items():
            self.log(
                f"test/{name}",
                val,
                prog_bar=self._should_show_in_progress_bar("test", name),
            )
        self._log_running_pipeline_metrics("test", x_out)

        return loss

    def configure_optimizers(self):
        optimizer = self.pipeline.create_optimizer()
        scheduler = self.pipeline.create_scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def on_fit_start(self):
        self.pipeline.call_layer_hook("on_fit_start")

    def on_fit_end(self):
        self.pipeline.call_layer_hook("on_fit_end")

    def on_train_epoch_end(self):
        self.pipeline.call_layer_hook("on_train_epoch_end")

    def on_validation_start(self):
        self.pipeline.call_layer_hook("on_validation_start")
        self._stage_metric_results["val"] = {}

    def on_validation_epoch_end(self):
        self.pipeline.call_layer_hook("on_validation_epoch_end")
        self._finalize_stage_metrics("val")

    def on_validation_end(self):
        self.pipeline.call_layer_hook("on_validation_end")

    def on_test_start(self):
        self.pipeline.call_layer_hook("on_test_start")
        self._stage_metric_results["test"] = {}

    def on_test_end(self):
        self.pipeline.call_layer_hook("on_test_end")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x_out, y_out = self.pipeline(x, y)
        return x_out

    def on_predict_start(self):
        self.pipeline.call_layer_hook("on_predict_start")

    def on_predict_end(self):
        self.pipeline.call_layer_hook("on_predict_end")

    def on_test_epoch_end(self) -> None:
        self._finalize_stage_metrics("test")
        self.pipeline.call_layer_hook("on_test_epoch_end")
        self._finalize_eval_phase()

    @property
    def evaluation_results(self) -> dict[str, float]:
        return dict(self._eval_results)

    def get_stage_metric_results(self, stage: str) -> dict[str, float]:
        return dict(self._stage_metric_results.get(stage, {}))

    @property
    def test_result_eval_metrics(self) -> dict[str, float]:
        return {
            f"test/{metric_path}": value
            for metric_path, value in self._eval_results.items()
            if self._should_show_eval_metric_in_test_results(*metric_path.split("/", 1))
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["nexuml_eval"] = {
            "results": dict(self._eval_results),
            "algorithms_blob": self._serialize_algorithms(),
        }
        # Persist fitted PostTrainFitLayer state (not captured by state_dict)
        from nexuml.core.post_train_layer import PostTrainFitLayer

        post_train_states: dict[str, dict] = {}
        for stage, name, layer in self.pipeline.iter_layers():
            if isinstance(layer, PostTrainFitLayer):
                key = f"{stage}/{name}"
                post_train_states[key] = {
                    "fitted": layer._fitted,
                    "state": layer._get_fit_state(),
                }
        if post_train_states:
            checkpoint["nexuml_post_train"] = post_train_states

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("nexuml_eval", {}) or {}
        self._eval_results = dict(state.get("results", {}))
        blob = state.get("algorithms_blob")
        if blob:
            self._evaluation_algorithms = self._deserialize_algorithms(blob)
        # Restore fitted PostTrainFitLayer state
        from nexuml.core.post_train_layer import PostTrainFitLayer

        post_train_states = checkpoint.get("nexuml_post_train") or {}
        if post_train_states:
            for stage, name, layer in self.pipeline.iter_layers():
                if isinstance(layer, PostTrainFitLayer):
                    key = f"{stage}/{name}"
                    saved = post_train_states.get(key) or {}
                    if saved.get("fitted"):
                        layer._set_fit_state(saved.get("state") or {})
                        layer._fitted = True

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Any,
        map_location: Any = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,
        weights_only: bool | None = None,
        allow_unsafe_globals: bool = True,
        **kwargs,
    ):
        if allow_unsafe_globals and hasattr(
            torch.serialization, "get_unsafe_globals_in_checkpoint"
        ):
            unsafe_globals = set()
            for value in torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint_path)):
                try:
                    unsafe_globals.add(_resolve_global(value))
                except Exception:
                    continue
            if unsafe_globals:
                torch.serialization.add_safe_globals(list(unsafe_globals))
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=False if strict is None else strict,
            weights_only=weights_only,
            **kwargs,
        )

    def _instantiate_evaluation_algorithms(
        self,
        algorithm_specs: list[EvalAlgorithmSpec],
    ) -> list[tuple[str, EvalAlgorithm]]:
        return [(spec.name or spec.type, create_algorithm(spec)) for spec in algorithm_specs]

    # TODO: Do we need _attach_eval_metadata?
    def _attach_eval_metadata(
        self, x_out: TensorDict, x_in: TensorDict, dataloader_idx: int = 0
    ) -> TensorDict:
        """Attach dataset metadata columns to an eval/test batch TensorDict.

        Metadata originates from a pandas DataFrame stored on the dataset object
        (``dataset.metadata``).  Each row corresponds to one sample; rows are
        identified via the integer ``sample_index`` tensor carried in *x_in*.

        Typical columns attached for DCASE scenarios:
        ``basename``, ``machine``, ``domain``, ``target``, ``section``.

        These columns are needed by grouped evaluation (e.g. per-machine AUC) where
        the grouping key comes from metadata rather than from a tensor in the batch.
        They map to ``AxisKeySpec(source="metadata")`` declarations in eval specs.

        Only available during eval/test phases (the test dataloader exposes the
        dataset with its metadata DataFrame).  Not populated during training or
        validation where sample identity is not required.

        Falls back to copying keys already present in *x_in* if ``sample_index``
        is absent (e.g. when the dataset does not expose a metadata DataFrame).

        Returns:
            The *x_out* TensorDict with metadata columns attached (if available).
        """
        from tensordict import NonTensorData

        # First: copy any direct x_in keys not already in x_out
        for key in x_in.keys():
            if key not in x_out.keys():
                try:
                    x_out[key] = x_in[key]
                except Exception:
                    pass

        # Then: if sample_index is available, do a full metadata lookup from the loader
        if "sample_index" not in x_in.keys():
            return x_out
        loader = None
        trainer = getattr(self, "_trainer", None) or getattr(self, "trainer", None)
        loaders = None
        if trainer:
            loaders = getattr(trainer, "test_dataloaders", None)
            if loaders is None:
                loaders = getattr(trainer, "predict_dataloaders", None)
        if isinstance(loaders, (list, tuple)):
            loader = loaders[dataloader_idx] if dataloader_idx < len(loaders) else None
        elif loaders is not None:
            loader = loaders
        if loader is None:
            return x_out
        meta = getattr(loader, "metadata", None)
        if meta is None:
            dataset = getattr(loader, "dataset", None)
            meta = getattr(dataset, "meta", None)
        if meta is None or (hasattr(meta, "empty") and meta.empty):
            return x_out
        if hasattr(meta, "reset_index"):
            meta = meta.reset_index(drop=True)
        try:
            indices = x_in["sample_index"].detach().cpu().reshape(-1).tolist()
            batch_meta = meta.iloc[indices].reset_index(drop=True)
            for column in batch_meta.columns:
                if column in x_out.keys():
                    continue
                values = batch_meta[column].tolist()
                try:
                    x_out[column] = torch.as_tensor(values, device=x_out.device)
                except Exception:
                    x_out[column] = NonTensorData(values, batch_size=x_out.batch_size)
        except Exception as exc:
            logger.debug("eval metadata attach skipped: %s", exc)
        return x_out

    def _run_eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        for prefix, algorithm in self._evaluation_algorithms:
            algorithm.eval_batch(x, y)

    def _finalize_eval_phase(self) -> None:
        for prefix, algorithm in self._evaluation_algorithms:
            algorithm.eval_end()

        logger_obj = getattr(self, "loggers", None) or self.logger
        for prefix, algorithm in self._evaluation_algorithms:
            algorithm.visualize(logger_obj)

        self._eval_results = {}
        for prefix, algorithm in self._evaluation_algorithms:
            algo_results = algorithm.results()
            for key, value in algo_results.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    scalar = float(value.detach().cpu().item())
                elif isinstance(value, (float, int)):
                    scalar = float(value)
                else:
                    continue
                self._eval_results[f"{prefix}/{key}"] = scalar
                self.log(
                    f"eval/{prefix}/{key}",
                    scalar,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=bool(self.logger),
                    sync_dist=True,
                )
                if self._should_show_eval_metric_in_test_results(prefix, key):
                    self.log(
                        f"test/{prefix}/{key}",
                        scalar,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=bool(self.logger),
                        sync_dist=True,
                    )
        self._eval_phase = "idle"

    def _should_show_eval_metric_in_test_results(self, prefix: str, key: str) -> bool:
        if self.scenario is None:
            return False

        policy = getattr(self.scenario.evaluation, "test_result_metrics", "none")
        if policy == "all":
            return True
        if policy == "none":
            return False

        metric_path = f"{prefix}/{key}"
        return key in policy or metric_path in policy

    def _serialize_algorithms(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(self._evaluation_algorithms, buffer)
        return buffer.getvalue()

    def _deserialize_algorithms(self, blob: bytes) -> list[tuple[str, EvalAlgorithm]]:
        buffer = io.BytesIO(blob)
        return torch.load(buffer, map_location="cpu", weights_only=False)

    def _finalize_stage_metrics(self, stage: str) -> None:
        requested_metrics = {metric.lower() for metric in self.pipeline.metric_keys}
        if not requested_metrics:
            return

        finalized_metrics: dict[str, float] = {}
        for _stage_name, _layer_name, layer in self.pipeline.iter_layers():
            get_epoch_metrics = getattr(layer, "get_epoch_metrics", None)
            if not callable(get_epoch_metrics):
                continue

            for metric_name, value in get_epoch_metrics(stage).items():
                if metric_name.lower() not in requested_metrics:
                    continue
                scalar = (
                    float(value.detach().cpu().item())
                    if isinstance(value, torch.Tensor)
                    else float(value)
                )
                finalized_metrics[f"{stage}/{metric_name}"] = scalar
                self.log(
                    f"{stage}/{metric_name}",
                    scalar,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=self._should_show_in_progress_bar(stage, metric_name),
                    logger=bool(self.logger),
                    sync_dist=True,
                )
        self._stage_metric_results[stage] = finalized_metrics

    def _log_running_pipeline_metrics(self, stage: str, x: TensorDict) -> None:
        requested_metrics = {metric.lower() for metric in self.pipeline.metric_keys}
        if not requested_metrics:
            return

        for metric_name in self.pipeline.metric_keys:
            if metric_name not in x.keys():
                continue
            raw_value = cast(torch.Tensor, x[metric_name])
            scalar: torch.Tensor | float = raw_value.mean() if raw_value.dim() > 0 else raw_value
            self.log(
                f"{stage}/{metric_name}",
                scalar,
                on_step=True,
                on_epoch=False,
                prog_bar=self._should_show_in_progress_bar(stage, metric_name),
                logger=False,
                sync_dist=False,
            )

    def _should_show_in_progress_bar(self, stage: str, name: str) -> bool:
        if self.scenario is None:
            return name == "loss"

        policy = getattr(self.scenario.training, "progress_bar_keys", ["loss"])
        if policy == "all":
            return True
        if policy == "none":
            return False

        metric_path = f"{stage}/{name}"
        lowered = {entry.lower() for entry in policy}
        return name.lower() in lowered or metric_path.lower() in lowered


class LightningFeatureExtractor(torch.nn.Module):
    """Partial pipeline wrapper for dataset preprocessing/export."""

    def __init__(
        self,
        lightning_module: NexuLightningModule,
        *,
        x_keys: Sequence[str] | None = None,
        y_keys: Sequence[str] | None = None,
    ):
        super().__init__()
        self.lightning_module = lightning_module
        self.x_keys = list(x_keys or [])
        self.y_keys = list(y_keys or [])

    def forward(
        self,
        x: TensorDict,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict, TensorDict | None]:
        return self.lightning_module.forward_until(
            x,
            y,
            x_keys=self.x_keys,
            y_keys=self.y_keys,
        )

    def on_predict_start(self) -> None:
        self.lightning_module.on_predict_start()

    def on_predict_end(self) -> None:
        self.lightning_module.on_predict_end()


@dataclass
class TrainResult:
    """Result of a training run."""

    pipeline: CompiledPipeline
    lightning_module: NexuLightningModule
    trainer: L.Trainer
    validation_results: list[dict[str, float]] = field(default_factory=list)
    test_results: list[dict[str, float]] = field(default_factory=list)
    load_report: dict[str, list[str]] | None = None
    eval_algorithm_results: dict[str, float] = field(default_factory=dict)


@dataclass
class RuntimeArtifacts:
    """Compiled runtime objects shared by training and dataset export."""

    pipeline: CompiledPipeline
    lightning_module: NexuLightningModule
    data_module: NexuDataModule
    load_report: dict[str, list[str]] | None = None


class NexuSession:
    """Thin orchestration layer for building and running a NexuML session."""

    def __init__(
        self,
        scenario: ScenarioSpec | None,
        registry: LayerRegistry | None = None,
        accelerator: str = "auto",
        devices: int | str = "auto",
        log_dir: str | Path = ".experiments",
        enable_progress_bar: bool = True,
        trainer_checkpoint: str | Path | None = None,
        run_name: str | None = None,
    ) -> None:
        if scenario is None and trainer_checkpoint is None:
            raise ValueError("Either 'scenario' or 'trainer_checkpoint' must be provided.")

        self.registry = registry
        self.accelerator = accelerator
        self.devices = devices
        self.log_dir = resolve_logs_root(log_dir)
        self.enable_progress_bar = enable_progress_bar
        self.trainer_checkpoint = (
            Path(trainer_checkpoint) if trainer_checkpoint is not None else None
        )
        self.run_name = run_name
        self.scenario = self._resolve_scenario(scenario)

        self._runtime: RuntimeArtifacts | None = None
        self._trainer: L.Trainer | None = None
        self._trainer_loggers: list[Any] | bool | None = None
        self._trainer_callbacks: list[Any] | None = None
        self._service_info_printed = False
        self._run_metadata_logged = False

    @classmethod
    def from_scenario(
        cls,
        scenario: ScenarioSpec,
        **kwargs,
    ) -> "NexuSession":
        """Create a session from a scenario definition.

        Returns:
            New ``NexuSession`` instance.
        """
        return cls(scenario=scenario, **kwargs)

    @classmethod
    def from_trainer_checkpoint(
        cls,
        trainer_checkpoint: str | Path,
        scenario: ScenarioSpec | None = None,
        **kwargs,
    ) -> "NexuSession":
        """Create a session from a Lightning Trainer checkpoint.

        Returns:
            New ``NexuSession`` instance restored from the checkpoint.
        """
        return cls(
            scenario=scenario,
            trainer_checkpoint=trainer_checkpoint,
            **kwargs,
        )

    @property
    def runtime(self) -> RuntimeArtifacts:
        return self.build_runtime()

    @property
    def pipeline(self) -> CompiledPipeline:
        return self.runtime.pipeline

    @property
    def lightning_module(self) -> NexuLightningModule:
        return self.runtime.lightning_module

    @property
    def data_module(self) -> NexuDataModule:
        return self.runtime.data_module

    @property
    def trainer(self) -> L.Trainer:
        return self.build_trainer()

    @property
    def trainer_loggers(self) -> list[Any] | bool:
        self._build_logging_components()
        return self._trainer_loggers if self._trainer_loggers is not None else False

    @property
    def trainer_callbacks(self) -> list[Any]:
        self._build_logging_components()
        return self._trainer_callbacks or []

    def setup(self) -> "NexuSession":
        """Build runtime objects and trainer.

        Returns:
            The same ``NexuSession`` instance (for chaining).
        """
        self.build_runtime()
        self.build_trainer()
        return self

    def build_runtime(self) -> RuntimeArtifacts:
        """Create runtime artifacts lazily.

        Returns:
            The ``RuntimeArtifacts`` (creating them on first call).
        """
        if self._runtime is None:
            if self.trainer_checkpoint is not None:
                self._runtime = create_runtime_artifacts_from_trainer_checkpoint(
                    self.trainer_checkpoint,
                    scenario=self.scenario,
                    registry=self.registry,
                )
            else:
                self._runtime = create_runtime_artifacts(
                    self.scenario,
                    registry=self.registry,
                )
        return self._runtime

    def build_trainer(self) -> L.Trainer:
        """Create the Lightning Trainer lazily.

        Returns:
            The ``lightning.Trainer`` instance (creating it on first call).
        """
        if self._trainer is not None:
            return self._trainer

        self.build_runtime()
        self._build_logging_components()
        self._log_run_metadata_artifacts()

        if not self._service_info_printed:
            from nexuml.tracking.logger import print_service_info

            data_backend = None
            if self._runtime is not None and self._runtime.data_module is not None:
                data_backend = self._runtime.data_module.loader_spec.backend

            print_service_info(
                trainer_loggers=(
                    self.trainer_loggers if isinstance(self.trainer_loggers, list) else []
                ),
                scenario_name=self.scenario.name,
                log_dir=self.log_dir,
                logging_spec=getattr(self.scenario, "logging", None),
                tuning_spec=getattr(self.scenario, "tuning", None),
                data_backend=data_backend,
                training_backend="lightning",
            )
            self._service_info_printed = True

        tr: TrainingSpec = self.scenario.training
        resolved_accelerator = self.accelerator if self.accelerator != "auto" else tr.accelerator
        resolved_devices = self.devices if self.devices != "auto" else tr.devices
        resolved_strategy = tr.strategy if tr.strategy != "auto" else "auto"
        resolved_precision = tr.precision if tr.precision != "32-true" else "32-true"

        self._trainer = L.Trainer(
            max_epochs=tr.max_epochs,
            accelerator=resolved_accelerator,
            devices=resolved_devices,
            strategy=resolved_strategy,
            precision=resolved_precision,  # ty: ignore[invalid-argument-type]
            default_root_dir=str(self.log_dir),
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=False,
            logger=self.trainer_loggers,
            callbacks=self.trainer_callbacks or None,
            num_sanity_val_steps=0 if tr.max_epochs == 0 else 2,
            log_every_n_steps=1,
        )
        return self._trainer

    def fit(self) -> "NexuSession":
        """Run Trainer.fit() for the current session.

        Returns:
            The same ``NexuSession`` instance (for chaining).
        """
        self.trainer.fit(
            self.lightning_module,
            datamodule=self.data_module,
            ckpt_path=str(self.trainer_checkpoint) if self.trainer_checkpoint is not None else None,
        )
        return self

    def validate(self) -> list[dict[str, float]]:
        """Run Trainer.validate() with the session datamodule.

        Returns:
            List of metric dictionaries from the validation stage.
        """
        raw_results = self.trainer.validate(
            self.lightning_module,
            datamodule=self.data_module,
        )
        results: list[dict[str, float]] = [dict(r) for r in raw_results]
        stage_metrics = self.lightning_module.get_stage_metric_results("val")
        if stage_metrics:
            for result in results:
                result.update(stage_metrics)
        return results

    def predict(
        self,
        dataloaders: Any = None,
        datamodule: NexuDataModule | None = None,
        return_predictions: bool = False,
    ):
        """Run Trainer.predict() with session defaults unless explicitly overridden.

        Returns:
            Prediction outputs from ``Trainer.predict()``, or ``None`` when
            *return_predictions* is ``False``.
        """
        if dataloaders is None and datamodule is None:
            datamodule = self.data_module
        return self.trainer.predict(
            self.lightning_module,
            dataloaders=dataloaders,
            datamodule=datamodule,
            return_predictions=return_predictions,
        )

    def test(
        self,
        dataloaders: Any = None,
        datamodule: NexuDataModule | None = None,
    ) -> list[dict[str, float]]:
        """Run Trainer.test() with session defaults unless explicitly overridden.

        Returns:
            List of metric dictionaries from the test stage.
        """
        if dataloaders is None and datamodule is None:
            datamodule = self.data_module
        raw_results = self.trainer.test(
            self.lightning_module,
            dataloaders=dataloaders,
            datamodule=datamodule,
        )
        results: list[dict[str, float]] = [dict(r) for r in raw_results]
        stage_metrics = self.lightning_module.get_stage_metric_results("test")
        if stage_metrics:
            for result in results:
                result.update(stage_metrics)
        mirrored_eval_metrics = self.lightning_module.test_result_eval_metrics
        if mirrored_eval_metrics:
            for result in results:
                result.update(mirrored_eval_metrics)
        return results

    def run(self) -> TrainResult:
        """Execute the standard fit → validate → post-train fit → test flow.

        Returns:
            ``TrainResult`` containing the pipeline, module, trainer, and
            metric results.
        """
        self.fit()
        # Skip validation for frozen eval runs (max_epochs=0) — no training happened
        if self.scenario is not None and self.scenario.training.max_epochs == 0:
            validation_results = []
        else:
            validation_results = self.validate()

        self._fit_post_train_layers(self.data_module.train_dataloader())

        test_results = self.test()
        eval_algorithm_results = self.lightning_module.evaluation_results
        if eval_algorithm_results:
            logger.info("Eval algorithm results: %s", eval_algorithm_results)

        return TrainResult(
            pipeline=self.pipeline,
            lightning_module=self.lightning_module,
            trainer=self.trainer,
            validation_results=validation_results,
            test_results=test_results,
            load_report=self.runtime.load_report,
            eval_algorithm_results=eval_algorithm_results,
        )

    def _fit_post_train_layers(self, train_loader) -> None:
        """Run one predict pass per unfitted PostTrainFitLayer in pipeline order.

        Raises:
            RuntimeError: If a ``PostTrainFitLayer`` is not fitted after its
                predict pass.
        """
        from nexuml.core.post_train_layer import PostTrainFitLayer

        layers = [
            layer
            for _stage, _name, layer in self.pipeline.iter_layers()
            if isinstance(layer, PostTrainFitLayer)
        ]
        if not layers:
            return

        # All already fitted (checkpoint resume) — skip all passes
        if all(layer._fitted for layer in layers):
            logger.info("Post-train fit: all layers already fitted (checkpoint). Skipping.")
            return

        for layer in layers:
            if layer._fitted:
                continue
            logger.info("Post-train fit: fitting %s...", layer.__class__.__name__)
            layer._armed = True
            self.predict(dataloaders=train_loader, return_predictions=False)
            if not layer._fitted:
                raise RuntimeError(
                    f"PostTrainFitLayer {layer.__class__.__name__} "
                    "was not fitted after predict pass."
                )
            logger.info("Post-train fit: %s done.", layer.__class__.__name__)

    def _resolve_scenario(self, scenario: ScenarioSpec | None) -> ScenarioSpec:
        if self.trainer_checkpoint is not None:
            return load_scenario_from_trainer_checkpoint(
                self.trainer_checkpoint,
                fallback=scenario,
            )
        assert scenario is not None
        return scenario

    def _build_logging_components(self) -> None:
        if self._trainer_loggers is not None and self._trainer_callbacks is not None:
            return

        from nexuml.tracking.logger import create_loggers
        from nexuml.training.callbacks import build_callbacks

        self._trainer_loggers = create_loggers(
            getattr(self.scenario, "logging", None),
            run_name=self.run_name or self.scenario.name,
        )
        self._trainer_callbacks = build_callbacks(getattr(self.scenario, "callbacks", []))

    def _log_run_metadata_artifacts(self) -> None:
        if self._run_metadata_logged:
            return
        if not self.trainer_loggers:
            self._run_metadata_logged = True
            return

        from nexuml.tracking.logger import log_text_artifact

        scenario = self.lightning_module.scenario
        if scenario is None:
            self._run_metadata_logged = True
            return
        scenario_dump = scenario.model_dump(mode="json")
        log_text_artifact(
            self.trainer_loggers,
            yaml.safe_dump(scenario_dump, sort_keys=False),
            "config.yaml",
        )
        runtime_metadata = self.lightning_module.hparams.get("runtime_metadata", {})
        if runtime_metadata:
            log_text_artifact(
                self.trainer_loggers,
                yaml.safe_dump(runtime_metadata, sort_keys=False),
                "runtime_metadata.yaml",
            )

        self._run_metadata_logged = True


def create_dataset_from_spec(scenario: ScenarioSpec) -> NexuDataset:
    """Create a dataset from a ScenarioSpec's data configuration.

    Returns:
        A ``NexuDataset`` built from the scenario's data config.
    """
    from nexuml.data.creator import NexuDataCreator

    creator = NexuDataCreator()
    return creator.build_dataset(scenario.data)


def create_data_module_from_spec(
    scenario: ScenarioSpec,
    registry: LayerRegistry | None = None,
) -> NexuDataModule:
    """Create a LightningDataModule from a ScenarioSpec.

    Returns:
        A ``NexuDataModule`` configured for the scenario.
    """
    if scenario.data.preprocessing.enabled:
        export_path = materialize_preprocessed_dataset(scenario, registry=registry)
        batch_size = scenario.data.loader.batch_size or scenario.training.batch_size
        if isinstance(batch_size, AutoBatchSizeSpec):
            batch_size = batch_size.min
        loader_spec = scenario.data.loader.model_copy(update={"batch_size": batch_size})
        dataset = ExportedDataset(export_path)
        return NexuDataModule(
            dataset=dataset,
            loader_spec=loader_spec,
            train_split=scenario.data.train_split,
            val_split=scenario.data.val_split,
            test_split=scenario.data.test_split,
            split_by_column=True,
        )

    return _create_base_data_module_from_spec(scenario)


def _create_base_data_module_from_spec(scenario: ScenarioSpec) -> NexuDataModule:
    """Create a data module without applying preprocessing materialization.

    Returns:
        A ``NexuDataModule`` without preprocessing materialization applied.
    """
    from nexuml.data.creator import NexuDataCreator

    creator = NexuDataCreator()
    return creator.build(scenario.data, default_batch_size=scenario.training.batch_size)


def materialize_preprocessed_dataset(
    scenario: ScenarioSpec,
    registry: LayerRegistry | None = None,
) -> Path:
    """Materialize the configured preprocessing view and return its export path.

    Returns:
        ``Path`` to the directory containing the materialized dataset.
    """
    if registry is None:
        registry = get_registry()

    preprocessing = scenario.data.preprocessing
    export_path = resolve_preprocessing_path(scenario)
    config_path = export_path / "config.yaml"
    if config_path.exists() and not preprocessing.overwrite:
        return export_path

    raw_scenario = scenario.model_copy(
        update={
            "data": scenario.data.model_copy(
                update={
                    "skip_pipeline_stages": [],
                    "preprocessing": scenario.data.preprocessing.model_copy(
                        update={"enabled": False}
                    ),
                }
            )
        }
    )

    raw_data_module = _create_base_data_module_from_spec(raw_scenario)
    hydrated_raw_scenario = _hydrate_scenario_from_dataset(raw_scenario, raw_data_module)
    raw_pipeline = compile(hydrated_raw_scenario, registry)
    raw_lightning_module = NexuLightningModule(raw_pipeline)

    transform = None
    if preprocessing.until_x_keys or preprocessing.until_y_keys:
        transform = LightningFeatureExtractor(
            raw_lightning_module,
            x_keys=preprocessing.until_x_keys,
            y_keys=preprocessing.until_y_keys,
        )

    export_data_module(
        raw_data_module,
        export_path,
        backend=preprocessing.writer,
        transform=transform,
        x_keys=preprocessing.x_keys,
        y_keys=preprocessing.y_keys,
        include_labels=preprocessing.include_labels,
        label_prefix=preprocessing.label_prefix,
        **preprocessing.writer_params,
    )
    return export_path


def resolve_preprocessing_path(scenario: ScenarioSpec) -> Path:
    """Resolve the configured preprocessing output directory.

    Returns:
        Resolved ``Path`` for the preprocessing output directory.
    """
    preprocessing = scenario.data.preprocessing
    if preprocessing.path:
        return resolve_logs_root(preprocessing.path)
    return (
        resolve_logs_root(".experiments")
        / "preprocessed"
        / scenario.name
        / preprocessing.target_view
        / preprocessing.writer
    )


def create_runtime_artifacts(
    scenario: ScenarioSpec,
    registry: LayerRegistry | None = None,
    apply_selective_checkpoint: bool = True,
) -> RuntimeArtifacts:
    """Compile the pipeline and create the matching Lightning/DataModule runtime.

    Returns:
        ``RuntimeArtifacts`` containing the compiled pipeline, Lightning module,
        data module, and optional load report.
    """
    if registry is None:
        registry = get_registry()

    data_module = create_data_module_from_spec(scenario, registry=registry)
    hydrated_scenario = _hydrate_scenario_from_dataset(scenario, data_module)
    pipeline = compile(hydrated_scenario, registry)

    load_report = None
    if (
        apply_selective_checkpoint
        and hydrated_scenario.checkpoint
        and hydrated_scenario.checkpoint.source
    ):
        from nexuml.core.export import load_weights

        report = load_weights(
            pipeline,
            hydrated_scenario.checkpoint.source,
            checkpoint=hydrated_scenario.checkpoint,
        )
        load_report = report.to_dict()
        logger.info(
            "Selective checkpoint load matched=%d missing=%d unexpected=%d mismatched=%d",
            len(report.matched),
            len(report.missing),
            len(report.unexpected),
            len(report.shape_mismatched),
        )

    runtime_metadata: dict[str, Any] = {"load_report": load_report or {}}
    lightning_module = NexuLightningModule(pipeline, scenario=hydrated_scenario)
    auto_result = _resolve_auto_batch_size_if_needed(
        scenario=hydrated_scenario,
        lightning_module=lightning_module,
        registry=registry,
    )
    if auto_result is not None:
        data_module, auto_metadata = auto_result
        runtime_metadata["auto_batch_size"] = auto_metadata
    lightning_module = NexuLightningModule(
        pipeline,
        scenario=hydrated_scenario,
        runtime_metadata=runtime_metadata,
    )
    return RuntimeArtifacts(
        pipeline=pipeline,
        lightning_module=lightning_module,
        data_module=data_module,
        load_report=load_report,
    )


def _resolve_auto_batch_size_if_needed(
    *,
    scenario: ScenarioSpec,
    lightning_module: NexuLightningModule,
    registry: LayerRegistry,
) -> tuple[NexuDataModule, dict[str, Any]] | None:
    """Probe structured auto training.batch_size and rebuild final datamodule.

    Returns:
        Tuple of ``(NexuDataModule, metadata_dict)`` with the auto-selected
        batch size, or ``None`` if auto batch-size probing is not applicable.

    Raises:
        RuntimeError: If CUDA is unavailable when auto batch-size probing is
            requested.
    """
    config = scenario.training.batch_size
    if not isinstance(config, AutoBatchSizeSpec):
        return None
    if scenario.data.loader.batch_size is not None:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("Automatic batch-size probing requires CUDA")

    def probe(candidate: int) -> None:
        candidate_scenario = scenario.model_copy(
            update={"training": scenario.training.model_copy(update={"batch_size": candidate})}
        )
        candidate_data_module = create_data_module_from_spec(candidate_scenario, registry=registry)
        candidate_data_module.setup("fit")
        batch = next(iter(candidate_data_module.train_dataloader()))
        batch = _move_batch_to_device(batch, torch.device("cuda"))
        lightning_module.to("cuda")
        lightning_module.zero_grad(set_to_none=True)
        x, y = batch
        x_out, _y_out = lightning_module.pipeline(x, y)
        loss, _loss_dict = lightning_module._compute_loss(x_out)
        loss.backward()
        lightning_module.zero_grad(set_to_none=True)

    result = resolve_with_probe(config, probe)
    final_scenario = scenario.model_copy(
        update={
            "training": scenario.training.model_copy(
                update={"batch_size": result.selected_batch_size}
            )
        }
    )
    return create_data_module_from_spec(final_scenario, registry=registry), result.to_dict()


def _move_batch_to_device(value: Any, device: torch.device) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    return value


def _hydrate_scenario_from_dataset(
    scenario: ScenarioSpec,
    data_module: NexuDataModule,
) -> ScenarioSpec:
    """Refresh data-related compile hints from the actual dataset view.

    Returns:
        Updated ``ScenarioSpec`` with hydrated ``input_shapes`` and
        ``num_classes``.
    """
    dataset = data_module.dataset
    if len(dataset) == 0:
        return scenario

    x_sample, _y_sample = dataset[0]
    input_shapes = {
        key: list(value.shape) for key, value in x_sample.items() if isinstance(value, torch.Tensor)
    }
    merged_input_shapes = dict(scenario.data.input_shapes)
    for key, shape in input_shapes.items():
        if key not in merged_input_shapes:
            merged_input_shapes[key] = shape

    update = {
        "input_shapes": merged_input_shapes,
    }

    if scenario.data.num_classes is None and len(getattr(dataset, "label_names", [])) == 1:
        label_name = dataset.label_names[0]
        num_classes = getattr(dataset, "num_classes", {}).get(label_name)
        if num_classes is not None:
            update["num_classes"] = num_classes

    return scenario.model_copy(
        update={
            "data": scenario.data.model_copy(update=update),
        }
    )


def train(
    scenario: ScenarioSpec | None,
    registry: LayerRegistry | None = None,
    accelerator: str = "auto",
    devices: int | str = "auto",
    log_dir: str | Path = ".experiments",
    enable_progress_bar: bool = True,
    trainer_checkpoint: str | Path | None = None,
    run_name: str | None = None,
) -> TrainResult:
    """Compatibility wrapper around ``NexuSession.run()``.

    Returns:
        ``TrainResult`` from the session run.
    """
    session = NexuSession(
        scenario=scenario,
        registry=registry,
        accelerator=accelerator,
        devices=devices,
        log_dir=log_dir,
        enable_progress_bar=enable_progress_bar,
        trainer_checkpoint=trainer_checkpoint,
        run_name=run_name,
    )
    return session.run()


def load_scenario_from_trainer_checkpoint(
    checkpoint_path: str | Path,
    fallback: ScenarioSpec | None = None,
) -> ScenarioSpec:
    """Load serialized scenario metadata from a Lightning Trainer checkpoint.

    Returns:
        ``ScenarioSpec`` deserialized from the checkpoint, or *fallback* if
        the checkpoint contains no scenario metadata and *fallback* is provided.

    Raises:
        ValueError: If the checkpoint lacks scenario metadata and no
            *fallback* is given.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", {}) or {}
    scenario_data = hyper_parameters.get("scenario")
    if scenario_data is None:
        if fallback is None:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' does not contain serialized scenario metadata."
            )
        return fallback
    return ScenarioSpec.model_validate(scenario_data)


def create_runtime_artifacts_from_trainer_checkpoint(
    checkpoint_path: str | Path,
    scenario: ScenarioSpec | None = None,
    registry: LayerRegistry | None = None,
) -> RuntimeArtifacts:
    """Rebuild runtime objects from a Lightning Trainer checkpoint.

    Returns:
        ``RuntimeArtifacts`` reconstructed from the checkpoint.
    """
    restored_scenario = load_scenario_from_trainer_checkpoint(checkpoint_path, fallback=scenario)
    restored_scenario = restored_scenario.model_copy(update={"checkpoint": None})
    data_module = create_data_module_from_spec(restored_scenario, registry=registry)
    lightning_module = NexuLightningModule.load_from_checkpoint(
        checkpoint_path,
        scenario=restored_scenario,
        registry=registry,
    )
    return RuntimeArtifacts(
        pipeline=lightning_module.pipeline,
        lightning_module=lightning_module,
        data_module=data_module,
        load_report=None,
    )


def _resolve_global(path: str) -> Any:
    if "." not in path:
        return getattr(builtins, path)

    parts = path.split(".")
    for idx in range(len(parts), 0, -1):
        module_name = ".".join(parts[:idx])
        try:
            module = __import__(module_name, fromlist=["_unused"])
        except Exception:
            continue
        obj: Any = module
        try:
            for attr in parts[idx:]:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"Could not resolve checkpoint global '{path}'")
