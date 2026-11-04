"""Evaluate packaged models through the existing NexuDataset/Lightning test path."""

from typing import Any

import torch

from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.types import ScenarioSpec, TrainingSpec


def evaluate_model(
    model: CompiledPipeline,
    scenario: ScenarioSpec,
    *,
    device: str | None = None,
    loggers: list[Any] | None = None,
) -> dict[str, float]:
    """Test exported computation without fitting or reconstructing its model.

    The caller owns logger finalization on failure; Lightning finalizes success.

    Returns:
        Test losses, pipeline metrics and configured evaluation-algorithm results.

    Raises:
        ValueError: If dataset configuration, its test split or cached preprocessing is absent.
    """
    import lightning as L
    from lightning.pytorch.utilities.exceptions import SIGTERMException

    from nexuml.training.lightning import (
        NexuLightningModule,
        create_data_module_from_spec,
        resolve_preprocessing_path,
    )
    from nexuml.evaluation.policy import _evaluation_state

    if not isinstance(scenario.training, TrainingSpec):
        raise ValueError("Dataset evaluation requires a model with dataset training configuration")
    if scenario.data.source is None and not scenario.data.datasets:
        raise ValueError("Model evaluation requires a NexuDataset source and a test split")

    class EvaluationFailureCleanup(L.Callback):
        def on_exception(self, trainer, pl_module, exception):
            if isinstance(exception, SIGTERMException) and exception.code is None:
                exception.code = 143
            # The caller finalizes failed loggers; Lightning must still finish teardown.
            trainer.loggers = []

    target = torch.device(device or "cpu")
    cuda_devices = (target.index or 0,) if target.type == "cuda" else ()
    with _evaluation_state(model, scenario.training.seed, cuda_devices=cuda_devices):
        if scenario.data.preprocessing.enabled:
            path = resolve_preprocessing_path(scenario)
            if not (path / "config.yaml").is_file():
                raise ValueError(
                    f"Evaluation requires the existing preprocessed dataset at {path}; "
                    "it will not rebuild or fit preprocessing"
                )
            scenario = scenario.model_copy(
                update={
                    "data": scenario.data.model_copy(
                        update={
                            "preprocessing": scenario.data.preprocessing.model_copy(
                                update={"overwrite": False}
                            )
                        }
                    )
                }
            )
        data = create_data_module_from_spec(scenario)
        data.setup("test")
        if data._test_ds is None or len(data._test_ds) == 0:
            raise ValueError(
                "NexuDataset test split is empty; evaluation will not use train/val samples"
            )

        module = NexuLightningModule(pipeline=model, scenario=scenario)
        trainer = L.Trainer(
            accelerator="gpu" if target.type == "cuda" else target.type,
            devices=[target.index or 0] if target.type == "cuda" else 1,
            logger=loggers or False,
            callbacks=[EvaluationFailureCleanup()],
            precision=scenario.training.precision,  # ty: ignore[invalid-argument-type]
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        results = trainer.test(module, datamodule=data, verbose=False)
        metrics = dict(results[0]) if results else {}
        metrics.update(module.get_stage_metric_results("test"))
        metrics.update({f"eval/{key}": value for key, value in module.evaluation_results.items()})
        metrics.update(module.test_result_eval_metrics)
        return metrics
