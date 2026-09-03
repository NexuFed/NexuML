"""Evaluation scenario fragments."""

from __future__ import annotations

from nexuml.core.types import EvalAlgorithmSpec, EvaluationSpec
from nexuml_library.evaluation.visualizers.class_histogram import ClassHistogramVisualizer
from nexuml_library.evaluation.visualizers.latent import LatentVisualizer
from nexuml_library.evaluation.visualizers.reconstruction import ReconstructionVisualizer


def merge_eval_specs(eval_specs: list[EvaluationSpec]) -> EvaluationSpec:
    """Merge a list of evaluation specs into a single combined spec.

    Returns:
        EvaluationSpec: Merged evaluation specification.
    """
    eval_spec = EvaluationSpec(metrics=[], algorithms=[], test_result_metrics=[])
    for spec in eval_specs:
        eval_spec.metrics.extend(spec.metrics)
        eval_spec.algorithms.extend(spec.algorithms)
        if not isinstance(eval_spec.test_result_metrics, list):
            continue
        if isinstance(spec.test_result_metrics, list):
            eval_spec.test_result_metrics.extend(spec.test_result_metrics)
        elif spec.test_result_metrics == "all":
            eval_spec.test_result_metrics = spec.test_result_metrics
    return eval_spec


def reconstruction_evaluation(
    reconstruction_feature_key: str = "features",
    reconstruction_key: str = "reconstructed",
    reconstruction_mask_key: str | None = None,
    reconstruction_patch_size: int | tuple[int, int] | None = None,
    reconstruction_label_keys: list[str] | None = None,
    metrics: list[str] = ["mse", "mae"],
) -> EvaluationSpec:
    """Evaluation spec for reconstruction metrics.

    Returns:
        EvaluationSpec: Evaluation specification with reconstruction metrics.
    """
    return EvaluationSpec(
        metrics=metrics,
        algorithms=[
            EvalAlgorithmSpec(
                algorithm=ReconstructionVisualizer(
                    reconstructed_key=reconstruction_key,
                    mask_key=reconstruction_mask_key,
                    patch_size=reconstruction_patch_size,
                    label_keys=reconstruction_label_keys,
                ),
                feature_key=reconstruction_feature_key,
            )
        ],
        test_result_metrics=metrics,
    )


def classification_evaluation(
    label_key: str = "class",
    feature_key: str = "latent",
    max_samples: int = 2000,
    metrics: list[str] = ["accuracy", "f1"],
) -> EvaluationSpec:
    """Evaluation spec for classification metrics.

    Returns:
        EvaluationSpec: Evaluation specification with classification metrics.
    """
    return EvaluationSpec(
        metrics=metrics,
        algorithms=[
            EvalAlgorithmSpec(
                algorithm=ClassHistogramVisualizer(),
                label_key=label_key,
            ),
            EvalAlgorithmSpec(
                algorithm=LatentVisualizer(method="tsne", max_samples=max_samples),
                feature_key=feature_key,
                label_key=label_key,
            ),
            EvalAlgorithmSpec(
                algorithm=LatentVisualizer(method="umap", max_samples=max_samples),
                feature_key=feature_key,
                label_key=label_key,
            ),
        ],
        test_result_metrics=metrics,
    )


def regression_evaluation() -> EvaluationSpec:
    """Evaluation spec for regression metrics.

    Returns:
        EvaluationSpec: Evaluation specification with regression metrics.
    """
    return EvaluationSpec(metrics=["mse", "mae", "r2"])
