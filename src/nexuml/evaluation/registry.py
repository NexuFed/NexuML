"""Explicit evaluation-algorithm materialization."""

from __future__ import annotations

from nexuml.core.components import EvalBuildContext
from nexuml.core.types import EvalAlgorithmSpec
from nexuml.evaluation.algorithm import EvalAlgorithm


def create_algorithm(spec: EvalAlgorithmSpec) -> EvalAlgorithm:
    """Materialize and validate an evaluation algorithm definition.

    Returns:
        Runtime evaluation algorithm.

    Raises:
        TypeError: If the definition builds the wrong runtime type.
    """
    algorithm = spec.algorithm.build(
        EvalBuildContext(feature_key=spec.feature_key, label_key=spec.label_key)
    )
    if not isinstance(algorithm, EvalAlgorithm):
        raise TypeError(
            f"{type(spec.algorithm).__name__}.build() must return EvalAlgorithm, "
            f"got {type(algorithm).__name__}"
        )
    return algorithm
