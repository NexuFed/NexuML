"""Evaluation module for NexuML."""

from nexuml.evaluation.algorithm import EvalAlgorithm

__all__ = [
    "EVAL_ALGORITHM_REGISTRY",
    "EvalAlgorithm",
    "create_algorithm",
]


def __getattr__(name: str):
    if name in ("EVAL_ALGORITHM_REGISTRY", "create_algorithm"):
        from nexuml.evaluation.registry import EVAL_ALGORITHM_REGISTRY, create_algorithm

        globals()["EVAL_ALGORITHM_REGISTRY"] = EVAL_ALGORITHM_REGISTRY
        globals()["create_algorithm"] = create_algorithm
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
