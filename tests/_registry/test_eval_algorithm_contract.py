"""Registry-driven contract tests for every discovered eval algorithm."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nexuml.evaluation.algorithm import EvalAlgorithm
from nexuml.evaluation.registry import EvalAlgorithmRegistry

# Eval algorithms known to be untestable with the generic synthetic batch in this
# file. Any other discovered eval algorithm that raises fails its parameter case
# instead of skipping.
_SKIP_ALLOWLIST: dict[str, str] = {
    "reconstruction_visualizer": (
        "ReconstructionVisualizer requires explicit feature_key/reconstructed_key "
        "that this generic contract test cannot infer"
    ),
}


def _eval_algorithm_skip_or_fail(key: str, exc: Exception, extra: str | None = None) -> None:
    """Skip allowlisted eval algorithms; fail others with rich, actionable context."""
    if key not in _SKIP_ALLOWLIST:
        detail = f"{type(exc).__name__}: {exc}"
        if extra:
            detail += extra
        raise AssertionError(
            f"Conformance failure for eval_algorithm {key!r}: {detail}\n"
            f"Hint: add {key!r} to the eval_algorithm skip allowlist only if the failure "
            f"requires a dependency or real data that synthetic fixtures cannot provide."
        ) from exc
    message = f"{_SKIP_ALLOWLIST[key]}: {exc}"
    if extra:
        message += extra
    pytest.skip(message)


def _make_eval_batch() -> tuple[TensorDict, TensorDict]:
    x = TensorDict(
        {
            "features": torch.randn(4, 16),
            "latent": torch.randn(4, 8),
        },
        batch_size=[4],
    )
    y = TensorDict(
        {
            "label": torch.randint(0, 2, (4,)),
            "target": torch.randn(4, 1),
        },
        batch_size=[4],
    )
    return x, y


@pytest.mark.conformance
def test_eval_algorithm_validate_params(
    eval_key: str,
    eval_registry: EvalAlgorithmRegistry,
) -> None:
    """Every eval algorithm must accept signature validation."""
    try:
        validated = eval_registry.validate_params(eval_key, {})
    except ValueError as exc:
        _eval_algorithm_skip_or_fail(eval_key, exc)
    assert isinstance(validated, dict)


@pytest.mark.conformance
def test_eval_algorithm_results_contract(
    eval_key: str,
    eval_registry: EvalAlgorithmRegistry,
    discovered_eval_algorithm: type[EvalAlgorithm],
) -> None:
    """Every eval algorithm must complete eval_batch -> eval_end -> results()."""
    cls = discovered_eval_algorithm

    # Try default params first, then fall back to required-key injection.
    try:
        validated = eval_registry.validate_params(eval_key, {})
        algorithm = cls(**validated)
    except (ValueError, TypeError) as exc:
        try:
            validated = eval_registry.validate_params(
                eval_key,
                {"feature_key": "features", "label_key": "label"},
            )
            algorithm = cls(**validated)
        except (ValueError, TypeError) as fallback_exc:
            _eval_algorithm_skip_or_fail(
                eval_key, exc, extra=f"; {type(fallback_exc).__name__}: {fallback_exc}"
            )

    x, y = _make_eval_batch()
    try:
        algorithm.eval_batch(x, y)
        algorithm.eval_end()
        results = algorithm.results()
    except (KeyError, ValueError, TypeError, RuntimeError) as exc:
        _eval_algorithm_skip_or_fail(eval_key, exc)

    assert isinstance(results, dict)
    assert all(isinstance(k, str) for k in results.keys())
    assert all(isinstance(v, (int, float, torch.Tensor)) for v in results.values())


def test_eval_algorithm_allowlist_is_self_auditing(eval_registry) -> None:
    """Every entry in the eval_algorithm skip allowlist must still exist in the registry."""
    registered = set(eval_registry.list().keys())
    stale = [key for key in _SKIP_ALLOWLIST if key not in registered]
    assert not stale, f"Stale eval_algorithm allowlist keys no longer in registry: {stale}"
