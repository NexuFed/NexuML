"""Registry-driven contract tests for every discovered layer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from nexuml.core.post_train_layer import PostTrainFitLayer
from nexuml.core.registry import LayerRegistry

# Generic input candidates tried in order for layers with unknown key expectations.
_INPUT_CANDIDATES: list[dict[str, tuple[int, ...]]] = [
    {"features": (16,)},
    {"features": (8, 16)},
    {"features": (16,), "reconstruction": (16,)},
    {"anomaly_score": (1,)},
    {"latent": (8,)},
    {"image": (3, 16, 16)},
    {"spectrogram": (1, 16, 16)},
    {"audio": (160,)},
]

# Layer-specific inputs that mimic upstream-produced keys.
_LAYER_INPUT_HINTS: dict[str, dict[str, tuple[int, ...]]] = {
    "decision_rule_pipeline_layer": {"anomaly_score": (1,)},
    "ConvolutionalDecoder": {"latent": (128,)},
}

# Layer-specific constructor params for validate_params / instantiation.
_LAYER_PARAM_HINTS: dict[str, dict[str, object]] = {
    "ConvolutionalDecoder": {
        "decoder_shape": [8, 4, 4],
        "output_shape": [3, 16, 16],
    },
}

# Layers known to be untestable with the generic synthetic inputs in this file.
# Any other discovered layer that raises fails its parameter case instead of skipping.
_SKIP_ALLOWLIST: dict[str, str] = {}


def _layer_skip_or_fail(key: str, exc: Exception) -> None:
    """Skip allowlisted layers; fail others with rich, actionable context."""
    if key not in _SKIP_ALLOWLIST:
        raise AssertionError(
            f"Conformance failure for layer {key!r}: "
            f"{type(exc).__name__}: {exc}\n"
            f"Hint: add {key!r} to the layer skip allowlist only if the failure "
            f"requires a dependency or real data that synthetic fixtures cannot provide."
        ) from exc
    pytest.skip(f"{_SKIP_ALLOWLIST[key]}: {exc}")


def _build_synthetic_input(shapes: dict[str, tuple[int, ...]], batch_size: int = 2) -> TensorDict:
    return TensorDict(
        {key: torch.randn(batch_size, *shape) for key, shape in shapes.items()},
        batch_size=[batch_size],
    )


def _param_hints_for(layer_key: str, input_sizes: dict[str, tuple[int, ...]]) -> dict[str, object]:
    """Provide minimal semantic params for layers with required constructor knobs."""
    feature_shape = next(iter(input_sizes.values()))
    feature_dim = feature_shape[-1]
    hints: dict[str, object] = {
        "target_dim": min(8, feature_dim),
        "output_sizes": {"out": (min(8, feature_dim),)},
        "decoder_shape": [min(8, feature_dim), feature_dim],
        "output_shape": [feature_dim],
        "metric": "accuracy",
        "metrics": ["accuracy"],
    }
    hints.update(_LAYER_PARAM_HINTS.get(layer_key, {}))
    return hints


def _candidate_inputs(key: str) -> list[dict[str, tuple[int, ...]]]:
    """Return generic input candidates, preferring a layer-specific hint if available."""
    hint = _LAYER_INPUT_HINTS.get(key)
    if hint is None:
        return _INPUT_CANDIDATES
    return [hint] + [c for c in _INPUT_CANDIDATES if c != hint]


def _try_build_and_forward(
    registry: LayerRegistry,
    key: str,
) -> tuple[nn.Module, TensorDict]:
    """Try to instantiate and forward a layer with generic synthetic inputs."""
    last_error: Exception | None = None
    for input_sizes in _candidate_inputs(key):
        keys_in = list(input_sizes)
        try:
            layer = registry.instantiate(
                key,
                input_sizes=input_sizes,
                keys_in=keys_in,
                keys_out=["out"],
                **_param_hints_for(key, input_sizes),
            )
            x = _build_synthetic_input(input_sizes)
            with torch.no_grad():
                layer(x, None)
            return layer, x
        except Exception as exc:  # noqa: BLE001 - exploratory candidate retry, see outer skip gate
            last_error = exc
    raise RuntimeError(f"Could not build/forward layer {key!r} with generic inputs: {last_error}")


@pytest.mark.conformance
def test_layer_validate_params(layer_key: str, layer_registry: LayerRegistry) -> None:
    """Every layer must accept signature validation for empty/default params."""
    try:
        validated = layer_registry.validate_params(layer_key, _LAYER_PARAM_HINTS.get(layer_key, {}))
    except ValueError as exc:
        _layer_skip_or_fail(layer_key, exc)
    assert isinstance(validated, dict)


@pytest.mark.conformance
def test_layer_forward_contract(
    layer_key: str,
    layer_registry: LayerRegistry,
    discovered_layer: type,
) -> None:
    """Every layer must instantiate, forward, and return
    (TensorDict|Tensor, Optional[TensorDict])."""
    try:
        layer, x = _try_build_and_forward(layer_registry, layer_key)
    except RuntimeError as exc:
        _layer_skip_or_fail(layer_key, exc)

    assert isinstance(layer, nn.Module)

    with torch.no_grad():
        x_out, y_out = layer(x, None)

    assert isinstance(x_out, (TensorDict, torch.Tensor))
    assert y_out is None or isinstance(y_out, TensorDict)


@pytest.mark.conformance
def test_post_train_fit_layer_lifecycle(
    layer_key: str,
    layer_registry: LayerRegistry,
    discovered_layer: type,
) -> None:
    """PostTrainFitLayer subclasses must support collect_batch / finalize_fit."""
    if not issubclass(discovered_layer, PostTrainFitLayer):
        pytest.skip("not a PostTrainFitLayer")

    try:
        layer, x = _try_build_and_forward(layer_registry, layer_key)
    except RuntimeError as exc:
        _layer_skip_or_fail(layer_key, exc)

    assert isinstance(layer, PostTrainFitLayer)
    layer._armed = True
    try:
        layer.collect_batch(x, None)
    except (KeyError, AttributeError) as exc:
        _layer_skip_or_fail(layer_key, exc)
    layer.on_predict_end()
    assert layer._fitted


def test_layer_allowlist_is_self_auditing(layer_registry: LayerRegistry) -> None:
    """Every entry in the layer skip allowlist must still exist in the registry."""
    registered = set(layer_registry.list().keys())
    stale = [key for key in _SKIP_ALLOWLIST if key not in registered]
    assert not stale, f"Stale layer allowlist keys no longer in registry: {stale}"
