"""Tests for nexuml.core.registry."""

from __future__ import annotations

import pytest

from nexuml.core.registry import LayerRegistry


def test_registry_list_and_get(layer_registry):
    items = layer_registry.list()
    assert "LinearEncoder" in items
    assert layer_registry.get("LinearEncoder") is items["LinearEncoder"]


def test_registry_validate_params(layer_registry):
    validated = layer_registry.validate_params("LinearEncoder", {"output_dim": 4})
    assert validated["output_dim"] == 4


def test_registry_validate_params_missing_required(layer_registry):
    # Layer with no required constructor params should validate empty params.
    validated = layer_registry.validate_params("IdentityLayer", {})
    assert isinstance(validated, dict)


def test_registry_instantiate(layer_registry):
    layer = layer_registry.instantiate(
        "LinearEncoder",
        input_sizes={"features": (16,)},
        keys_in=["features"],
        keys_out=["latent"],
        output_dim=4,
    )
    assert layer is not None


def test_registry_duplicate_registration_raises():
    registry = LayerRegistry()
    registry.register("foo", dict)
    with pytest.raises(ValueError):
        registry.register("foo", list)
