"""Tests for the component identity registry."""

from __future__ import annotations

import pytest

from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.registry import ComponentRegistry


class _FirstDefinition(LayerDefinition):
    component_name = "first"

    def build(self, context: LayerBuildContext):
        return object()


class _SecondDefinition(LayerDefinition):
    component_name = "second"

    def build(self, context: LayerBuildContext):
        return object()


def test_registry_duplicate_identity_raises() -> None:
    registry = ComponentRegistry()
    registry.register("same", _FirstDefinition, kind="layer", version="1")

    with pytest.raises(ValueError, match="Component registry conflict"):
        registry.register("same", _SecondDefinition, kind="layer", version="1")


def test_registry_rejects_duplicate_type_identity() -> None:
    registry = ComponentRegistry()
    registry.register("first", _FirstDefinition, kind="layer", version="1")

    with pytest.raises(ValueError, match="already registered"):
        registry.register("second", _FirstDefinition, kind="layer", version="1")
