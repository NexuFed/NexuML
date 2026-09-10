"""Tests for the component identity registry."""

from __future__ import annotations

import pytest

from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import DiscoveredItem
from nexuml.core.registry import ComponentRegistry
from nexuml.core.serialization import lower_component, restore_component


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


def test_known_registry_entries_skip_discovery_for_serialization(monkeypatch) -> None:
    registry = ComponentRegistry()
    registry.register("first", _FirstDefinition, kind="layer", version="1")

    def fail_scan() -> None:
        pytest.fail("known component lookup should not scan")

    monkeypatch.setattr(registry, "scan", fail_scan)
    monkeypatch.setattr("nexuml.core.serialization.get_component_registry", lambda: registry)

    lowered = lower_component(_FirstDefinition())
    restored = restore_component(kind="layer", value=lowered)

    assert lowered == {"type": "first", "version": "1", "params": {}}
    assert type(restored) is _FirstDefinition


def test_unknown_identity_discovers_and_resolves_registered_entry(monkeypatch) -> None:
    registry = ComponentRegistry()
    scan_calls = 0

    def scan() -> None:
        nonlocal scan_calls
        scan_calls += 1
        registry.register("first", _FirstDefinition, kind="layer", version="1")
        registry._loaded = True

    monkeypatch.setattr(registry, "scan", scan)

    assert registry.get_type("layer", "first") is _FirstDefinition
    assert registry.get_entry("layer", "first").definition_type is _FirstDefinition
    assert scan_calls == 1


def test_unknown_type_discovers_and_resolves_registered_entry(monkeypatch) -> None:
    registry = ComponentRegistry()
    scan_calls = 0

    def scan() -> None:
        nonlocal scan_calls
        scan_calls += 1
        registry.register("first", _FirstDefinition, kind="layer", version="1")
        registry._loaded = True

    monkeypatch.setattr(registry, "scan", scan)

    assert registry.entry_for_type(_FirstDefinition).name == "first"
    assert scan_calls == 1


def test_unknown_identity_still_raises_useful_key_error(monkeypatch) -> None:
    registry = ComponentRegistry()

    def scan() -> None:
        registry._loaded = True

    monkeypatch.setattr(registry, "scan", scan)

    with pytest.raises(KeyError) as exc_info:
        registry.get_entry("layer", "missing")

    assert "Unknown component 'layer'/'missing'/'1'. Available: []" in str(exc_info.value)


@pytest.mark.parametrize("collection", ["entries", "errors"])
def test_explicit_registry_collections_still_scan(monkeypatch, collection) -> None:
    registry = ComponentRegistry()
    registry.register("first", _FirstDefinition, kind="layer", version="1")
    scan_calls = 0

    def scan() -> None:
        nonlocal scan_calls
        scan_calls += 1
        registry._loaded = True

    monkeypatch.setattr(registry, "scan", scan)

    if collection == "entries":
        registry.entries()
    else:
        assert registry.errors == []

    assert scan_calls == 1


def test_explicit_entries_scan_and_preserve_conflict_errors(monkeypatch) -> None:
    class FakeScanner:
        def __init__(self) -> None:
            self.errors = []

        def scan_package(self, package_path: str) -> None:
            return None

        def by_kind(self, kind: str) -> list[DiscoveredItem]:
            if kind == "layer":
                return [DiscoveredItem("layer", "same", _SecondDefinition, "fake")]
            return []

    monkeypatch.setattr("nexuml.core.discovery.Scanner", FakeScanner)
    monkeypatch.setattr("nexuml.core.discovery.discover_library_packages", lambda: ["fake"])

    registry = ComponentRegistry()
    registry.register("same", _FirstDefinition, kind="layer", version="1")

    assert [entry.name for entry in registry.entries()] == ["same"]
    errors = registry.errors
    assert len(errors) == 1
    assert errors[0].error_type == "ValueError"
    assert "Component registry conflict" in errors[0].message
