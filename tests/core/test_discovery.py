"""Tests for nexuml.core.discovery."""

from __future__ import annotations


from nexuml.core.discovery import (
    LibraryConfig,
    Scanner,
    discover_entry_point_packages,
    discover_library_packages,
    scan_all,
)
from nexuml.core.registry import ComponentRegistry, get_component_registry
from nexuml.core.serialization import lower_component, restore_component


def test_scan_all_returns_items():
    scanner = scan_all()
    assert scanner.items
    assert scanner.by_kind("layer")
    assert scanner.by_kind("scenario")


def test_registry_has_unique_component_identities():
    identities = [
        (entry.kind, entry.name, entry.version) for entry in get_component_registry().entries()
    ]
    assert len(identities) == len(set(identities))


def test_scanner_records_errors_without_aborting():
    scanner = Scanner()
    scanner.scan_package("definitely.not.a.real.package")
    assert scanner.errors


def test_discover_entry_point_packages_is_list():
    packages = discover_entry_point_packages()
    assert isinstance(packages, list)


def test_discover_library_packages_includes_nexuml_library():
    packages = discover_library_packages()
    assert "nexuml_library" in packages


def test_library_config_roundtrip(tmp_path):
    config = LibraryConfig()
    root = tmp_path / "lib"
    root.mkdir()
    config.add_root(str(root))
    path = tmp_path / "libraries.json"
    config.save(path)

    loaded = LibraryConfig.load(path)
    assert str(root.resolve()) in loaded.roots


def test_scan_all_includes_local_library_keys_for_registry_parametrization(
    isolated_library_config, minimal_local_library
):
    """`tests/_registry/conftest.py`'s pytest_generate_tests parametrizes
    contract tests over scan_all().by_kind(...); a local library root's
    decorated elements must show up there too."""
    config = LibraryConfig.load()
    config.add_root(str(minimal_local_library.root))
    config.save()

    scanner = scan_all()

    assert minimal_local_library.layer_key in {item.key for item in scanner.by_kind("layer")}
    assert minimal_local_library.scenario_key in {item.key for item in scanner.by_kind("scenario")}
    assert minimal_local_library.dataset_key in {
        item.key for item in scanner.by_kind("data_source")
    }
    assert minimal_local_library.eval_key in {
        item.key for item in scanner.by_kind("eval_algorithm")
    }


def test_discover_entry_point_packages_returns_loadable_entries(monkeypatch):
    class FakeEntryPoint:
        def __init__(self, value: str):
            self.value = value
            self.group = "nexuml.libraries"

        def load(self):
            return object()

    fake_eps = [FakeEntryPoint("fake_pkg_a"), FakeEntryPoint("fake_pkg_b")]
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group=None: fake_eps if group == "nexuml.libraries" else [],
    )

    packages = discover_entry_point_packages()
    assert packages == ["fake_pkg_a", "fake_pkg_b"]


def test_discover_entry_point_packages_skips_failed_loads(monkeypatch):
    class FailingEntryPoint:
        value = "broken_pkg"
        group = "nexuml.libraries"

        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group=None: [FailingEntryPoint()],
    )

    packages = discover_entry_point_packages()
    assert packages == []


def test_entry_point_component_identity_round_trip(
    isolated_library_config, minimal_local_library, monkeypatch
):
    class FakeEntryPoint:
        value = minimal_local_library.package_name
        group = "nexuml.libraries"

        def load(self):
            return object()

    monkeypatch.syspath_prepend(str(minimal_local_library.root))
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group=None: [FakeEntryPoint()] if group == "nexuml.libraries" else [],
    )

    registry = ComponentRegistry()
    registry.scan()
    definition_type = registry.get_type("layer", minimal_local_library.layer_key)
    definition = definition_type.model_validate({"scale": 3.0})
    monkeypatch.setattr("nexuml.core.serialization.get_component_registry", lambda: registry)

    restored = restore_component(kind="layer", value=lower_component(definition))

    assert type(restored) is definition_type
    assert restored.model_dump()["scale"] == 3.0
