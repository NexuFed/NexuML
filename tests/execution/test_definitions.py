"""Execution definitions use the existing discovery sources and identity registry."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nexuml.core.components import ExecutionBackendDefinition
from nexuml.core.discovery import DISCOVERED_ATTR, LibraryConfig, execution_backend
from nexuml.core.registry import ComponentRegistry


def test_core_only_discovery():
    registry = ComponentRegistry()
    registry.scan([])
    entry = registry.get_entry("execution_backend", "local")
    assert entry.definition_type.label == "Local"
    assert entry.definition_type.model_json_schema()["properties"] == {}
    assert not registry.errors


def test_catalog_import_does_not_load_cluster_runtimes():
    code = """
import sys
from nexuml.core.registry import ComponentRegistry
registry = ComponentRegistry()
registry.scan([])
assert registry.entries(kind='execution_backend')
assert not {'ray', 'kubernetes', 'fastapi'} & sys.modules.keys()
assert 'nexuml.execution.ray' not in sys.modules
import nexuml.core.registry as registry_module
registry_module._default_registry = registry
from nexuml.execution import catalog
assert len(catalog()) == 5
assert not {'ray', 'kubernetes', 'fastapi'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )


@pytest.mark.parametrize("source", ["entry-point", "local-root"])
def test_library_execution_discovery(tmp_path, monkeypatch, source):
    package = tmp_path / "execution_test_library"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "backend.py").write_text(
        textwrap.dedent("""
        from nexuml.core.components import ExecutionBackendDefinition
        from nexuml.core.discovery import execution_backend

        @execution_backend('test-execution')
        class TestExecution(ExecutionBackendDefinition):
            label = 'Library execution'
            workers: int = 2

            def run(self, scenario, **kwargs):
                return self.workers
    """)
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import nexuml.core.registry as registry_module

    registry = ComponentRegistry()
    monkeypatch.setattr(registry_module, "_default_registry", registry)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    if source == "entry-point":

        class EntryPoint:
            value = "execution_test_library"

            def load(self):
                return object()

        monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: [EntryPoint()])
        monkeypatch.setattr(LibraryConfig, "load", lambda *args: LibraryConfig())
    else:
        monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: [])
        monkeypatch.setattr(LibraryConfig, "load", lambda *args: LibraryConfig([str(tmp_path)]))
    try:
        registry.scan()
        entry = registry.get_entry("execution_backend", "test-execution")
        assert entry.definition_type.model_json_schema()["properties"]["workers"]["default"] == 2
        assert entry.definition_type().run(None) == 2
        from nexuml.core.config import ResolvedConfig
        from nexuml.core.types import ScenarioSpec

        config = ResolvedConfig.from_scenario(
            ScenarioSpec(name="extension", execution=entry.definition_type(workers=7))
        )
        assert ResolvedConfig.from_yaml(config.to_yaml()).execution.workers == 7
        assert not registry.errors
    finally:
        for name in tuple(sys.modules):
            if name.startswith("execution_test_library"):
                sys.modules.pop(name)


def test_execution_decorator_rejects_wrong_role():
    with pytest.raises(TypeError, match="ExecutionBackendDefinition"):
        execution_backend("invalid")(object)


def test_execution_conflict_is_reported_without_replacing_core(monkeypatch):
    class ConflictingExecution(ExecutionBackendDefinition):
        component_name = "local"

        def run(self, scenario, **kwargs):
            return None

    setattr(ConflictingExecution, DISCOVERED_ATTR, {"kind": "execution_backend", "key": "local"})
    from types import ModuleType

    module = ModuleType("conflicting_execution")
    module.ConflictingExecution = ConflictingExecution
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = ComponentRegistry()
    registry.scan([module.__name__])
    assert registry.get_type("execution_backend", "local").__name__ == "LocalExecution"
    assert any(error.key == "local" and error.phase == "register" for error in registry.errors)
