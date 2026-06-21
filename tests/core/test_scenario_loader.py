"""Pure-logic tests for nexuml.core.scenario_loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexuml.core.scenario_loader import (
    _build_factory,
    _export,
    _load_python_module,
    _module_name_for_path,
    _search_space,
    _tags,
    _tuning_spec,
    _validate_search_space_entry,
    load_scenario_file,
    project_root_for,
)
from nexuml.core.types import ScenarioSpec, TuningSpec


def test_project_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "nested" / "file.py"
    sub.parent.mkdir(parents=True)
    sub.write_text("")
    assert project_root_for(sub) == tmp_path


def test_project_root_finds_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    assert project_root_for(tmp_path) == tmp_path


def test_project_root_defaults_to_cwd(tmp_path):
    # tmp_path has neither .git nor pyproject.toml
    assert project_root_for(tmp_path) == Path.cwd().resolve()


def test_module_name_is_deterministic(tmp_path):
    path = tmp_path / "my_scenario.py"
    path.write_text("")
    name1 = _module_name_for_path(path, "source")
    name2 = _module_name_for_path(path, "source")
    assert name1 == name2
    assert name1.startswith("nexuml_scenario_file_my_scenario_")


def test_load_python_module(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("x = 42\n")
    module = _load_python_module(path, path.read_text())
    assert module.x == 42


def test_export_prefers_constant(tmp_path):
    module = _load_python_module(tmp_path / "mod.py", "VALUE = 7\ndef value(): return 9\n")
    assert _export(module, "VALUE", "value") == 7


def test_export_falls_back_to_callable(tmp_path):
    module = _load_python_module(tmp_path / "mod.py", "def value(): return 11\n")
    assert _export(module, "VALUE", "value") == 11


def test_export_returns_none_when_missing(tmp_path):
    module = _load_python_module(tmp_path / "mod.py", "")
    assert _export(module, "VALUE", "value") is None


def test_tags_from_string():
    assert _tags("fast") == ["fast"]


def test_tags_from_iterable():
    assert _tags(["a", "b"]) == ["a", "b"]


def test_tags_from_none():
    assert _tags(None) == []


def test_search_space_accepts_valid_dict(tmp_path):
    space = {
        "lr": {"type": "float", "low": 1e-5, "high": 1e-1},
        "mode": {"type": "categorical", "choices": ["a", "b"]},
    }
    assert _search_space(space, tmp_path / "scenario.py") == space


def test_search_space_with_int_type(tmp_path):
    space = {"units": {"type": "int", "low": 8, "high": 128}}
    _validate_search_space_entry("units", space["units"], tmp_path / "scenario.py")


def test_search_space_rejects_invalid_type(tmp_path):
    with pytest.raises(ValueError, match="type 'float', 'int', or 'categorical'"):
        _search_space({"x": {"type": "unknown"}}, tmp_path / "scenario.py")


def test_search_space_rejects_non_dict(tmp_path):
    with pytest.raises(TypeError, match="must be a dict"):
        _search_space([1, 2, 3], tmp_path / "scenario.py")


def test_search_space_derived_entry(tmp_path):
    space = {"x": {"derived": "lambda p: p['lr'] * 10"}}
    assert _search_space(space, tmp_path / "scenario.py") == space


def test_search_space_conditional_branch(tmp_path):
    space = {
        "mode": {
            "type": "categorical",
            "choices": ["a", "b"],
            "when": {"a": {"x": {"type": "int", "low": 0, "high": 1}}},
        }
    }
    assert _search_space(space, tmp_path / "scenario.py") == space


def test_build_factory_validates_return_type(tmp_path):
    def bad(**kwargs):
        return SimpleNamespace()

    factory = _build_factory(bad, tmp_path / "scenario.py")
    assert factory is not None
    with pytest.raises(ValueError, match="must return ScenarioSpec"):
        factory()


def test_build_factory_returns_scenario(tmp_path):
    def good(**kwargs):
        return ScenarioSpec(name="ok")

    factory = _build_factory(good, tmp_path / "scenario.py")
    assert factory is not None
    assert factory().name == "ok"


def test_tuning_spec_from_dict(tmp_path):
    spec = _tuning_spec({"n_trials": 5}, tmp_path / "scenario.py")
    assert isinstance(spec, TuningSpec)
    assert spec.n_trials == 5


def test_tuning_spec_already_object(tmp_path):
    original = TuningSpec(n_trials=3)
    assert _tuning_spec(original, tmp_path / "scenario.py") is original


def test_tuning_spec_rejects_invalid_type(tmp_path):
    with pytest.raises(TypeError, match="TuningSpec or dict"):
        _tuning_spec("bad", tmp_path / "scenario.py")


def test_load_scenario_file(tmp_path):
    path = tmp_path / "scenario.py"
    path.write_text(
        "from nexuml.core.types import ScenarioSpec\n"
        "def scenario():\n"
        "    return ScenarioSpec(name='loaded')\n"
    )
    loaded = load_scenario_file(path)
    assert loaded.scenario.name == "loaded"
    assert loaded.source == path.read_text()


def test_load_scenario_file_missing():
    with pytest.raises(FileNotFoundError):
        load_scenario_file("/nonexistent/path/scenario.py")


def test_load_scenario_file_not_python(tmp_path):
    path = tmp_path / "scenario.txt"
    path.write_text("")
    with pytest.raises(ValueError, match="Expected a .py file"):
        load_scenario_file(path)
