"""Trusted Python scenario/tuning file loading."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from nexuml.core.types import ScenarioSpec, TuningSpec


@dataclass(frozen=True)
class LoadedScenarioFile:
    """Loaded trusted scenario file plus optional agent metadata."""

    path: Path
    scenario: ScenarioSpec
    source: str
    hypothesis: str | None = None
    parent: str | None = None
    tags: list[str] = field(default_factory=list)
    search_space: dict[str, dict[str, Any]] | None = None
    tuning_spec: TuningSpec | None = None
    build_factory: Callable[..., ScenarioSpec] | None = None


def project_root_for(path: Path) -> Path:
    """Return the project root directory for a given path."""
    start = path if path.is_dir() else path.parent
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def _module_name_for_path(path: Path, source: str) -> str:
    digest = hashlib.sha256(f"{path.resolve()}\0{source}".encode("utf-8")).hexdigest()[:16]
    return f"nexuml_scenario_file_{path.stem}_{digest}"


def _load_python_module(path: Path, source: str) -> ModuleType:
    module_name = _module_name_for_path(path, source)
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules.pop(module_name, None)
    sys.modules[module_name] = module
    added_sys_paths: list[str] = []
    for import_root in (project_root_for(path), path.parent):
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)
            added_sys_paths.append(import_root_str)
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        for import_root_str in added_sys_paths:
            try:
                sys.path.remove(import_root_str)
            except ValueError:
                pass
    return module


def _export(module: ModuleType, constant: str, function: str) -> Any:
    value = getattr(module, constant, None)
    if value is None:
        value = getattr(module, function, None)
    return value() if callable(value) else value


def _tags(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(tag) for tag in list(value or [])]


def _search_space(value: Any, path: Path) -> dict[str, dict[str, Any]] | None:
    """Validate Python scenario-file SEARCH_SPACE.

    Scalar dotted-path entries stay plain dictionaries and remain YAML-exportable.
    Conditional ``when`` branches and ``derived`` entries are Python-only.

    Returns:
        Validated search space dict, or ``None`` if *value* is ``None``.

    Raises:
        TypeError: If *value* is not a dict or an entry has an invalid type.
        ValueError: If a search space key is empty or entry type is unknown.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"Search space in {path} must be a dict, got {type(value).__name__}")
    for key, spec in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Search space keys in {path} must be non-empty strings")
        _validate_search_space_entry(key, spec, path)
    return {str(key): dict(spec) for key, spec in value.items()}


def _validate_search_space_entry(key: str, spec: Any, path: Path) -> None:
    if not isinstance(spec, dict):
        raise TypeError(f"Search space entry for {key!r} must be a dict")

    if "derived" in spec:
        rule = spec["derived"]
        if not isinstance(rule, str) and not callable(rule):
            raise TypeError(f"Search space derived entry for {key!r} must be a string or callable")
        return

    suggest_type = spec.get("type")
    if suggest_type is None and "choices" in spec:
        suggest_type = "categorical"
    if suggest_type not in {"float", "int", "categorical"}:
        raise ValueError(
            f"Search space entry for {key!r} must have type 'float', 'int', or 'categorical'"
        )

    if "when" in spec:
        when = spec["when"]
        if not isinstance(when, dict):
            raise TypeError(f"Search space 'when' for {key!r} must be a dict")
        for choice, subspace in when.items():
            if not isinstance(subspace, dict):
                raise TypeError(f"Search space 'when' branch {key!r}={choice!r} must be a dict")
            for child_key, child_spec in subspace.items():
                if not isinstance(child_key, str) or not child_key:
                    raise ValueError(f"Search space keys in {path} must be non-empty strings")
                _validate_search_space_entry(child_key, child_spec, path)


def _build_factory(value: Any, path: Path) -> Callable[..., ScenarioSpec] | None:
    if value is None:
        return None
    if not callable(value):
        raise TypeError(f"build in {path} must be callable")

    def checked_build(**params: Any) -> ScenarioSpec:
        scenario = value(**params)
        if not isinstance(scenario, ScenarioSpec):
            raise ValueError(
                f"build() in {path} must return ScenarioSpec, got {type(scenario).__name__}"
            )
        return scenario

    return checked_build


def _tuning_spec(value: Any, path: Path) -> TuningSpec | None:
    if value is None:
        return None
    if isinstance(value, TuningSpec):
        return value
    if isinstance(value, dict):
        return TuningSpec.model_validate(value)
    raise TypeError(f"Tuning spec in {path} must be TuningSpec or dict, got {type(value).__name__}")


def load_scenario_file(path: str | Path) -> LoadedScenarioFile:
    """Load a trusted Python file exposing ``scenario() -> ScenarioSpec``.

    Returns:
        Loaded scenario file with metadata, search space, and tuning spec.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not a ``.py`` file or lacks a callable ``scenario()``.
        TypeError: If ``scenario()`` does not return a ``ScenarioSpec``.
    """
    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Python file not found: {module_path}")
    if module_path.suffix != ".py":
        raise ValueError(f"Expected a .py file: {module_path}")

    source = module_path.read_text(encoding="utf-8")
    module = _load_python_module(module_path, source)
    scenario_fn = getattr(module, "scenario", None)
    if not callable(scenario_fn):
        raise ValueError(f"Scenario file must define callable scenario(): {module_path}")

    scenario = scenario_fn()
    if not isinstance(scenario, ScenarioSpec):
        raise TypeError(
            f"scenario() in {module_path} must return ScenarioSpec, got {type(scenario).__name__}"
        )

    return LoadedScenarioFile(
        path=module_path,
        scenario=scenario,
        source=source,
        hypothesis=getattr(module, "HYPOTHESIS", None),
        parent=getattr(module, "PARENT", None),
        tags=_tags(getattr(module, "TAGS", [])),
        search_space=_search_space(_export(module, "SEARCH_SPACE", "search_space"), module_path),
        tuning_spec=_tuning_spec(_export(module, "TUNING_SPEC", "tuning_spec"), module_path),
        build_factory=_build_factory(getattr(module, "build", None), module_path),
    )
