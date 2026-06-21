"""Core discovery module with decorator-based registration and scanning.

Provides decorators (@layer, @data_source, @scenario, @eval_algorithm) that attach
metadata to objects, and a Scanner that collects decorated objects from imported
modules. Supports entry-point and local-root discovery.
"""

from __future__ import annotations

import importlib
import importlib.util
import importlib.metadata
import inspect
import json
import logging
import pkgutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

DISCOVERED_ATTR = "__nexuml_discovered__"


@dataclass(frozen=True)
class DiscoveredItem:
    """Metadata for a decorated discovery item."""

    kind: str
    key: str
    obj: Any
    module: str
    origin: str = "decorator"


@dataclass(frozen=True)
class DiscoveryError:
    """A failure encountered while importing or registering a discovery item.

    Collected instead of raised so that one broken module cannot hide every
    other scenario/layer/dataset/eval algorithm in the registry.
    """

    module: str
    phase: str  # "import" (failed to import) or "register" (failed to register)
    error_type: str
    message: str
    traceback: str = ""
    key: str | None = None  # discovery key, when known (register phase)

    def short(self) -> str:
        """One-line, human-readable summary for tables/log lines.

        Returns:
            Compact ``module (key): error_type: message`` string.
        """
        where = f"{self.module}" if self.key is None else f"{self.module} ({self.key})"
        return f"{where}: {self.error_type}: {self.message}"


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def _validate_key(key: str, obj: Any) -> None:
    if not isinstance(key, str) or not key:
        raise ValueError(f"Discovery key must be a non-empty string, got {key!r} for {obj!r}")


def _attach_metadata(obj: Any, kind: str, key: str) -> None:
    _validate_key(key, obj)
    existing = getattr(obj, "__dict__", {}).get(DISCOVERED_ATTR)
    if existing is not None:
        # Duplicate or conflicting key on same object
        if existing.get("key") == key and existing.get("kind") == kind:
            raise ValueError(
                f"Duplicate discovery decorator on {obj!r}: kind={kind}, key={key} already attached"
            )
        raise ValueError(
            f"Conflicting discovery decorators on {obj!r}: "
            f"existing={existing}, new=(kind={kind}, key={key})"
        )
    setattr(obj, DISCOVERED_ATTR, {"kind": kind, "key": key})


def layer(key: str) -> Callable[[Any], Any]:
    """Decorator to register a PipelineLayer subclass.

    Returns:
        Decorator that attaches discovery metadata to *cls*.
    """

    def decorator(cls: Any) -> Any:
        _attach_metadata(cls, "layer", key)
        return cls

    return decorator


def data_source(key: str) -> Callable[[Any], Any]:
    """Decorator to register a dataset class.

    Returns:
        Decorator that attaches discovery metadata to *cls*.
    """

    def decorator(cls: Any) -> Any:
        _attach_metadata(cls, "data_source", key)
        return cls

    return decorator


def scenario(key: str) -> Callable[[Any], Any]:
    """Decorator to register a scenario function.

    Returns:
        Decorator that attaches discovery metadata to *fn*.
    """

    def decorator(fn: Any) -> Any:
        _attach_metadata(fn, "scenario", key)
        return fn

    return decorator


def eval_algorithm(key: str) -> Callable[[Any], Any]:
    """Decorator to register an EvalAlgorithm subclass.

    Returns:
        Decorator that attaches discovery metadata to *cls*.
    """

    def decorator(cls: Any) -> Any:
        _attach_metadata(cls, "eval_algorithm", key)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class Scanner:
    """Collects decorated objects from imported modules."""

    def __init__(self) -> None:
        self._items: list[DiscoveredItem] = []
        self._errors: list[DiscoveryError] = []

    def _record_error(self, module: str, phase: str, exc: BaseException) -> None:
        self._errors.append(
            DiscoveryError(
                module=module,
                phase=phase,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
        )
        logger.warning(
            "Discovery %s error in %s: %s: %s",
            phase,
            module,
            type(exc).__name__,
            exc,
        )

    def scan_module(self, module: Any) -> list[DiscoveredItem]:
        """Scan a single already-imported module for decorated objects.

        Returns:
            Newly discovered items found in *module*.
        """
        found: list[DiscoveredItem] = []
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            meta = getattr(obj, "__dict__", {}).get(DISCOVERED_ATTR)
            if meta is None:
                continue
            item = DiscoveredItem(
                kind=meta["kind"],
                key=meta["key"],
                obj=obj,
                module=getattr(module, "__name__", "<unknown>"),
                origin="decorator",
            )
            found.append(item)
        self._items.extend(found)
        return found

    def scan_package(self, package_path: str) -> list[DiscoveredItem]:
        """Import a package (if possible) and recursively scan its modules.

        Resilient by design: a module that fails to import (SyntaxError,
        NameError, validation errors at import time, etc.) is recorded as a
        :class:`DiscoveryError` and skipped, so the remaining modules in the
        package are still discovered. Inspect :attr:`errors` afterwards to see
        what was skipped and why.

        Returns:
            Newly discovered items across all modules in the package.
        """
        found: list[DiscoveredItem] = []
        try:
            package = importlib.import_module(package_path)
        except Exception as exc:  # noqa: BLE001 - one bad package must not abort discovery
            self._record_error(package_path, "import", exc)
            return found

        if not hasattr(package, "__path__"):
            found.extend(self.scan_module(package))
            return found

        def _on_walk_error(name: str) -> None:
            # walk_packages routes *all* exceptions raised while descending into
            # a subpackage's __init__ here (not just ImportError) when onerror
            # is supplied, so the walk continues past broken subpackages.
            exc = sys.exc_info()[1] or RuntimeError("unknown import error")
            self._record_error(name, "import", exc)

        for _importer, modname, _ispkg in pkgutil.walk_packages(
            package.__path__, prefix=package.__name__ + ".", onerror=_on_walk_error
        ):
            try:
                module = importlib.import_module(modname)
            except Exception as exc:  # noqa: BLE001 - skip & record, never abort the scan
                self._record_error(modname, "import", exc)
                continue
            found.extend(self.scan_module(module))
        return found

    def scan_packages(self, package_paths: list[str]) -> list[DiscoveredItem]:
        """Scan multiple package paths.

        Returns:
            Combined list of discovered items from all packages.
        """
        found: list[DiscoveredItem] = []
        for path in package_paths:
            found.extend(self.scan_package(path))
        return found

    @property
    def items(self) -> list[DiscoveredItem]:
        return list(self._items)

    @property
    def errors(self) -> list[DiscoveryError]:
        """Import/walk failures collected during scanning."""
        return list(self._errors)

    def by_kind(self, kind: str) -> list[DiscoveredItem]:
        return [item for item in self._items if item.kind == kind]


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


def discover_entry_point_packages(group: str = "nexuml.libraries") -> list[str]:
    """Return package names advertised via importlib.metadata entry points."""
    packages: list[str] = []
    try:
        eps = importlib.metadata.entry_points(group=group)
    except TypeError:
        # Python < 3.12 fallback
        all_eps = importlib.metadata.entry_points()
        eps = (
            all_eps.get(group, [])
            if isinstance(all_eps, dict)
            else [ep for ep in all_eps if getattr(ep, "group", None) == group]
        )
    for ep in eps:
        try:
            ep.load()
            packages.append(ep.value)
        except Exception as exc:
            logger.warning("Failed to load entry point %s: %s", ep, exc)
    return packages


# ---------------------------------------------------------------------------
# Local library root discovery
# ---------------------------------------------------------------------------


DEFAULT_CONFIG_DIR = Path.home() / ".config" / "nexuml"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "libraries.json"


@dataclass
class LibraryConfig:
    """User-level configuration for local library roots."""

    roots: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "LibraryConfig":
        path = path or DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            roots = data.get("roots", [])
            # Normalize to absolute paths
            normalized = [str(Path(r).expanduser().resolve()) for r in roots if r]
            return cls(roots=normalized)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load library config from %s: %s", path, exc)
            return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"roots": self.roots}, indent=2))

    def add_root(self, root: str) -> None:
        abs_root = str(Path(root).expanduser().resolve())
        if abs_root not in self.roots:
            self.roots.append(abs_root)

    def remove_root(self, root: str) -> None:
        abs_root = str(Path(root).expanduser().resolve())
        if abs_root in self.roots:
            self.roots.remove(abs_root)


def _is_package_dir(dir_path: Path) -> bool:
    return (dir_path / "__init__.py").exists()


def _collect_package_paths_under_root(root: str) -> list[str]:
    """Return importable package paths under a local library root.

    Scans flat and nested packages. Assumes the root is on PYTHONPATH or
    will be added temporarily during scanning.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.debug("Library root does not exist or is not a directory: %s", root)
        return []

    scan_root = root_path / "src" if (root_path / "src").is_dir() else root_path

    # Add root to sys.path temporarily so subpackages are importable
    import sys

    str_root = str(scan_root)
    if str_root not in sys.path:
        sys.path.insert(0, str_root)

    packages: list[str] = []
    for child in scan_root.iterdir():
        if child.is_dir() and _is_package_dir(child):
            package_name = child.name
            packages.append(package_name)
            # Recurse into nested packages
            for subdir in child.rglob(""):
                if subdir == child:
                    continue
                if _is_package_dir(subdir):
                    rel = subdir.relative_to(scan_root)
                    packages.append(".".join(rel.parts))
    return packages


def discover_local_packages(config: LibraryConfig | None = None) -> list[str]:
    """Discover all importable package paths under configured local roots.

    Returns:
        Dotted package path strings discovered under all configured roots.
    """
    config = config or LibraryConfig.load()
    packages: list[str] = []
    for root in config.roots:
        packages.extend(_collect_package_paths_under_root(root))
    return packages


def discover_library_packages(
    include_entry_points: bool = True, include_local_roots: bool = True
) -> list[str]:
    """Return package paths from installed entry points and configured roots."""
    packages: list[str] = []
    if importlib.util.find_spec("nexuml_library") is not None:
        packages.append("nexuml_library")
    if include_entry_points:
        packages.extend(discover_entry_point_packages())
    if include_local_roots:
        packages.extend(discover_local_packages())
    return list(dict.fromkeys(packages))


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def register_items(
    items: list[DiscoveredItem],
    register: Callable[[str, Any], None],
    errors: list[DiscoveryError],
) -> None:
    """Register each discovered item, collecting (not raising) failures.

    Mirrors the resilience of :meth:`Scanner.scan_package`: a single key
    conflict or bad item must not wipe out every other item in the registry.
    Appends a :class:`DiscoveryError` (phase="register") for each failure.
    """
    for item in items:
        try:
            register(item.key, item.obj)
        except Exception as exc:  # noqa: BLE001 - record & continue, never abort
            errors.append(
                DiscoveryError(
                    module=item.module,
                    phase="register",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    key=item.key,
                )
            )
            logger.warning(
                "Discovery register error for %s (%s): %s: %s",
                item.key,
                item.module,
                type(exc).__name__,
                exc,
            )


def scan_all(
    extra_package_paths: list[str] | None = None,
    include_entry_points: bool = True,
    include_local_roots: bool = True,
) -> Scanner:
    """Scan built-in packages, entry points, and local roots for decorated items.

    No persistent object cache is used; this always performs fresh imports.

    Returns:
        Scanner populated with all discovered items.
    """
    scanner = Scanner()

    for pkg in discover_library_packages(include_entry_points, include_local_roots):
        scanner.scan_package(pkg)

    # Extra manual paths
    if extra_package_paths:
        scanner.scan_packages(extra_package_paths)

    return scanner
