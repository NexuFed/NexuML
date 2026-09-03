"""Generate API reference pages for the two published NexuML packages.

The public documentation intentionally indexes only ``nexuml`` and
``nexuml_library``. Repository-local/private packages are not part of the
published API reference.
"""

from pathlib import Path
import importlib
import sys
from typing import Any

mkdocs_gen_files = importlib.import_module("mkdocs_gen_files")

PACKAGE_ROOTS = [
    (Path("src"), "nexuml"),
    (Path("library/src"), "nexuml_library"),
]

for src_root, _package_name in PACKAGE_ROOTS:
    resolved = str(src_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

nav = mkdocs_gen_files.Nav()


def set_nav_item(nav_obj: Any, parts: list[str], nav_path: str) -> None:
    """Set one generated literate-nav item."""
    nav_obj[parts] = nav_path


def all_parents_are_packages(path: Path, src_root: Path) -> bool:
    """Return whether every parent between root and module is a Python package."""
    current = path.parent
    while current != src_root:
        if not (current / "__init__.py").exists():
            return False
        current = current.parent
    return True


for src_root, package_name in PACKAGE_ROOTS:
    package_dir = src_root / package_name
    for path in sorted(package_dir.rglob("*.py")):
        module_path = path.relative_to(src_root).with_suffix("")
        parts = list(module_path.parts)

        if any(part.startswith("_") for part in parts):
            continue
        if not all_parents_are_packages(path, src_root):
            continue

        full_doc_path = Path("reference/api") / module_path.with_suffix(".md")
        nav_path = module_path.with_suffix(".md")
        set_nav_item(nav, parts, nav_path.as_posix())

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            dotted = ".".join(parts)
            fd.write(f"# `{dotted}`\n\n")
            fd.write(f"::: {dotted}\n")

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
