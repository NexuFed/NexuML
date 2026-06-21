"""Generate API reference pages for mkdocstrings using mkdocs-gen-files.

Walks both package roots and emits one docs page per public module,
plus a literate-nav SUMMARY.md so the nav is auto-maintained.
Only modules inside proper Python packages (with __init__.py chain) are included.
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


def add_package_root(src_root: Path, package_name: str) -> None:
    """Add an import root/package pair once."""
    package_root = (src_root.resolve(), package_name)
    existing = {(root.resolve(), name) for root, name in PACKAGE_ROOTS}
    if package_root not in existing:
        PACKAGE_ROOTS.append((src_root, package_name))


external_root = Path("external")
if external_root.exists():
    for library_root in sorted(external_root.iterdir()):
        if not library_root.is_dir():
            continue

        # src layout: external/<library>/src/<package>/__init__.py
        src_root = library_root / "src"
        if src_root.is_dir():
            for package_dir in sorted(src_root.iterdir()):
                if (package_dir / "__init__.py").exists():
                    add_package_root(src_root, package_dir.name)

        # flat layout: external/<library>/<package>/__init__.py
        else:
            if (library_root / "__init__.py").exists():
                add_package_root(external_root, library_root.name)

for src_root, _package_name in PACKAGE_ROOTS:
    resolved = str(src_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

nav = mkdocs_gen_files.Nav()


def set_nav_item(nav_obj: Any, parts: list[str], nav_path: str) -> None:
    """Set a mkdocs-gen-files nav item."""
    nav_obj[parts] = nav_path


def is_proper_package(directory: Path) -> bool:
    """Return True if every directory up the chain to root has an __init__.py."""
    return (directory / "__init__.py").exists()


def all_parents_are_packages(path: Path, src_root: Path) -> bool:
    """Return True if every parent directory between src_root and path has __init__.py."""
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

        # Skip private modules and __main__
        if any(part.startswith("_") for part in parts):
            continue

        # Skip modules not in a proper package hierarchy
        if not all_parents_are_packages(path, src_root):
            continue

        # Full virtual path in docs (written by gen-files)
        full_doc_path = Path("reference/api") / module_path.with_suffix(".md")

        # Nav path is relative to the SUMMARY.md location (reference/api/)
        nav_path = module_path.with_suffix(".md")
        set_nav_item(nav, parts, nav_path.as_posix())

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            dotted = ".".join(parts)
            fd.write(f"# `{dotted}`\n\n")
            fd.write(f"::: {dotted}\n")

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
