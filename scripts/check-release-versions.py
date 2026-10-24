"""Require one stable version for both Python distributions and Studio."""

import argparse
import json
from pathlib import Path
import re
import tomllib


def main():
    """Validate checked-in versions and an optional tag without changing metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Stable vX.Y.Z tag to validate against all packages")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    versions = {
        str(path): tomllib.loads((root / path).read_text())["project"]["version"]
        for path in (Path("pyproject.toml"), Path("library/pyproject.toml"))
    }
    for path in ("studio/package.json", "studio/package-lock.json"):
        package = json.loads((root / path).read_text())
        versions[path] = package["version"]
        if path.endswith("package-lock.json"):
            versions["studio lock root"] = package["packages"][""]["version"]
    version = versions["pyproject.toml"]
    if len(set(versions.values())) != 1:
        parser.error(f"Package versions must match: {versions}")
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        parser.error(f"Expected stable X.Y.Z package version: {version!r}")
    if args.tag is not None and args.tag != f"v{version}":
        parser.error(f"Tag {args.tag!r} must match package version v{version}")
    print(version)


if __name__ == "__main__":
    main()
