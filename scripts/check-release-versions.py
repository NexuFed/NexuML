"""Require one stable version for both Python distributions and Studio."""

import argparse
from pathlib import Path

from release_versions import validate_versions


def main():
    """Validate checked-in versions and an optional tag without changing metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Stable vX.Y.Z tag to validate against all packages")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        version = validate_versions(args.root.resolve(), args.tag)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(version)


if __name__ == "__main__":
    main()
