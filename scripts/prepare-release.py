"""Prepare shared release versions in a disposable stable-tag checkout."""

import argparse
from pathlib import Path

from release_versions import prepare_release


def main():
    """Stamp the existing tag's version without creating commits or tags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Existing full refs/tags/vX.Y.Z ref")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        version = prepare_release(args.root.resolve(), args.ref)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(version)


if __name__ == "__main__":
    main()
