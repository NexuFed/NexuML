"""Verify fixture-free build identity before trusted artifact staging."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit


def verify(directory: Path, revision: str, version: str) -> None:
    """Reject mismatched provenance, unexpected targets, and acceptance fixtures.

    Raises:
        ValueError: If the downloaded candidate does not match its checked revision.
    """
    config = None
    for target in ("versioned", "changelog"):
        root = directory / target
        identity = json.loads((root / "build.json").read_text())
        expected = {"schemaVersion", "target", "version", "revision", "site", "mount"}
        if (
            set(identity) != expected
            or identity["schemaVersion"] != 1
            or identity["target"] != target
            or identity["version"] != version
            or identity["revision"] != revision
        ):
            raise ValueError(
                f"{target}: build identity does not match the published revision/config"
            )
        origin = urlsplit(identity["site"])
        if (
            origin.scheme not in {"http", "https"}
            or not origin.netloc
            or origin.path
            or origin.query
            or origin.fragment
            or origin.username
            or origin.password
            or not re.fullmatch(r"/(?:[A-Za-z0-9_-]+/)*", identity["mount"])
        ):
            raise ValueError(f"{target}: invalid hosting configuration")
        current = (identity["site"], identity["mount"])
        if config is not None and config != current:
            raise ValueError("Versioned and global builds use different hosting configuration")
        config = current
        probes = {
            "zz-api-proof",
            "zz-invalid-proof",
            "zz-generated-pipeline",
            "docs_api_probe",
            "zz-new-proof",
            "aa-order-proof",
            "zz-media-proof",
            "zz-proof",
        }
        if any(path.stem in probes for path in root.rglob("*")):
            raise ValueError(f"{target}: acceptance fixture leaked into publication output")
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError(f"{target}: publication candidate contains symlinks")
    if not (directory / "versioned/index.html").is_file():
        raise ValueError("Missing versioned documentation home")
    if not (directory / "changelog/changelog/index.html").is_file():
        raise ValueError("Missing global changelog home")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    verify(args.directory, args.revision, args.version)
    print("Exact-revision, fixture-free documentation candidates verified")
