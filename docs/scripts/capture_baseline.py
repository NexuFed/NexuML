"""Capture reproducible source/route and published-archive migration evidence."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args: str) -> bytes:
    """Read git objects without checking out or modifying archive files.

    Returns:
        The requested git object's raw output.
    """
    return subprocess.check_output(["git", *args], cwd=ROOT)


def capture(source: str, archive: str, routes: dict) -> dict:
    """Return authored headings and git tree digests, including modes and paths.

    Raises:
        ValueError: If a page, asset, or version is missing from the inventory.
    """
    source = git("rev-parse", source).decode().strip()
    archive = git("rev-parse", archive).decode().strip()
    files = git("ls-tree", "-r", "--name-only", source, "docs/").decode().splitlines()
    pages = {p.removeprefix("docs/") for p in files if p.endswith(".md")}
    if pages != set(routes["pages"]):
        raise ValueError(f"Unmapped or unexpected pages: {pages ^ set(routes['pages'])}")
    inventory = {}
    for page in sorted(pages):
        content = git("show", f"{source}:docs/{page}")
        inventory[page] = {
            "target": routes["pages"][page],
            "sha256": hashlib.sha256(content).hexdigest(),
            "headings": re.findall(r"^#{1,6} (.+)$", content.decode(), re.MULTILINE),
        }
    for asset in routes["downloads"]:
        if f"docs/{asset}" not in files:
            raise ValueError(f"Missing download: {asset}")
    versions = json.loads(git("show", f"{archive}:versions.json"))
    trees = {}
    entries = git("ls-tree", archive).decode().splitlines()
    for entry in entries:
        metadata, name = entry.split("\t", 1)
        if metadata.split()[1] == "tree":
            listing = git("ls-tree", "-r", "-z", f"{archive}:{name}")
            trees[name] = {
                "gitTree": metadata.split()[2],
                "manifestSha256": hashlib.sha256(listing).hexdigest(),
                "files": len(listing.rstrip(b"\0").split(b"\0")),
            }
    for version in versions:
        if version["version"] not in trees:
            raise ValueError(f"Manifest version has no archive tree: {version}")
    return {
        "sourceCommit": source,
        "archiveCommit": archive,
        "archiveRootEntries": entries,
        "versions": versions,
        "latest": git("show", f"{archive}:latest").decode(),
        "trees": trees,
        "pages": inventory,
        "downloads": routes["downloads"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    routes = json.loads((ROOT / "docs/migration/route-map.json").read_text())
    result = capture(args.source, args.archive, routes)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Captured {len(result['pages'])} pages and {len(result['trees'])} archive trees")
