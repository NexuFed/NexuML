"""Verify public documentation bytes rather than trusting a Pages deployment status."""

import argparse
import json
from pathlib import Path
from time import monotonic, sleep
from urllib.request import Request, urlopen


def verify(directory: Path, url: str) -> None:
    """Compare the served version list, build identities and latest home with the archive.

    Raises:
        ValueError: If the public site serves different bytes.
    """
    paths = {"versions.json": directory / "versions.json", "": directory / "index.html"}
    for entry in json.loads(paths["versions.json"].read_bytes()):
        version = entry["version"]
        identity = directory / version / "build.json"
        if identity.is_file():
            paths[f"{version}/build.json"] = identity
        if "latest" in entry["aliases"]:
            paths[f"{version}/"] = directory / version / "index.html"
            paths["latest/"] = directory / "latest/index.html"
            if identity.is_file():
                paths["latest/build.json"] = directory / "latest/build.json"
    for route, path in paths.items():
        expected = path.read_bytes()
        request = Request(url.rstrip("/") + "/" + route, headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=10) as response:
            if response.read(len(expected) + 1) != expected:
                raise ValueError(
                    f"Live documentation differs from the validated archive: {route or '/'}"
                )


def main() -> None:
    """Allow bounded CDN propagation, then fail if the deployment is not actually served.

    Raises:
        SystemExit: If public delivery has not matched the archive within five minutes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    deadline = monotonic() + 300
    while True:
        try:
            verify(args.directory, args.url)
            break
        except (OSError, ValueError) as error:
            if monotonic() >= deadline:
                raise SystemExit(
                    f"Public site is not serving the validated archive: {error}"
                ) from error
            print(f"Waiting for public documentation delivery: {error}", flush=True)
            sleep(5)
    print("Public documentation matches the validated archive")


if __name__ == "__main__":
    main()
