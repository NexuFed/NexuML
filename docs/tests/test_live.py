"""A successful Pages API response is not proof that the built site is served."""

from email.message import Message
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_live", ROOT / "docs/scripts/verify_live.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def archive(tmp_path):
    versions = [
        {"version": "main", "aliases": []},
        {"version": "v0.2.3", "aliases": ["latest"]},
        {"version": "v0.2.1", "aliases": []},
    ]
    (tmp_path / "versions.json").write_text(json.dumps(versions))
    (tmp_path / "index.html").write_text('<meta http-equiv="refresh" content="0;url=v0.2.3/">')
    for version in ("main", "v0.2.3"):
        root = tmp_path / version
        root.mkdir()
        (root / "build.json").write_text(json.dumps({"version": version, "revision": "1" * 40}))
        (root / "index.html").write_text(f"<h1>{version}</h1>")
    latest = tmp_path / "latest"
    latest.mkdir()
    for name in ("index.html", "build.json"):
        (latest / name).write_bytes((tmp_path / "v0.2.3" / name).read_bytes())
    return tmp_path


def serve(monkeypatch, archive, changed=None):
    responses = {
        "versions.json": (archive / "versions.json").read_bytes(),
        "": (archive / "index.html").read_bytes(),
        "latest/": (archive / "latest/index.html").read_bytes(),
        "latest/build.json": (archive / "latest/build.json").read_bytes(),
        "main/build.json": (archive / "main/build.json").read_bytes(),
        "v0.2.3/build.json": (archive / "v0.2.3/build.json").read_bytes(),
        "v0.2.3/": (archive / "v0.2.3/index.html").read_bytes(),
    }
    if changed is not None:
        responses[changed] = b"stale deployment"

    def open_request(request, timeout):
        assert request.get_header("Cache-control") == "no-cache"
        assert timeout == 10
        return BytesIO(responses[request.full_url.removeprefix("https://docs.example/NexuML/")])

    monkeypatch.setattr(module, "urlopen", open_request)


def test_live_archive_matches_including_legacy_versions_without_build_identity(
    monkeypatch, archive
):
    serve(monkeypatch, archive)
    module.verify(archive, "https://docs.example/NexuML/")


@pytest.mark.parametrize(
    "route",
    [
        "versions.json",
        "",
        "latest/",
        "latest/build.json",
        "main/build.json",
        "v0.2.3/build.json",
        "v0.2.3/",
    ],
)
def test_live_stale_bytes_fail(monkeypatch, archive, route):
    serve(monkeypatch, archive, route)
    with pytest.raises(ValueError, match="differs"):
        module.verify(archive, "https://docs.example/NexuML")


def test_live_missing_release_route_fails(monkeypatch, archive):
    def missing(request, timeout):
        raise HTTPError(request.full_url, 404, "Not found", Message(), None)

    monkeypatch.setattr(module, "urlopen", missing)
    with pytest.raises(HTTPError):
        module.verify(archive, "https://docs.example/NexuML/")


def test_cli_retries_transient_stale_delivery(monkeypatch, archive):
    monkeypatch.setattr(
        sys, "argv", ["verify_live", "--directory", str(archive), "--url", "https://docs.example/"]
    )
    attempts = []

    def verify(directory, url):
        attempts.append((directory, url))
        if len(attempts) == 1:
            raise ValueError("stale")

    delays = []
    monkeypatch.setattr(module, "verify", verify)
    monkeypatch.setattr(module, "sleep", delays.append)
    module.main()
    assert attempts == [(archive, "https://docs.example/")] * 2
    assert delays == [5]


def test_cli_stops_after_propagation_deadline(monkeypatch, archive):
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_live", "--directory", str(archive), "--url", "https://docs.example/NexuML/"],
    )
    serve(monkeypatch, archive, "versions.json")
    clock = iter((0, 300))
    monkeypatch.setattr(module, "monotonic", lambda: next(clock))
    with pytest.raises(SystemExit, match="not serving the validated archive"):
        module.main()
