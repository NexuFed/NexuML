"""Hermetic public-snapshot byte boundaries; no S3, reader, native or GPU calls."""

import hashlib
import io
import json
import runpy
from types import SimpleNamespace
from unittest import mock

import pytest


@pytest.fixture
def probe():
    return runpy.run_path("tests/fixtures/physical_ai_public_data.py")


def entry(payload=b"public"):
    return {
        "path": "meta/info.json",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_manifest_rejects_unpinned_bytes(probe):
    with pytest.raises(ValueError, match="pinned"):
        probe["manifest"](b'{"files": []}')


@pytest.mark.parametrize("path", ["../escape", "/escape", "meta\\escape"])
def test_manifest_rejects_unsafe_paths_even_with_matching_digest(probe, path):
    files = [{**entry(), "path": f"file-{index}"} for index in range(15)]
    files[0].update(path=path, bytes=624607954 - 14 * 6)
    payload = json.dumps({"files": files}).encode()
    with mock.patch.dict(
        probe["manifest"].__globals__, MANIFEST_SHA256=hashlib.sha256(payload).hexdigest()
    ):
        with pytest.raises(ValueError, match="unsafe"):
            probe["manifest"](payload)


@pytest.mark.parametrize(
    "payload,size,digest,match",
    [
        (b"public", 6, hashlib.sha256(b"public").hexdigest(), None),
        (b"public!", 6, hashlib.sha256(b"public").hexdigest(), "exceeds"),
        (b"publi", 6, hashlib.sha256(b"public").hexdigest(), "content"),
        (b"public", 6, "0" * 64, "content"),
    ],
)
def test_download_exact_bytes_and_hash(probe, tmp_path, payload, size, digest, match):
    client = mock.Mock()
    client.get_object.return_value = {"Body": io.BytesIO(payload), "ContentLength": size}
    files = [{"path": "data/file.parquet", "bytes": size, "sha256": digest}]
    with mock.patch.object(
        probe["shutil"], "disk_usage", return_value=SimpleNamespace(free=8 * 1024**3)
    ):
        if match:
            with pytest.raises(ValueError, match=match):
                probe["download"](client, {"files": files}, tmp_path)
        else:
            probe["download"](client, {"files": files}, tmp_path)
            assert (tmp_path / "data/file.parquet").read_bytes() == payload
    client.get_object.assert_called_once_with(
        Bucket="rglitza", Key=probe["PREFIX"] + "data/file.parquet"
    )


def test_download_refuses_disk_floor_before_network(probe, tmp_path):
    client = mock.Mock()
    with mock.patch.object(
        probe["shutil"], "disk_usage", return_value=SimpleNamespace(free=probe["FLOOR"])
    ):
        with pytest.raises(RuntimeError, match="disk floor"):
            probe["download"](client, {"files": [entry()]}, tmp_path)
    client.get_object.assert_not_called()


def test_download_does_not_overwrite_existing_file(probe, tmp_path):
    path = tmp_path / "meta/info.json"
    path.parent.mkdir()
    path.write_bytes(b"original")
    client = mock.Mock()
    client.get_object.return_value = {"Body": io.BytesIO(b"public"), "ContentLength": 6}
    with mock.patch.object(
        probe["shutil"], "disk_usage", return_value=SimpleNamespace(free=8 * 1024**3)
    ):
        with pytest.raises(FileExistsError):
            probe["download"](client, {"files": [entry()]}, tmp_path)
    assert path.read_bytes() == b"original"
