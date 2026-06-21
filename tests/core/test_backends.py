"""Tests for backend discovery/registration."""

from __future__ import annotations

from nexuml.data.export import list_export_backends
from nexuml.data.loaders import list_loader_backends


def test_export_backends():
    backends = list_export_backends()
    assert "numpy" in backends
    assert "numpy_mmap" in backends
    assert "torch" in backends
    assert "tensordict_memmap" in backends
    assert "webdataset" in backends


def test_loader_backends():
    backends = list_loader_backends()
    assert "torch" in backends


def test_tracking_backend_imports():
    from nexuml.tracking import logger

    assert logger.create_loggers is not None
