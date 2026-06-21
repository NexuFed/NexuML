"""Contract tests for every backend category exposed by ``nexuml backend list``."""

from __future__ import annotations

import pytest

from nexuml.data.export import get_export_backend, list_export_backends
from nexuml.data.loaders import get_loader_backend, list_loader_backends

# Categories without a dedicated list_*_backends() registry function. The
# data-export and data-loader categories are parametrized directly below over
# list_export_backends()/list_loader_backends() so newly registered backends
# are automatically covered.
_BACKENDS: dict[str, list[str]] = {
    "training": ["lightning"],
    "tracking": ["tensorboard", "dvclive", "mlflow"],
    "eval-storage": ["memory", "memmap"],
    "pipeline-export": ["package", "safetensors", "onnx"],
}


@pytest.fixture(params=sorted(_BACKENDS.keys()), ids=sorted(_BACKENDS.keys()))
def backend_category(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.mark.conformance
def test_backend_category_resolves(backend_category: str) -> None:
    """Every backend in a category resolves to an implementation."""
    names = _BACKENDS[backend_category]
    assert names

    if backend_category == "training":
        from nexuml.training.lightning import NexuSession

        assert NexuSession is not None

    elif backend_category == "tracking":
        from nexuml.tracking import logger

        assert logger is not None

    elif backend_category == "eval-storage":
        from nexuml.evaluation import storage

        assert storage is not None

    elif backend_category == "pipeline-export":
        from nexuml.core import export

        assert export.export_package is not None
        assert export.export_safetensors is not None
        assert export.export_onnx is not None


@pytest.mark.conformance
@pytest.mark.parametrize("backend_name", list_export_backends(), ids=list_export_backends())
def test_data_export_backend_resolves(backend_name: str) -> None:
    """Every registered data-export backend resolves to an implementation."""
    assert get_export_backend(backend_name) is not None


@pytest.mark.conformance
@pytest.mark.parametrize("backend_name", list_loader_backends(), ids=list_loader_backends())
def test_data_loader_backend_resolves(backend_name: str) -> None:
    """Every registered data-loader backend resolves to an implementation."""
    assert get_loader_backend(backend_name) is not None
