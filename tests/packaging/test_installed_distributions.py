import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    "NEXUML_INSTALLED_DIST_TEST" not in os.environ,
    reason="installed distribution environment was not requested",
)


def _assert_no_source_paths() -> None:
    source_roots = {Path("src").resolve(), Path("library/src").resolve()}
    assert source_roots.isdisjoint(Path(path).resolve() for path in sys.path if path)


def test_core_distribution() -> None:
    _assert_no_source_paths()
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.distribution("nexuml-library")

    import nexuml
    import torch.nn as nn

    assert nexuml.__version__ == importlib.metadata.version("nexuml") == "0.2.0"
    assert nexuml.nn_module(nn.Identity).factory == "torch.nn.modules.linear:Identity"
    subprocess.run([sys.executable, "-I", "-m", "nexuml", "--help"], check=True)


def test_library_distribution() -> None:
    _assert_no_source_paths()
    assert importlib.metadata.version("nexuml-library") == "0.2.0"
    entry_points = importlib.metadata.entry_points(group="nexuml.libraries")
    base = next((entry for entry in entry_points if entry.name == "base"), None)
    assert base is not None and base.value == "nexuml_library"
    base.load()

    from nexuml.core.discovery import Scanner

    scanner = Scanner()
    scanner.scan_package("nexuml_library")
    discovered = {(item.kind, item.key) for item in scanner.items}
    assert {("layer", "LinearEncoder"), ("scenario", "cifar-resnet")} <= discovered

    from nexuml.data.creator import NexuDataCreator
    from nexuml.data.loaders.definitions import TorchLoader
    from nexuml_library.scenarios.data.synthetic import synthetic_vector_data

    data_spec = synthetic_vector_data(feature_shape=(4,), num_samples=12)
    assert isinstance(data_spec.loader.backend, TorchLoader)

    data_module = NexuDataCreator().build(data_spec, default_batch_size=4)
    data_module.setup("fit")
    features, targets = next(iter(data_module.train_dataloader()))
    assert features.batch_size == (4,)
    assert features["features"].shape == (4, 4)
    assert targets is not None
    assert targets["reconstruction_target"].shape == (4, 4)
