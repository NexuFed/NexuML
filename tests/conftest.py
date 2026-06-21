"""Shared fixtures and gating helpers for the NexuML test suite."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.discovery import scan_all
from nexuml.core.registry import get_registry
from nexuml.core.scenario_registry import get_scenario_registry
from nexuml.data.auto_batch import cuda_device_info
from nexuml.data.registry import get_dataset_registry
from nexuml.evaluation.registry import get_eval_registry
from nexuml_library.data.synthetic import SyntheticDataset


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Apply environment/dependency gating markers before each test."""
    for marker in item.iter_markers("requires_gpu"):
        if not cuda_device_info().get("available"):
            pytest.skip("CUDA not available")

    for marker in item.iter_markers("requires_data"):
        if os.environ.get("NEXUML_RUN_DATA_TESTS") != "1":
            pytest.skip("NEXUML_RUN_DATA_TESTS not set")
        root = os.environ.get("NEXUML_DATA_ROOT")
        if not root or not Path(root).exists():
            pytest.skip("NEXUML_DATA_ROOT missing or invalid")

    for marker in item.iter_markers("requires_optional"):
        if not marker.args:
            continue
        dep = marker.args[0]
        try:
            __import__(dep)
        except ImportError:
            pytest.skip(f"optional dependency {dep!r} not installed")


@pytest.fixture(scope="session")
def layer_registry():
    """Default layer registry, fully loaded."""
    return get_registry()


@pytest.fixture(scope="session")
def scenario_registry():
    """Default scenario registry, fully loaded."""
    return get_scenario_registry()


@pytest.fixture(scope="session")
def dataset_registry():
    """Default dataset registry, fully loaded."""
    return get_dataset_registry()


@pytest.fixture(scope="session")
def eval_registry():
    """Default evaluation-algorithm registry, fully loaded."""
    return get_eval_registry()


@pytest.fixture(scope="session")
def scanner():
    """Fresh discovery scan of all built-in/entry-point/local-root sources."""
    return scan_all()


@pytest.fixture(scope="session")
def cuda_info():
    """CUDA device metadata."""
    return cuda_device_info()


@pytest.fixture
def synthetic_dataset(
    feature_shape: tuple[int, ...] = (16,),
    num_samples: int = 32,
    seed: int = 0,
):
    """Small synthetic dataset for fast, deterministic tests."""
    return SyntheticDataset(feature_shape=feature_shape, num_samples=num_samples, seed=seed)


@pytest.fixture
def synthetic_batch(synthetic_dataset: SyntheticDataset, batch_size: int = 4):
    """Batched (x, y) TensorDict pair from the synthetic dataset."""
    xs: list[TensorDict] = []
    ys: list[TensorDict | None] = []
    for i in range(min(batch_size, len(synthetic_dataset))):
        x, y = synthetic_dataset[i]
        xs.append(x)
        ys.append(y)
    x_batch = torch.stack(xs)  # ty: ignore[invalid-argument-type]
    y_batch = (
        torch.stack([y for y in ys if y is not None])  # ty: ignore[invalid-argument-type]
        if any(y is not None for y in ys)
        else None
    )
    return x_batch, y_batch


@pytest.fixture
def vector_scenario_spec():
    """Minimal vector scenario spec backed by synthetic data."""
    from nexuml.core.types import PipelineSpec, ScenarioSpec, TrainingSpec
    from nexuml_library.scenarios.data.synthetic import synthetic_vector_data

    return ScenarioSpec(
        name="test_vector",
        pipeline=PipelineSpec(),
        training=TrainingSpec(max_epochs=1, batch_size=4),
        data=synthetic_vector_data(feature_shape=(16,), num_samples=32),
    )


@pytest.fixture
def isolated_library_config(tmp_path, monkeypatch):
    """Redirect ``LibraryConfig``'s default path to a temp file.

    Prevents tests from reading or writing the user's real
    ``~/.config/nexuml/libraries.json``.
    """
    config_path = tmp_path / "nexuml_config" / "libraries.json"
    monkeypatch.setattr("nexuml.core.discovery.DEFAULT_CONFIG_PATH", config_path)
    return config_path


@pytest.fixture
def minimal_local_library(tmp_path):
    """Temporary local library root with one minimal decorated layer, scenario,
    data source, and eval algorithm, for discovery/registry end-to-end tests."""
    pkg_name = "nexuml_test_lib_" + uuid.uuid4().hex[:8]
    root = tmp_path / "local_lib"
    pkg_dir = root / pkg_name
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "items.py").write_text(
        f'''
from nexuml.core.discovery import data_source, eval_algorithm, layer, scenario


@layer("{pkg_name}.layer")
class MinimalLayer:
    pass


@data_source("{pkg_name}.dataset")
class MinimalDataset:
    pass


@scenario("{pkg_name}.scenario")
def minimal_scenario():
    return None


@eval_algorithm("{pkg_name}.eval")
class MinimalEval:
    pass
'''
    )
    return SimpleNamespace(
        root=root,
        package_name=pkg_name,
        layer_key=f"{pkg_name}.layer",
        dataset_key=f"{pkg_name}.dataset",
        scenario_key=f"{pkg_name}.scenario",
        eval_key=f"{pkg_name}.eval",
    )


@pytest.fixture
def small_pipeline_spec():
    """Small pipeline spec: linear encoder + reconstruction loss."""
    from nexuml.core.types import LayerSpec, PipelineSpec

    return PipelineSpec(
        stages={
            "encode": [
                LayerSpec(
                    type_key="linear_encoder",
                    keys_in=["features"],
                    keys_out=["latent"],
                    params={"hidden_dims": [32], "latent_dim": 8},
                ),
            ],
            "decode": [
                LayerSpec(
                    type_key="linear_encoder",
                    keys_in=["latent"],
                    keys_out=["reconstructed"],
                    params={"hidden_dims": [32], "latent_dim": 16},
                ),
            ],
        }
    )
