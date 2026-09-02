"""Tests for NexuML data loaders."""

from __future__ import annotations

import pydantic
import pytest

from nexuml.core.types import LoaderSpec
from nexuml.data.loaders.definitions import DaliLoader, TorchLoader
from nexuml.data.module import NexuDataModule
from nexuml_library.data.synthetic import SyntheticDataset


def test_torch_loader_definition_builds_backend():
    assert TorchLoader().build() is not None


def test_torch_data_module():
    dataset = SyntheticDataset(feature_shape=(16,), num_samples=32).build()
    module = NexuDataModule(
        dataset=dataset,
        loader_spec=LoaderSpec(backend=TorchLoader(), batch_size=4, num_workers=0),
    )
    module.setup()
    loader = module.train_dataloader()
    batch = next(iter(loader))
    assert batch is not None
    x, y = batch
    assert x is not None


def test_torch_data_module_val_and_test_splits():
    dataset = SyntheticDataset(feature_shape=(16,), num_samples=32).build()
    module = NexuDataModule(
        dataset=dataset,
        loader_spec=LoaderSpec(backend=TorchLoader(), batch_size=4, num_workers=0),
    )
    module.setup()

    val_batch = next(iter(module.val_dataloader()))
    test_batch = next(iter(module.test_dataloader()))

    assert val_batch is not None
    assert test_batch is not None
    val_x, _ = val_batch
    test_x, _ = test_batch
    assert val_x is not None
    assert test_x is not None


def test_loader_spec_rejects_non_positive_batch_size():
    with pytest.raises(pydantic.ValidationError):
        LoaderSpec(backend=TorchLoader(), batch_size=0)


@pytest.mark.requires_optional("nvidia.dali")
def test_dali_loader_definition_builds_backend():
    assert DaliLoader().build() is not None
