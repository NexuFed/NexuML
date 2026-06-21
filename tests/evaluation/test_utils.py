"""Pure-logic tests for nexuml.evaluation.utils."""

from __future__ import annotations

import pytest
import torch

from nexuml.evaluation.utils import (
    RAMFeatureStore,
    ReservoirSampler,
    create_feature_store,
)


def test_reservoir_sampler_collects_items():
    sampler = ReservoirSampler(max_samples=5)
    sampler.add(torch.randn(10, 3))
    assert sampler.n_seen == 10
    assert sampler.n_sampled == 5
    out = sampler.get()
    assert out is not None
    assert out.shape == (5, 3)


def test_reservoir_sampler_returns_none_when_empty():
    sampler = ReservoirSampler(max_samples=5)
    assert sampler.get() is None
    assert sampler.n_seen == 0


def test_ram_feature_store_concatenates_chunks():
    store = RAMFeatureStore()
    store.append(torch.randn(4, 8))
    store.append(torch.randn(3, 8))
    data = store.as_array()
    assert data is not None
    assert data.shape == (7, 8)


def test_ram_feature_store_respects_max_samples():
    store = RAMFeatureStore(max_samples=3)
    store.append(torch.randn(5, 4))
    data = store.as_array()
    assert data is not None
    assert data.shape == (3, 4)


def test_ram_feature_store_cleanup():
    store = RAMFeatureStore()
    store.append(torch.randn(2, 2))
    store.cleanup()
    assert store.as_array() is None


def test_create_feature_store_ram():
    store = create_feature_store("ram", max_samples=10)
    assert isinstance(store, RAMFeatureStore)


def test_create_feature_store_memmap(tmp_path):
    store = create_feature_store(
        "memmap",
        max_samples=10,
        storage_path=tmp_path / "features",
        retain_storage=False,
    )
    store.append(torch.randn(3, 4))
    arr = store.as_array()
    assert arr is not None
    assert arr.shape == (3, 4)
    store.cleanup()


def test_create_feature_store_unknown_backend():
    with pytest.raises(ValueError, match="Unknown feature storage backend"):
        create_feature_store("unknown")


def test_memmap_feature_store_grows_past_initial_capacity(tmp_path):
    store = create_feature_store(
        "memmap",
        storage_path=tmp_path / "features",
        retain_storage=False,
    )
    # Initial capacity is 1024; write more to exercise _grow.
    store.append(torch.randn(1500, 2))
    arr = store.as_array()
    assert arr is not None
    assert arr.shape[0] == 1500
    store.cleanup()


def test_memmap_feature_store_max_samples_is_initial_capacity(tmp_path):
    store = create_feature_store(
        "memmap",
        max_samples=5,
        storage_path=tmp_path / "features",
        retain_storage=False,
    )
    store.append(torch.randn(10, 2))
    arr = store.as_array()
    assert arr is not None
    # max_samples is initial capacity, not a hard limit.
    assert arr.shape == (10, 2)
    store.cleanup()
