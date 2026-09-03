"""Pure-logic tests for nexuml.evaluation.utils."""

from __future__ import annotations

import pytest
import torch

from nexuml.core.types import DistanceEstimatorSpec
from nexuml.evaluation.storage import create_temporary_storage
from nexuml.evaluation.utils import (
    MemmapFeatureStore,
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


def test_distance_estimator_spec_creates_ram_feature_store():
    spec = DistanceEstimatorSpec(storage_backend="ram", max_samples=2)
    restored = DistanceEstimatorSpec.model_validate(spec.model_dump())
    store = restored.create_feature_store()

    assert restored.storage_backend == "ram"
    assert isinstance(store, RAMFeatureStore)
    store.append(torch.randn(3, 4))
    data = store.as_array()
    assert data is not None
    assert data.shape == (2, 4)


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


def test_distance_estimator_spec_creates_memmap_feature_store(tmp_path):
    storage_path = tmp_path / "estimator-features"
    spec = DistanceEstimatorSpec(
        storage_backend="memmap",
        storage_path=str(storage_path),
        max_samples=10,
        retain_storage=True,
    )
    restored = DistanceEstimatorSpec.model_validate(spec.model_dump())
    store = restored.create_feature_store()

    assert restored.storage_backend == "memmap"
    assert isinstance(store, MemmapFeatureStore)
    assert store.storage_path == storage_path
    assert store.max_samples == 10
    assert store.retain_storage is True
    store.append(torch.randn(3, 4))
    data = store.as_array()
    assert data is not None
    assert data.shape == (3, 4)
    store.cleanup()


def test_create_feature_store_unknown_backend():
    with pytest.raises(ValueError, match="Unknown feature storage backend"):
        create_feature_store("unknown")


def test_in_memory_storage_names_are_not_aliases():
    with pytest.raises(ValueError, match="Unknown feature storage backend"):
        create_feature_store("memory")
    with pytest.raises(ValueError, match="Unknown temporary storage backend"):
        create_temporary_storage("ram")


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
