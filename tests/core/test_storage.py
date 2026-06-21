"""Pure-logic tests for nexuml.core.storage."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.storage import SharedStorage


def test_storage_set_and_get():
    store = SharedStorage(memory_mapped=False)
    tensor = torch.randn(2, 4)
    store.set("x", tensor)
    got = store.get("x")
    assert got is not None
    assert torch.equal(got, tensor)


def test_storage_append_existing_key():
    store = SharedStorage(memory_mapped=False)
    store.set("x", torch.randn(2, 3))
    store.append("x", torch.randn(3, 3))
    got = store.get("x")
    assert got is not None
    assert got.shape == (5, 3)


def test_storage_append_new_key_initializes():
    store = SharedStorage(memory_mapped=False)
    store.append("y", torch.randn(2, 2))
    got = store.get("y")
    assert got is not None
    assert got.shape == (2, 2)


def test_storage_clear():
    store = SharedStorage(memory_mapped=False)
    store.set("x", torch.randn(2, 2))
    store.clear("x")
    assert store.get("x") is None


def test_storage_set_same_shape_updates_in_place():
    store = SharedStorage(memory_mapped=False)
    original = torch.randn(2, 2)
    store.set("x", original)
    replacement = torch.randn(2, 2)
    store.set("x", replacement)
    got = store.get("x")
    assert got is not None
    assert torch.equal(got, replacement)


def test_storage_ring_buffer():
    store = SharedStorage(memory_mapped=False)
    store.set("buf", torch.zeros(5, 2))
    counter = 0
    for i in range(3):
        value = torch.full((2, 2), float(i + 1))
        counter = store.set_ring_buffer("buf", value, counter)
    got = store.get("buf")
    assert got is not None
    # Three writes of (2,2) == 6 scalar values into a buffer of 5*2 == 10 scalars.
    assert got.sum().item() == 22.0
    assert counter == 1  # (0 + 6) % 5 = 1


def test_storage_ring_buffer_uninitialized_raises():
    store = SharedStorage(memory_mapped=False)
    with pytest.raises(ValueError, match="must be initialized"):
        store.set_ring_buffer("missing", torch.randn(1, 1), 0)


def test_storage_set_tensordict():
    store = SharedStorage(memory_mapped=False)
    td = TensorDict({"a": torch.randn(2, 3)}, batch_size=[2])
    store.set("td", td)
    got = store.get("td")
    assert isinstance(got, TensorDict)
