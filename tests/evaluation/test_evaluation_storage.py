"""Tests for nexuml.evaluation.storage temporary TensorDict storage."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nexuml.evaluation.storage import (
    AppendableTensorDictBuffer,
    ReservoirTensorDictBuffer,
    create_temporary_storage,
)


def _batch(values: list[float]) -> TensorDict:
    return TensorDict({"x": torch.tensor(values).unsqueeze(-1)}, batch_size=[len(values)])


def test_memory_storage_set_get_roundtrip():
    storage = create_temporary_storage("memory", max_samples=4)

    for i, row in enumerate(_batch([1.0, 2.0, 3.0])):
        storage.set(i, row.unsqueeze(0))

    result = storage.get()
    assert result is not None
    assert len(storage) == 3
    assert result.batch_size[0] == 3
    assert set(result.keys()) == {"x"}
    x = result["x"]
    assert isinstance(x, torch.Tensor)
    assert torch.equal(x.squeeze(-1), torch.tensor([1.0, 2.0, 3.0]))


def test_memory_storage_rejects_multi_row_item():
    storage = create_temporary_storage("memory", max_samples=4)

    with pytest.raises(ValueError):
        storage.set(0, _batch([1.0, 2.0]))


def test_memory_storage_rejects_out_of_range_index():
    storage = create_temporary_storage("memory", max_samples=4)

    with pytest.raises(IndexError):
        storage.set(5, _batch([1.0]))


def test_create_temporary_storage_unknown_backend():
    with pytest.raises(ValueError):
        create_temporary_storage("not-a-backend")


def test_appendable_buffer_truncates_at_capacity():
    buffer = AppendableTensorDictBuffer(max_samples=2)

    buffer.add_batch(_batch([1.0, 2.0, 3.0]))

    assert len(buffer) == 2
    assert buffer.truncated is True
    result = buffer.get()
    assert result is not None
    x = result["x"]
    assert isinstance(x, torch.Tensor)
    assert torch.equal(x.squeeze(-1), torch.tensor([1.0, 2.0]))


def test_appendable_buffer_under_capacity_not_truncated():
    buffer = AppendableTensorDictBuffer(max_samples=4)

    buffer.add_batch(_batch([1.0, 2.0]))

    assert len(buffer) == 2
    assert buffer.truncated is False


def test_reservoir_buffer_caps_at_max_samples():
    buffer = ReservoirTensorDictBuffer(max_samples=3)

    buffer.add_batch(_batch([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert len(buffer) == 3
    assert buffer.n_seen == 5
    result = buffer.get()
    assert result is not None
    assert result.batch_size[0] == 3


@pytest.mark.requires_optional("torchrl")
def test_memmap_storage_set_get_and_finalize(tmp_path):
    storage = create_temporary_storage("memmap", max_samples=4, storage_path=tmp_path / "storage")

    for i, row in enumerate(_batch([1.0, 2.0])):
        storage.set(i, row.unsqueeze(0))

    result = storage.get()
    assert result is not None
    assert len(storage) == 2
    x = result["x"]
    assert isinstance(x, torch.Tensor)
    assert torch.equal(x.squeeze(-1).cpu(), torch.tensor([1.0, 2.0]))

    storage.finalize()
