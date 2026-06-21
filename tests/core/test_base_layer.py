"""Tests for nexuml.core.base_layer."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from typing import cast

from nexuml.core.base_layer import PipelineLayer


class _DoubleLayer(PipelineLayer):
    def forward_tensor(self, x: torch.Tensor, y=None) -> torch.Tensor:
        return x * 2


@pytest.fixture
def double_layer():
    return _DoubleLayer(
        input_sizes={"a": (4,)},
        keys_in=["a"],
        keys_out=["b"],
    )


def test_pipeline_layer_forward_returns_tuple(double_layer):
    x = TensorDict({"a": torch.randn(2, 4)}, batch_size=[2])
    x_out, y_out = double_layer(x, None)
    assert isinstance(x_out, TensorDict)
    assert y_out is None
    assert "b" in x_out.keys()


def test_merge_td_single_key(double_layer):
    x = TensorDict({"a": torch.randn(2, 4)}, batch_size=[2])
    merged = double_layer.merge_td(x)
    assert torch.equal(merged, cast(torch.Tensor, x["a"]))


def test_merge_td_multiple_keys():
    layer = _DoubleLayer(
        input_sizes={"a": (4,), "c": (4,)},
        keys_in=["a", "c"],
        keys_out=["b"],
    )
    x = TensorDict(
        {"a": torch.randn(2, 4), "c": torch.randn(2, 4)},
        batch_size=[2],
    )
    merged = layer.merge_td(x)
    expected = torch.cat([cast(torch.Tensor, x["a"]), cast(torch.Tensor, x["c"])], dim=1)
    assert torch.equal(merged, expected)
