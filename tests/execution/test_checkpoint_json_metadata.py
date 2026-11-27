"""Checkpoint metadata conversion only; no models, training or native execution."""

import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from nexuml.core.export import _load_checkpoint_metadata, _make_json_safe


@pytest.mark.parametrize(
    "value,label", [(float("inf"), "inf"), (-float("inf"), "-inf"), (float("nan"), "nan")]
)
@pytest.mark.parametrize("tensor", [False, True])
def test_checkpoint_sentinels_remain_explicit_in_strict_json(value, label, tensor):
    sentinel = torch.tensor(value) if tensor else value
    safe = _make_json_safe({"callbacks": {"kth_value": sentinel}, "finite_score": 1.25})
    assert safe == {"callbacks": {"kth_value": label}, "finite_score": 1.25}
    assert json.loads(json.dumps(safe, allow_nan=False)) == safe


def test_checkpoint_tensor_metadata_recurses_without_changing_finite_values():
    assert _make_json_safe(torch.tensor([[1.0, float("inf")], [-float("inf"), float("nan")]])) == [
        [1.0, "inf"],
        ["-inf", "nan"],
    ]


def test_model_checkpoint_callback_provenance_is_strict_json_without_touching_state():
    sentinel = torch.tensor(float("inf"))
    raw = {
        "epoch": 0,
        "global_step": 4,
        "callbacks": {"ModelCheckpoint": {"kth_value": sentinel, "best_model_score": None}},
        "state_dict": {"untouched": sentinel},
    }
    with mock.patch("nexuml.core.export.torch.load", return_value=raw):
        metadata = _load_checkpoint_metadata(Path("owned.ckpt"))
    assert metadata["callbacks"]["ModelCheckpoint"]["kth_value"] == "inf"
    assert metadata["global_step"] == 4
    assert "state_dict" not in metadata
    json.dumps({"provenance": {"source_checkpoint": metadata}}, allow_nan=False)
    assert torch.isinf(raw["state_dict"]["untouched"])
