"""Distributional validation must ignore endpoint pair assignments and test data."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nexuml.core.components import LayerBuildContext
from nexuml_library.evaluation.maintenance.ablation_validation import (
    DistributionalTransportValidation,
    score_distributional_transport,
)
from nexuml_library.layers.model.flow import Flow, _FlowRuntime


def _flow() -> _FlowRuntime:
    """Build a small untrained vector field without external artifacts."""
    model = Flow(hidden_dims=[8, 8], num_steps=4).build(
        LayerBuildContext(
            input_sizes={"state": (2,), "time": (1,)},
            keys_in=["state", "time"],
            keys_out=["velocity"],
        )
    )
    assert isinstance(model, _FlowRuntime)
    return model


def test_distributional_score_is_pairing_invariant_and_deterministic() -> None:
    """Reordering faulty observations cannot change distributional validation."""
    healthy = torch.tensor(
        [[0.0, 0.0], [1.0, 0.3], [2.0, -0.2], [3.0, 0.4]],
    )
    faulty = healthy + torch.tensor([0.5, -0.4])
    model = _flow()
    first = score_distributional_transport(
        model,
        healthy,
        faulty,
        device=torch.device("cpu"),
        batch_size=2,
        projections=32,
        seed=12,
    )
    reordered = score_distributional_transport(
        model,
        healthy,
        faulty[[2, 0, 3, 1]],
        device=torch.device("cpu"),
        batch_size=3,
        projections=32,
        seed=12,
    )
    assert first == pytest.approx(reordered, abs=1e-7)
    assert first["val/transport_swd"] >= 0.0
    assert first["val/swd_relative_reduction"] == pytest.approx(
        1.0 - first["val/transport_swd"] / first["val/healthy_to_faulty_swd"]
    )


def test_distributional_callback_uses_only_validation_features(tmp_path: Path) -> None:
    """Train/test values and pair assignments cannot enter selection data."""
    features = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0]],
            [[5.0, 5.0], [6.0, 6.0]],
            [[7.0, 7.0], [8.0, 8.0]],
        ]
    )
    rows = [
        {"split": "train", "group_id": "fan:id_00"},
        {"split": "val", "group_id": "fan:id_04"},
        {"split": "val", "group_id": "fan:id_04"},
        {"split": "test", "group_id": "fan:id_06"},
    ]
    file = tmp_path / "prepared.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": features,
            "rows": rows,
            "settings": {"seed": 42},
        },
        file,
    )
    callback = DistributionalTransportValidation(file, seed=42)
    healthy, faulty = callback._get_validation()
    assert torch.equal(healthy, features[1:3, 0])
    assert torch.equal(faulty, features[1:3, 1])
    assert len(healthy) == 2
