"""Distributional validation must ignore endpoint pair assignments and test data."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nexuml.core.components import LayerBuildContext
from nexuml.training.lightning import NexuSession
from nexuml_library.evaluation.maintenance.ablation_evaluate import (
    evaluate_prepared_checkpoints,
)
from nexuml_library.evaluation.maintenance.ablation_validation import (
    DistributionalTransportValidation,
    score_distributional_transport,
)
from nexuml_library.layers.model.flow import Flow, _FlowRuntime
from nexuml_library.scenarios.flow.mimii_acoustic_ablation import mimii_acoustic_ablation


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
        healthy[[1, 3, 0, 2]],
        faulty[[2, 0, 3, 1]],
        device=torch.device("cpu"),
        batch_size=3,
        projections=32,
        seed=12,
    )
    assert first == pytest.approx(reordered, rel=1e-5, abs=1e-6)
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


def test_validation_swd_reaches_checkpoint_selection(tmp_path: Path) -> None:
    """A two-epoch CPU run selects on distributional SWD and seals the test split."""
    features = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.5]],
            [[0.2, 0.1], [1.2, 0.6]],
            [[0.4, 0.3], [1.4, 0.8]],
            [[0.6, 0.4], [1.6, 0.9]],
            [[2.0, 1.0], [3.0, 1.5]],
            [[2.2, 1.1], [3.2, 1.6]],
            [[20.0, 10.0], [30.0, 15.0]],
            [[22.0, 11.0], [32.0, 16.0]],
        ]
    )
    splits = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
    groups = ["fan:id_00"] * 4 + ["fan:id_04"] * 2 + ["fan:id_06"] * 2
    rows = [
        {"split": split, "group_id": group, "paired": 1}
        for split, group in zip(splits, groups, strict=True)
    ]
    prepared_file = tmp_path / "prepared-seed7.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": features,
            "rows": rows,
            "settings": {"seed": 7},
        },
        prepared_file,
    )

    spec = mimii_acoustic_ablation(
        prepared_file,
        hidden_dims=[8, 8],
        num_steps=4,
        batch_size=2,
        max_epochs=2,
        patience=3,
        seed=7,
        checkpoint_root=tmp_path / "checkpoints",
    )
    session = NexuSession.from_scenario(
        spec,
        accelerator="cpu",
        devices=1,
        log_dir=tmp_path / "logs",
        enable_loggers=False,
        enable_progress_bar=False,
    )
    session.setup().fit()
    validation = session.validate()
    assert validation
    assert "val/transport_swd" in validation[0]
    assert "val/swd_relative_reduction" in validation[0]

    directory = tmp_path / "checkpoints" / "prepared-seed7-mlp-8x8"
    last = torch.load(directory / "last.ckpt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 1
    assert last["global_step"] == 4
    assert any(
        "val/transport_swd" in name and state.get("best_model_score") is not None
        for name, state in last["callbacks"].items()
        if isinstance(state, dict)
    )

    report = evaluate_prepared_checkpoints(
        directory, device=torch.device("cpu"), projections=8, seed=7
    )
    assert report["test_is_sealed"]
    assert "held_out_test_best" not in report
    assert report["training_best"]["num_samples"] == 4
    assert report["training_last"]["num_samples"] == 4
    assert set(report["training_best"]["by_machine"]) == {"fan:id_00"}
    assert report["validation_best"]["num_samples"] == 2
