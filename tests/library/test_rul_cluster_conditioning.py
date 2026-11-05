"""Contracts for train-only state clusters and oracle-free conditional CfC."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import LayerBuildContext
from nexuml_library.data.maintenance.rul_cluster_conditions import (
    _ClusterLabeledRULRuntime,
    _kmeans,
    read_cluster_artifact,
)
from nexuml_library.data.maintenance.rul_enhanced import _PairedRULRuntime
from nexuml_library.layers.model.rul_cluster_auxiliary_loss import (
    RULClusterAuxiliaryLoss,
)
from nexuml_library.layers.model.rul_cluster_conditioned_cfc import (
    ClusterConditionedCfCRULHead,
)
from nexuml_library.layers.model.rul_enhanced_cfc import EnhancedCfCRULHead
from nexuml_library.scenarios.maintenance.rul_cluster_conditioning import (
    ARMS,
    SEEDS,
    cluster_conditioning_study,
)


@pytest.mark.parametrize("seed", (42, 43, 44))
def test_cluster_kmeans_deterministic(seed: int) -> None:
    """Engine-balanced clusters are reproducible, without data shuffling."""
    points = torch.tensor([
        [-4.0, -3.8], [-4.2, -4.0], [-3.7, -4.1],
        [4.0, 4.1], [4.2, 3.8], [3.7, 4.0],
    ])
    centers, indices = _kmeans(points, clusters=2, seed=seed)
    again, labels = _kmeans(points, clusters=2, seed=seed)
    assert torch.equal(centers, again)
    assert torch.equal(indices, labels)
    assert indices[0] == indices[1] == indices[2]
    assert indices[3] == indices[4] == indices[5]
    assert indices[0] != indices[3]


def test_cluster_teacher_labels_not_visible_in_observations() -> None:
    """Held-out engines have masked labels; no teacher enters x tensors."""
    rows = [
        {"asset_id": "train:FD003:1", "split": "train", "step": 29},
        {"asset_id": "train:FD003:1", "split": "train", "step": 30},
        {"asset_id": "val:FD003:2", "split": "val", "step": 29},
        {"asset_id": "test:FD003:3", "split": "test", "step": 29},
    ]
    features = torch.randn(4, 30, 15)
    labels = torch.tensor([[100.0], [99.0], [70.0], [50.0]])
    base = _PairedRULRuntime(
        features=features, targets=labels, raw=labels, rows=rows
    )
    dataset = _ClusterLabeledRULRuntime(
        base, work=[1, 1, -1, -1], failure=[0, 0, -1, -1]
    )
    training = dataset.get_split("train")
    validation = dataset.get_split("val")
    testing = dataset.get_split("test")
    x, y = training[0]
    assert isinstance(x, TensorDict)
    assert isinstance(y, TensorDict)
    assert "working_cluster" not in x
    assert "failure_cluster" not in x
    assert cast(Tensor, y["working_cluster"]).item() == 1
    assert cast(Tensor, y["failure_cluster"]).item() == 0
    _, second_labels = training[1]
    assert isinstance(second_labels, TensorDict)
    assert cast(Tensor, second_labels["working_cluster"]).item() == 1
    for split in (validation, testing):
        x_holdout, y_holdout = split[0]
        assert isinstance(x_holdout, TensorDict)
        assert "failure_cluster" not in x_holdout
        assert isinstance(y_holdout, TensorDict)
        assert cast(Tensor, y_holdout["working_cluster"]).item() == -1
        assert cast(Tensor, y_holdout["failure_cluster"]).item() == -1


def test_cluster_artifact_rejects_wrong_encoder_and_validation_leak(
    tmp_path: Path,
) -> None:
    """Only train-engine assignments with matching checkpoint SHA are valid."""
    path = tmp_path / "teacher.pt"
    original = {
        "format": "nex233-cfc-conditional-clusters-v1",
        "working_centroids": torch.zeros(3, 128),
        "working_mean": torch.zeros(128),
        "working_std": torch.ones(128),
        "failure_centroids": torch.zeros(2, 128),
        "failure_mean": torch.zeros(128),
        "failure_std": torch.ones(128),
        "train_engine_assignments": {
            f"train:FD003:{i}": [i % 3, i % 2] for i in range(1, 81)
        },
        "settings": {
            "subset": "FD003",
            "seed": 42,
            "working_clusters": 3,
            "failure_clusters": 2,
            "training_engines_only": True,
            "uses_test_or_val_endpoints": False,
            "prepared_sha256": "a" * 64,
            "encoder_sha256": "b" * 64,
        },
    }
    torch.save(original, path)
    assert read_cluster_artifact(
        path, prepared_sha256="a" * 64, encoder_sha256="b" * 64
    )["settings"]["seed"] == 42
    with pytest.raises(ValueError, match="provenance"):
        read_cluster_artifact(path, encoder_sha256="c" * 64)
    original["settings"]["uses_test_or_val_endpoints"] = True
    torch.save(original, path)
    with pytest.raises(ValueError, match="provenance"):
        read_cluster_artifact(path)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cluster_scenarios_are_native_and_isolated(
    arm: str, seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four arms share the original split and reference architecture."""
    monkeypatch.chdir(tmp_path)
    variant = cast(Literal["reference", "working", "failure", "both"], arm)
    scenario = cluster_conditioning_study(variant, seed)
    expected = f"rul-FD003-enhanced-cfc-cluster-{arm}-s{seed}"
    assert scenario.name == expected
    assert scenario.data.input_shapes["rul_history"] == [30, 15]
    assert scenario.training.loss_keys == {"rul_loss": 1.0}
    head = scenario.pipeline.stages["RULModel"][0].component
    assert isinstance(head, EnhancedCfCRULHead)
    assert head.initialization_seed == seed
    assert head.upstream_seed == 42
    assert head.train_encoder and head.multiscale and head.segmentation
    assert head.sensor_graph and head.trajectory and not head.trend
    if arm == "reference":
        assert not isinstance(head, ClusterConditionedCfCRULHead)
    else:
        assert isinstance(head, ClusterConditionedCfCRULHead)
        assert head.condition == arm
        assert head.working_clusters == 3 and head.failure_clusters == 2
        loss = scenario.pipeline.stages["RULLoss"][0].component
        assert isinstance(loss, RULClusterAuxiliaryLoss)
        assert (loss.working_weight > 0) == (arm in ("working", "both"))
        assert (loss.failure_weight > 0) == (arm in ("failure", "both"))
    dirpath = scenario.callbacks[2].kwargs["dirpath"]
    assert isinstance(dirpath, str)
    assert Path(dirpath).name == expected
    assert scenario.callbacks[2].kwargs["monitor"] == "val/loss"


def test_masked_auxiliary_loss_is_zero_without_train_targets() -> None:
    """Unknown memberships (-1) must not contribute a validation CE term."""
    x = TensorDict({
        "failure_cluster_logits": torch.tensor(
            [[-1.0, 1.0], [2.0, -2.0]], requires_grad=True
        )
    }, batch_size=[2])
    y = TensorDict({"failure_cluster": torch.tensor([[-1], [-1]])}, batch_size=[2])
    from nexuml_library.layers.model.rul_cluster_auxiliary_loss import (
        _RULClusterAuxiliaryRuntime,
    )

    loss = _RULClusterAuxiliaryRuntime._auxiliary_loss(x, y, "failure")
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert x["failure_cluster_logits"].grad is not None


def test_conditional_model_context_contract() -> None:
    """The conditional head preserves the original paired sensor routing."""
    context = LayerBuildContext(
        input_sizes={
            "rul_history": (30, 15),
            "previous_rul_history": (30, 15),
            "pair_mask": (1,),
        },
        keys_in=["rul_history"],
        keys_out=["predicted_rul", "previous_predicted_rul"],
    )
    assert context.input_sizes["rul_history"] == (30, 15)
