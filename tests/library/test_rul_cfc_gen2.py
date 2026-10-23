"""Tests for engine-equal CfC supervision and validation-only optimizer studies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.components import LayerBuildContext
from nexuml_library.data.maintenance.rul_sequence_flow import (
    AutoPatchFormerSequenceFlowRULFeatures,
    _SequenceFlowRULRuntime,
    _inverse_engine_weights,
)
from nexuml_library.evaluation.maintenance.rul_generalization_diagnostics import (
    _validation_engine_groups,
)
from nexuml_library.evaluation.maintenance.rul_gen2_summary import (
    summarize_cfc_gen2,
)
from nexuml_library.layers.model.rul_engine_loss import (
    RULInverseEngineFrequencyLoss,
    _RULInverseEngineFrequencyRuntime,
)
from nexuml_library.scenarios.maintenance.rul_gen2 import (
    cfc_rul_optimization_study,
)


def test_engine_frequency_weights_are_train_only() -> None:
    """Every training engine contributes the same sum of sample weights."""
    rows = [
        *[{"split": "train", "asset_id": "a"} for _ in range(4)],
        *[{"split": "train", "asset_id": "b"} for _ in range(2)],
        {"split": "val", "asset_id": "v"},
        {"split": "test", "asset_id": "t"},
    ]
    weights, counts = _inverse_engine_weights(rows)
    assert counts == {"a": 4, "b": 2}
    assert weights.tolist() == pytest.approx(
        [0.75] * 4 + [1.5] * 2 + [1.0, 1.0]
    )
    assert float(weights[:4].sum()) == pytest.approx(3)
    assert float(weights[4:6].sum()) == pytest.approx(3)
    with pytest.raises(ValueError, match="training assets"):
        _inverse_engine_weights([{"split": "val", "asset_id": "v"}])


def test_engine_weight_labels_stay_aligned_after_asset_split() -> None:
    """NexuML take() must retain inverse-engine weights for each row."""
    rows = [
        *[{"split": "train", "asset_id": "a"} for _ in range(4)],
        *[{"split": "train", "asset_id": "b"} for _ in range(2)],
        {"split": "val", "asset_id": "v"},
        {"split": "test", "asset_id": "t"},
    ]
    weights, _ = _inverse_engine_weights(rows)
    data = _SequenceFlowRULRuntime(
        features=torch.zeros(8, 30, 256),
        targets=torch.arange(8, dtype=torch.float32).view(-1, 1),
        raw_targets=torch.arange(8, dtype=torch.float32).view(-1, 1),
        meta=pd.DataFrame.from_records(rows),
        weights=weights,
    )
    split = data.take([5, 3, 6])
    assert len(split) == 3
    assert split.load_labels(0, split.meta.iloc[0])["rul_weight"].item() == 1.5
    assert split.load_labels(1, split.meta.iloc[1])["rul_weight"].item() == 0.75
    assert split.load_labels(2, split.meta.iloc[2])["rul_weight"].item() == 1.0
    assert split.load_labels(0, split.meta.iloc[0])["rul"].item() == 5.0


def test_engine_frequency_rul_loss_is_differentiable() -> None:
    """The loss uses global inverse counts, not per-minibatch normalization."""
    runtime = RULInverseEngineFrequencyLoss().build(
        LayerBuildContext(
            input_sizes={"predicted_rul": (1,)},
            keys_in=["predicted_rul"],
            keys_out=["rul_loss"],
        )
    )
    assert isinstance(runtime, _RULInverseEngineFrequencyRuntime)
    prediction = torch.tensor([[2.0], [3.0]], requires_grad=True)
    x = TensorDict({"predicted_rul": prediction}, batch_size=[2])
    y = TensorDict({
        "rul": torch.zeros(2, 1),
        "rul_weight": torch.tensor([[0.5], [2.0]]),
    }, batch_size=[2])
    output, _ = runtime(x, y)
    assert isinstance(output, TensorDict)
    assert output["rul_loss"].shape == (2,)
    assert output["rul_loss"].mean().item() == pytest.approx(10.0)
    output["rul_loss"].mean().backward()
    assert prediction.grad is not None
    assert prediction.grad.flatten().tolist() == pytest.approx([1.0, 6.0])


def test_validation_stress_groups_partition_engines() -> None:
    """Four stable disjoint groups cover held-out engines without test labels."""
    engines = {
        f"val:{i}": (float(i * i), i + 2)
        for i in range(8)
    }
    groups = _validation_engine_groups(engines)
    assert groups == _validation_engine_groups(dict(reversed(list(engines.items()))))
    assert len(groups) == 4
    assert sum(int(item["engines"]) for item in groups) == len(engines)
    assert sum(int(item["windows"]) for item in groups) == sum(
        windows for _, windows in engines.values()
    )
    assert all(item["engines"] == 2 for item in groups)
    with pytest.raises(ValueError, match="nonempty"):
        _validation_engine_groups({"v": (1.0, 1)})


@pytest.mark.parametrize("arm", ["flow", "encoder"])
@pytest.mark.parametrize("dropout", ["d010", "d030"])
@pytest.mark.parametrize("choice,lr,weight_decay", [
    ("adam_1e3", 1e-3, 0.0),
    ("adamw_3e4", 3e-4, 1e-2),
    ("adamw_1e4", 1e-4, 1e-2),
])
@pytest.mark.parametrize("weighting", ["window", "engine"])
def test_gen2_scenarios_resolve_optimizer_and_training_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: Literal["encoder", "flow"],
    dropout: Literal["d010", "d030"],
    choice: Literal["adam_1e3", "adamw_3e4", "adamw_1e4"],
    lr: float,
    weight_decay: float,
    weighting: Literal["window", "engine"],
) -> None:
    """Compiler receives explicit training.lr and correct optimizer decay."""
    monkeypatch.chdir(tmp_path)
    study = cfc_rul_optimization_study(arm, dropout, choice, weighting, 42)
    source = study.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerSequenceFlowRULFeatures)
    assert source.seed == 42
    assert source.use_flow is (arm == "flow")
    assert source.engine_balanced is (weighting == "engine")
    assert study.training.lr == pytest.approx(lr)
    assert study.training.optimizer.kwargs["lr"] == pytest.approx(lr)
    assert study.training.optimizer.kwargs.get("weight_decay", 0.0) == pytest.approx(
        weight_decay
    )
    assert study.pipeline.stages["RULLoss"][0].keys_out == ["rul_loss"]
    assert isinstance(
        study.pipeline.stages["RULLoss"][0].component,
        RULInverseEngineFrequencyLoss,
    ) is (weighting == "engine")
    compiled = compile_pipeline(study)
    runtime_optimizer = compiled.create_optimizer()
    assert runtime_optimizer.param_groups[0]["lr"] == pytest.approx(lr)
    assert runtime_optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        weight_decay
    )
    assert compiled.resolved_config.to_scenario().name == study.name


def _write_run(
    root: Path, audits: Path, *,
    arm: str, dropout: str, opt: str, weighting: str, seed: int,
) -> None:
    """Make one synthetic matched native validation run and provenance audit."""
    name = (
        f"rul-FD001-cfc-gen2-{arm}-{dropout}-{opt}-"
        f"{weighting}-head-s{seed}"
    )
    score = 100.0 + (20.0 if arm == "encoder" else 0.0)
    score += (4.0 if weighting == "window" else 0.0)
    score += (3.0 if dropout == "d030" else 0.0)
    score += (2.0 if opt == "adam_1e3" else 0.0)
    with (root / f"{name}.curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("epoch", "train_eval_mse", "val_eval_mse")
        )
        writer.writeheader()
        writer.writerow({
            "epoch": 0, "train_eval_mse": 60.0, "val_eval_mse": score
        })
        writer.writerow({
            "epoch": 1, "train_eval_mse": 20.0, "val_eval_mse": score + 30
        })
    (root / f"{name}.curves.validation_engines.json").write_text(
        json.dumps({
            "best_epoch": 0,
            "validation_engines": [
                {"asset_id": f"val:{i}", "windows": 10, "mse": float(i)}
                for i in range(8)
            ],
            "validation_engine_groups": [
                {
                    "group": i, "engines": 2, "windows": 20,
                    "window_mse": score + i,
                }
                for i in range(4)
            ],
        }),
        encoding="utf-8",
    )
    (audits / f"{name}.geometry.json").write_text(
        json.dumps({
            "subset": "FD001", "seed": 42, "architecture": "cfc",
            "use_flow": arm == "flow",
            "engine_balanced": weighting == "engine",
            "test_labels_used_in_feature_fitting": False,
            "patchformer_checkpoint_sha256": "same-encoder",
            "endpoint_artifact_sha256": "same-endpoint",
            "flow_checkpoint_sha256": (
                "same-flow" if arm == "flow" else None
            ),
        }),
        encoding="utf-8",
    )


def test_gen2_summary_is_validation_only_and_checks_provenance(
    tmp_path: Path,
) -> None:
    """Paired summary never accesses official test predictions."""
    root = tmp_path / "curves"
    audits = tmp_path / "audits"
    root.mkdir()
    audits.mkdir()
    for arm in ("encoder", "flow"):
        for dropout in ("d010", "d030"):
            for opt in ("adam_1e3", "adamw_3e4", "adamw_1e4"):
                for weighting in ("window", "engine"):
                    _write_run(
                        root, audits,
                        arm=arm, dropout=dropout, opt=opt, weighting=weighting,
                        seed=42,
                    )
    summary = summarize_cfc_gen2(root, audits, (42,))
    assert summary["test_predictions_read"] is False
    assert summary["best_validation_configuration"] == (
        "d010_adamw_3e4_engine"
    )
    assert len(summary["ranking"]) == 12
    assert summary["ranking"][0]["paired_flow_minus_encoder_mean"] == -20.0
    broken = audits / (
        "rul-FD001-cfc-gen2-flow-d010-adamw_3e4-engine-head-s42.geometry.json"
    )
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["endpoint_artifact_sha256"] = "mismatch"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="upstream checkpoints"):
        summarize_cfc_gen2(root, audits, (42,))
