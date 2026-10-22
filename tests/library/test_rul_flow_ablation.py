"""Flow-specific RUL ablation, identity, and no-flow baseline tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from nexuml.core.compiler import compile as compile_pipeline
from nexuml_library.data.maintenance.rul_flow_calibration import (
    AutoPatchFormerFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_flow_ablation_summary import (
    _run_name,
    summarize_flow_ablation,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel
from nexuml_library.layers.model.rul_flow_calibration import (
    PatchFormerFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_flow_calibration import (
    patchformer_flow_rul_calibration,
)


@pytest.mark.parametrize(
    ("mode", "uses_geometry", "uses_baseline"),
    [
        ("flow_time", True, False),
        ("flow_geometry", True, False),
        ("baseline_only", False, True),
        ("hybrid", True, True),
    ],
)
def test_model_ablation_blocks_unselected_input_gradients(
    mode: Literal["flow_time", "flow_geometry", "baseline_only", "hybrid"],
    uses_geometry: bool,
    uses_baseline: bool,
) -> None:
    """MLP cannot infer an ablated input via another feature pathway."""
    from nexuml.core.components import LayerBuildContext

    runtime = PatchFormerFlowRULHead(mode=mode).build(
        LayerBuildContext(
            input_sizes={"flow_rul_features": (3,)},
            keys_in=["flow_rul_features"],
            keys_out=["predicted_rul"],
        )
    )
    x = torch.tensor(
        [[0.2, 0.4, 75.0], [0.6, 0.8, 30.0]], requires_grad=True
    )
    predicted = runtime.forward_tensor(x)
    assert predicted.shape == (2, 1)
    predicted.sum().backward()
    assert x.grad is not None
    if not uses_geometry:
        assert torch.count_nonzero(x.grad[:, :2]).item() == 0
    if mode == "flow_time":
        assert torch.count_nonzero(x.grad[:, 1:]).item() == 0
    if mode == "flow_geometry":
        assert torch.count_nonzero(x.grad[:, 2]).item() == 0
    if mode == "baseline_only":
        assert torch.count_nonzero(x.grad[:, 2]).item() > 0
    if uses_baseline:
        assert bool(((predicted - x[:, 2:3]).abs() <= 30.0).all())
    else:
        assert bool(((predicted >= 0) & (predicted <= 130)).all())


@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("mode", ["baseline_only", "flow_geometry"])
def test_new_ablation_scenarios_preserve_native_comparison_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    subset: Literal["FD001", "FD003"],
    seed: int,
    mode: Literal["baseline_only", "flow_geometry"],
) -> None:
    """Every FD001/FD003/seed recipe resolves without manual model path."""
    monkeypatch.chdir(tmp_path)
    model = patchformer_flow_rul_calibration(subset, seed, mode)
    source = model.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerFlowRULFeatures)
    assert source.feature_mode == (
        "baseline_only" if mode == "baseline_only" else "with_flow"
    )
    assert source.seed == seed
    assert source.subset == subset
    assert model.data.input_shapes == {"flow_rul_features": [3]}
    assert model.training.loss_keys == {"rul_loss": 1.0}
    assert model.evaluation.algorithms[0].label_key == "rul"
    assert model.evaluation.algorithms[1].label_key == "rul_raw"
    assert model.evaluation.algorithms[2].name == "engine_audit"
    assert mode.replace("_", "-") in model.name
    compiled = compile_pipeline(model)
    assert compiled.resolved_config.to_scenario().name == model.name


def _battery_free_cmapss(root: Path) -> None:
    """Generate legal C-MAPSS data without any pre-trained flow model."""
    train = [
        [engine, cycle, *([float(engine + cycle / 100)] * 24)]
        for engine in range(1, 9)
        for cycle in range(1, 70)
    ]
    test = [
        [engine, cycle, *([float(engine + cycle / 100)] * 24)]
        for engine in (1, 2)
        for cycle in range(1, 31)
    ]
    np.savetxt(root / "train_FD001.txt", np.asarray(train))
    np.savetxt(root / "test_FD001.txt", np.asarray(test))
    np.savetxt(root / "RUL_FD001.txt", [5, 12])


def test_baseline_only_builds_without_flow_or_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Original frozen PatchFormer calibration works without a Flow checkpoint."""
    monkeypatch.chdir(tmp_path)
    _battery_free_cmapss(tmp_path)
    run = "rul-FD001-controlled-patchformer-seed42"
    ckpt = tmp_path / ".experiments/rul/checkpoints" / run / "epoch=0.ckpt"
    ckpt.parent.mkdir(parents=True)
    patch = _PatchFormerModel(30, 14, 64, 5, 2, 0.1)
    torch.save(
        {
            "state_dict": {
                f"pipeline.RULModel.0.backbone.{name}": tensor
                for name, tensor in patch.state_dict().items()
            }
        },
        ckpt,
    )
    manifest = tmp_path / ".experiments/rul/audits" / f"{run}.checkpoint.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"best_model_path": str(ckpt)}) + "\n",
        encoding="utf-8",
    )
    source = AutoPatchFormerFlowRULFeatures(
        root=str(tmp_path),
        prepared_rul_path=str(tmp_path / "prepared.pt"),
        endpoint_artifact=str(tmp_path / "absent-flow.pt"),
        diagnostics_file=str(tmp_path / "baseline.json"),
        feature_mode="baseline_only",
    )
    dataset = source.build()
    assert len(dataset.get_split("train")) > 0
    assert len(dataset.get_split("val")) > 0
    assert len(dataset.get_split("test")) == 2
    assert not (tmp_path / "absent-flow.pt").exists()
    first, y = dataset.get_split("test")[0]
    assert y is not None
    assert torch.count_nonzero(first["flow_rul_features"][:2]) == 0
    assert torch.isfinite(first["flow_rul_features"]).all()
    saved = json.loads((tmp_path / "baseline.json").read_text())
    assert saved["flow_checkpoint_sha256"] is None
    assert saved["patchformer_upstream_run"] == run
    assert saved["test_labels_used_in_feature_fitting"] is False


def _write_audit(
    directory: Path, run: str, error: float,
) -> None:
    """Write 100 independent engine rows with their unchanged official labels."""
    with (directory / f"{run}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["asset_id", "predicted_rul", "true_rul_capped", "true_rul_raw"]
        )
        for engine in range(1, 101):
            truth = float(50 + engine % 40)
            writer.writerow([f"test:FD003:{engine}", truth + error, truth, truth])


def _write_geometry(directory: Path, run: str, *, arm: str) -> None:
    """Save native calibration provenance for a fabricated paired seed."""
    (directory / f"{run}.geometry.json").write_text(
        json.dumps(
            {
                "subset": "FD003",
                "seed": 42,
                "patchformer_upstream_run": "rul-FD003-controlled-patchformer-seed42",
                "patchformer_checkpoint_sha256": "same-backbone",
                "flow_checkpoint_sha256": (
                    None if arm == "baseline_only" else "same-flow"
                ),
                "test_labels_used_in_feature_fitting": False,
            }
        ),
        encoding="utf-8",
    )


def test_paired_five_arm_report_checks_flow_identity(
    tmp_path: Path,
) -> None:
    """Hybrid improvement is measured against calibrated, not raw, baseline."""
    run = "rul-FD003-controlled-patchformer-seed42"
    _write_audit(tmp_path, run, 8.0)
    for arm, residual in (
        ("baseline_only", 4.0),
        ("flow_time", 14.0),
        ("flow_geometry", 11.0),
        ("hybrid", 2.0),
    ):
        name = _run_name("FD003", 42, arm)
        _write_audit(tmp_path, name, residual)
        _write_geometry(tmp_path, name, arm=arm)
    report = summarize_flow_ablation(
        tmp_path, subsets=("FD003",), seeds=(42,)
    )
    result = report["comparisons"]
    assert len(result) == 2
    for summary in result:
        assert summary["flow_specific_comparison"]["rmse"][
            "mean_hybrid_minus_baseline_only"
        ] == pytest.approx(-2.0)
        assert summary["flow_specific_comparison"]["nasa"][
            "seeds_hybrid_better"
        ] == 1
    geom = tmp_path / (
        _run_name("FD003", 42, "hybrid") + ".geometry.json"
    )
    payload = json.loads(geom.read_text())
    payload["flow_checkpoint_sha256"] = "different-flow"
    geom.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different frozen Flow"):
        summarize_flow_ablation(tmp_path, subsets=("FD003",), seeds=(42,))


def test_existing_calibrations_recover_upstream_from_immutable_endpoints(
    tmp_path: Path,
) -> None:
    """Seed-42 prior runs need no retraining just to add a provenance field."""
    audit = tmp_path / "audits"
    audit.mkdir()
    flow_run = "rul-FD003-patchformer-latent-flow-seed42"
    prepared = tmp_path / "prepared" / f"{flow_run}-endpoints.pt"
    prepared.parent.mkdir()
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": torch.ones(2, 2, 128),
            "rows": [
                {"split": "train", "group_id": "train-1"},
                {"split": "val", "group_id": "val-1"},
            ],
            "settings": {
                "subset": "FD003",
                "seed": 42,
                "held_out_test_engine_labels_used": False,
                "frozen_patchformer_upstream_run":
                    "rul-FD003-controlled-patchformer-seed42",
                "frozen_patchformer_checkpoint_sha256": "same-backbone",
            },
        },
        prepared,
    )
    _write_audit(
        audit, "rul-FD003-controlled-patchformer-seed42", 8.0
    )
    for arm in ("baseline_only", "flow_time", "flow_geometry", "hybrid"):
        name = _run_name("FD003", 42, arm)
        _write_audit(audit, name, 4.0)
        _write_geometry(audit, name, arm=arm)
        if arm != "baseline_only":
            path = audit / f"{name}.geometry.json"
            data = json.loads(path.read_text())
            del data["patchformer_upstream_run"]
            path.write_text(json.dumps(data))
    report = summarize_flow_ablation(
        audit, subsets=("FD003",), seeds=(42,)
    )
    assert len(report["comparisons"]) == 2
    endpoint = torch.load(prepared, weights_only=True)
    endpoint["settings"]["frozen_patchformer_checkpoint_sha256"] = "wrong"
    torch.save(endpoint, prepared)
    with pytest.raises(ValueError, match="identity mismatch"):
        summarize_flow_ablation(audit, subsets=("FD003",), seeds=(42,))
