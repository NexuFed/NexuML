"""Leakage and trainability tests for NEX-233 healthy-reference conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.training.lightning import (
    NexuLightningModule,
    NexuSession,
    create_runtime_artifacts,
)
from nexuml_library.data.maintenance.healthy_reference import (
    read_healthy_reference_assessment,
    read_healthy_reference_training,
)
from nexuml_library.evaluation.maintenance.reference_conditioned_prepare import (
    build_reference_conditioned_fold,
    save_reference_conditioned_folds,
)
from nexuml_library.evaluation.maintenance.reference_conditioned_run import (
    run_reference_conditioned_folds,
)
from nexuml_library.evaluation.maintenance.reference_conditioned_validation import (
    evaluate_reference_transport,
    evaluate_training_machine_validation,
)
from nexuml_library.layers.model.healthy_reference_flow import (
    HealthyReferenceConditionedFlow,
    _HealthyReferenceConditionedFlowRuntime,
)
from nexuml_library.scenarios.flow.healthy_reference_fisher import (
    healthy_reference_flow_scenario,
)


def _raw_source(tmp_path: Path) -> Path:
    """Create four synthetic FISHER machine groups with unique recordings.

    Returns:
        Original machine-disjoint independent/raw artifact.
    """
    generator = torch.Generator().manual_seed(11)
    features: list[Tensor] = []
    rows: list[dict[str, Any]] = []
    for index, (name, split) in enumerate(
        (
            ("fan:id_00", "train"),
            ("fan:id_02", "train"),
            ("fan:id_04", "val"),
            ("fan:id_06", "test"),
        )
    ):
        for recording in range(20):
            healthy = torch.randn(8, generator=generator) * 0.3
            healthy[index] += 2.0
            faulty = healthy + 0.6 * torch.randn(8, generator=generator)
            features.append(torch.stack((healthy, faulty)))
            rows.append(
                {
                    "group_id": name,
                    "split": split,
                    "paired": 1,
                    "healthy_file": f"{name}/normal/{recording}.wav",
                    "faulty_file": f"{name}/abnormal/{recording}.wav",
                }
            )
    source = tmp_path / "fisher-independent-raw-seed42.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": torch.stack(features),
            "rows": rows,
            "settings": {
                "frontend": "fisher",
                "coupling": "independent",
                "scaling": "raw",
                "seed": 42,
                "feature_dim": 8,
            },
        },
        source,
    )
    return source


def _runtime(mode: str = "reference") -> _HealthyReferenceConditionedFlowRuntime:
    """Build a small differentiable flow through the NexuML layer contract."""
    runtime = HealthyReferenceConditionedFlow(
        hidden_dims=[12, 12],
        context_dim=5,
        encoder_hidden_dim=10,
        num_steps=4,
        context_mode=mode,
    ).build(
        LayerBuildContext(
            input_sizes={
                "flow_state": (8,),
                "flow_time": (1,),
                "healthy_reference": (4, 8),
            },
            keys_in=["flow_state", "flow_time", "healthy_reference"],
            keys_out=["predicted_velocity"],
        )
    )
    return cast(_HealthyReferenceConditionedFlowRuntime, runtime)


def test_reference_encoder_is_permutation_invariant_and_trainable() -> None:
    """Backprop reaches the DeepSets encoder and the Flow Matching backbone."""
    torch.manual_seed(7)
    runtime = _runtime()
    state = torch.randn(3, 8)
    reference = torch.randn(3, 4, 8)
    time = torch.full((3, 1), 0.4)
    x = TensorDict(
        {
            "flow_state": state,
            "flow_time": time,
            "healthy_reference": reference,
        },
        batch_size=[3],
    )
    prediction = cast(Tensor, runtime(x)[0]["predicted_velocity"])
    reordered = runtime.velocity(state, time, reference[:, [3, 1, 0, 2]])
    assert torch.allclose(prediction, reordered, atol=1e-6)
    prediction.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in runtime.encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in runtime.backbone.parameters()
    )
    trajectory = runtime.sample(state, reference, record_trajectory=True)
    assert trajectory.shape == (3, 5, 8)
    assert torch.isfinite(trajectory).all()
    assert torch.allclose(trajectory[:, 0], state)


def test_context_ablation_does_not_depend_on_reference_values() -> None:
    """Constant mode deletes conditioning information while preserving flow."""
    torch.manual_seed(8)
    state = torch.randn(3, 8)
    time = torch.full((3, 1), 0.5)
    source = torch.ones(4, 8)
    alternative = -torch.ones(4, 8)
    conditioned = _runtime("reference")
    zeroed = _runtime("constant")
    assert not torch.allclose(
        conditioned.velocity(state, time, source),
        conditioned.velocity(state, time, alternative),
    )
    assert torch.equal(
        zeroed.velocity(state, time, source),
        zeroed.velocity(state, time, alternative),
    )
    assert torch.equal(zeroed.sample(state, source), zeroed.sample(state, alternative))
    with pytest.raises(ValueError, match="Reference"):
        conditioned.velocity(state, time, torch.randn(2, 7))


def test_leave_one_out_preparation_does_not_leak_faults_or_support(tmp_path: Path) -> None:
    """Held-out endpoints never enter training and support/query files are disjoint."""
    source = _raw_source(tmp_path)
    before = torch.load(source, map_location="cpu", weights_only=True)
    training, assessment = build_reference_conditioned_fold(
        source,
        heldout_machine="fan:id_06",
        reference_size=4,
        validation_fraction=0.25,
        seed=42,
    )
    assert training["machine_groups"] == ["fan:id_00", "fan:id_02", "fan:id_04"]
    assert training["features"].shape[0] == 36
    assert training["references"].shape == (3, 4, 8)
    assert assessment["features"].shape == (16, 2, 8)
    assert assessment["references"].shape == (4, 8)
    assert all(row["group_id"] != "fan:id_06" for row in training["rows"])
    assert all(row["group_id"] == "fan:id_06" for row in assessment["rows"])
    assert all(row["split"] in {"train", "val"} for row in training["rows"])
    query_files = {
        row[key] for row in training["rows"] for key in ("healthy_file", "faulty_file")
    }
    support_files = {
        filename for support in training["support_healthy_files"] for filename in support
    }
    assert query_files.isdisjoint(support_files)
    assert set(assessment["support_healthy_files"]).isdisjoint({
        row[key] for row in assessment["rows"] for key in ("healthy_file", "faulty_file")
    })
    assert torch.equal(
        before["features"], torch.load(source, map_location="cpu", weights_only=True)["features"]
    )


def test_unseen_faulty_modification_cannot_change_training_payload(tmp_path: Path) -> None:
    """Fitting statistics and training context never access held-out faults."""
    source = _raw_source(tmp_path)
    first, _ = build_reference_conditioned_fold(
        source, heldout_machine="fan:id_06", reference_size=4
    )
    raw = torch.load(source, map_location="cpu", weights_only=True)
    for index, row in enumerate(raw["rows"]):
        if row["group_id"] == "fan:id_06":
            raw["features"][index, 1] = 1e4
    torch.save(raw, source)
    second, _ = build_reference_conditioned_fold(
        source, heldout_machine="fan:id_06", reference_size=4
    )
    assert torch.equal(first["features"], second["features"])
    assert torch.equal(first["references"], second["references"])
    assert torch.equal(first["training_mean"], second["training_mean"])
    assert torch.equal(first["training_std"], second["training_std"])


def test_prepared_artifacts_roundtrip_and_refuse_overwrite(tmp_path: Path) -> None:
    """Persist immutable train and separate unseen-machine assessment artifacts."""
    source = _raw_source(tmp_path)
    directory = tmp_path / "folds"
    manifest = save_reference_conditioned_folds(
        source, directory, reference_size=4, validation_fraction=0.25
    )
    assert len(manifest["folds"]) == 4
    for machine, paths in manifest["folds"].items():
        training = read_healthy_reference_training(paths["training_file"])
        assessment = read_healthy_reference_assessment(paths["assessment_file"])
        assert machine not in training["machine_groups"]
        assert machine == assessment["heldout_machine"]
    with pytest.raises(FileExistsError, match="overwrite"):
        save_reference_conditioned_folds(source, directory, reference_size=4)


def test_scenario_compiles_one_training_step_and_machine_wise_validation(
    tmp_path: Path,
) -> None:
    """Train within a normal NexuML scenario with context and checkpoint callbacks."""
    source = _raw_source(tmp_path)
    manifest = save_reference_conditioned_folds(
        source,
        tmp_path / "folds",
        heldout_machine="fan:id_06",
        reference_size=4,
        validation_fraction=0.25,
    )
    training_file = manifest["folds"]["fan:id_06"]["training_file"]
    spec = healthy_reference_flow_scenario(
        training_file,
        width=12,
        context_dim=5,
        encoder_width=10,
        num_steps=2,
        batch_size=8,
        max_epochs=1,
        patience=2,
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert (
        spec.pipeline.stages["Flow"][0].component.__class__
        is HealthyReferenceConditionedFlow
    )
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(spec).to_yaml())
    assert restored.pipeline.stages["Flow"][0].component.__class__ is (
        HealthyReferenceConditionedFlow
    )
    artifacts = create_runtime_artifacts(spec)
    artifacts.data_module.setup("fit")
    x, y = next(iter(artifacts.data_module.train_dataloader()))
    output, _ = artifacts.lightning_module.pipeline(x, y)
    loss, _ = artifacts.lightning_module._compute_loss(output)
    loss.backward()
    assert bool(torch.isfinite(loss))
    session = NexuSession.from_scenario(
        spec,
        accelerator="cpu",
        devices=1,
        log_dir=tmp_path / "logs",
        enable_progress_bar=False,
        enable_loggers=False,
    )
    session.setup().fit()
    scores = session.validate()
    assert scores and "val/transport_swd" in scores[0]
    trained_module = cast(NexuLightningModule, session.trainer.model)
    flow = cast(
        _HealthyReferenceConditionedFlowRuntime,
        next(
            layer for stage, _, layer in trained_module.pipeline.iter_layers()
            if stage == "Flow"
        ),
    )
    machine_scores = evaluate_training_machine_validation(
        flow,
        training_file,
        device=torch.device("cpu"),
        batch_size=8,
        projections=8,
        seed=42,
    )
    assert len(machine_scores["by_machine"]) == 3
    assert machine_scores["mean_transported_swd"] >= 0
    assessment = read_healthy_reference_assessment(
        manifest["folds"]["fan:id_06"]["assessment_file"]
    )
    score = evaluate_reference_transport(
        flow,
        assessment["features"][:, 0],
        assessment["features"][:, 1],
        assessment["references"],
        device=torch.device("cpu"),
        projections=8,
        batch_size=8,
    )
    assert score["initial_swd"] > 0
    assert score["transported_swd"] >= 0


def test_one_fold_runner_selects_checkpoint_before_unseen_assessment(
    tmp_path: Path,
) -> None:
    """End-to-end CPU run isolates unseen faults and emits one assessment report."""
    source = _raw_source(tmp_path)
    prepared = tmp_path / "prepared"
    save_reference_conditioned_folds(
        source,
        prepared,
        heldout_machine="fan:id_06",
        reference_size=4,
        validation_fraction=0.25,
    )
    result = run_reference_conditioned_folds(
        prepared / "manifest.json",
        checkpoint_root=tmp_path / "run-checkpoints",
        reports_dir=tmp_path / "run-reports",
        heldout_machine="fan:id_06",
        context_mode="reference",
        width=12,
        context_dim=5,
        encoder_width=10,
        num_steps=2,
        batch_size=8,
        max_epochs=1,
        patience=2,
        accelerator="cpu",
    )
    assert list(result["folds"]) == ["fan:id_06"]
    report = result["folds"]["fan:id_06"]
    assert Path(report["best_checkpoint"]).is_file()
    assert report["unseen_machine_initial_swd"] > 0
    assert (tmp_path / "run-reports" / "summary-reference.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        run_reference_conditioned_folds(
            prepared / "manifest.json",
            checkpoint_root=tmp_path / "run-checkpoints",
            reports_dir=tmp_path / "run-reports",
            heldout_machine="fan:id_06",
            accelerator="cpu",
        )
