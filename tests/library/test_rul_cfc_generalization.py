"""CfC full-sequence RUL generalization ablation and measurement contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.training.lightning import NexuLightningModule
from nexuml_library.evaluation.maintenance.rul_generalization_diagnostics import (
    _evaluate_split,
)
from nexuml_library.evaluation.maintenance.rul_generalization_summary import (
    summarize_cfc_generalization,
)
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_generalization import (
    cfc_generalization_study,
)


@pytest.mark.parametrize("head_seed", [42, 43, 44])
@pytest.mark.parametrize("arm", ["flow", "encoder"])
@pytest.mark.parametrize(
    "variant,width,dropout",
    [
        ("w128_d010", 128, 0.1),
        ("w128_d030", 128, 0.3),
        ("w064_d010", 64, 0.1),
        ("w064_d030", 64, 0.3),
    ],
)
def test_cfc_generalization_scenarios_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head_seed: int,
    arm: Literal["flow", "encoder"],
    variant: Literal["w128_d010", "w128_d030", "w064_d010", "w064_d030"],
    width: int,
    dropout: float,
) -> None:
    """Every matched head shares the frozen upstream but has isolated outputs."""
    monkeypatch.chdir(tmp_path)
    spec = cfc_generalization_study(arm, variant, head_seed)
    source = spec.data.datasets[0].source
    assert source.seed == 42 and source.architecture == "cfc"
    assert source.use_flow is (arm == "flow")
    head = spec.pipeline.stages["RULModel"][0].component
    assert head == PatchFormerSequenceFlowRULHead(
        hidden_dim=width, attention_heads=4,
        dropout=dropout, initialization_seed=head_seed,
    )
    assert spec.training.max_epochs == 150
    assert spec.callbacks[0].kwargs["seed"] == head_seed
    assert spec.callbacks[1].kwargs["output_file"].endswith(
        f"{arm}-{variant}-head-s{head_seed}.curves.csv"
    )
    assert spec.callbacks[2].kwargs["monitor"] == "val/loss"
    assert spec.evaluation.algorithms[2].algorithm.output_file.endswith(
        f"{arm}-{variant}-head-s{head_seed}.csv"
    )
    assert compile_pipeline(spec).resolved_config.to_scenario().name == spec.name


class _ToyModel:
    """Simple predictor with native TensorDict pipeline semantics."""

    device = torch.device("cpu")

    def pipeline(
        self, x: TensorDict, y: TensorDict
    ) -> tuple[TensorDict, TensorDict]:
        """Return known predictions and preserve labels.

        Returns:
            Predicted features and corresponding unchanged labels.
        """
        x["predicted_rul"] = x["features"]
        return x, y


def _batch(
    predictions: list[float],
    labels: list[float],
    indices: list[int],
) -> tuple[TensorDict, TensorDict]:
    """Make one variable-size native RUL batch.

    Returns:
        Features and labels with corresponding local sample indices.
    """
    n = len(predictions)
    return (
        TensorDict(
            {
                "features": torch.tensor(predictions).reshape(n, 1),
                "sample_index": torch.tensor(indices),
            },
            batch_size=[n],
        ),
        TensorDict(
            {"rul": torch.tensor(labels).reshape(n, 1)}, batch_size=[n]
        ),
    )


def test_evaluation_mode_epoch_mse_weights_every_window() -> None:
    """Last partial batches and per-engine window counts remain exact."""
    batches = [
        _batch([2, 2], [1, 1], [0, 1]),
        _batch([4], [1], [2]),
    ]
    metadata = pd.DataFrame({
        "asset_id": ["val:a", "val:a", "val:b"],
        "split": ["val", "val", "val"],
    })
    total, count, engines = _evaluate_split(
        cast(NexuLightningModule, _ToyModel()),
        batches,
        metadata=metadata,
    )
    assert total == pytest.approx(11.0)
    assert count == 3
    assert engines == {"val:a": (2.0, 2), "val:b": (9.0, 1)}
    metadata.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="non-val"):
        _evaluate_split(
            cast(NexuLightningModule, _ToyModel()),
            batches,
            metadata=metadata,
        )


def test_only_validation_chooses_cfc_regularization(
    tmp_path: Path,
) -> None:
    """Summary rejects changed upstream normalization and ignores test CSVs."""
    root = tmp_path / "curves"
    audits = tmp_path / "audits"
    root.mkdir()
    audits.mkdir()
    for seed in (42, 43):
        for arm in ("encoder", "flow"):
            for variant in (
                "w128_d010", "w128_d030", "w064_d010", "w064_d030"
            ):
                name = (
                    f"rul-FD001-cfc-generalization-{arm}-"
                    f"{variant}-head-s{seed}"
                )
                val = 100.0 + (
                    0 if variant == "w064_d030" else 10
                ) + (0 if arm == "flow" else 20)
                csv_path = root / f"{name}.curves.csv"
                with csv_path.open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=(
                            "epoch", "train_eval_mse", "val_eval_mse", "mse_gap"
                        )
                    )
                    writer.writeheader()
                    for epoch, train, validation in (
                        (0, 40.0, val),
                        (1, 30.0, val + 15),
                    ):
                        writer.writerow({
                            "epoch": epoch,
                            "train_eval_mse": train,
                            "val_eval_mse": validation,
                            "mse_gap": validation - train,
                        })
                (root / f"{name}.curves.validation_engines.json").write_text(
                    json.dumps({
                        "best_epoch": 0,
                        "validation_engines": [
                            {"asset_id": "val:1", "windows": 6, "mse": 2.0},
                            {"asset_id": "val:2", "windows": 4, "mse": 6.0},
                        ],
                    }),
                    encoding="utf-8",
                )
                (audits / f"{name}.geometry.json").write_text(
                    json.dumps({
                        "subset": "FD001", "seed": 42,
                        "architecture": "cfc", "use_flow": arm == "flow",
                        "test_labels_used_in_feature_fitting": False,
                        "patchformer_checkpoint_sha256": "encoder",
                        "endpoint_artifact_sha256": "endpoints",
                    }),
                    encoding="utf-8",
                )
    results = summarize_cfc_generalization(root, audits, (42, 43))
    assert results["test_predictions_read"] is False
    assert results["preferred_on_validation"] == "w064_d030"
    assert results["per_seed"]["42"]["flow_w064_d030"][
        "val_mse_increase_after_best"
    ] == 15.0
    corrupted = audits / (
        "rul-FD001-cfc-generalization-flow-w064_d030-head-s42.geometry.json"
    )
    info = json.loads(corrupted.read_text(encoding="utf-8"))
    info["endpoint_artifact_sha256"] = "wrong"
    corrupted.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent frozen input"):
        summarize_cfc_generalization(root, audits, (42, 43))


def test_epoch_callback_writes_full_splits_and_best_validation_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callback restores eval state, preserves RNG and excludes test engines."""
    import nexuml_library.evaluation.maintenance.rul_generalization_diagnostics as diag

    class ToyModel(_ToyModel):
        training = False

        def eval(self) -> ToyModel:
            """Switch to dropout-free inference.

            Returns:
                The same model.
            """
            self.training = False
            return self

        def train(self, mode: bool = True) -> ToyModel:
            """Restore initial model mode after measuring metrics.

            Returns:
                The same model.
            """
            self.training = mode
            return self

        def log(self, name: str, value: float, **kwargs: object) -> None:
            """Record a scalar without a Lightning trainer."""
            assert name.startswith("val/generalization_")
            assert value >= 0 or name.endswith("_gap")

    class ToyData:
        def __init__(self) -> None:
            self._train_ds = type(
                "TrainSplit", (),
                {"meta": pd.DataFrame({
                    "asset_id": ["train:1", "train:1"],
                    "split": ["train", "train"],
                })},
            )()
            self._val_ds = type(
                "ValSplit", (),
                {"meta": pd.DataFrame({
                    "asset_id": ["val:1", "val:2"],
                    "split": ["val", "val"],
                })},
            )()

        def _loader(
            self, dataset: object, split: str, shuffle: bool = False
        ) -> list[tuple[TensorDict, TensorDict]]:
            """Return two unshuffled training examples.

            Returns:
                One deterministic training batch.
            """
            assert not shuffle
            return [_batch([1.0, 2.0], [1.0, 2.0], [0, 1])]

        def val_dataloader(self) -> list[tuple[TensorDict, TensorDict]]:
            """Return two validation examples.

            Returns:
                One deterministic validation batch.
            """
            return [_batch([1.0, 4.0], [2.0, 1.0], [0, 1])]

    class ToyTrainer:
        sanity_checking = False
        world_size = 1
        current_epoch = 0
        datamodule = ToyData()

    monkeypatch.setattr(diag, "NexuLightningModule", ToyModel)
    monkeypatch.setattr(diag, "NexuDataModule", ToyData)
    model = ToyModel()
    model.train(True)
    callback = diag.SequenceRULGeneralizationDiagnostics(
        str(tmp_path / "run.curves.csv")
    )
    trainer = ToyTrainer()
    state_before = torch.random.get_rng_state().clone()
    callback.on_fit_start(trainer, model)
    callback.on_validation_epoch_end(trainer, model)
    assert model.training is True
    assert torch.equal(torch.random.get_rng_state(), state_before)
    with (tmp_path / "run.curves.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["train_eval_mse"]) == pytest.approx(0.0)
    assert float(rows[0]["val_eval_mse"]) == pytest.approx(5.0)
    engines = json.loads(
        (tmp_path / "run.curves.validation_engines.json").read_text()
    )
    assert engines["test_labels_used"] is False
    assert len(engines["validation_engines"]) == 2
    assert {entry["asset_id"] for entry in engines["validation_engines"]} == {
        "val:1", "val:2"
    }
