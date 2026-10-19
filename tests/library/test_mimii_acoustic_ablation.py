"""No-download ablation tests for prepared MIMII endpoint transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from nexuml.core.config import ResolvedConfig
from nexuml.training.lightning import create_runtime_artifacts
from nexuml_library.data.maintenance.prepared_endpoints import (
    PreparedMaintenanceEndpointPairs,
    read_prepared_artifact,
)
from nexuml_library.evaluation.maintenance import ablation_train
from nexuml_library.evaluation.maintenance.ablation_evaluate import (
    evaluate_prepared_split,
)
from nexuml_library.evaluation.maintenance.ablation_prepare import (
    couple_training_endpoints,
    feature_variants,
    training_normalizer,
)
from nexuml_library.scenarios.flow.mimii_acoustic_ablation import (
    mimii_acoustic_ablation,
    scenario as ablation_scenario_file,
)


def _example_features() -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Return visibly mismatched training assignments with sealed partitions.

    Returns:
        Feature matrix and metadata with two training machine IDs.
    """
    vectors = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [11.0, 0.0]],
            [[11.0, 0.0], [1.0, 0.0]],
            [[70.0, 1.0], [71.0, 1.0]],
            [[80.0, 1.0], [81.0, 1.0]],
        ]
    )
    rows: list[dict[str, Any]] = []
    for i in range(len(vectors)):
        group = "id_00" if i < 2 else "id_02" if i < 4 else "id_04" if i == 4 else "id_06"
        split = "train" if i < 4 else "val" if i == 4 else "test"
        rows.append(
            {
                "group_id": f"fan:{group}",
                "healthy_group": f"fan:{group}",
                "faulty_group": f"fan:{group}",
                "healthy_file": f"normal_{i}.wav",
                "faulty_file": f"faulty_{i}.wav",
                "pair_id": f"pair_{i}",
                "split": split,
                "paired": 1,
            }
        )
    return vectors, rows


def test_train_only_standardization_does_not_use_validation_or_test() -> None:
    features, rows = _example_features()
    mean, std = training_normalizer(features, rows)
    assert mean.shape == (1, 1, 2)
    assert std.shape == (1, 1, 2)
    assert float(mean[0, 0, 0]) == pytest.approx(5.5)
    assert float(std[0, 0, 1]) == pytest.approx(1e-6)
    normalized = (features - mean) / std
    assert torch.allclose(normalized[:4].mean(dim=(0, 1)), torch.zeros(2), atol=1e-6)
    assert normalized[4:, 0, 0].min() > 5


def test_machinewise_ot_reassigns_training_targets_but_never_val_or_test() -> None:
    pytest.importorskip("scipy.optimize")
    features, rows = _example_features()
    ot, updated, metrics = couple_training_endpoints(features, rows, coupling="ot")
    assert metrics["selected_train_pair_mse"] < metrics["independent_train_pair_mse"]
    assert ot[:4, 1, 0].tolist() == [0.0, 10.0, 1.0, 11.0]
    assert torch.equal(ot[4:], features[4:])
    assert updated[4:] == rows[4:]
    assert updated[0]["faulty_file"] == rows[1]["faulty_file"]
    assert updated[2]["faulty_file"] == rows[3]["faulty_file"]
    assert torch.equal(ot[:4, 1].mean(0), features[:4, 1].mean(0))


def test_all_four_variants_use_identical_splits_and_train_statistics() -> None:
    pytest.importorskip("scipy.optimize")
    features, rows = _example_features()
    variants = feature_variants(features, rows, frontend="fisher", seed=7)
    assert set(variants) == {
        "fisher-independent-raw",
        "fisher-independent-standardized",
        "fisher-ot-raw",
        "fisher-ot-standardized",
    }
    for key, artifact in variants.items():
        assert artifact["format"] == "nex233-features-v1"
        assert artifact["settings"]["normalization_fit_split"] == "train"
        assert "unconditional_cancellation_fraction" in artifact["settings"][
            "training_velocity_statistics"
        ]
        assert len(artifact["settings"]["training_neighbor_alignment"]) == 3
        assert [row["split"] for row in artifact["rows"]] == [
            row["split"] for row in rows
        ]
        if "standardized" in key:
            assert artifact["features"][4, 0, 0] != features[4, 0, 0]
        if "independent" in key:
            assert artifact["settings"]["ot_fit_split"] == "none"


def test_prepared_tensor_source_handles_take_and_machine_disjointness(tmp_path: Path) -> None:
    pytest.importorskip("scipy.optimize")
    features, rows = _example_features()
    artifact = feature_variants(features, rows, frontend="logmel", seed=7)[
        "logmel-ot-standardized"
    ]
    file = tmp_path / "prepared.pt"
    torch.save(artifact, file)
    assert read_prepared_artifact(file)["settings"]["seed"] == 7
    source = PreparedMaintenanceEndpointPairs(path=str(file)).build()
    train = source.get_split("train")
    valid = source.get_split("val")
    test = source.get_split("test")
    assert len(train) == 4 and len(valid) == 1 and len(test) == 1
    inputs, labels = train[0]
    assert inputs["healthy_spectrum"].shape == (2,)
    assert inputs["faulty_spectrum"].shape == (2,)
    assert labels is not None
    paired_label = labels["paired"]
    assert isinstance(paired_label, torch.Tensor)
    assert paired_label.item() == 1.0
    assert valid[0][0]["healthy_spectrum"].shape == (2,)


def test_scenario_compiles_without_fisher_download_and_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("scipy.optimize")
    features, rows = _example_features()
    payload = feature_variants(features, rows, frontend="fisher", seed=7)[
        "fisher-ot-standardized"
    ]
    file = tmp_path / "fisher-ot-standardized-seed7.pt"
    torch.save(payload, file)
    scenario = mimii_acoustic_ablation(
        file,
        hidden_dims=[16, 16],
        batch_size=2,
        max_epochs=1,
        patience=2,
        seed=7,
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert "AudioFeatures" not in scenario.pipeline.stages
    assert scenario.pipeline.stages["Flow"][0].component.hidden_dims == [16, 16]
    assert "fisher-ot-standardized" in scenario.callbacks[0].kwargs["dirpath"]
    parsed = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())
    assert parsed.data.input_shapes["healthy_spectrum"] == [2]
    artifacts = create_runtime_artifacts(scenario)
    assert artifacts.data_module.dataset.get_split("val")[0][0][
        "healthy_spectrum"
    ].shape == (2,)
    assert len(list(artifacts.pipeline.iter_layers())) == 3
    metrics = evaluate_prepared_split(
        artifacts,
        split="val",
        device=torch.device("cpu"),
        projections=8,
        seed=7,
    )
    assert metrics["split"] == "val"
    assert metrics["num_samples"] == 1
    assert set(metrics["fixed_time_fm_mse"]) == {"0.25", "0.5", "0.75"}
    assert metrics["swd_healthy_to_faulty"] >= 0.0
    assert metrics["training_mean_shift_swd"] >= 0.0
    with pytest.raises(ValueError, match="Only val"):
        evaluate_prepared_split(
            artifacts, split="train", device=torch.device("cpu")
        )


def test_prepared_artifact_rejects_overlap_and_tampering(tmp_path: Path) -> None:
    features, rows = _example_features()
    artifact = {
        "format": "nex233-features-v1",
        "features": features,
        "rows": rows,
        "settings": {"seed": 7},
    }
    path = tmp_path / "invalid.pt"
    artifact["rows"][4]["group_id"] = "fan:id_00"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="overlaps"):
        read_prepared_artifact(path)
    artifact["features"] = torch.full_like(features, float("nan"))
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="finite"):
        read_prepared_artifact(path)


def test_scenario_rejects_inconsistent_seed(tmp_path: Path) -> None:
    features, rows = _example_features()
    file = tmp_path / "variant.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": features,
            "rows": rows,
            "settings": {"seed": 7},
        },
        file,
    )
    with pytest.raises(ValueError, match="seed"):
        mimii_acoustic_ablation(file, seed=42)


def test_ablation_trainer_never_calls_automatic_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scientific training path deliberately calls fit and validate only."""
    features, rows = _example_features()
    file = tmp_path / "variant-seed7.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": features,
            "rows": rows,
            "settings": {"seed": 7},
        },
        file,
    )
    operations: list[str] = []

    class FakeSession:
        @classmethod
        def from_scenario(cls, spec: Any, **kwargs: Any) -> FakeSession:
            assert spec.name.startswith("mimii_acoustic_ablation")
            return cls()

        def setup(self) -> FakeSession:
            operations.append("setup")
            return self

        def fit(self) -> FakeSession:
            operations.append("fit")
            return self

        def validate(self) -> list[dict[str, float]]:
            operations.append("validate")
            return []

        def test(self) -> None:
            raise AssertionError("Test must not run in an ablation training session")

    monkeypatch.setattr(ablation_train, "NexuSession", FakeSession)
    ablation_train.train_ablation(
        file,
        checkpoint_root=tmp_path / "checkpoints",
        width=128,
        max_epochs=1,
        seed=7,
        accelerator="cpu",
    )
    assert operations == ["setup", "fit", "validate"]

def test_ablation_factory_is_explicit_and_scenario_file_uses_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the artifact-dependent factory is not globally registered."""
    from nexuml.core.discovery import DISCOVERED_ATTR

    assert getattr(mimii_acoustic_ablation, DISCOVERED_ATTR, None) is None

    features, rows = _example_features()
    artifact = {
        "format": "nex233-features-v1",
        "features": features,
        "rows": rows,
        "settings": {"seed": 7},
    }
    prepared = tmp_path / "prepared-seed7.pt"
    torch.save(artifact, prepared)
    monkeypatch.setenv("NEX233_PREPARED_FILE", str(prepared))
    monkeypatch.setenv("NEX233_SEED", "7")
    monkeypatch.setenv("NEX233_FLOW_WIDTH", "16")
    built = ablation_scenario_file()
    assert built.name.startswith("mimii_acoustic_ablation")
    assert built.pipeline.stages["Flow"][0].component.hidden_dims == [16, 16]
