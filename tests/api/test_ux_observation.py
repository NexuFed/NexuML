"""Observed progress and definition metadata, without compiling or downloading data."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lightning")

from nexuml.api.observation import Observation
from nexuml_library.data.image.cifar import CIFAR10Dataset
from nexuml_library.evaluation.visualizers.reconstruction import ReconstructionVisualizer
from nexuml_library.evaluation.visualizers.latent import LatentVisualizer
from nexuml_library.layers.model.resnet import ResNet


def test_real_counter_payloads_and_throttling(tmp_path, monkeypatch):
    path = tmp_path / "observations.jsonl"
    callback = Observation(path)
    trainer = SimpleNamespace(
        current_epoch=0,
        global_step=2,
        max_epochs=3,
        num_training_batches=10,
        num_val_batches=[2, 3],
        num_sanity_val_batches=[1],
        sanity_checking=False,
        callback_metrics={},
    )
    clock = iter([10.0, 10.1, 10.3, 10.4, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 12.0])
    monkeypatch.setattr("nexuml.api.observation.time.monotonic", lambda: next(clock))
    callback.record("progress", phase="preparation / data setup")
    callback.on_train_epoch_start(trainer, None)
    callback.on_train_batch_end(trainer, None, None, None, 0)
    callback.on_train_batch_end(trainer, None, None, None, 1)
    callback.on_train_batch_end(trainer, None, None, None, 9)
    callback.on_validation_epoch_start(trainer, None)
    for index in range(5):
        callback.on_validation_batch_end(trainer, None, None, None, index, 0)
    callback.on_train_batch_end(trainer, None, None, None, 5)
    callback.progress(trainer, "unknown total", 2, float("inf"), force=True)
    events = [json.loads(line)["payload"] for line in path.read_text().splitlines()]
    train = [event for event in events if event["phase"] == "train"]
    assert [event["batch"] for event in train] == [0, 2, 10, 6]
    assert all(event["total"] == 10 and event["max_epochs"] == 3 for event in train)
    validation = [event for event in events if event["phase"] == "validate"]
    assert validation[-1]["batch"] == validation[-1]["total"] == 5
    assert events[-1]["total"] is None


def test_schema_metadata_is_authoritative_and_preserves_model_validation():
    assert CIFAR10Dataset.model_json_schema()["x-nexuml-outputs"] == [
        {"domain": "y", "key": "class_labels"}
    ]
    routing = ReconstructionVisualizer.model_json_schema()["x-nexuml-routing"]
    assert routing["algorithm.params.reconstructed_key"] == {"domain": "x", "required": True}
    assert (
        LatentVisualizer.model_json_schema()["x-nexuml-routing"]["feature_key"]["default"]
        == "latent"
    )
    assert "resnet18" in ResNet.model_json_schema()["properties"]["resnet_type"]["enum"]
    assert ResNet.model_json_schema()["x-nexuml-category"] == ["Models", "Vision"]
    assert ResNet.model_config["frozen"] and ResNet.model_config["extra"] == "forbid"


def test_training_metrics_stream_before_epoch_end_without_stale_validation(tmp_path, monkeypatch):
    path = tmp_path / "observations.jsonl"
    callback = Observation(path)
    trainer = SimpleNamespace(
        current_epoch=0,
        global_step=1,
        max_epochs=1,
        num_training_batches=10,
        callback_metrics={
            "train/loss": torch.tensor(2.0),
            "train/classification_loss": torch.tensor(2.0),
            "val/loss": torch.tensor(9.0),
            "lr-Adam": 0.001,
            "train/vector": torch.tensor([1.0, 2.0]),
            "train/nonfinite": float("inf"),
        },
    )
    now = [10.0]
    monkeypatch.setattr("nexuml.api.observation.time.monotonic", lambda: now[0])
    for index, elapsed in [(0, 0), (1, 0.1), (2, 0.3), (9, 0.31)]:
        now[0] = 10.0 + elapsed
        trainer.global_step = index + 1
        trainer.callback_metrics["train/loss"] = torch.tensor(2.0 / (index + 1))
        callback.on_train_batch_end(trainer, None, None, None, index)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    metrics = [event["payload"] for event in events if event["kind"] == "metrics"]
    assert [sample["step"] for sample in metrics] == [1, 3, 10]
    assert [sample["values"]["train/loss"] for sample in metrics] == pytest.approx([2, 2 / 3, 0.2])
    assert all(
        set(sample["values"]) == {"train/loss", "train/classification_loss", "lr-Adam"}
        for sample in metrics
    )


def test_native_classification_observes_finalized_accuracy_and_f1(tmp_path, monkeypatch):
    from nexuml.training.lightning import NexuSession
    from nexuml_library.scenarios.asd.synthetic_linear_ae import synthetic_linear_ae_multiclass

    monkeypatch.chdir(tmp_path)
    scenario = synthetic_linear_ae_multiclass(
        feature_shape=(8,),
        num_samples=64,
        num_classes=3,
        hidden_dims=[8],
        latent_dim=2,
        batch_size=8,
        max_epochs=1,
    )
    scenario.training.accelerator = "cpu"
    scenario.training.devices = 1
    scenario.training.metric_keys = ["accuracy", "f1"]
    scenario.evaluation.algorithms = []
    path = tmp_path / "observations.jsonl"
    session = NexuSession(scenario, enable_progress_bar=False)
    session.trainer.callbacks.append(Observation(path))
    session.run()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    samples = [event["payload"] for event in events if event["kind"] == "metrics"]
    for stage in ("val", "test"):
        final = [sample["values"] for sample in samples if f"{stage}/accuracy" in sample["values"]][
            -1
        ]
        expected = session.lightning_module.get_stage_metric_results(stage)
        for key in (f"{stage}/accuracy", f"{stage}/f1"):
            assert final[key] == pytest.approx(expected[key])
    training = [
        sample
        for sample in samples
        if sample.get("batch") is not None and "train/loss" in sample["values"]
    ]
    assert len(training) >= 2
    assert training[0]["step"] < training[-1]["step"]
