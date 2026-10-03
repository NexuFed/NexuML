"""Observed progress and definition metadata, without compiling or downloading data."""

import json
from types import SimpleNamespace

import pytest

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
