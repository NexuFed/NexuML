"""SDK submission is shared and never mistaken for remote completion."""

import pytest

pytest.importorskip("fastapi")

from nexuml.api import worker
from nexuml.core.serialization import lower_model
from nexuml.core.types import ScenarioSpec
from nexuml.execution.definitions import RayJobsSDKExecution


def request():
    return {
        "data": lower_model(ScenarioSpec(name="sdk-api", execution=RayJobsSDKExecution())),
        "trainer_checkpoint": "/worker only/partial.ckpt",
    }


def test_remote_resume_preparation_never_loads_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "nexuml.training.lightning.load_scenario_from_trainer_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("read worker checkpoint locally"),
    )
    result = worker.dispatch("prepare_train", request())
    assert result["data"]["execution"]["type"] == "ray-jobs-sdk"
    assert result["launch_review"]["argv"][-1] == "/worker only/partial.ckpt"


def test_api_submission_receipt_is_not_training_success(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "nexuml.execution.run",
        lambda scenario, **kwargs: calls.append((scenario, kwargs)) or "sdk_submission",
    )
    value = request()
    value["observations"] = str(tmp_path / "observations")
    result = worker.execute("train", value)
    assert result["status"] == "submitted"
    assert result["submission_id"] == "sdk_submission"
    assert calls[0][1]["trainer_checkpoint"] == "/worker only/partial.ckpt"


def test_api_export_dispatches_without_a_local_session(monkeypatch):
    monkeypatch.setattr(
        "nexuml.training.lightning.NexuSession",
        lambda *_args, **_kwargs: pytest.fail("constructed local export session"),
    )
    calls = []
    monkeypatch.setattr(
        "nexuml.execution.run",
        lambda scenario, **kwargs: calls.append(kwargs) or "sdk_export",
    )
    value = {**request(), "export_kind": "train_package", "output": "/worker/policy"}
    result = worker.execute("export", value)
    assert result == {"status": "submitted", "submission_id": "sdk_export"}
    assert calls[0]["operation"] == "export"
    assert calls[0]["checkpoint"] == value["trainer_checkpoint"]
