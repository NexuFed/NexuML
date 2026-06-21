"""Pure-logic tests for nexuml.tracking.logger helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import nexuml.tracking.logger as logger_module
from nexuml.tracking.logger import (
    _augment_mlflow_tags,
    _bold,
    _collect_git_metadata,
    _cyan,
    _local_artifact_root,
    _next_version,
    _normalize_mlflow_tracking_uri,
    _resolve_mlflow_artifact_location,
    _to_hwc_uint8,
    get_temp_artifact_root,
    iter_loggers,
    log_artifact,
    log_image,
    log_text_artifact,
    staged_artifact_path,
)


def test_color_helpers_disabled_when_no_color(monkeypatch):
    monkeypatch.setattr(logger_module, "_USE_COLOR", False)
    assert _bold("x") == "x"
    assert _cyan("x") == "x"
    assert logger_module._dim("x") == "x"


def test_get_temp_artifact_root_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUML_TEMP_ARTIFACT_DIR", str(tmp_path))
    root = get_temp_artifact_root()
    assert root == tmp_path


def test_staged_artifact_path_creates_file_under_temp_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUML_TEMP_ARTIFACT_DIR", str(tmp_path))
    with staged_artifact_path("model.pt") as path:
        assert path.name == "model.pt"
        path.write_text("weights")
    assert not path.exists()


def test_normalize_mlflow_tracking_uri_passthrough_remote():
    assert _normalize_mlflow_tracking_uri("http://remote:5000") == "http://remote:5000"


def test_normalize_mlflow_tracking_uri_fixes_relative_sqlite():
    uri = _normalize_mlflow_tracking_uri("sqlite:///mlflow.db")
    assert uri.startswith("sqlite:////")
    assert uri.endswith("mlflow.db")


def test_normalize_mlflow_tracking_uri_fixes_sqlite_colon():
    uri = _normalize_mlflow_tracking_uri("sqlite:mlflow.db")
    assert uri.startswith("sqlite:////")


def test_resolve_mlflow_artifact_location_for_file_uri():
    loc = _resolve_mlflow_artifact_location("file:///tmp/mlruns", None)
    assert loc is not None
    assert "artifacts" in loc


def test_resolve_mlflow_artifact_location_for_sqlite_uri(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    loc = _resolve_mlflow_artifact_location(uri, None)
    assert loc is not None
    assert "mlflow_artifacts" in loc


def test_resolve_mlflow_artifact_location_with_explicit_location(tmp_path):
    loc = _resolve_mlflow_artifact_location("file:///tmp/mlruns", str(tmp_path / "arts"))
    assert loc is not None
    assert "arts" in loc


def test_next_version_empty_directory(tmp_path):
    assert _next_version(tmp_path, "run") == 0


def test_next_version_counts_existing(tmp_path):
    (tmp_path / "run_v0").mkdir()
    (tmp_path / "run_v3").mkdir()
    (tmp_path / "run_v10").mkdir()
    assert _next_version(tmp_path, "run") == 11


def test_iter_loggers_with_none():
    assert iter_loggers(None) == []


def test_iter_loggers_with_list():
    assert iter_loggers([1, 2, None]) == [1, 2]


def test_iter_loggers_with_logger_container():
    container = SimpleNamespace(loggers=[1, 2])
    assert iter_loggers(container) == [1, 2]


def test_iter_loggers_singleton():
    assert iter_loggers("x") == ["x"]


def test_to_hwc_uint8_from_float_numpy():
    arr = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    out = _to_hwc_uint8(arr)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 127


def test_to_hwc_uint8_from_chw_tensor():
    tensor = torch.ones(3, 4, 4) * 0.5
    out = _to_hwc_uint8(tensor)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8


def test_collect_git_metadata():
    meta = _collect_git_metadata()
    assert isinstance(meta, dict)
    if meta:
        assert "git.commit" in meta
        assert "git.dirty" in meta


def test_local_artifact_root_for_tensorboard(tmp_path):
    logger_obj = type("TensorBoardLogger", (), {"log_dir": str(tmp_path)})()
    assert _local_artifact_root(logger_obj) == tmp_path / "artifacts"


def test_local_artifact_root_for_dvclive(tmp_path):
    logger_obj = type("DVCLiveLogger", (), {"dir": str(tmp_path)})()
    assert _local_artifact_root(logger_obj) == tmp_path / "artifacts"


def test_local_artifact_root_unknown():
    assert _local_artifact_root(object()) is None


def test_augment_mlflow_tags_adds_git():
    tags = {"custom": "value"}
    augmented = _augment_mlflow_tags(tags)
    assert augmented["custom"] == "value"
    assert "git.commit" in augmented


def test_log_artifact_calls_mlflow_style_experiment(tmp_path):
    calls: list[tuple[str, str, str | None]] = []
    experiment = SimpleNamespace(
        log_artifact=lambda run_id, path, artifact_path=None: calls.append(
            (run_id, path, artifact_path)
        )
    )
    fake_logger = SimpleNamespace(experiment=experiment, run_id="run-1")

    source = tmp_path / "metrics.json"
    source.write_text("{}")

    log_artifact(fake_logger, source, artifact_path="metrics")

    assert calls == [("run-1", str(source), "metrics")]


def test_log_artifact_copies_file_for_tensorboard_style_logger(tmp_path):
    fake_logger = type("TensorBoardLogger", (), {"log_dir": str(tmp_path)})()

    source = tmp_path / "summary.txt"
    source.write_text("hello")

    log_artifact(fake_logger, source)

    copied = tmp_path / "artifacts" / "summary.txt"
    assert copied.exists()
    assert copied.read_text() == "hello"


def test_log_artifact_missing_source_raises(tmp_path):
    fake_logger = type("TensorBoardLogger", (), {"log_dir": str(tmp_path)})()
    with pytest.raises(FileNotFoundError):
        log_artifact(fake_logger, tmp_path / "missing.txt")


def test_log_text_artifact_writes_text_and_logs_it(tmp_path):
    fake_logger = type("TensorBoardLogger", (), {"log_dir": str(tmp_path)})()

    log_text_artifact(fake_logger, "hello world", "notes.txt")

    written = tmp_path / "artifacts" / "notes.txt"
    assert written.exists()
    assert written.read_text(encoding="utf-8") == "hello world"


def test_log_image_calls_tensorboard_native_add_image():
    calls: list[tuple[str, object, int | None, str]] = []
    experiment = SimpleNamespace(
        add_image=lambda tag, image, global_step=None, dataformats="HWC": calls.append(
            (tag, image.shape, global_step, dataformats)
        )
    )
    fake_logger = SimpleNamespace(experiment=experiment)

    image = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    log_image(fake_logger, "val/reconstruction", image, step=3)

    assert len(calls) == 1
    tag, shape, step, dataformats = calls[0]
    assert tag == "val/reconstruction"
    assert shape == (4, 4, 3)
    assert step == 3
    assert dataformats == "HWC"


def test_log_image_file_fallback_writes_png(tmp_path):
    fake_logger = type("TensorBoardLogger", (), {"log_dir": str(tmp_path)})()

    image = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    log_image(fake_logger, "val/reconstruction", image)

    destination = tmp_path / "artifacts" / "val" / "val_reconstruction.png"
    assert destination.exists()
