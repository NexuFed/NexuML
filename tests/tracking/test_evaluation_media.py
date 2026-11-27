"""Bounded media encoding and native logger delivery proofs, without simulators."""

from __future__ import annotations

import builtins
import copy
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nexuml.core.types import RLVideoSpec
import nexuml.tracking.logger as media


class NativeMLFlowLogger:
    def __init__(self):
        self.uploads = []
        self.run_id = "existing-run"
        self.experiment = GuardedMLflowClient(self.uploads)


class GuardedMLflowClient:
    def __init__(self, uploads):
        self.uploads = uploads

    def log_artifact(self, run_id, path, artifact_path):
        self.uploads.append((run_id, Path(path).name, artifact_path, Path(path).read_bytes()))

    def __getattr__(self, name):
        raise AssertionError(f"MLflow must not be probed for TensorBoard capability: {name}")


class FakeFileWriter:
    def __init__(self):
        self.summaries = []
        self.flushes = 0

    def add_summary(self, summary, step):
        self.summaries.append((summary, step))

    def flush(self):
        self.flushes += 1


class NativeTensorBoardLogger:
    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self.experiment = SimpleNamespace(file_writer=FakeFileWriter())


@pytest.fixture
def capture(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    staging = tmp_path / "staging"
    monkeypatch.setenv("NEXUML_TEMP_ARTIFACT_DIR", str(staging))
    frames = [np.full((33, 65, 3), value, dtype=np.uint8) for value in (0, 120, 240)]
    metadata = {
        "fps": 10.0,
        "captured_frames": 3,
        "capture_capped": True,
        "source_size": [130, 66],
        "captured_size": [65, 33],
        "frame_step": 16,
        "metrics": {"eval/rl/return_mean": 2.0},
        "evaluation": {"episodes": 2, "max_steps_per_episode": 3, "deterministic": True},
        "seeds": [7, 8],
        "environment": {"provider": "synthetic"},
        "config_hash": "config-sha256",
        "contract_hash": "contract-sha256",
        "package": {"path": "trusted.pt", "source_frames": None},
    }
    return frames, metadata, staging


@pytest.fixture
def mp4_stack():
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    try:
        media._evaluation_mp4_info()
    except RuntimeError as exc:
        pytest.skip(f"optional compatible FFmpeg unavailable: {exc}")
    return imageio


def decode_gif(data):
    from PIL import Image, ImageSequence

    with Image.open(io.BytesIO(data)) as image:
        size = image.size
        decoded = []
        durations = []
        for frame in ImageSequence.Iterator(image):
            decoded.append(np.array(frame.convert("RGB")))
            durations.append(frame.info["duration"])
    return size, decoded, durations


@pytest.mark.parametrize("target", ["mlflow", "tensorboard", "both"])
@pytest.mark.parametrize("format_name", ["gif", "mp4", "both"])
def test_selected_outputs_encoded_once_and_native_dispatch(
    target, format_name, capture, tmp_path, monkeypatch, request
):
    if format_name != "gif":
        request.getfixturevalue("mp4_stack")
    frames, metadata, staging = capture
    original_metadata = copy.deepcopy(metadata)
    original_frames = [frame.copy() for frame in frames]
    mlflow = NativeMLFlowLogger()
    tensorboard = NativeTensorBoardLogger(tmp_path / "tensorboard")
    loggers = {
        "mlflow": [mlflow],
        "tensorboard": [tensorboard],
        "both": [mlflow, tensorboard],
    }[target]
    encode_calls = {"gif": 0, "mp4": 0}
    for kind in encode_calls:
        original = getattr(media, f"_encode_evaluation_{kind}")

        def encode(buffer, path, fps, *, kind=kind, original=original):
            assert buffer is frames  # no second rollout buffer
            encode_calls[kind] += 1
            return original(buffer, path, fps)

        monkeypatch.setattr(media, f"_encode_evaluation_{kind}", encode)
    native_dispatch = media.log_artifact
    dispatches = []

    def dispatch(logger_obj, path, artifact_path):
        assert logger_obj is loggers
        assert Path(path).is_file()
        dispatches.append(Path(path).suffix)
        native_dispatch(logger_obj, path, artifact_path)

    monkeypatch.setattr(media, "log_artifact", dispatch)
    video = RLVideoSpec(enabled=True, format=format_name)
    media.preflight_evaluation_media(loggers, video)
    media.log_evaluation_media(loggers, frames, metadata, video, step=16)
    selected = ["gif", "mp4"] if format_name == "both" else [format_name]
    names = {f"frame_000000000016.{kind}" for kind in [*selected, "json"]}
    assert dispatches == [f".{kind}" for kind in [*selected, "json"]]
    assert encode_calls == {
        "gif": int("gif" in selected or target != "mlflow"),
        "mp4": int("mp4" in selected),
    }
    uploaded = {name: data for _, name, _, data in mlflow.uploads}
    local = Path(tensorboard.log_dir) / "artifacts" / "evaluation"
    if target != "tensorboard":
        assert len(mlflow.uploads) == len(names)
        assert {run for run, _, _, _ in mlflow.uploads} == {"existing-run"}
        assert {directory for _, _, directory, _ in mlflow.uploads} == {"evaluation"}
        assert set(uploaded) == names
    else:
        assert mlflow.uploads == []
    if target != "mlflow":
        assert {path.name for path in local.iterdir()} == names
        writer = tensorboard.experiment.file_writer
        assert writer.flushes == 1
        assert len(writer.summaries) == 1
        summary, step = writer.summaries[0]
        assert step == 16
        assert summary.value[0].tag == "eval/rl/episode"
        preview = summary.value[0].image
        assert (preview.width, preview.height, preview.colorspace) == (65, 33, 3)
        size, decoded, durations = decode_gif(preview.encoded_image_string)
        assert size == (65, 33) and len(decoded) == 3 and durations == [100] * 3
        if "gif" in selected:
            assert preview.encoded_image_string == (local / "frame_000000000016.gif").read_bytes()
        if target == "both":
            assert uploaded == {path.name: path.read_bytes() for path in local.iterdir()}
        artifacts = {path.name: path.read_bytes() for path in local.iterdir()}
    else:
        assert not local.exists()
        assert tensorboard.experiment.file_writer.summaries == []
        artifacts = uploaded
    saved = json.loads(artifacts["frame_000000000016.json"])
    for key, value in original_metadata.items():
        assert saved[key] == value
    assert saved["formats"] == selected
    if "gif" in selected:
        size, decoded, durations = decode_gif(artifacts["frame_000000000016.gif"])
        assert size == (65, 33) and durations == [100] * 3
        for actual, expected in zip(decoded, frames, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert saved["encoders"]["gif"]["library"] == "Pillow"
    if "mp4" in selected:
        info = saved["encoders"]["mp4"]
        assert info["encoded_size"] == [66, 34] and info["padding"] == [1, 1]
        assert info["codec"] == "libx264" and info["pixel_format"] == "yuv420p"
        assert info["version"] and info["backend_version"] and info["ffmpeg_version"]
        movie = tmp_path / "decoded.mp4"
        movie.write_bytes(artifacts["frame_000000000016.mp4"])
        imageio = request.getfixturevalue("mp4_stack")
        with imageio.get_reader(str(movie), format="FFMPEG") as reader:
            video_metadata = reader.get_meta_data()
            decoded = [frame for frame in reader]
        assert len(decoded) == 3
        assert video_metadata["fps"] == pytest.approx(10)
        assert video_metadata["duration"] == pytest.approx(0.3)
        assert video_metadata["codec"] == "h264"
        assert video_metadata["pix_fmt"].startswith("yuv420p")
        assert "audio_codec" not in video_metadata
        for actual, expected in zip(decoded, frames, strict=True):
            assert actual.shape == (34, 66, 3)
            padded = np.pad(expected, ((0, 1), (0, 1), (0, 0)), mode="edge")
            assert np.abs(actual.astype(int) - padded.astype(int)).max() <= 4
    assert metadata == original_metadata
    for frame, original in zip(frames, original_frames, strict=True):
        np.testing.assert_array_equal(frame, original)
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize("fps,duration", [(24, 40), (1000, 10)])
def test_gif_timing_quantization(capture, fps, duration):
    frames, metadata, _ = capture
    metadata["fps"] = fps
    logger = NativeMLFlowLogger()
    media.log_evaluation_media(logger, frames, metadata, RLVideoSpec(enabled=True), step=0)
    artifacts = {name: data for _, name, _, data in logger.uploads}
    _, _, durations = decode_gif(artifacts["frame_000000000000.gif"])
    assert durations == [duration] * 3
    info = json.loads(artifacts["frame_000000000000.json"])["encoders"]["gif"]
    assert info["frame_duration_ms"] == duration and info["timing_quantization_ms"] == 10


def test_disabled_media_does_not_import_encoders_access_loggers_or_stage(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("PIL", "imageio", "tensorboard"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled media has no side effects")

    monkeypatch.setattr(media, "iter_loggers", forbidden)
    monkeypatch.setattr(media, "staged_artifact_path", forbidden)
    video = RLVideoSpec()
    media.preflight_evaluation_media(object(), video)
    media.log_evaluation_media(object(), [], {}, video, step=0)


@pytest.mark.parametrize("failure", [RuntimeError("write failed"), KeyboardInterrupt("cancelled")])
def test_encoder_cleanup_failure_preserves_primary_failure(
    capture, mp4_stack, monkeypatch, failure
):
    frames, metadata, staging = capture
    logger = NativeMLFlowLogger()
    closed = []

    class Writer:
        def append_data(self, frame):
            raise failure

        def close(self):
            closed.append(True)
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(mp4_stack, "get_writer", lambda *args, **kwargs: Writer())
    with pytest.raises(type(failure)) as caught:
        media.log_evaluation_media(
            logger, frames, metadata, RLVideoSpec(enabled=True, format="mp4"), step=0
        )
    assert caught.value is failure and closed == [True]
    assert "cleanup failed" in " ".join(caught.value.__notes__)
    assert not logger.uploads and list(staging.iterdir()) == []


def test_playable_but_truncated_mp4_is_not_published(capture, mp4_stack, monkeypatch):
    frames, metadata, staging = capture
    logger = NativeMLFlowLogger()
    original = mp4_stack.get_writer

    class DropFrames:
        def __init__(self, *args, **kwargs):
            self.writer = original(*args, **kwargs)
            self.count = 0

        def append_data(self, frame):
            if self.count == 0:
                self.writer.append_data(frame)
            self.count += 1

        def close(self):
            self.writer.close()

    monkeypatch.setattr(mp4_stack, "get_writer", DropFrames)
    with pytest.raises(RuntimeError, match="missing captured frames"):
        media.log_evaluation_media(
            logger, frames, metadata, RLVideoSpec(enabled=True, format="mp4"), step=0
        )
    assert not logger.uploads and list(staging.iterdir()) == []


@pytest.mark.parametrize("target", [None, False, [], object()])
def test_explicit_recording_requires_artifact_target(target):
    with pytest.raises(ValueError, match="artifact-capable logger"):
        media.preflight_evaluation_media(target, RLVideoSpec(enabled=True))


def test_unsupported_target_retains_native_diagnostic(caplog):
    class UnsupportedMediaLogger:
        pass

    with caplog.at_level(logging.INFO, logger=media.__name__):
        with pytest.raises(ValueError, match="artifact-capable"):
            media.preflight_evaluation_media(UnsupportedMediaLogger(), RLVideoSpec(enabled=True))
    assert "Skipping artifact logging for unsupported logger backend 'UnsupportedMediaLogger'" in (
        caplog.text
    )


def test_supported_local_target_and_unsupported_combination(capture, tmp_path, caplog):
    frames, metadata, _ = capture
    local = type("DVCLiveLogger", (), {"log_dir": str(tmp_path / "local")})()

    class UnsupportedAlongsideLocalLogger:
        pass

    loggers = [local, UnsupportedAlongsideLocalLogger()]
    with caplog.at_level(logging.INFO, logger=media.__name__):
        media.preflight_evaluation_media(loggers, RLVideoSpec(enabled=True))
        media.log_evaluation_media(loggers, frames, metadata, RLVideoSpec(enabled=True), step=0)
    assert "UnsupportedAlongsideLocalLogger" in caplog.text
    assert len(list((tmp_path / "local" / "artifacts" / "evaluation").iterdir())) == 2


@pytest.mark.parametrize("writer", [None, SimpleNamespace(add_summary=lambda *_: None)])
def test_tensorboard_requires_active_compatible_writer(tmp_path, writer):
    logger = NativeTensorBoardLogger(tmp_path)
    logger.experiment.file_writer = writer
    with pytest.raises(RuntimeError, match="active TensorBoard experiment.file_writer"):
        media.preflight_evaluation_media(logger, RLVideoSpec(enabled=True))


@pytest.mark.parametrize(
    "missing,format_name,diagnostic",
    [
        ("PIL", "gif", "nexuml\\[rendering\\]"),
        ("imageio", "mp4", "nexuml\\[video\\]"),
        ("imageio_ffmpeg", "both", "nexuml\\[video\\]"),
    ],
)
def test_missing_prerequisite_fails_preflight(monkeypatch, missing, format_name, diagnostic):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == missing or name.startswith(missing + "."):
            raise ImportError("blocked prerequisite")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match=diagnostic):
        media.preflight_evaluation_media(
            NativeMLFlowLogger(), RLVideoSpec(enabled=True, format=format_name)
        )


def test_gif_never_imports_video_stack(capture, monkeypatch):
    frames, metadata, _ = capture
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("imageio", "tensorboard"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    video = RLVideoSpec(enabled=True)
    logger = NativeMLFlowLogger()
    media.preflight_evaluation_media(logger, video)
    media.log_evaluation_media(logger, frames, metadata, video, step=0)
    assert len(logger.uploads) == 2


@pytest.mark.parametrize("failure", ["executable", "codec"])
def test_ffmpeg_preflight_missing_executable_or_h264(capture, monkeypatch, failure):
    backend = pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("imageio.v2")
    if failure == "executable":
        monkeypatch.setattr(backend, "get_ffmpeg_exe", lambda: "/no/such/ffmpeg")
    else:
        monkeypatch.setattr(
            media.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(stdout="mpeg4 only")
        )
    with pytest.raises(RuntimeError, match="FFmpeg executable with libx264 support"):
        media.preflight_evaluation_media(
            NativeMLFlowLogger(), RLVideoSpec(enabled=True, format="mp4")
        )


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("failed_format", ["gif", "mp4"])
def test_failed_encoder_never_publishes_and_cleans_staging(
    capture, monkeypatch, exception, failed_format
):
    frames, metadata, staging = capture
    logger = NativeMLFlowLogger()

    def fail(_frames, path, _fps):
        path.write_bytes(b"partial")
        raise exception("encoder failed")

    monkeypatch.setattr(media, f"_encode_evaluation_{failed_format}", fail)
    with pytest.raises(exception, match="encoder failed"):
        media.log_evaluation_media(
            logger, frames, metadata, RLVideoSpec(enabled=True, format="both"), step=0
        )
    assert logger.uploads == [] and list(staging.iterdir()) == []


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_mp4_writer_closes_after_failure(capture, mp4_stack, monkeypatch, exception):
    frames, metadata, staging = capture
    closed = []

    class BrokenWriter:
        def close(self):
            closed.append(True)

        def append_data(self, _frame):
            raise exception("interrupted writer")

    def get_writer(path, **kwargs):
        assert kwargs["codec"] == "libx264" and kwargs["pixelformat"] == "yuv420p"
        assert kwargs["macro_block_size"] is None and kwargs["output_params"] == ["-an"]
        Path(path).write_bytes(b"partial")
        return BrokenWriter()

    monkeypatch.setattr(mp4_stack, "get_writer", get_writer)
    logger = NativeMLFlowLogger()
    with pytest.raises(exception, match="interrupted writer"):
        media.log_evaluation_media(
            logger, frames, metadata, RLVideoSpec(enabled=True, format="mp4"), step=0
        )
    assert closed == [True] and logger.uploads == [] and list(staging.iterdir()) == []


def test_failed_mp4_finalization_never_uploads(capture, mp4_stack, monkeypatch):
    frames, metadata, staging = capture

    class BrokenFinalization:
        def append_data(self, _frame):
            pass

        def close(self):
            raise RuntimeError("failed finalization")

    monkeypatch.setattr(mp4_stack, "get_writer", lambda *_, **__: BrokenFinalization())
    logger = NativeMLFlowLogger()
    with pytest.raises(RuntimeError, match="failed finalization"):
        media.log_evaluation_media(
            logger, frames, metadata, RLVideoSpec(enabled=True, format="mp4"), step=0
        )
    assert logger.uploads == [] and list(staging.iterdir()) == []


def test_silent_ffmpeg_failure_cannot_upload_empty_mp4(capture, mp4_stack, monkeypatch):
    frames, metadata, staging = capture

    class EmptyWriter:
        def close(self):
            pass

        def append_data(self, _frame):
            pass

    monkeypatch.setattr(mp4_stack, "get_writer", lambda *_, **__: EmptyWriter())
    logger = NativeMLFlowLogger()
    with pytest.raises((OSError, RuntimeError)):
        media.log_evaluation_media(
            logger, frames, metadata, RLVideoSpec(enabled=True, format="mp4"), step=0
        )
    assert logger.uploads == [] and list(staging.iterdir()) == []


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_failed_upload_visible_and_staging_cleaned(capture, monkeypatch, exception):
    frames, metadata, staging = capture
    logger = NativeMLFlowLogger()

    def fail(*_):
        raise exception("upload rejected")

    monkeypatch.setattr(logger.experiment, "log_artifact", fail)
    with pytest.raises(exception, match="upload rejected"):
        media.log_evaluation_media(logger, frames, metadata, RLVideoSpec(enabled=True), step=0)
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize("stage", ["add_summary", "flush"])
def test_failed_preview_retains_delivered_artifacts(capture, tmp_path, monkeypatch, stage):
    frames, metadata, staging = capture
    tensorboard = NativeTensorBoardLogger(tmp_path / "tensorboard")
    mlflow = NativeMLFlowLogger()

    def fail(*args):
        raise OSError("preview unavailable")

    monkeypatch.setattr(tensorboard.experiment.file_writer, stage, fail)
    with pytest.raises(OSError, match="preview unavailable") as caught:
        media.log_evaluation_media(
            [mlflow, tensorboard], frames, metadata, RLVideoSpec(enabled=True), step=16
        )
    assert len(mlflow.uploads) == 2
    assert (tmp_path / "tensorboard/artifacts/evaluation/frame_000000000016.gif").is_file()
    assert "TensorBoard preview delivery failed" in " ".join(caught.value.__notes__)
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps_never_uploads(capture, fps):
    frames, metadata, _ = capture
    metadata["fps"] = fps
    logger = NativeMLFlowLogger()
    with pytest.raises(ValueError, match="finite positive FPS"):
        media.log_evaluation_media(logger, frames, metadata, RLVideoSpec(enabled=True), step=0)
    assert logger.uploads == []


def test_empty_capture_never_uploads(capture):
    _, metadata, _ = capture
    logger = NativeMLFlowLogger()
    with pytest.raises(ValueError, match="requires frames"):
        media.log_evaluation_media(logger, [], metadata, RLVideoSpec(enabled=True), step=0)
    assert logger.uploads == []


@pytest.mark.requires_optional("tensorboard")
def test_real_native_tensorboard_events_hold_animated_bytes(capture, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    frames, metadata, staging = capture
    logger = media._create_tensorboard_logger(str(tmp_path / "tb"), "test-media", "existing")
    assert logger is not None
    try:
        experiment = logger.experiment
        writer = experiment.file_writer
        for step in (0, 16):
            media.preflight_evaluation_media(logger, RLVideoSpec(enabled=True))
            media.log_evaluation_media(
                logger, frames, metadata, RLVideoSpec(enabled=True), step=step
            )
            assert logger.experiment is experiment and logger.experiment.file_writer is writer
        events = EventAccumulator(logger.log_dir, size_guidance={"images": 0}).Reload()
        images = events.Images("eval/rl/episode")
        assert [event.step for event in images] == [0, 16]
        for event in images:
            size, decoded, durations = decode_gif(event.encoded_image_string)
            assert size == (65, 33) and durations == [100] * 3
            assert len(decoded) == 3
            artifact = Path(logger.log_dir) / "artifacts" / "evaluation"
            assert (
                event.encoded_image_string
                == (artifact / f"frame_{event.step:012d}.gif").read_bytes()
            )
    finally:
        logger.finalize("success")
    assert list(staging.iterdir()) == []


@pytest.mark.requires_optional("mlflow")
def test_real_native_mlflow_local_sqlite_artifact_roundtrip(capture, tmp_path, monkeypatch):
    import mlflow
    from mlflow.tracking import MlflowClient

    from nexuml.core.types import LoggingSpec, MLflowSpec

    frames, metadata, staging = capture
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    loggers = []
    try:
        loggers = media.create_loggers(
            LoggingSpec(
                experiment_name="local-media-proof",
                temp_artifact_dir=str(staging),
                mlflow=MLflowSpec(
                    tracking_uri=uri,
                    artifact_location=str(tmp_path / "artifacts"),
                    auto_nested_runs=False,
                ),
            ),
            run_name="existing-media-run",
        )
        assert len(loggers) == 1
        native = loggers[0]
        run_id = native.run_id
        media.preflight_evaluation_media(loggers, RLVideoSpec(enabled=True))
        native.log_metrics(metadata["metrics"], step=16)
        media.log_evaluation_media(loggers, frames, metadata, RLVideoSpec(enabled=True), step=16)
        assert native.run_id == run_id
        client = MlflowClient(tracking_uri=uri)
        assert {artifact.path for artifact in client.list_artifacts(run_id, "evaluation")} == {
            "evaluation/frame_000000000016.gif",
            "evaluation/frame_000000000016.json",
        }
        destination = tmp_path / "downloaded"
        destination.mkdir()
        gif = client.download_artifacts(
            run_id, "evaluation/frame_000000000016.gif", str(destination)
        )
        size, decoded, durations = decode_gif(Path(gif).read_bytes())
        assert size == (65, 33) and len(decoded) == 3 and durations == [100] * 3
        provenance = client.download_artifacts(
            run_id, "evaluation/frame_000000000016.json", str(destination)
        )
        assert json.loads(Path(provenance).read_text())["config_hash"] == "config-sha256"
        history = client.get_metric_history(run_id, "eval/rl/return_mean")
        assert [(point.step, point.value) for point in history] == [(16, 2.0)]
        experiment = client.get_run(run_id).info.experiment_id
        assert len(client.search_runs([experiment])) == 1
    finally:
        for logger in loggers:
            logger.finalize("success")
        mlflow.set_tracking_uri(previous_uri)
    assert list(staging.iterdir()) == []
