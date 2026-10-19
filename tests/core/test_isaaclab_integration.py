"""Opt-in prepared-profile acceptance; each native app owns a fresh child process."""

import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys

import pytest


pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.requires_simulator,
    pytest.mark.requires_optional("isaaclab"),
]
ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "scripts/isaaclab/run.py"
JOB = ROOT / "tests/fixtures/isaaclab_job.py"


def _run(args, output, label, *, expected=0, process_scope=True):
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NEXUML_LOGS_ROOT": str(output / "native-logs")}
    result = subprocess.run(
        [sys.executable, *([str(ENTRYPOINT)] if process_scope else []), *map(str, args)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=420,
    )
    (output / f"{label}.log").write_text(result.stdout)
    assert result.returncode == expected, result.stdout
    return result.stdout


@pytest.mark.parametrize("task", ["Isaac-Cartpole", "Isaac-Cartpole-Direct"])
@pytest.mark.parametrize("physics", ["newton_mjwarp", "isaacsim_physx", "ovphysx"])
def test_native_manager_and_direct_transition_oracle(tmp_path, task, physics):
    log = _run(
        [ROOT / "tests/fixtures/isaaclab_transitions.py", task, physics],
        tmp_path,
        "transitions",
    )
    assert log.count("NATIVE_TRANSITION_GAE") == 7
    assert "NATIVE_ADAPTER_TRANSITIONS_VERIFIED" in log


@pytest.mark.parametrize(
    "reference", ["cartpole", "cartpole-newton", "cartpole-direct", "franka", "so101"]
)
def test_native_checkpoint_remaining_budget_export_and_scalar_replay(tmp_path, reference):
    for stage in ("interrupt", "resume", "complete", "infer"):
        log = _run(
            [JOB, reference, stage, tmp_path],
            tmp_path,
            stage,
            expected=1 if stage == "interrupt" else 0,
            process_scope=stage != "infer",
        )
        if stage == "interrupt":
            assert "INTERRUPTED_CHECKPOINT_AND_FAILED_LOGGER_FINALIZED" in log
        elif stage == "resume":
            assert "ACTOR_CRITIC_OPTIMIZERS_COUNTERS_RESTORED" in log
        elif stage == "complete":
            assert "COMPLETED_BUDGET_NO_SIMULATOR" in log
        else:
            assert "EXACT_PORTABLE_CPU_INFERENCE_VERIFIED" in log
    log = _run(
        [
            "-m",
            "nexuml",
            "evaluate",
            tmp_path / reference / "resume/policy",
            "--device",
            "cuda:0",
            "-O",
            "training.evaluation.episodes=2",
            "-O",
            "training.evaluation.max_steps_per_episode=8",
        ],
        tmp_path,
        "scalar-replay",
    )
    assert "Evaluation complete!" in log and "frame step 0" in log
    if reference == "so101":
        assert "task_success_rate" in log and "final_joint_error_rad_mean" in log


def test_native_frame_budget_counts_rounded_actual_transitions(tmp_path):
    log = _run([JOB, "cartpole", "rounded", tmp_path], tmp_path, "rounded")
    assert '"frames": 256' in log and '"collected_after_batches": [0, 1]' in log


@pytest.mark.parametrize("reference", ["cartpole", "cartpole-newton"])
def test_native_sigterm_finalizes_failed_logger_before_owned_shutdown(tmp_path, reference):
    log = _run([JOB, reference, "terminate", tmp_path], tmp_path, "sigterm", expected=143)
    assert "INTERRUPTED_CHECKPOINT_AND_FAILED_LOGGER_FINALIZED" in log
    if reference == "cartpole":
        assert "SimulationContext cleared" in log


def test_native_kit_async_cancellation_cannot_exit_successfully(tmp_path):
    log = _run([JOB, "cartpole", "cancel-native", tmp_path], tmp_path, "async-cancel", expected=1)
    assert "NATIVE_RUNTIME_CLOSED_BEFORE_ASYNC_CANCEL" in log and "CancelledError" in log


def test_native_recorded_replay_sigterm_never_marks_mlflow_success(tmp_path):
    _run([JOB, "cartpole", "train", tmp_path], tmp_path, "training")
    log = _run(
        [JOB, "cartpole", "cancel-replay", tmp_path], tmp_path, "cancel-replay", expected=143
    )
    assert "Evaluation complete!" not in log and "SimulationContext cleared" in log
    with sqlite3.connect(tmp_path / "cartpole/cancel-replay.db") as database:
        assert database.execute("SELECT count(*), min(status) FROM runs").fetchone() == (
            1,
            "FAILED",
        )
    assert not list((tmp_path / "cartpole/cancel-artifacts").rglob("*.gif"))


@pytest.mark.parametrize("reference", ["cartpole", "franka", "so101"])
@pytest.mark.parametrize("format", ["gif", "mp4", "both"])
def test_native_recorded_replay_uses_existing_media_pipeline(tmp_path, reference, format):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    image = pytest.importorskip("PIL.Image")
    _run([JOB, reference, "train", tmp_path], tmp_path, "training")
    log_root = tmp_path / "native-logs"
    prior_events = len(list(log_root.rglob("events.out.tfevents.*")))
    backend = ("tensorboard", "mlflow", "both")[
        (("cartpole", "franka", "so101").index(reference) + ("gif", "mp4", "both").index(format))
        % 3
    ]
    logging = []
    if backend != "tensorboard":
        mlflow = {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "artifact_location": str(log_root / "mlflow-artifacts"),
            "auto_nested_runs": False,
        }
        logging = [
            "-O",
            "logging="
            + json.dumps(
                {
                    "tensorboard": None if backend == "mlflow" else {},
                    "mlflow": mlflow,
                }
            ),
        ]
    log = _run(
        [
            "-m",
            "nexuml",
            "evaluate",
            tmp_path / reference / "train/policy",
            "--device",
            "cuda:0",
            "-O",
            "training.evaluation.episodes=2",
            "-O",
            "training.evaluation.max_steps_per_episode=8",
            "-O",
            "training.evaluation.video.enabled=true",
            "-O",
            f"training.evaluation.video.format={format}",
            "-O",
            "training.evaluation.video.max_frames=9",
            "-O",
            "training.evaluation.video.max_size=320",
            *logging,
        ],
        tmp_path,
        "recorded-replay",
    )
    assert "Evaluation complete!" in log
    artifacts = list((tmp_path / "native-logs").rglob("*.gif"))
    videos = list((tmp_path / "native-logs").rglob("*.mp4"))
    assert bool(artifacts) == (format in {"gif", "both"})
    assert bool(videos) == (format in {"mp4", "both"})
    metadata = []
    for path in log_root.rglob("*.json"):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "captured_frames" in data:
            metadata.append(data)
    assert metadata and all(m["renderer"] == "isaac_rtx" for m in metadata)
    assert all(1 <= m["captured_frames"] <= 9 and m["frame_step"] == 0 for m in metadata)
    expected = metadata[0]["captured_frames"] / metadata[0]["fps"]
    for path in artifacts:
        duration = 0
        with image.open(path) as clip:
            assert clip.size == (320, 240)
            assert 1 <= clip.n_frames <= metadata[0]["captured_frames"]
            for index in range(clip.n_frames):
                clip.seek(index)
                duration += clip.info["duration"] / 1000
        gif_duration = metadata[0]["encoders"]["gif"]["frame_duration_ms"] / 1000
        assert duration == pytest.approx(metadata[0]["captured_frames"] * gif_duration, abs=0.011)
    for path in videos:
        reader = ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            header = next(reader)
            assert header["size"] == (320, 240) and header["codec"] == "h264"
            assert sum(1 for _ in reader) == metadata[0]["captured_frames"]
            assert header["duration"] == pytest.approx(expected, abs=0.02)
        finally:
            reader.close()
    if backend == "mlflow":
        assert len(list(log_root.rglob("events.out.tfevents.*"))) == prior_events
    if backend != "tensorboard":
        with sqlite3.connect(tmp_path / "mlflow.db") as database:
            assert database.execute("SELECT count(*), min(status) FROM runs").fetchone() == (
                1,
                "FINISHED",
            )
