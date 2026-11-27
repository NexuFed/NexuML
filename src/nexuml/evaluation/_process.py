"""Fresh snapshot evaluation and its owned process lifetime."""

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

from nexuml.core.config import ResolvedConfig
from nexuml.core.export import (
    PACKAGE_FILENAME,
    _restore_packaged_policy,
    export_package,
    load_inference_package,
)
from nexuml.evaluation.policy import evaluate_policy, resolve_policy_evaluation


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate_snapshot(
    root: Path, digest: str, *, device: str = "cpu", media_sink=None, receipt_sink=None
) -> dict:
    """Evaluate a verified actor using prepared inputs.

    Returns:
        Metrics, episode report, and immutable source provenance.

    Raises:
        ValueError: If the digest or inference-only package requirement fails.
    """
    artifact = root / "policy" / PACKAGE_FILENAME
    if _digest(artifact) != digest:
        raise ValueError("Evaluation snapshot digest mismatch")
    model, _, metadata = load_inference_package(artifact)
    if metadata["training_state_available"]:
        raise ValueError("Snapshot worker requires a complete inference-only policy")
    actor = _restore_packaged_policy(model, metadata)
    scenario = ResolvedConfig.load(root / "evaluation.yaml").to_scenario()
    assert scenario.interaction is not None
    definition = scenario.interaction.evaluation_environment or scenario.interaction.environment
    frame, version = metadata["training_frames"], metadata["training_batches"]
    reports: list[dict] = []
    with definition.job_scope():
        metrics = evaluate_policy(
            actor.to(device),
            scenario,
            frame_step=frame,
            media_sink=media_sink,
            episode_report_sink=reports.append,
            provenance={
                "source_policy_files_sha256": metadata["policy_files_sha256"],
                "source_training_frames": frame,
            },
        )
        (report,) = reports
        receipt = {
            "snapshot_sha256": digest,
            "source_training_frames": frame,
            "source_training_batches": version,
            "evaluator_pid": os.getpid(),
            "evaluator_device": device,
            "metrics": metrics,
            "episode_report": report,
        }
        if receipt_sink is not None:
            receipt_sink(receipt)
        return receipt


def _worker(root: Path, digest: str, device: str) -> None:
    def cancel(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel)

    def capture(frames, metadata):
        import numpy as np

        np.save(root / "frames.npy", frames, allow_pickle=False)
        (root / "media.json").write_text(json.dumps(metadata, allow_nan=False))

    def write_receipt(receipt):
        (root / "result.json").write_text(json.dumps(receipt, allow_nan=False))

    # Kit's standard fast shutdown may exit the process on leaving its scope.
    evaluate_snapshot(root, digest, device=device, media_sink=capture, receipt_sink=write_receipt)


def prepare_snapshot(root: Path, policy, scenario, owner) -> tuple[Path, str, int, int]:
    """Capture independent evaluation inputs on the learner's update boundary.

    Returns:
        Actor package path, digest, source frame, and source update.
    """
    frame, version = int(owner.frames), int(owner.batches)
    package = export_package(
        policy, root / "policy", metadata={"training_batches": version}, lightning_module=owner
    )
    artifact = package / PACKAGE_FILENAME
    ResolvedConfig.from_scenario(scenario).save(root / "evaluation.yaml")
    return artifact, _digest(artifact), frame, version


def validate_receipt(result: dict, sha256: str, frame: int, version: int) -> None:
    if (
        result["snapshot_sha256"] != sha256
        or result["source_training_frames"] != frame
        or result["source_training_batches"] != version
    ):
        raise RuntimeError("Snapshot evaluation returned unrelated source provenance")


def _terminate(worker: subprocess.Popen) -> None:
    if worker.poll() is not None:
        return
    worker.terminate()
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait(timeout=5)


@contextmanager
def _snapshot_process(root, digest, device):
    with (root / "worker.log").open("w+b") as log:
        worker = subprocess.Popen(
            [sys.executable, "-m", __name__, str(root), digest, device],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else None,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            yield worker, log
        finally:
            _terminate(worker)


def _read_result(root, worker, log, device):
    result_path = root / "result.json"
    if worker.returncode != 0 or not result_path.is_file():
        log.seek(max(0, log.seek(0, os.SEEK_END) - 3000))
        output = log.read().decode(errors="replace")
        raise RuntimeError(
            f"{device.upper()} snapshot evaluation exited {worker.returncode}: {output}"
        )
    return json.loads(result_path.read_text())


class ProcessEvaluation:
    """One immutable snapshot and its evaluator; caller owns close on every path."""

    def __init__(self, policy, scenario, *, owner, timeout_s: float, device: str = "cpu"):
        self.timeout_s = timeout_s
        self.device = device
        _, spec, _ = resolve_policy_evaluation(scenario)
        self.video = spec.video
        with ExitStack() as stack:
            self.root = Path(stack.enter_context(TemporaryDirectory(prefix="nexuml-evaluation-")))
            self.artifact, self.sha256, self.frame, self.version = prepare_snapshot(
                self.root, policy, scenario, owner
            )
            self.worker, self.log = stack.enter_context(
                _snapshot_process(self.root, self.sha256, device)
            )
            self.deadline = monotonic() + timeout_s
            self.resources = stack.pop_all()

    def close(self) -> None:
        self.resources.close()

    def result(
        self, *, wait: bool, check_cancelled: Callable[[], None] | None = None
    ) -> dict | None:
        while True:
            if check_cancelled is not None:
                check_cancelled()
            remaining = max(0.0, self.deadline - monotonic())
            if not wait and remaining and self.worker.poll() is None:
                return None
            try:
                self.worker.wait(
                    timeout=remaining if check_cancelled is None else min(remaining, 0.25)
                )
                break
            except subprocess.TimeoutExpired as error:
                if monotonic() >= self.deadline:
                    raise RuntimeError(
                        f"{self.device.upper()} snapshot evaluation exceeded {self.timeout_s:g} s"
                    ) from error
        if check_cancelled is not None:
            check_cancelled()
        result = _read_result(self.root, self.worker, self.log, self.device)
        if _digest(self.artifact) != self.sha256:
            raise RuntimeError("CPU snapshot was modified during evaluation")
        validate_receipt(result, self.sha256, self.frame, self.version)
        if result["evaluator_device"] != self.device:
            raise RuntimeError("Snapshot evaluation returned unrelated device provenance")
        return result

    def publish_media(self, loggers, report: dict) -> None:
        if not self.video.enabled:
            return
        import numpy as np

        from nexuml.tracking.logger import log_evaluation_media

        metadata = json.loads((self.root / "media.json").read_text())
        frames = np.load(self.root / "frames.npy", allow_pickle=False, mmap_mode="r")
        if (
            frames.ndim != 4
            or frames.dtype != np.uint8
            or frames.shape[-1] != 3
            or not 0 < len(frames) <= self.video.max_frames
            or not 0 < min(frames.shape[1:3]) <= max(frames.shape[1:3]) <= self.video.max_size
            or metadata["captured_frames"] != len(frames)
            or metadata["frame_step"] != self.frame
            or metadata["provenance"]["source_policy_files_sha256"]
            != report["episode_report"]["source_policy_files_sha256"]
        ):
            raise RuntimeError("Snapshot evaluator returned invalid media")
        log_evaluation_media(loggers, list(frames), metadata, self.video, step=self.frame)


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
