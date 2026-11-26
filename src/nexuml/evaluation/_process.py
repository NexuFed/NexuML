"""Isolated CPU worker and owned process lifetime for fit evaluation."""

from collections.abc import Callable
from contextlib import ExitStack
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
from nexuml.evaluation.policy import evaluate_policy


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate_snapshot(root: Path, digest: str) -> dict:
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
            actor,
            scenario,
            frame_step=frame,
            episode_report_sink=reports.append,
            provenance={
                "source_policy_files_sha256": metadata["policy_files_sha256"],
                "source_training_frames": frame,
            },
        )
    (report,) = reports
    return {
        "snapshot_sha256": digest,
        "source_training_frames": frame,
        "source_training_batches": version,
        "evaluator_pid": os.getpid(),
        "metrics": metrics,
        "episode_report": report,
    }


def _worker(root: Path, digest: str) -> None:
    def cancel(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel)
    (root / "result.json").write_text(json.dumps(evaluate_snapshot(root, digest), allow_nan=False))


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


class ProcessEvaluation:
    """One immutable snapshot and its evaluator; caller owns close on every path."""

    def __init__(self, policy, scenario, *, owner, timeout_s: float):
        self.timeout_s = timeout_s
        with ExitStack() as stack:
            self.root = Path(stack.enter_context(TemporaryDirectory(prefix="nexuml-evaluation-")))
            self.artifact, self.sha256, self.frame, self.version = prepare_snapshot(
                self.root, policy, scenario, owner
            )
            self.log = stack.enter_context((self.root / "worker.log").open("w+b"))
            self.worker = subprocess.Popen(
                [sys.executable, "-m", __name__, str(self.root), self.sha256],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            stack.callback(_terminate, self.worker)
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
                        f"CPU snapshot evaluation exceeded {self.timeout_s:g} s"
                    ) from error
        if check_cancelled is not None:
            check_cancelled()
        result_path = self.root / "result.json"
        if self.worker.returncode != 0 or not result_path.is_file():
            self.log.seek(max(0, self.log.seek(0, os.SEEK_END) - 3000))
            output = self.log.read().decode(errors="replace")
            raise RuntimeError(f"CPU snapshot evaluation exited {self.worker.returncode}: {output}")
        result = json.loads(result_path.read_text())
        if _digest(self.artifact) != self.sha256:
            raise RuntimeError("CPU snapshot was modified during evaluation")
        validate_receipt(result, self.sha256, self.frame, self.version)
        return result


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), sys.argv[2])
