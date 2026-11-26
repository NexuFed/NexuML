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


def _worker(root: Path, digest: str) -> None:
    def cancel(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel)
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
    (root / "result.json").write_text(
        json.dumps(
            {
                "snapshot_sha256": digest,
                "source_training_frames": frame,
                "source_training_batches": version,
                "evaluator_pid": os.getpid(),
                "metrics": metrics,
                "episode_report": report,
            },
            allow_nan=False,
        )
    )


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
        self.frame, self.version = int(owner.frames), int(owner.batches)
        self.timeout_s = timeout_s
        with ExitStack() as stack:
            self.root = Path(stack.enter_context(TemporaryDirectory(prefix="nexuml-evaluation-")))
            package = export_package(
                policy,
                self.root / "policy",
                metadata={"training_batches": self.version},
                lightning_module=owner,
            )
            self.artifact = package / PACKAGE_FILENAME
            self.sha256 = _digest(self.artifact)
            ResolvedConfig.from_scenario(scenario).save(self.root / "evaluation.yaml")
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
        if (
            result["snapshot_sha256"] != self.sha256
            or _digest(self.artifact) != self.sha256
            or result["source_training_frames"] != self.frame
            or result["source_training_batches"] != self.version
        ):
            raise RuntimeError("CPU snapshot evaluation returned unrelated source provenance")
        return result


def evaluate_in_fresh_process(
    policy, scenario, *, owner, timeout_s: float, check_cancelled: Callable[[], None] | None = None
) -> dict:
    """Export the post-update actor and block on one fresh CPU evaluator.

    Returns:
        Evaluation receipt with verified source frame, update, and digest.
    """
    pending = ProcessEvaluation(policy, scenario, owner=owner, timeout_s=timeout_s)
    try:
        result = pending.result(wait=True, check_cancelled=check_cancelled)
        assert result is not None
        return result
    finally:
        pending.close()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), sys.argv[2])
