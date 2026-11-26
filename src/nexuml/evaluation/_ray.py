"""Single-use CPU evaluation tasks on an existing caller-owned Ray runtime."""

from collections.abc import Callable
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from nexuml.core.export import PACKAGE_FILENAME
from nexuml.core.types import RayEvaluationSpec
from nexuml.evaluation._process import evaluate_snapshot, prepare_snapshot, validate_receipt


def require_ray_runtime():
    """Require an existing caller connection.

    Returns:
        The imported Ray module with its runtime already initialized.

    Raises:
        ImportError: If the Ray extra is missing.
        ValueError: If no caller-owned Ray connection exists.
    """
    try:
        import ray
    except ImportError as error:
        raise ImportError("Ray snapshot evaluation requires nexuml[ray]") from error
    if not ray.is_initialized():
        raise ValueError(
            "Connect to an existing Ray runtime before selecting Ray snapshot evaluation"
        )
    return ray


def _evaluate_on_worker(package: bytes, config: bytes, digest: str) -> str:
    import ray

    with TemporaryDirectory(prefix="nexuml-ray-evaluation-") as directory:
        root = Path(directory)
        (root / "policy").mkdir()
        (root / "policy" / PACKAGE_FILENAME).write_bytes(package)
        (root / "evaluation.yaml").write_bytes(config)
        report = evaluate_snapshot(root, digest)
    context = ray.get_runtime_context()
    report["evaluator_node_id"] = context.get_node_id()
    report["evaluator_resources"] = context.get_assigned_resources()
    return json.dumps(report, allow_nan=False)


class RayEvaluation:
    """One bounded task reference; neither starts nor shuts down the Ray runtime."""

    def __init__(self, policy, scenario, *, owner, spec: RayEvaluationSpec):
        self.ray = require_ray_runtime()
        self.timeout_s = spec.timeout_s
        with TemporaryDirectory(prefix="nexuml-ray-evaluation-") as directory:
            root = Path(directory)
            artifact, self.sha256, self.frame, self.version = prepare_snapshot(
                root, policy, scenario, owner
            )
            package = artifact.read_bytes()
            config = (root / "evaluation.yaml").read_bytes()
        task = self.ray.remote(max_calls=1, max_retries=0, num_cpus=1, num_gpus=0)(
            _evaluate_on_worker
        )
        self.reference = task.options(resources=spec.resources).remote(package, config, self.sha256)
        self.deadline = monotonic() + self.timeout_s

    def result(self, *, wait: bool, check_cancelled: Callable[[], None]) -> dict | None:
        while True:
            check_cancelled()
            remaining = max(0.0, self.deadline - monotonic())
            ready, _ = self.ray.wait([self.reference], timeout=min(0.25, remaining) if wait else 0)
            if ready:
                check_cancelled()
                result = json.loads(self.ray.get(self.reference))
                validate_receipt(result, self.sha256, self.frame, self.version)
                if result["evaluator_device"] != "cpu":
                    raise RuntimeError("Ray CPU evaluation returned unrelated device provenance")
                return result
            if monotonic() >= self.deadline:
                raise RuntimeError(f"Ray snapshot evaluation exceeded {self.timeout_s:g} s")
            if not wait:
                return None

    def close(self) -> None:
        reference = self.reference
        ready, _ = self.ray.wait([reference], timeout=0, fetch_local=False)
        if not ready:
            self.ray.cancel(reference)
            ready, _ = self.ray.wait([reference], timeout=5, fetch_local=False)
            if not ready:
                self.ray.cancel(reference, force=True)
                ready, _ = self.ray.wait([reference], timeout=5, fetch_local=False)
                if not ready:
                    raise RuntimeError("Ray snapshot evaluation did not confirm cancellation")
