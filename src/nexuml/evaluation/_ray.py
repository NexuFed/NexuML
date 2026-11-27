"""Single-use evaluation tasks on an existing caller-owned Ray runtime."""

from collections.abc import Callable
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from nexuml.core.export import PACKAGE_FILENAME
from nexuml.core.types import RayEvaluationSpec
from nexuml.evaluation._process import (
    _read_result,
    _snapshot_process,
    evaluate_snapshot,
    prepare_snapshot,
    validate_receipt,
)


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


def _evaluate_on_worker(package: bytes, config: bytes, digest: str, timeout_s: float = 120) -> str:
    import ray
    import torch

    context = ray.get_runtime_context()
    resources = context.get_assigned_resources()
    device = "cuda:0" if resources.get("GPU") == 1 else "cpu"
    if device == "cuda:0" and (torch.cuda.device_count() != 1 or not torch.cuda.is_available()):
        raise RuntimeError("Ray snapshot GPU assignment did not expose exactly one CUDA device")
    with TemporaryDirectory(prefix="nexuml-ray-evaluation-") as directory:
        root = Path(directory)
        (root / "policy").mkdir()
        (root / "policy" / PACKAGE_FILENAME).write_bytes(package)
        (root / "evaluation.yaml").write_bytes(config)
        if device == "cpu":
            report = evaluate_snapshot(root, digest)
        else:
            # The Ray task retains its worker protocol; the child owns native Kit shutdown.
            with _snapshot_process(root, digest, device) as (worker, log):
                worker.wait(timeout=timeout_s)
                report = _read_result(root, worker, log, device)
    report["evaluator_node_id"] = context.get_node_id()
    report["evaluator_resources"] = resources
    return json.dumps(report, allow_nan=False)


class RayEvaluation:
    """One bounded task reference; neither starts nor shuts down the Ray runtime."""

    def __init__(self, policy, scenario, *, owner, spec: RayEvaluationSpec):
        self.ray = require_ray_runtime()
        self.timeout_s = spec.timeout_s
        self.num_gpus = spec.num_gpus
        with TemporaryDirectory(prefix="nexuml-ray-evaluation-") as directory:
            root = Path(directory)
            artifact, self.sha256, self.frame, self.version = prepare_snapshot(
                root, policy, scenario, owner
            )
            package = artifact.read_bytes()
            config = (root / "evaluation.yaml").read_bytes()
        task = self.ray.remote(max_calls=1, max_retries=0, num_cpus=1, num_gpus=spec.num_gpus)(
            _evaluate_on_worker
        )
        options: dict[str, Any] = {"resources": spec.resources}
        if spec.py_executable is not None:
            options["runtime_env"] = {"py_executable": spec.py_executable}
        inputs = (package, config, self.sha256)
        if spec.num_gpus:
            inputs = (*inputs, self.timeout_s)
        self.reference = task.options(**options).remote(*inputs)
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
                expected = "cuda:0" if self.num_gpus else "cpu"
                if result["evaluator_device"] != expected:
                    raise RuntimeError(
                        "Ray snapshot evaluation returned unrelated device provenance"
                    )
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
