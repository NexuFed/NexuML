"""Submit fixed qualification cases through the registered SDK backend; inert by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4
from urllib.parse import urlsplit


CASES = ("preflight", "regression", "package", "recipe", "so101", "browser", "recovery")
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
KNOWN_STATES = TERMINAL | {"PENDING", "RUNNING"}


def _bounded_rpc(function, *args, timeout_seconds=30, **kwargs):
    """Bound a control-plane call without changing the SDK or its worker environment.

    Returns:
        The native SDK result.

    Raises:
        RuntimeError: If another controller already owns the process alarm.
    """
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("Campaign must own its control-plane deadline timer")

    def expired(_signal, _frame):
        raise TimeoutError("Ray control-plane request exceeded its deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return function(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def source_identity(root: Path) -> str:
    """Hash the uploaded implementation and test-owned worker fixtures.

    Returns:
        A content identity independent of the upload directory name.
    """
    paths = {
        path
        for directory in ("src", "library/src", "tests", "scripts")
        for path in (root / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    }
    paths.update(root / name for name in ("pyproject.toml", "library/pyproject.toml", "uv.lock"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def case_result(logs: str) -> dict | None:
    """Read the worker's final structured receipt, never infer success from other logs.

    Returns:
        The last valid fixed-fixture receipt, or unknown.

    Raises:
        ValueError: If a worker publishes malformed structured metadata.
    """
    for line in reversed(logs.splitlines()):
        if line.startswith("NEXUML_CASE_RESULT="):
            value = json.loads(line.partition("=")[2])
            if not isinstance(value, dict):
                raise ValueError("Worker receipt must be a mapping")
            return value
    return None


def run_campaign(
    attempts: list[dict],
    client,
    output: Path,
    *,
    concurrency: int = 1,
    queue_seconds: float = 900,
    poll_seconds: float = 2,
) -> dict:
    """Bound submissions, retain terminal receipts, and cancel only owned job IDs.

    Returns:
        An explicit campaign result; submission is never a passing case.

    Raises:
        ValueError: If the budget or attempt ownership is invalid.
        RuntimeError: If an owned deadline cancellation remains unconfirmed.
    """
    from nexuml.execution import run

    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be between one and four")
    if any(not math.isfinite(value) or value <= 0 for value in (queue_seconds, poll_seconds)):
        raise ValueError("Monitoring budgets must be finite and positive")
    roots = [str(item["attempt_root"]) for item in attempts]
    if len(set(roots)) != len(roots):
        raise ValueError("Attempts must not share output roots")
    output.mkdir(parents=True, exist_ok=False)
    pending, active, results = list(attempts), {}, []

    def metadata(job_id, attempt):
        return {
            "job_id": job_id,
            "case": attempt["case"],
            "attempt_root": str(attempt["attempt_root"]),
            "source_revision": attempt["options"]["source_revision"],
            "source_dirty": attempt["options"].get("source_dirty"),
            "source_tree_sha256": attempt["options"]["source_tree_sha256"],
            **{
                key: attempt[key]
                for key in ("submission_error_type", "reconciliation", "cleanup")
                if key in attempt
            },
        }

    def save():
        value = {
            "concurrency": concurrency,
            "pool": "pool_gpu_workers",
            "results": results,
            "active": list(active),
            "active_attempts": [metadata(job_id, attempt) for job_id, attempt in active.items()],
            "remaining": len(pending),
        }
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "summary.json")
        return value

    try:
        while pending or active:
            while pending and len(active) < concurrency:
                attempt = pending.pop(0)
                scenario = attempt["scenario"]
                execution = scenario.execution
                if (
                    execution.component_name != "ray-jobs-sdk"
                    or execution.entrypoint_num_gpus != 1
                    or execution.entrypoint_resources != {"pool_gpu_workers": 1}
                ):
                    raise ValueError("Acceptance requires SDK Jobs, one GPU and GPU workers only")
                option_names = {
                    "preflight": {"data_root"},
                    "regression": {"suite"},
                    "recipe": {"recipe"},
                    "recovery": {"source_attempt", "source_case_sha256", "source_guard_sha256"},
                }.get(attempt["case"], set())
                worker_options = {
                    key: value
                    for key, value in attempt["options"].items()
                    if key in option_names and value is not None
                }
                job_id = uuid4().hex
                if job_id in active or any(row["job_id"] == job_id for row in results):
                    raise ValueError("Attempt UUID must not reuse an owned job ID")
                active[job_id] = {**attempt, "submitted_at": time.monotonic()}
                attempt = active[job_id]
                save()  # Persist ownership before an RPC can accept the job and lose its response.
                try:
                    returned_id = _bounded_rpc(
                        run,
                        scenario,
                        timeout_seconds=120,
                        operation="acceptance",
                        case=attempt["case"],
                        attempt_root=attempt["attempt_root"],
                        wall_seconds=attempt["wall_seconds"],
                        case_options=worker_options,
                        submission_id=job_id,
                    )
                except Exception as error:
                    attempt["submission_error_type"] = type(error).__name__
                    try:
                        observed = str(_bounded_rpc(client.get_job_status, job_id)).upper()
                        reconciled = observed in KNOWN_STATES
                        attempt["reconciliation"] = observed if reconciled else "unknown"
                    except BaseException as reconciliation_error:
                        reconciled = False
                        attempt["reconciliation"] = type(reconciliation_error).__name__
                    if not reconciled:
                        results.append(
                            {**metadata(job_id, attempt), "status": "submission_unknown"}
                        )
                        raise
                else:
                    if returned_id != job_id:
                        raise ValueError("Submission returned a different ID than the owned UUID")
                print(f"Submitted {attempt['case']}: {job_id}", flush=True)
                save()
            for job_id, attempt in list(active.items()):
                status = str(_bounded_rpc(client.get_job_status, job_id)).upper()
                elapsed = time.monotonic() - attempt["submitted_at"]
                timed_out = "stop_requested_at" in attempt
                if status not in TERMINAL:
                    if elapsed <= queue_seconds + attempt["wall_seconds"]:
                        continue
                    if not timed_out:
                        _bounded_rpc(client.stop_job, job_id)
                        attempt["stop_requested_at"] = time.monotonic()
                    elif time.monotonic() - attempt["stop_requested_at"] > 30:
                        raise RuntimeError(f"Owned job cancellation remains unconfirmed: {job_id}")
                    continue
                logs = _bounded_rpc(client.get_job_logs, job_id)
                (output / f"{job_id}.log").write_text(logs)
                receipt = case_result(logs)
                identity = attempt["options"]["source_tree_sha256"]
                passed = (
                    not timed_out
                    and status == "SUCCEEDED"
                    and receipt is not None
                    and receipt.get("status") == "PASS"
                    and receipt.get("source_tree_sha256") == identity
                )
                row = {
                    "case": attempt["case"],
                    "job_id": job_id,
                    "native_status": status,
                    "status": "passed"
                    if passed
                    else (
                        "blocked" if receipt and receipt.get("status") == "BLOCKED" else "failed"
                    ),
                    "deadline_exceeded": timed_out,
                    "elapsed_seconds": elapsed,
                    "attempt_root": attempt["attempt_root"],
                    "source_revision": attempt["options"]["source_revision"],
                    "source_dirty": attempt["options"].get("source_dirty"),
                    "source_tree_sha256": identity,
                    "budgets": attempt["options"].get("budgets"),
                    "receipt": receipt,
                    **{
                        key: attempt[key]
                        for key in ("submission_error_type", "reconciliation")
                        if key in attempt
                    },
                }
                (output / f"{job_id}.json").write_text(json.dumps(row, indent=2) + "\n")
                results.append(row)
                del active[job_id]
                print(f"{job_id}: {row['status']} ({status})", flush=True)
                save()
            if active:
                time.sleep(poll_seconds)
    finally:
        primary = sys.exception()
        for job_id, attempt in active.items():
            cleanup = {"status": "unknown"}
            attempt["cleanup"] = cleanup
            try:
                _bounded_rpc(client.stop_job, job_id)
            except BaseException as error:
                cleanup["stop_error_type"] = type(error).__name__
            try:
                status = str(_bounded_rpc(client.get_job_status, job_id)).upper()
                if status in TERMINAL:
                    cleanup.update(status="confirmed_terminal", native_status=status)
            except BaseException as error:
                cleanup["status_error_type"] = type(error).__name__
            try:
                logs = _bounded_rpc(client.get_job_logs, job_id)
                (output / f"{job_id}.log").write_text(logs)
            except BaseException as error:
                cleanup["logs_error_type"] = type(error).__name__
        try:
            save()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"Campaign receipt also failed: {type(error).__name__}")
    return save()


def main() -> None:
    """Resolve trusted settings and budgets; submit only with explicit --submit.

    Raises:
        SystemExit: If any submitted case fails or input validation fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES, required=True)
    parser.add_argument("--recipe", help="Registered matrix recipe; required for recipe cases")
    parser.add_argument("--profile", type=Path, default=Path("configs/execution/ray.yaml"))
    parser.add_argument("--address", default="https://ray.ika.rub.de")
    parser.add_argument("--ca", type=Path, required=True)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, default=Path("/mnt/data/nexuml-qualification"))
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--wall-seconds", type=float, default=3600)
    parser.add_argument("--queue-seconds", type=float, default=900)
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--source-attempt", type=str)
    parser.add_argument("--source-case-sha256", type=str)
    parser.add_argument("--source-guard-sha256", type=str)
    parser.add_argument("--suite", choices=("full", "focused", "static"), default="full")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not args.ca.is_file():
        parser.error("The approved CA file must exist on the submitter")
    address = urlsplit(args.address)
    if address.username or address.password or address.query or address.fragment:
        parser.error("The endpoint must not contain credentials, query parameters or fragments")
    if not args.durable_root.is_absolute() or ".." in args.durable_root.parts:
        parser.error("Durable root must be an absolute worker-owned path")
    if not args.durable_root.is_relative_to("/mnt/data/nexuml-qualification"):
        parser.error("This campaign requires its approved /mnt/data/nexuml-qualification root")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        parser.error("wall-seconds must be finite and positive")
    if min(args.frames, args.batch, args.envs) <= 0:
        parser.error("Recipe budgets must be positive")
    if "recipe" in args.case and not args.recipe:
        parser.error("recipe cases require --recipe")
    from nexuml.core.types import ReinforcementLearningSpec, ScenarioSpec
    from nexuml.execution import load_ray_job_profile
    from nexuml.execution.definitions import RayJobsSDKExecution

    base = load_ray_job_profile(args.profile)
    if base.setup_command is None and args.extra:
        parser.error("Dependency extras require enabled locked setup")
    if any(not extra.replace("-", "").replace("_", "").isalnum() for extra in args.extra):
        parser.error("Extras must be project dependency-extra names")
    root = Path(__file__).resolve().parents[1]
    identity = source_identity(root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    campaign = uuid4().hex
    attempts = []
    for case in args.case:
        attempt_root = args.durable_root / campaign / f"{case}-{uuid4().hex}"
        values = base.model_dump()
        values.update(address=args.address, verify=str(args.ca))
        values["entrypoint_num_gpus"] = 1
        values["entrypoint_resources"] = {"pool_gpu_workers": 1}
        if args.extra:
            assert base.setup_command is not None
            values["setup_command"] = [
                *base.setup_command,
                *(part for extra in args.extra for part in ("--extra", extra)),
            ]
        runtime = values["runtime_env"]
        runtime["working_dir"] = str(root)
        runtime["excludes"] = [
            *runtime.get("excludes", []),
            "/.venv/",
            "/rl-runs/",
            "/reports/",
            "/data/",
        ]
        runtime["env_vars"] = {
            **runtime.get("env_vars", {}),
            "NEXUML_LOGS_ROOT": str(attempt_root / "logs"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        execution = RayJobsSDKExecution.model_validate(values)
        if case == "recipe":
            from nexuml.core.scenario_registry import get_scenario_registry

            scenario = get_scenario_registry().get(args.recipe)()
            if not isinstance(scenario.training, ReinforcementLearningSpec):
                parser.error("Recipe cases require a registered RL matrix scenario")
            scenario.training.total_frames = args.frames
            scenario.training.frames_per_batch = args.batch
            scenario.training.collector.num_envs_per_collector = args.envs
            scenario.execution = execution
        else:
            scenario = ScenarioSpec(name=f"acceptance-{case}", execution=execution)
        attempts.append(
            {
                "case": case,
                "scenario": scenario,
                "attempt_root": str(attempt_root),
                "wall_seconds": args.wall_seconds,
                "options": {
                    "source_revision": revision,
                    "source_dirty": dirty,
                    "source_tree_sha256": identity,
                    "data_root": args.data_root,
                    "suite": args.suite,
                    "recipe": args.recipe,
                    "source_attempt": args.source_attempt,
                    "source_case_sha256": args.source_case_sha256,
                    "source_guard_sha256": args.source_guard_sha256,
                    "budgets": {
                        "cpu_threads": 1,
                        "gpus": 1,
                        "ram_bytes": 32 * 1024**3,
                        "vram_bytes": 20 * 1024**3,
                        "output_bytes": 4 * 1024**3,
                        "wall_seconds": args.wall_seconds,
                        "disk_floor_bytes": 4 * 1024**3,
                        "hard_limits": ["wall_seconds", "disk_floor_bytes"],
                    },
                },
            }
        )
    print(
        json.dumps(
            {
                "campaign": campaign,
                "source_revision": revision,
                "source_dirty": dirty,
                "source_tree_sha256": identity,
                "pool": "pool_gpu_workers",
                "concurrency": args.concurrency,
                "deployment_inputs": {
                    "address": args.address,
                    "ca_path": str(args.ca),
                    "durable_root": str(args.durable_root),
                    "gpu_reservation_per_job": 1,
                    "maximum_in_flight": 4,
                    "initial_qualification_concurrency": 1,
                    "parallel_capacity": "UNVERIFIED; ceiling is not capacity approval",
                    "worker_prerequisites": {
                        "native_assets_licences": "UNVERIFIED; prepared worker profile required",
                        "graphics": "UNVERIFIED; submitter graphics do not qualify workers",
                        "display_private_route": "UNVERIFIED; prepared display/access required",
                        "browser_private_route": "UNVERIFIED; remote browser/access required",
                        "ros_world_controllers_reset_clock": "UNVERIFIED; ROS profile required",
                        "private_data": (
                            "SUPPLIED_UNVERIFIED; approval and worker admission still required"
                            if args.data_root
                            else "MISSING; no approved remote data root supplied"
                        ),
                    },
                    "boundaries": (
                        "No provisioning, image/driver/shared-environment changes, "
                        "DGX, separate CPU pools or local application/GPU fallback"
                    ),
                },
                "attempts": [
                    {key: value for key, value in item.items() if key != "scenario"}
                    for item in attempts
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.submit:
        return
    from ray.job_submission import JobSubmissionClient

    os.environ["RAY_API_SERVER_ADDRESS"] = args.address
    client = _bounded_rpc(JobSubmissionClient, args.address, verify=str(args.ca))
    result = run_campaign(
        attempts,
        client,
        args.output,
        concurrency=args.concurrency,
        queue_seconds=args.queue_seconds,
    )
    if any(row["status"] != "passed" for row in result["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
