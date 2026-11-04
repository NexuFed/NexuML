"""Fixed remote partial-delivery experiment; original CLI/job failure is preserved."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time


EXPERIMENT = "physical-ai-logging-partial-delivery"
BLOCKER = b"intentional owned qualification blocker; not evaluation media\n"


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def _inspect_failure(record, owned, client, run_id, experiment_id, expected_gif, tb_runs, gif_info):
    stage = record["stages"][-1]
    if (
        stage["name"] != "logging-partial-delivery-evaluate"
        or stage["status"] != "FAIL"
        or stage["exit_code"] != 1
    ):
        raise AssertionError("original evaluation stage must remain FAIL/exit1")
    text = Path(stage["log_path"]).read_text()
    for token in (
        "Artifact delivery failed",
        "MLFlowLogger",
        run_id,
        "TensorBoardLogger",
        "already delivered",
        "Evaluation failed",
    ):
        if token not in text:
            raise AssertionError(f"partial-delivery CLI diagnostic missing: {token}")
    if _sha256(stage["log_path"]) != stage["log_sha256"]:
        raise AssertionError("original failed CLI log changed")
    run = client.get_run(run_id)
    artifacts = owned / "artifact-root" / run_id / "artifacts"
    if (
        run.info.status != "FAILED"
        or run.info.experiment_id != experiment_id
        or run.info.artifact_uri != artifacts.as_uri()
    ):
        raise AssertionError("native CLI finalization must leave the same run FAILED")
    if (artifacts / "evaluation").read_bytes() != BLOCKER or not (
        artifacts / "evaluation"
    ).is_file():
        raise AssertionError("owned failed destination was replaced")
    entries = client.list_artifacts(run_id)
    if {entry.path for entry in entries} != {
        "source_package.json",
        "config.yaml",
        "evaluation",
    } or any(entry.is_dir for entry in entries):
        raise AssertionError(
            "failed MLflow run must retain prior files, not completed rollout media"
        )
    runs = tb_runs(owned)
    if len(runs) != 1 or runs[0]["previews"]:
        raise AssertionError("one TensorBoard writer without an unpublished preview required")
    tb = Path(runs[0]["log_dir"]) / "artifacts"
    gif = tb / "evaluation/frame_000000000064.gif"
    if (
        sorted(path.name for path in gif.parent.iterdir()) != [gif.name]
        or _sha256(gif) != expected_gif
    ):
        raise AssertionError("only the same-policy GIF may have reached TensorBoard before failure")
    decoded = gif_info(gif)
    if not decoded["changing_nonuniform_frames"]:
        raise AssertionError("retained partial GIF must be genuinely animated")
    prior = []
    for name in ("source_package.json", "config.yaml"):
        if (tb / name).read_bytes() != (artifacts / name).read_bytes():
            raise AssertionError("successful prior artifact delivery differs across backends")
        prior.append({"filename": name, "sha256": _sha256(tb / name)})
    if any((owned / "staging").iterdir()):
        raise AssertionError(
            "failed media staging must be cleaned without deleting delivered outputs"
        )
    record.setdefault("partial_delivery", {})["gif_decoded"] = decoded
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "status": run.info.status,
        "experiment_name": EXPERIMENT,
        "cli": {
            "exit_code": stage["exit_code"],
            "status": stage["status"],
            "log_path": stage["log_path"],
            "log_sha256": stage["log_sha256"],
        },
        "tensorboard_gif": {"path": str(gif), "sha256": expected_gif},
        "prior_artifacts": prior,
    }


def _observe_failed_ui(record, root, env, stage, group_members, manifest):
    browser = runpy.run_path(str(Path(__file__).with_name("physical_ai_browser.py")))
    setup = json.loads((root / "browser-setup.json").read_text())
    if setup["status"] != "success":
        raise ValueError("successful private browser setup required")
    paths = setup["paths"]
    ui = root / "failure-ui"
    ui.mkdir(mode=0o700)
    original_db = root / "partial/logs/mlflow.db"
    database = root / "failed-ui.db"
    shutil.copyfile(original_db, database)
    if _sha256(database) != _sha256(original_db):
        raise AssertionError("private failed-run UI store differs")
    server_env = {
        **env,
        "CUDA_VISIBLE_DEVICES": "",
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "LD_LIBRARY_PATH": paths["libraryPrefix"]
        + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
    }
    with socket.socket() as reservation:
        reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reservation.bind(("127.0.0.1", 5001))
    baseline = group_members()
    state = {"cleanup": "UNVERIFIED", "scope": "native failed-run UI only; no TensorBoard service"}
    record["partial_delivery"]["browser"] = state
    primary = None
    with (root / "failed-mlflow-ui.log").open("xb") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--backend-store-uri",
                f"sqlite:///{database}",
                "--host",
                "127.0.0.1",
                "--port",
                "5001",
                "--workers",
                "1",
            ],
            env=server_env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            state["listener"] = browser["_wait_server"](process, 5001)
            stage(
                record,
                "partial-delivery-native-failed-ui",
                [
                    paths["node"],
                    str(Path(__file__).with_name("physical_ai_logger_browser.cjs")),
                    str(ui),
                    str(manifest),
                    "--partial-delivery",
                ],
                server_env,
            )
            with browser["_owned_listener"](process, 5001) as listener:
                if listener != state["listener"]:
                    raise AssertionError("failed-run UI listener changed")
            proof = json.loads((ui / "probe.json").read_text())
            expected = json.loads(manifest.read_text())
            observed = proof.get("mlflow_failure", {})
            if (
                proof["status"] != "success"
                or proof.get("claim") != "UI_OPENED_ONLY"
                or proof["V_gate"] != "UNVERIFIED"
                or observed.get("run_id") != expected["run_id"]
                or observed.get("experiment_id") != expected["experiment_id"]
                or observed.get("status", {}).get("actual") != "Failed"
                or observed.get("cli") != expected["cli"]
                or observed.get("tensorboard_gif") != expected["tensorboard_gif"]
            ):
                raise AssertionError("native failed-run observation incomplete")
            record["checks"]["partial_delivery_native_failed_ui"] = "PASS"
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                browser["_signal_members"](group_members() - baseline, signal.SIGTERM)
                deadline = time.monotonic() + 15
                while group_members() - baseline and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(0.1)
                survivors = group_members() - baseline
                browser["_signal_members"](survivors, signal.SIGKILL)
                process.wait(timeout=5)
                if survivors:
                    raise RuntimeError("failed-run UI needed forced cleanup")
                with socket.socket() as reservation:
                    reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    reservation.bind(("127.0.0.1", 5001))
                state["cleanup"] = "PASS"
            except BaseException as error:
                state["cleanup"] = "FAIL"
                if primary is None:
                    raise
                primary.add_note(f"Failed-run UI cleanup also failed: {error}")


def run_logging_failure(
    record, root, options, env, stage, recover, evaluation_metrics, group_members
):
    """Observe real delivery failure, then re-raise the original nonzero CLI stage.

    Raises:
        AssertionError: Source identities or required real failure evidence differ.
    """
    recover(record, root, options, env)
    source = Path(options["source_attempt"])
    case_data = (source / "case.json").read_bytes()
    if hashlib.sha256(case_data).hexdigest() != options["source_case_sha256"]:
        raise AssertionError("original producer receipt changed after admission")
    previous = json.loads(case_data)
    data = (source / "logging-manifest.json").read_bytes()
    if (
        previous["case"] != "logging"
        or previous["checks"].get("logger_payloads") != "PASS"
        or hashlib.sha256(data).hexdigest() != previous["logging"]["manifest_sha256"]
    ):
        raise AssertionError("qualified logger producer required")
    payload = json.loads(data)
    source_manifest = root / "source-logging-manifest.json"
    source_manifest.write_bytes(data)
    policy = source / "both/checkpoint-policy"
    if (
        payload["modes"]["both"]["policies"]["final"]["path"] != str(policy)
        or payload["modes"]["both"]["policies"]["final"]["source_frames"] != 64
    ):
        raise AssertionError("fixed original final policy path differs")
    disabled = root / "disabled"
    disabled.mkdir(mode=0o700)
    disabled_config = disabled / "scalar-case.yaml"
    disabled_config.write_bytes((source / "disabled/scalar-case.yaml").read_bytes())
    mode_env = {
        **env,
        "CUDA_VISIBLE_DEVICES": "",
        "SDL_VIDEODRIVER": "dummy",
        "NEXUML_LOGS_ROOT": str(disabled / "logs"),
    }
    cli = [sys.executable, "-m", "nexuml.cli.main", "evaluate", str(policy)]
    result = stage(
        record,
        "logging-disabled-fresh-evaluate",
        [*cli, "--config", str(disabled_config), "--device", "cpu"],
        mode_env,
    )
    metrics = evaluation_metrics(Path(result["log_path"]), expected_frames=64)
    if (
        metrics != payload["modes"]["disabled"]["final_scalar_metrics"]
        or (disabled / "logs").exists()
    ):
        raise AssertionError("disabled evaluation changed metrics or constructed logging outputs")
    record["checks"]["disabled_logging_fresh_evaluation"] = "PASS"

    from mlflow.tracking import MlflowClient
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.types import LoggingSpec

    owned = root / "partial"
    owned.mkdir(mode=0o700)
    (owned / "logs").mkdir()
    client = MlflowClient(tracking_uri=f"sqlite:///{owned / 'logs/mlflow.db'}")
    artifact_root = owned / "artifact-root"
    experiment_id = client.create_experiment(EXPERIMENT, artifact_location=artifact_root.as_uri())
    run = client.create_run(experiment_id, run_name="cartpole-ppo-partial-delivery")
    artifacts = artifact_root / run.info.run_id / "artifacts"
    if run.info.artifact_uri != artifacts.as_uri() or run.info.status != "RUNNING":
        raise AssertionError("fresh native run must target its exact owned artifact root")
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "evaluation").open("xb") as blocker:
        blocker.write(BLOCKER)
    config = ResolvedConfig.load(source / "both/eval-case.yaml")
    config.logging = LoggingSpec.model_validate(
        {
            "tensorboard": {"log_dir": str(owned / "logs/tb")},
            "mlflow": {
                "tracking_uri": f"sqlite:///{owned / 'logs/mlflow.db'}",
                "artifact_location": artifact_root.as_uri(),
                "experiment_name": EXPERIMENT,
                "run_id": run.info.run_id,
                "auto_nested_runs": False,
                "log_model": False,
                "synchronous": True,
            },
            "experiment_name": EXPERIMENT,
            "run_name": "cartpole-ppo-partial-delivery",
            "log_system_metrics": False,
            "temp_artifact_dir": str(owned / "staging"),
        }
    )
    config_path = owned / "eval-case.yaml"
    config.save(config_path)
    mode_env["NEXUML_LOGS_ROOT"] = str(owned / "logs")
    record["partial_delivery"] = {
        "run_id": run.info.run_id,
        "experiment_id": experiment_id,
        "scope": "intentional owned evaluation-path obstruction; original CLI/job failure retained",
    }
    helpers = runpy.run_path(str(Path(__file__).with_name("physical_ai_logging.py")))
    try:
        stage(
            record,
            "logging-partial-delivery-evaluate",
            [*cli, "--config", str(config_path), "--device", "cpu"],
            mode_env,
        )
    except Exception as original:
        try:
            expected = payload["modes"]["both"]["native"]["tensorboard"]["fresh"]["final"]["media"][
                "64"
            ]["files"]["gif"]["sha256"]
            proof = _inspect_failure(
                record,
                owned,
                client,
                run.info.run_id,
                experiment_id,
                expected,
                helpers["_tensorboard_runs"],
                helpers["_gif_info"],
            )
            manifest = root / "partial-delivery.json"
            _write_json(manifest, proof)
            record["partial_delivery"]["manifest_sha256"] = _sha256(manifest)
            record["checks"]["partial_delivery_preserved_outputs"] = "PASS"
            stage(
                record,
                "sandboxed-browser-cancellation",
                [
                    sys.executable,
                    str(Path(__file__).with_name("physical_ai_browser_cancel.py")),
                    str(root / "browser-cancellation"),
                    str(source_manifest),
                ],
                env,
            )
            cancellation = json.loads((root / "browser-cancellation/proof.json").read_text())
            if cancellation["cleanup"] != "PASS" or cancellation["V_gate"] != "UNVERIFIED":
                raise AssertionError("failed-run browser cancellation proof absent")
            _observe_failed_ui(record, root, env, stage, group_members, manifest)
            recover(record, root, options, env)
        except Exception as observation:
            original.add_note(
                f"Failure observation also failed: {type(observation).__name__}: {observation}"
            )
            try:
                with (root / "partial-delivery-observation-error.txt").open("x") as diagnostic:
                    diagnostic.write(str(observation))
            except OSError as diagnostic_error:
                original.add_note(f"Failure diagnostic could not be saved: {diagnostic_error}")
        raise
    raise AssertionError("intentional native artifact destination failure did not fail the CLI")
