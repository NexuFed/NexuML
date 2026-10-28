"""Hermetic failure receipt checks; no actual CLI, MLflow or browser."""

import json
import contextlib
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest import mock
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_logging_failure.py"


@pytest.fixture
def delivery(tmp_path):
    """Create inert file/metadata doubles without a logger or decoder runtime.

    Returns:
        Inert delivery files, metadata doubles and the observation callable.
    """
    functions = runpy.run_path(str(SCRIPT))
    root = tmp_path / "partial"
    root.mkdir()
    run_id, experiment_id = "a" * 32, "1"
    artifacts = root / "artifact-root" / run_id / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "evaluation").write_bytes(functions["BLOCKER"])
    tb = root / "logs/tb/experiment/eval_v0"
    (tb / "artifacts/evaluation").mkdir(parents=True)
    gif = tb / "artifacts/evaluation/frame_000000000064.gif"
    gif.write_bytes(b"mock GIF; native decoding deliberately mocked")
    expected_gif = functions["_sha256"](gif)
    for name in ("source_package.json", "config.yaml"):
        (artifacts / name).write_text("retained " + name)
        (tb / "artifacts" / name).write_text("retained " + name)
    (root / "staging").mkdir()
    log = tmp_path / "original-cli.log"
    log.write_text(
        f"Artifact delivery failed MLFlowLogger {run_id}; "
        "already delivered TensorBoardLogger; Evaluation failed"
    )
    stage = {
        "name": "logging-partial-delivery-evaluate",
        "status": "FAIL",
        "exit_code": 1,
        "log_path": str(log),
        "log_sha256": functions["_sha256"](log),
    }
    record = {"stages": [stage], "checks": {}, "gates": {key: "UNVERIFIED" for key in "CVLRD"}}
    run = SimpleNamespace(
        info=SimpleNamespace(
            status="FAILED", experiment_id=experiment_id, artifact_uri=artifacts.as_uri()
        )
    )
    entries = [
        SimpleNamespace(path=name, is_dir=False)
        for name in ("source_package.json", "config.yaml", "evaluation")
    ]
    client = SimpleNamespace(
        get_run=mock.Mock(return_value=run), list_artifacts=mock.Mock(return_value=entries)
    )
    runs = [{"log_dir": str(tb), "previews": []}]
    decoded = {"changing_nonuniform_frames": True}

    def inspect():
        return functions["_inspect_failure"](
            record,
            root,
            client,
            run_id,
            experiment_id,
            expected_gif,
            lambda _: runs,
            lambda _: decoded,
        )

    return SimpleNamespace(**locals())


def test_partial_receipt_preserves_failed_status_and_prior_hashes(delivery):
    """Valid observation never promotes the failed CLI or overall gates."""
    value = delivery.inspect()
    assert value["status"] == "FAILED"
    assert value["cli"]["exit_code"] == 1 and value["cli"]["status"] == "FAIL"
    assert [row["filename"] for row in value["prior_artifacts"]] == [
        "source_package.json",
        "config.yaml",
    ]
    assert delivery.record["stages"][0]["status"] == "FAIL"
    assert set(delivery.record["gates"].values()) == {"UNVERIFIED"}
    assert delivery.client.get_run.call_count == 1
    assert delivery.client.list_artifacts.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        "cli-success",
        "wrong-gif",
        "wrong-stage",
        "missing-backend",
        "changed-log",
        "running",
        "wrong-run-root",
        "wrong-experiment",
        "extra-artifact",
        "directory",
        "preview",
        "different-prior",
        "extra-media",
        "static-gif",
        "staging",
        "blocker",
    ],
)
def test_missing_partial_failure_evidence_is_rejected(delivery, change):
    """Missing native/CLI/retention evidence must fail observation."""
    if change == "cli-success":
        delivery.stage["exit_code"] = 0
    if change == "wrong-gif":
        delivery.gif.write_bytes(b"different recorded bytes")
    if change == "wrong-stage":
        delivery.stage["name"] = "logging-disabled-fresh-evaluate"
    if change == "missing-backend":
        delivery.log.write_text("Evaluation failed without destination identity")
        delivery.stage["log_sha256"] = delivery.functions["_sha256"](delivery.log)
    if change == "changed-log":
        delivery.log.write_text(delivery.log.read_text() + "changed")
    if change == "running":
        delivery.run.info.status = "RUNNING"
    if change == "wrong-run-root":
        delivery.run.info.artifact_uri = "file:///foreign/artifacts"
    if change == "wrong-experiment":
        delivery.run.info.experiment_id = "2"
    if change == "extra-artifact":
        delivery.entries.append(
            SimpleNamespace(path="evaluation/frame_000000000064.gif", is_dir=False)
        )
    if change == "directory":
        delivery.entries[2].is_dir = True
    if change == "preview":
        delivery.runs[0]["previews"] = [{"step": 64}]
    if change == "different-prior":
        (delivery.tb / "artifacts/config.yaml").write_text("different")
    if change == "extra-media":
        (delivery.gif.parent / "frame_000000000064.json").write_text("{}")
    if change == "static-gif":
        delivery.decoded["changing_nonuniform_frames"] = False
    if change == "staging":
        (delivery.root / "staging/leftover.mp4").write_bytes(b"partial")
    if change == "blocker":
        (delivery.artifacts / "evaluation").write_text("replaced")
    with pytest.raises(AssertionError):
        delivery.inspect()


@pytest.mark.parametrize("observer_error", [False, True, "diagnostic-write-fails"])
def test_real_stage_exception_is_reraised_even_after_observing_expected_failure(
    tmp_path, monkeypatch, observer_error
):
    """Success or failure of observation cannot replace the original stage failure."""
    functions = runpy.run_path(str(SCRIPT))
    execute = functions["run_logging_failure"]
    globals_ = execute.__globals__
    root, source = tmp_path / "attempt", tmp_path / "producer"
    root.mkdir()
    source.mkdir()
    for mode, name in (("disabled", "scalar-case.yaml"), ("both", "eval-case.yaml")):
        (source / mode).mkdir()
        (source / mode / name).write_text("mock config")
    policy = str(source / "both/checkpoint-policy")
    payload = {
        "modes": {
            "both": {
                "policies": {"final": {"path": policy, "source_frames": 64}},
                "native": {
                    "tensorboard": {
                        "fresh": {
                            "final": {"media": {"64": {"files": {"gif": {"sha256": "b" * 64}}}}}
                        }
                    }
                },
            },
            "disabled": {"final_scalar_metrics": {"eval/rl/return_mean": 9.5}},
        }
    }
    manifest = source / "logging-manifest.json"
    manifest.write_text(json.dumps(payload))
    case = source / "case.json"
    case.write_text(
        json.dumps(
            {
                "case": "logging",
                "checks": {"logger_payloads": "PASS"},
                "logging": {"manifest_sha256": functions["_sha256"](manifest)},
            }
        )
    )
    run_id = "a" * 32
    native = SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            status="RUNNING",
            artifact_uri=(root / "partial/artifact-root" / run_id / "artifacts").as_uri(),
        )
    )
    client = SimpleNamespace(
        create_experiment=mock.Mock(return_value="1"), create_run=mock.Mock(return_value=native)
    )
    mlflow = SimpleNamespace(MlflowClient=mock.Mock(return_value=client))
    config = SimpleNamespace(logging=None, save=lambda path: path.write_text("owned config"))
    monkeypatch.setitem(sys.modules, "mlflow.tracking", mlflow)
    monkeypatch.setitem(
        sys.modules,
        "nexuml.core.config",
        SimpleNamespace(ResolvedConfig=SimpleNamespace(load=lambda _: config)),
    )
    monkeypatch.setitem(
        sys.modules,
        "nexuml.core.types",
        SimpleNamespace(LoggingSpec=SimpleNamespace(model_validate=lambda value: value)),
    )
    monkeypatch.setattr(
        globals_["runpy"], "run_path", lambda _: {"_tensorboard_runs": None, "_gif_info": None}
    )
    proof = {"run_id": run_id}
    monkeypatch.setitem(globals_, "_inspect_failure", lambda *args: proof)

    def observe(*args):
        if observer_error:
            raise RuntimeError("native observation unavailable")
        args[0]["checks"]["partial_delivery_native_failed_ui"] = "PASS"

    monkeypatch.setitem(globals_, "_observe_failed_ui", observe)
    if observer_error == "diagnostic-write-fails":
        original_open = Path.open

        def open_path(path, *args, **kwargs):
            if path == root / "partial-delivery-observation-error.txt":
                raise OSError("evidence disk full")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_path)
    original = RuntimeError("original native evaluation stage failed")
    record = {"stages": [], "checks": {}, "gates": {key: "UNVERIFIED" for key in "CVLRD"}}

    def stage(record, name, argv, env):
        if name == "logging-partial-delivery-evaluate":
            record["stages"].append({"name": name, "exit_code": 1, "status": "FAIL"})
            raise original
        if name == "sandboxed-browser-cancellation":
            directory = root / "browser-cancellation"
            directory.mkdir()
            (directory / "proof.json").write_text(
                json.dumps({"cleanup": "PASS", "V_gate": "UNVERIFIED"})
            )
        return {"log_path": str(root / "disabled.log")}

    options = {"source_attempt": str(source), "source_case_sha256": functions["_sha256"](case)}
    with pytest.raises(RuntimeError) as caught:
        execute(
            record,
            root,
            options,
            {},
            stage,
            mock.Mock(),
            lambda *_args, **_kwargs: {"eval/rl/return_mean": 9.5},
            lambda: set(),
        )
    assert caught.value is original
    assert record["stages"][0]["exit_code"] == 1 and record["stages"][0]["status"] == "FAIL"
    assert set(record["gates"].values()) == {"UNVERIFIED"}
    assert not hasattr(client, "set_terminated")
    if observer_error:
        assert any("observation unavailable" in note for note in original.__notes__)
        assert "partial_delivery_native_failed_ui" not in record["checks"]
    if observer_error == "diagnostic-write-fails":
        assert any("evidence disk full" in note for note in original.__notes__)


@pytest.mark.parametrize(
    "wrong", [None, "run_id", "experiment_id", "status", "cli", "tensorboard_gif"]
)
def test_failed_run_ui_uses_private_store_and_cleans_owned_server(tmp_path, monkeypatch, wrong):
    """Only matching failed-run proof passes; every path cleans only owned resources."""
    functions = runpy.run_path(str(SCRIPT))
    observe = functions["_observe_failed_ui"]
    globals_ = observe.__globals__
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "partial/logs").mkdir(parents=True)
    (root / "partial/logs/mlflow.db").write_bytes(b"inert failed native store")
    (root / "browser-setup.json").write_text(
        json.dumps(
            {
                "status": "success",
                "paths": {"node": "/private/node", "libraryPrefix": "/private/lib"},
            }
        )
    )
    expected = {
        "run_id": "a" * 32,
        "experiment_id": "1",
        "cli": {"exit_code": 1, "status": "FAIL"},
        "tensorboard_gif": {"sha256": "b" * 64},
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(expected))
    listener = {"pid": 123, "start_ticks": 456, "socket_inode": "789"}
    signal_owned = mock.Mock()
    helpers = {
        "_wait_server": mock.Mock(return_value=listener),
        "_owned_listener": lambda *_: contextlib.nullcontext(listener),
        "_signal_members": signal_owned,
    }
    monkeypatch.setattr(globals_["runpy"], "run_path", lambda _: helpers)
    process = mock.Mock(pid=123, returncode=0)
    launch = mock.Mock(return_value=process)
    monkeypatch.setattr(globals_["subprocess"], "Popen", launch)
    reserve = mock.MagicMock()
    monkeypatch.setattr(globals_["socket"], "socket", mock.Mock(return_value=reserve))
    record = {"partial_delivery": {}, "checks": {}, "gates": {key: "UNVERIFIED" for key in "CVLRD"}}

    def stage(record, name, argv, env):
        assert name == "partial-delivery-native-failed-ui"
        assert argv[-1] == "--partial-delivery" and argv[0] == "/private/node"
        assert env["LD_LIBRARY_PATH"] == "/private/lib" and env["CUDA_VISIBLE_DEVICES"] == ""
        actual = {**expected, "status": {"actual": "Failed"}}
        if wrong is not None:
            actual[wrong] = {"actual": "Running"} if wrong == "status" else None
        (root / "failure-ui/probe.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "claim": "UI_OPENED_ONLY",
                    "V_gate": "UNVERIFIED",
                    "mlflow_failure": actual,
                }
            )
        )

    if wrong:
        with pytest.raises(AssertionError, match="observation incomplete"):
            observe(record, root, {}, stage, lambda: {999}, manifest)
        assert "partial_delivery_native_failed_ui" not in record["checks"]
    else:
        observe(record, root, {}, stage, lambda: {999}, manifest)
        assert record["checks"]["partial_delivery_native_failed_ui"] == "PASS"
    assert record["partial_delivery"]["browser"]["cleanup"] == "PASS"
    assert (root / "failed-ui.db").read_bytes() == b"inert failed native store"
    argv = launch.call_args.args[0]
    assert (
        "tensorboard" not in argv
        and argv[argv.index("--backend-store-uri") + 1] == f"sqlite:///{root / 'failed-ui.db'}"
    )
    assert set(record["gates"].values()) == {"UNVERIFIED"}
    assert all(call.args[0] == set() for call in signal_owned.call_args_list)
    process.wait.assert_called_once_with(timeout=5)
