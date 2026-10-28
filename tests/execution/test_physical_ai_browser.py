"""Hermetic UI orchestration only: no servers, browsers, native runtime or models."""

import hashlib
import contextlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from typing import Any


SCRIPT = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_browser.py"


class BrowserControlTest(unittest.TestCase):
    """Verify source admission, native commands and failure cleanup with mocks."""

    def setUp(self):
        self.run_browser = runpy.run_path(str(SCRIPT))["run_browser"]
        self.g = self.run_browser.__globals__
        scratch = tempfile.TemporaryDirectory(prefix="browser-control-")
        self.addCleanup(scratch.cleanup)
        self.base = Path(scratch.name)
        self.source, self.root = self.base / "producer", self.base / "attempt"
        self.source.mkdir()
        self.root.mkdir()
        database = self.source / "mlflow-only/logs/mlflow.db"
        database.parent.mkdir(parents=True)
        database.write_bytes(b"inert database bytes")
        both_database = self.source / "both/logs/mlflow.db"
        both_database.parent.mkdir(parents=True)
        both_database.write_bytes(b"inert both database bytes")
        manifest: dict[str, dict[str, dict[str, Any]]] = {
            "modes": {
                "tb-only": {
                    "ui_inputs": {"tensorboard_log_root": str(self.source / "tb-only/logs/tb")}
                },
                "mlflow-only": {"ui_inputs": {"mlflow_tracking_uri": f"sqlite:///{database}"}},
                "both": {
                    "ui_inputs": {
                        "tensorboard_log_root": str(self.source / "both/logs/tb"),
                        "mlflow_tracking_uri": f"sqlite:///{both_database}",
                    }
                },
            }
        }
        for mode in ("mlflow-only", "both"):
            manifest["modes"][mode].update(
                logging={"experiment_name": f"physical-ai-logging-{mode}"},
                configs={"case.yaml": {"path": f"/producer/{mode}/case.yaml", "sha256": "d" * 64}},
                native={
                    "mlflow": {
                        "main": {
                            "run_id": ("a" if mode == "mlflow-only" else "b") * 32,
                            "experiment_id": "1",
                            "scalar_history": {"eval/rl/return_mean": [{"step": 64, "value": 9.5}]},
                        }
                    }
                },
            )
        (self.source / "logging-manifest.json").write_text(json.dumps(manifest))
        receipt = {
            "case": "logging",
            "checks": {"logger_payloads": "PASS"},
            "source_tree_sha256": "producer-tree",
            "logging": {
                "manifest_sha256": hashlib.sha256(
                    (self.source / "logging-manifest.json").read_bytes()
                ).hexdigest()
            },
        }
        (self.source / "case.json").write_text(json.dumps(receipt))
        self.options = {
            "source_attempt": str(self.source),
            "source_case_sha256": hashlib.sha256(
                (self.source / "case.json").read_bytes()
            ).hexdigest(),
        }
        (self.root / "browser-setup.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "paths": {
                        "node": "/private/node",
                        "libraryPrefix": "/private/browser-clients:/private/native",
                    },
                }
            )
        )
        self.record = {"checks": {}, "gates": {key: "UNVERIFIED" for key in "CVLRD"}}
        self.recover = mock.Mock()
        self.stage = mock.Mock(side_effect=self.probe)
        self.processes = []

        def launch(argv, **kwargs):
            process = mock.Mock(pid=1000 + len(self.processes), returncode=0)
            self.processes.append(process)
            return process

        self.popen = mock.patch.object(self.g["subprocess"], "Popen", side_effect=launch)
        self.launch = self.popen.start()
        self.addCleanup(self.popen.stop)
        self.socket = mock.patch.object(self.g["socket"], "socket")
        self.reservation = self.socket.start()
        self.addCleanup(self.socket.stop)
        listener = {"pid": 1001, "start_ticks": 123, "socket_inode": "456"}
        self.ready = mock.patch.dict(
            self.g,
            _wait_server=mock.Mock(return_value=listener),
            _owned_listener=mock.Mock(side_effect=lambda *args: contextlib.nullcontext(listener)),
            _signal_members=mock.Mock(),
        )
        self.ready.start()
        self.addCleanup(self.ready.stop)

    def probe(self, record, name, argv, env):
        if name == "sandboxed-browser-cancellation":
            directory = self.root / "browser-cancellation"
            directory.mkdir()
            (directory / "proof.json").write_text(
                json.dumps({"cleanup": "PASS", "V_gate": "UNVERIFIED"})
            )
            return
        self.assertEqual(name, "logger-ui-reconnaissance")
        self.assertEqual(argv[0], "/private/node")
        self.assertEqual(env["MLFLOW_DISABLE_TELEMETRY"], "true")
        mode = self.options.get("logger_mode", "separate")
        tb_mode = "both" if mode == "both" else "tb-only"
        ml_mode = "both" if mode == "both" else "mlflow-only"
        source = json.loads((self.source / "logging-manifest.json").read_text())["modes"][ml_mode]
        self.assertEqual("--both-loggers" in argv, mode == "both")
        (self.root / "ui/probe.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "claim": "UI_OPENED_ONLY",
                    "V_gate": "UNVERIFIED",
                    "logger_mode": mode,
                    "preview_selection": "PASS",
                    "scalar_inspection": "PASS",
                    "overview_inspection": "PASS",
                    "mlflow_overview": {
                        "mode": ml_mode,
                        "run_id": source["native"]["mlflow"]["main"]["run_id"],
                        "experiment_id": "1",
                        "source_frames": 64,
                        "source_config": source["configs"]["case.yaml"],
                        "status": {"actual": "Finished"},
                        "rows": [
                            {"key": key, "actual": value}
                            for key, value in [
                                ("train/rl/frames", "64"),
                                ("eval/rl/return_mean", "9.5"),
                                ("scenario/training/total_frames", "64"),
                                ("scenario/interaction/environment/params/env_name", "CartPole-v1"),
                                (
                                    "scenario/logging/experiment_name",
                                    f"physical-ai-logging-{ml_mode}",
                                ),
                            ]
                        ],
                    },
                    "tensorboard_scalars": {
                        "mode": tb_mode,
                        "metrics": [{"tag": "train/rl/frames"}, {"tag": "eval/rl/return_mean"}],
                    },
                    "mlflow_artifacts": {
                        "mode": ml_mode,
                        "source_frames": 64,
                        "artifacts": [
                            {"filename": f"frame_000000000064.{ext}"}
                            for ext in ("mp4", "gif", "json")
                        ],
                        "mp4": {"ended": True},
                        "gif": {"samples": [{}, {}]},
                    },
                    "tensorboard_previews": {
                        "mode": tb_mode,
                        "stages": [{"step": step} for step in (0, 32, 64)],
                        "reopened": {"step": 0, "reopened": True},
                    },
                }
            )
        )

    def execute(self, members=lambda: {1}):
        return self.run_browser(
            self.record, self.root, self.options, {}, self.stage, self.recover, members
        )

    def test_only_native_loopback_servers_and_cloned_database(self):
        self.execute()
        self.recover.assert_called_once()
        self.assertEqual(self.launch.call_count, 2)
        for call in self.launch.call_args_list:
            (argv,) = call.args
            self.assertIn("127.0.0.1", argv)
            self.assertEqual(call.kwargs["stdin"], subprocess.DEVNULL)
            self.assertNotIn("start_new_session", call.kwargs)
        mlflow = self.launch.call_args_list[1].args[0]
        self.assertIn(f"sqlite:///{self.root / 'mlflow-ui.db'}", mlflow)
        self.assertNotIn(f"sqlite:///{self.source / 'mlflow-only/logs/mlflow.db'}", mlflow)
        self.assertEqual((self.root / "mlflow-ui.db").read_bytes(), b"inert database bytes")
        self.assertEqual(self.record["browser"]["cleanup"], "PASS")
        self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})
        self.assertEqual(self.record["checks"]["logger_ui_reconnaissance"], "PASS")
        self.assertEqual(self.record["checks"]["tensorboard_native_previews"], "PASS")
        self.assertEqual(self.record["checks"]["tensorboard_scalar_history"], "PASS")
        self.assertEqual(self.record["checks"]["mlflow_native_artifacts"], "PASS")
        self.assertEqual(self.record["checks"]["mlflow_native_overview"], "PASS")

    def test_incomplete_or_cross_mode_overview_proof_never_passes(self):
        original = self.stage.side_effect
        replacements = {
            "mode": "both",
            "run_id": "b" * 32,
            "experiment_id": "99",
            "source_frames": 63,
            "source_config": {},
            "status": {"actual": "Running"},
            "rows": [],
        }
        for index, (field, replacement) in enumerate(replacements.items()):
            with self.subTest(field=field):
                self.root = self.base / f"overview-failure-{index}"
                self.root.mkdir()
                (self.root / "browser-setup.json").write_bytes(
                    (self.base / "attempt/browser-setup.json").read_bytes()
                )
                self.record = {"checks": {}, "gates": {key: "UNVERIFIED" for key in "CVLRD"}}

                def probe(*args):
                    original(*args)
                    if args[1] == "logger-ui-reconnaissance":
                        file = self.root / "ui/probe.json"
                        value = json.loads(file.read_text())
                        value["mlflow_overview"][field] = replacement
                        file.write_text(json.dumps(value))

                self.stage.side_effect = probe
                with self.assertRaisesRegex(AssertionError, "Overview status/metrics/config proof"):
                    self.execute()
                self.assertNotIn("mlflow_native_overview", self.record["checks"])
                self.assertEqual(self.record["browser"]["cleanup"], "PASS")
                self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})

    def test_both_mode_uses_only_its_own_stores_and_explicit_probe_flag(self):
        self.options["logger_mode"] = "both"
        self.execute()
        self.assertEqual(self.record["browser"]["logger_mode"], "both")
        tb_argv = self.launch.call_args_list[0].args[0]
        self.assertEqual(tb_argv[tb_argv.index("--logdir") + 1], str(self.source / "both/logs/tb"))
        self.assertEqual((self.root / "mlflow-ui.db").read_bytes(), b"inert both database bytes")
        self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})

    def test_unsupported_mode_fails_before_recovery_or_servers(self):
        self.options["logger_mode"] = "disabled"
        with self.assertRaisesRegex(ValueError, "logger mode"):
            self.execute()
        self.recover.assert_not_called()
        self.launch.assert_not_called()
        self.stage.assert_not_called()

    def test_both_mode_cannot_accept_a_separate_mode_probe(self):
        self.options["logger_mode"] = "both"
        original = self.stage.side_effect

        def probe(*args):
            original(*args)
            if args[1] == "logger-ui-reconnaissance":
                file = self.root / "ui/probe.json"
                value = json.loads(file.read_text())
                value["logger_mode"] = "separate"
                file.write_text(json.dumps(value))

        self.stage.side_effect = probe
        with self.assertRaisesRegex(AssertionError, "status/claim"):
            self.execute()
        self.assertNotIn("tensorboard_native_previews", self.record["checks"])
        self.assertEqual(self.record["browser"]["cleanup"], "PASS")

    def test_both_mode_cannot_accept_single_logger_media_proof(self):
        self.options["logger_mode"] = "both"
        original = self.stage.side_effect

        def probe(*args):
            original(*args)
            if args[1] == "logger-ui-reconnaissance":
                file = self.root / "ui/probe.json"
                value = json.loads(file.read_text())
                value["mlflow_artifacts"]["mode"] = "mlflow-only"
                file.write_text(json.dumps(value))

        self.stage.side_effect = probe
        with self.assertRaisesRegex(AssertionError, "MLflow artifact retrieval/playback"):
            self.execute()
        self.assertNotIn("mlflow_native_artifacts", self.record["checks"])
        self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})

    def test_native_mlflow_playback_is_required_without_promoting_visual_gate(self):
        original = self.stage.side_effect

        def probe(*args):
            original(*args)
            if args[1] == "logger-ui-reconnaissance":
                destination = self.root / "ui/probe.json"
                value = json.loads(destination.read_text())
                value["mlflow_artifacts"]["mp4"]["ended"] = False
                destination.write_text(json.dumps(value))

        self.stage.side_effect = probe
        with self.assertRaisesRegex(AssertionError, "MLflow artifact retrieval/playback proof"):
            self.execute()
        self.assertNotIn("mlflow_native_artifacts", self.record["checks"])
        self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})

    def test_native_csv_history_is_required_without_promoting_visual_gate(self):
        original = self.stage.side_effect

        def probe(*args):
            original(*args)
            if args[1] == "logger-ui-reconnaissance":
                destination = self.root / "ui/probe.json"
                value = json.loads(destination.read_text())
                value.pop("tensorboard_scalars")
                destination.write_text(json.dumps(value))

        self.stage.side_effect = probe
        with self.assertRaisesRegex(AssertionError, "CSV/history proof"):
            self.execute()
        self.assertNotIn("tensorboard_scalar_history", self.record["checks"])
        self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})

    def test_preview_stages_and_reopening_are_required_without_promoting_visual_gate(self):
        original = self.stage.side_effect

        def probe(*args):
            original(*args)
            if args[1] == "logger-ui-reconnaissance":
                destination = self.root / "ui/probe.json"
                value = json.loads(destination.read_text())
                value["tensorboard_previews"]["reopened"]["reopened"] = False
                destination.write_text(json.dumps(value))

        self.stage.side_effect = probe
        with self.assertRaisesRegex(AssertionError, "reopening proof"):
            self.execute()
        self.assertNotIn("tensorboard_native_previews", self.record["checks"])
        self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})

    def test_source_receipt_change_rejected_before_servers(self):
        (self.source / "case.json").write_text("changed")
        with self.assertRaisesRegex(AssertionError, "receipt changed"):
            self.execute()
        self.launch.assert_not_called()

    def test_source_manifest_change_rejected_before_servers(self):
        (self.source / "logging-manifest.json").write_text("changed")
        with self.assertRaisesRegex(AssertionError, "manifest changed"):
            self.execute()
        self.launch.assert_not_called()

    def test_port_conflict_does_not_start_or_move_servers(self):
        self.reservation.return_value.__enter__.return_value.bind.side_effect = OSError(
            "occupied port"
        )
        with self.assertRaisesRegex(OSError, "occupied port"):
            self.execute()
        self.launch.assert_not_called()

    def test_probe_failure_preserved_when_cleanup_also_fails(self):
        failure = RuntimeError("original browser failure")

        def fail_browser(*args):
            if args[1] == "sandboxed-browser-cancellation":
                return self.probe(*args)
            raise failure

        self.stage.side_effect = fail_browser
        calls = 0

        def members():
            nonlocal calls
            calls += 1
            return {1} if calls == 1 else {1, 1000, 1001}

        with mock.patch.object(self.g["time"], "monotonic", side_effect=[0, 99]):
            with mock.patch.dict(self.g, _signal_members=mock.Mock()) as patched:
                with self.assertRaises(RuntimeError) as caught:
                    self.execute(members)
        self.assertIs(caught.exception, failure)
        self.assertTrue(any("cleanup also failed" in note for note in caught.exception.__notes__))
        self.assertTrue(
            all(call.args[0] == {1000, 1001} for call in patched["_signal_members"].call_args_list)
        )
        self.assertNotIn("logger_ui_reconnaissance", self.record["checks"])

    def test_unexpected_visual_pass_is_not_accepted_as_reconnaissance(self):
        def probe(*args):
            if args[1] == "sandboxed-browser-cancellation":
                return self.probe(*args)
            (self.root / "ui/probe.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "claim": "UI_OPENED_ONLY",
                        "V_gate": "PASS",
                    }
                )
            )

        self.stage.side_effect = probe
        with self.assertRaisesRegex(AssertionError, "status/claim"):
            self.execute()
        self.assertNotIn("logger_ui_reconnaissance", self.record["checks"])

    def test_failed_actual_browser_launch_never_starts_native_ui_servers(self):
        self.stage.side_effect = AssertionError("real sandbox launch failed")
        with self.assertRaisesRegex(AssertionError, "sandbox launch failed"):
            self.execute()
        self.launch.assert_not_called()
        self.assertNotIn("sandboxed_browser_cancellation", self.record["checks"])

    def test_private_ui_clients_affect_only_child_servers_not_parent_environment(self):
        before = dict(os.environ)
        self.execute()
        self.assertEqual(dict(os.environ), before)
        for call in self.launch.call_args_list:
            self.assertEqual(
                call.kwargs["env"]["LD_LIBRARY_PATH"], "/private/browser-clients:/private/native"
            )


class ServerReadinessTest(unittest.TestCase):
    """Readiness never accepts a foreign or replaced socket's successful HTTP reply."""

    def setUp(self):
        self.wait = runpy.run_path(str(SCRIPT))["_wait_server"]
        self.g = self.wait.__globals__
        self.process = mock.Mock(pid=1000, returncode=None)
        self.process.poll.return_value = None
        self.opener = mock.Mock()
        self.opener.open.side_effect = lambda *args, **kwargs: contextlib.nullcontext(
            SimpleNamespace(status=200)
        )
        patcher = mock.patch.object(
            self.g["urllib"].request, "build_opener", return_value=self.opener
        )
        self.build_opener = patcher.start()
        self.addCleanup(patcher.stop)

    def test_foreign_listener_is_rejected_before_request(self):
        with (
            mock.patch.dict(
                self.g, _owned_listener=mock.Mock(side_effect=RuntimeError("foreign listener"))
            ),
            mock.patch.object(
                self.g["urllib"].request,
                "urlopen",
                return_value=contextlib.nullcontext(SimpleNamespace(status=200)),
            ) as request,
        ):
            with self.assertRaisesRegex(RuntimeError, "foreign listener"):
                self.wait(self.process, 6007)
        request.assert_not_called()

    def test_listener_replacement_after_http_is_rejected(self):
        @contextlib.contextmanager
        def listener(*args):
            yield {"pid": 1001, "socket_inode": "123"}
            raise RuntimeError("owned listener changed during HTTP readiness")

        with (
            mock.patch.dict(self.g, _owned_listener=listener),
            mock.patch.object(
                self.g["urllib"].request,
                "urlopen",
                return_value=contextlib.nullcontext(SimpleNamespace(status=200)),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "listener changed"):
                self.wait(self.process, 6007)

    def test_readiness_disables_inherited_proxies(self):
        identity = {"pid": 1001, "socket_inode": "123"}
        with (
            mock.patch.dict(self.g, _owned_listener=lambda *args: contextlib.nullcontext(identity)),
            mock.patch.object(
                self.g["urllib"].request,
                "urlopen",
                side_effect=AssertionError("default client retains proxy configuration"),
            ),
        ):
            self.assertEqual(self.wait(self.process, 6007), identity)
        proxy = next(
            handler
            for handler in self.build_opener.call_args.args
            if isinstance(handler, self.g["urllib"].request.ProxyHandler)
        )
        self.assertEqual(proxy.proxies, {})
        self.opener.open.assert_called_once_with("http://127.0.0.1:6007/", timeout=1)

    def test_readiness_redirects_never_leave_the_exact_origin(self):
        handler = self.g["_ReadinessRedirect"](6007)
        request = self.g["urllib"].request.Request("http://127.0.0.1:6007/")
        for url in (
            "https://127.0.0.1:6007/",
            "http://127.0.0.1:5001/",
            "http://example.invalid/",
            "http://user@127.0.0.1:6007/",
            "http://localhost:6007/",
        ):
            with self.subTest(url=url), self.assertRaises(self.g["urllib"].error.URLError):
                handler.redirect_request(request, None, 302, "redirect", {}, url)
        redirected = handler.redirect_request(
            request, None, 302, "redirect", {}, "http://127.0.0.1:6007/scalars"
        )
        self.assertEqual(redirected.full_url, "http://127.0.0.1:6007/scalars")


class ListenerOwnershipTest(unittest.TestCase):
    """Synthetic proc files and inert pidfds exercise the exact ownership checks."""

    def setUp(self):
        self.listener = runpy.run_path(str(SCRIPT))["_owned_listener"]
        self.g = self.listener.__wrapped__.__globals__
        scratch = tempfile.TemporaryDirectory(prefix="listener-proc-")
        self.addCleanup(scratch.cleanup)
        self.proc = Path(scratch.name)
        (self.proc / "net").mkdir()
        (self.proc / "net/tcp6").write_text("header\n")
        self.table()
        self.process = mock.Mock(pid=1000, returncode=None)
        self.process.poll.return_value = None
        for pid, parent, children in ((1000, os.getpid(), "1001"), (1001, 1000, "")):
            directory = self.proc / str(pid)
            (directory / "fd").mkdir(parents=True)
            (directory / f"task/{pid}").mkdir(parents=True)
            (directory / f"task/{pid}/children").write_text(children)
            (directory / "stat").write_text(
                f"{pid} (python worker) S {parent} " + "0 " * 17 + "123"
            )
        (self.proc / "1001/fd/9").symlink_to("socket:[456]")
        patcher = mock.patch.dict(
            self.g,
            PROC=self.proc,
            _pidfd_call=lambda *args: os.open(os.devnull, os.O_RDONLY),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(self.g["select"], "select", return_value=([], [], []))
        self.select = patcher.start()
        self.addCleanup(patcher.stop)

    def table(self, inode="456", address="0100007F"):
        (self.proc / "net/tcp").write_text(
            "header\n" + f"0: {address}:1777 00000000:0000 0A 0 0 0 0 0 {inode}\n"
        )

    def test_owned_server_child_has_exact_pid_start_and_socket_identity(self):
        with self.listener(self.process, 6007) as identity:
            self.assertEqual(identity, {"pid": 1001, "start_ticks": 123, "socket_inode": "456"})

    def test_port_without_a_listener_is_not_ready(self):
        (self.proc / "net/tcp").write_text("header\n")
        with self.listener(self.process, 6007) as identity:
            self.assertIsNone(identity)

    def test_foreign_or_reparented_listener_is_not_owned(self):
        (self.proc / "1000/task/1000/children").write_text("")
        with self.assertRaisesRegex(RuntimeError, "foreign listener"):
            with self.listener(self.process, 6007):
                self.fail("foreign listener admitted")
        (self.proc / "1000/task/1000/children").write_text("1001")
        (self.proc / "1001/stat").write_text("1001 (foreign) S 999 " + "0 " * 17 + "123")
        with self.assertRaisesRegex(RuntimeError, "foreign listener"):
            with self.listener(self.process, 6007):
                self.fail("reparented listener admitted")

    def test_wildcard_listener_is_rejected(self):
        self.table(address="00000000")
        with self.assertRaisesRegex(RuntimeError, "exclusively bound"):
            with self.listener(self.process, 6007):
                self.fail("wildcard listener admitted")

    def test_socket_replacement_during_request_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "listener changed"):
            with self.listener(self.process, 6007):
                self.table(inode="789")

    def test_live_cli_does_not_hide_exited_listener_child(self):
        with self.assertRaisesRegex(RuntimeError, "listener changed"):
            with self.listener(self.process, 6007):
                self.select.side_effect = lambda descriptors, *args: ([descriptors[-1]], [], [])
        self.assertIsNone(self.process.poll())

    def test_original_request_failure_is_not_masked_by_ownership_scan(self):
        with self.assertRaisesRegex(FileNotFoundError, "original request"):
            with self.listener(self.process, 6007):
                raise FileNotFoundError("original request")

    def test_cleanup_rejects_reused_pid_outside_owned_group(self):
        signal_members = self.g["_signal_members"]
        stat = self.proc / "1001/stat"
        fields = stat.read_text().rsplit(")", 1)[1].split()
        fields[2] = str(os.getpgrp() + 1)
        stat.write_text("1001 (foreign) " + " ".join(fields))
        with self.assertRaisesRegex(RuntimeError, "outside its owned group"):
            signal_members({1001}, 15)

    def test_cleanup_signals_only_pinned_current_group_members(self):
        stat = self.proc / "1001/stat"
        fields = stat.read_text().rsplit(")", 1)[1].split()
        fields[2] = str(os.getpgrp())
        stat.write_text("1001 (owned) " + " ".join(fields))
        calls = []

        def syscall(number, *args):
            calls.append((number, args))
            return os.open(os.devnull, os.O_RDONLY) if number == 434 else 0

        with mock.patch.dict(self.g, _pidfd_call=syscall):
            self.g["_signal_members"]({1001}, 15)
        self.assertEqual(calls[0], (434, (1001, 0)))
        self.assertEqual(calls[1], (424, (calls[1][1][0], 15, 0, 0)))


if __name__ == "__main__":
    unittest.main()
