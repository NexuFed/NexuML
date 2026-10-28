"""Mocked cancellation proof accounting only; never launches any browser/process."""

import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


class BrowserCancelTest(unittest.TestCase):
    """Reject missing launch and altered native guard failures without fallback."""

    def setUp(self):
        self.main = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "fixtures/physical_ai_browser_cancel.py")
        )["main"]
        self.g = self.main.__globals__
        scratch = tempfile.TemporaryDirectory(prefix="browser-cancel-test-")
        self.addCleanup(scratch.cleanup)
        self.base = Path(scratch.name)
        runtime = self.base / "venv/browser-runtime"
        runtime.mkdir(parents=True)
        (runtime / "setup.json").write_text(
            json.dumps({"status": "success", "paths": {"node": "/private/node"}})
        )
        patcher = mock.patch.dict(os.environ, UV_PROJECT_ENVIRONMENT=str(runtime.parent))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.guard = {
            "pid": 123,
            "exit_code": 137,
            "child_returncode": -9,
            "reason": "signal",
            "process_ownership": "dedicated-linux-child-subreaper-pidfds",
            "errors": [],
        }
        self.marker = 'NEXUML_BROWSER_LAUNCHED={"version":"153.0.8010.12","pid":123}\n'

    def execute(self):
        root = self.base / "owned"

        def launch(argv, **kwargs):
            self.assertEqual(argv[-1], "--guard-forced-exit")
            self.assertEqual(argv[argv.index("--disk-floor-bytes") + 1], str(4 * 2**30))
            self.assertEqual(argv[argv.index("--deadline-seconds") + 1], "90")
            self.assertNotIn("shell", kwargs)
            self.assertNotIn("start_new_session", kwargs)
            kwargs["stdout"].write(self.marker.encode())
            (root / "guard.json").write_text(json.dumps(self.guard))
            return SimpleNamespace(returncode=self.guard["exit_code"])

        with mock.patch.object(self.g["subprocess"], "run", side_effect=launch):
            self.main(root, self.base / "manifest.json")
        return root

    def test_expected_parent_kill_keeps_viewing_unverified(self):
        root = self.execute()
        proof = json.loads((root / "proof.json").read_text())
        self.assertEqual((proof["cleanup"], proof["V_gate"]), ("PASS", "UNVERIFIED"))
        self.assertEqual(proof["expected_exit_code"], 137)

    def test_real_browser_marker_is_required(self):
        self.marker = "ordinary setup output\n"
        with self.assertRaisesRegex(AssertionError, "launch marker absent"):
            self.execute()
        self.assertFalse((self.base / "owned/proof.json").exists())

    def test_browser_version_pid_and_duplicate_markers_cannot_be_forged(self):
        original = self.marker
        for marker in (
            original.replace("153.0.8010.12", "152.0.0.0"),
            original.replace('"pid":123', '"pid":124'),
            original + original,
        ):
            self.marker = marker
            with self.subTest(marker=marker), self.assertRaises(AssertionError):
                self.execute()
            self.base = self.base / "next"
            self.base.mkdir()

    def test_native_failure_or_cleanup_error_cannot_be_promoted(self):
        for key, value in (
            ("exit_code", 1),
            ("child_returncode", 0),
            ("errors", ["owned child survived"]),
            ("process_ownership", "fresh-process-group"),
        ):
            original = self.guard[key]
            self.guard[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.execute()
            self.guard[key] = original
            # Each probe owns a fresh destination; never overwrite the failed attempt.
            self.base = self.base / "next"
            self.base.mkdir()


if __name__ == "__main__":
    unittest.main()
