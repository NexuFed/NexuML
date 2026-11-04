"""Hermetic private-runtime setup checks; no downloads, native execution or Ray."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_native_setup.py"


class PhysicalAINativeSetupTest(unittest.TestCase):
    def setUp(self):
        self.main = runpy.run_path(str(SCRIPT))["main"]
        self.g = self.main.__globals__
        self.pins = self.g["PACKAGES"]
        scratch = tempfile.TemporaryDirectory(prefix="native setup ; $(literal)-")
        self.addCleanup(scratch.cleanup)
        self.base = Path(scratch.name)
        self.venv = self.base / ("a" * 32)
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
        (self.venv / "bin").mkdir()
        (self.venv / "bin/python").touch()
        self.destination = self.venv / "native-runtime"
        self.content = b"authenticated mock package"
        self.packages = tuple(
            {**pin, "size": len(self.content), "sha256": hashlib.sha256(self.content).hexdigest()}
            for pin in self.pins
        )
        self.patch(mock.patch.dict(self.g, ENV_BASE=self.base, PACKAGES=self.packages))
        self.patch(
            mock.patch.dict(
                os.environ,
                {
                    "UV_PROJECT_ENVIRONMENT": str(self.venv),
                    "LD_LIBRARY_PATH": "/existing libraries:/vendor/nvidia",
                    "PRIVATE_CREDENTIAL": "never-emit-this",
                },
            )
        )
        self.patch(mock.patch.object(self.g["sys"], "argv", [str(SCRIPT)]))
        self.patch(mock.patch.object(self.g["sys"], "platform", "linux"))
        self.release = self.patch(
            mock.patch.object(
                self.g["platform"],
                "freedesktop_os_release",
                return_value={"ID": "ubuntu", "VERSION_ID": "22.04"},
            )
        )
        self.machine = self.patch(
            mock.patch.object(
                self.g["platform"],
                "machine",
                return_value="x86_64",
            )
        )
        self.which = self.patch(
            mock.patch.object(
                self.g["shutil"],
                "which",
                side_effect=lambda name: "/usr/bin/" + name,
            )
        )
        self.http = self.patch(
            mock.patch.object(
                self.g["urllib"].request,
                "urlopen",
                side_effect=self.download,
            )
        )
        self.launch = self.patch(
            mock.patch.object(
                self.g["subprocess"],
                "run",
                side_effect=self.command,
            )
        )
        self.output = io.StringIO()
        self.patch(contextlib.redirect_stdout(self.output))

    def patch(self, patch):
        value = patch.__enter__()
        self.addCleanup(patch.__exit__, None, None, None)
        return value

    def download(self, url, timeout):
        self.assertEqual(timeout, 30)
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = io.BytesIO(self.content).read
        response.geturl.return_value = url
        return response

    def command(self, argv, **kwargs):
        if argv[0] == "dpkg-deb":
            self.assertEqual(self.http.call_count, 13)
            self.assertEqual(argv[1], "--extract")
            self.assertEqual(argv[3], str(self.destination))
            self.assertEqual(kwargs, {"check": True, "timeout": 60})
            self.assertEqual(Path(argv[2]).read_bytes(), self.content)
            libraries = self.destination / "usr/lib/x86_64-linux-gnu"
            libraries.mkdir(parents=True, exist_ok=True)
            for name in ("libGL.so.1", "libEGL.so.1", "libXt.so.6", "libXrandr.so.2"):
                (libraries / name).write_bytes(b"mock shared library")
            return SimpleNamespace(returncode=0)
        self.assertEqual(argv[0], "ldd")
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["timeout"], 30)
        self.assertFalse(kwargs.get("shell", False))
        return SimpleNamespace(returncode=0, stdout="libc.so.6 => /lib/libc.so.6 (0x123)\n")

    def receipt(self):
        return json.loads((self.destination / "setup.json").read_text())

    def test_lock_preserves_twelve_client_pins_and_adds_only_authenticated_egl_dispatch(self):
        expected = {
            "libgl1": ("1.4.0-1", 110500),
            "libglx0": ("1.4.0-1", 41024),
            "libglvnd0": ("1.4.0-1", 73582),
            "libegl1": ("1.4.0-1", 28564),
            "libxt6": ("1:1.2.1-1", 176536),
            "libice6": ("2:1.0.10-1build2", 42606),
            "libsm6": ("2:1.2.3-1build2", 16736),
            "libxrandr2": ("2:1.5.2-1build1", 20394),
            "libxrender1": ("1:0.9.10-1build4", 19678),
            "libxfixes3": ("1:6.0.0-1", 11656),
            "libbsd0": ("0.11.5-1", 44760),
            "libmd0": ("1.0.4-1build1", 22964),
            "libuuid1": ("2.37.2-4ubuntu3.6", 23978),
        }
        self.assertEqual(len(self.pins), 13)
        self.assertEqual({p["name"]: (p["version"], p["size"]) for p in self.pins}, expected)
        self.assertEqual(
            hashlib.sha256(
                json.dumps(
                    tuple(pin for pin in self.pins if pin["name"] != "libegl1"), sort_keys=True
                ).encode()
            ).hexdigest(),
            "ad9b213ba2055cbe54053499f589124cd0d01d0eeefdda66376c478d6a58d0c4",
        )
        egl = next(pin for pin in self.pins if pin["name"] == "libegl1")
        self.assertEqual(
            egl["sha256"], "4a35c0e925e15a076e7ce11d7c76f8ecb16615fc347c2c09afa4c205a7ca8ec4"
        )
        self.assertEqual(egl["filename"], "pool/main/libg/libglvnd/libegl1_1.4.0-1_amd64.deb")
        self.assertEqual(self.g["ARCHIVE"], "https://archive.ubuntu.com/ubuntu/")
        for pin in self.pins:
            self.assertRegex(pin["sha256"], r"^[a-f0-9]{64}$")
            self.assertTrue(pin["filename"].startswith("pool/main/"))
            self.assertTrue(pin["filename"].endswith("_amd64.deb"))

    def test_success_is_private_exclusive_and_does_not_mutate_environment(self):
        before = dict(os.environ)
        self.main()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.destination / "cache").stat().st_mode & 0o777, 0o700)
        self.assertEqual(len(list(self.venv.iterdir())), 3)
        self.assertEqual(self.launch.call_count, 17)
        for call in self.launch.call_args_list[13:]:
            self.assertEqual(call.args[0][0], "ldd")
            self.assertEqual(
                call.kwargs["env"]["LD_LIBRARY_PATH"],
                str(self.destination / "usr/lib/x86_64-linux-gnu")
                + ":"
                + before["LD_LIBRARY_PATH"],
            )
        for pin, call in zip(self.packages, self.http.call_args_list, strict=True):
            self.assertEqual(call.args, (self.g["ARCHIVE"] + pin["filename"],))
        receipt = self.receipt()
        self.assertEqual(receipt["status"], "success")
        self.assertEqual(receipt["directory"], str(self.destination))
        self.assertEqual(receipt["packages"], list(self.packages))
        self.assertEqual(len(receipt["ldd"]), 4)
        self.assertEqual(len(receipt["libraries"]), 4)
        self.assertEqual(
            set(receipt["libraries"].values()),
            {
                hashlib.sha256(b"mock shared library").hexdigest(),
            },
        )
        self.assertEqual(
            json.loads(self.output.getvalue().removeprefix("NEXUML_NATIVE_RUNTIME_SETUP=")), receipt
        )
        self.assertNotIn("never-emit-this", self.output.getvalue())
        with self.assertRaises(FileExistsError):
            self.main()
        self.assertEqual(self.http.call_count, 13)

    def test_egl_dispatch_client_is_pinned_and_its_actual_library_closure_is_checked(self):
        self.assertIn("libegl1", {pin["name"] for pin in self.pins})
        self.main()
        self.assertIn("libEGL.so.1", {item["library"] for item in self.receipt()["ldd"]})

    def test_empty_library_path_adds_no_current_directory_search(self):
        os.environ.pop("LD_LIBRARY_PATH", None)
        self.main()
        for call in self.launch.call_args_list[13:]:
            self.assertEqual(
                call.kwargs["env"]["LD_LIBRARY_PATH"],
                str(self.destination / "usr/lib/x86_64-linux-gnu"),
            )
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)

    def test_invalid_platform_is_rejected_before_download_or_destination(self):
        for release, machine in (
            ({"ID": "debian", "VERSION_ID": "22.04"}, "x86_64"),
            ({"ID": "ubuntu", "VERSION_ID": "24.04"}, "x86_64"),
            ({"ID": "ubuntu", "VERSION_ID": "22.04"}, "aarch64"),
        ):
            with self.subTest(release=release, machine=machine):
                self.release.return_value = release
                self.machine.return_value = machine
                with self.assertRaisesRegex(RuntimeError, "Ubuntu 22.04 amd64"):
                    self.main()
        self.http.assert_not_called()
        self.launch.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_arguments_and_duplicate_pins_are_rejected(self):
        with mock.patch.object(self.g["sys"], "argv", [str(SCRIPT), "--url", "https://other"]):
            with self.assertRaisesRegex(ValueError, "no arguments"):
                self.main()
        for packages in (
            self.packages[:-1],
            (*self.packages[:-1], self.packages[0]),
            (*self.packages[:-1], {**self.packages[-1], "filename": self.packages[0]["filename"]}),
        ):
            with mock.patch.dict(self.g, PACKAGES=packages):
                with self.assertRaisesRegex(ValueError, "13 unique"):
                    self.main()
        self.http.assert_not_called()

    def test_invalid_paths_and_nonvenv_are_rejected(self):
        for value in (
            "relative",
            str(self.base),
            str(SCRIPT.parents[2]),
            str(self.base / "not-a-run-id"),
            str(self.base.parent / ("b" * 32)),
        ):
            with mock.patch.dict(os.environ, UV_PROJECT_ENVIRONMENT=value):
                with self.assertRaises(ValueError):
                    self.main()
        (self.venv / "pyvenv.cfg").unlink()
        with self.assertRaises(ValueError):
            self.main()
        self.http.assert_not_called()

    def test_symlink_root_and_ancestry_are_rejected(self):
        link = self.base / ("b" * 32)
        link.symlink_to(self.venv, target_is_directory=True)
        with mock.patch.dict(os.environ, UV_PROJECT_ENVIRONMENT=str(link)):
            with self.assertRaises(ValueError):
                self.main()
        alias = self.base / "alias"
        alias.symlink_to(self.base, target_is_directory=True)
        with mock.patch.dict(self.g, ENV_BASE=alias):
            with mock.patch.dict(os.environ, UV_PROJECT_ENVIRONMENT=str(alias / self.venv.name)):
                with self.assertRaises(ValueError):
                    self.main()
        self.http.assert_not_called()

    def test_other_owners_and_missing_tools_are_rejected(self):
        with mock.patch.object(self.g["os"], "getuid", return_value=self.venv.stat().st_uid + 1):
            with self.assertRaises(ValueError):
                self.main()
        self.which.return_value = None
        self.which.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "already be installed"):
            self.main()
        self.http.assert_not_called()

    def test_existing_destination_symlink_is_never_followed_or_cleaned(self):
        unrelated = self.base / "unrelated"
        unrelated.mkdir()
        self.destination.symlink_to(unrelated, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.main()
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(list(unrelated.iterdir()), [])
        self.http.assert_not_called()

    def test_last_bad_hash_prevents_all_extraction(self):
        packages = (*self.packages[:-1], {**self.packages[-1], "sha256": "0" * 64})
        with mock.patch.dict(self.g, PACKAGES=packages):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch: libuuid1"):
                self.main()
        self.assertEqual(self.http.call_count, 13)
        self.launch.assert_not_called()
        self.assertFalse(
            (self.destination / "cache" / Path(self.pins[-1]["filename"]).name).exists()
        )
        self.assertEqual(self.receipt()["status"], "failed")

    def test_bad_size_prevents_all_extraction_and_writing_unverified_bytes(self):
        packages = (*self.packages[:-1], {**self.packages[-1], "size": len(self.content) - 1})
        with mock.patch.dict(self.g, PACKAGES=packages):
            with self.assertRaisesRegex(ValueError, "size mismatch: libuuid1"):
                self.main()
        self.launch.assert_not_called()
        self.assertEqual(len(list((self.destination / "cache").iterdir())), 12)

    def test_http_failure_is_not_retried(self):
        self.http.side_effect = TimeoutError("original download deadline")
        with self.assertRaisesRegex(TimeoutError, "original download deadline"):
            self.main()
        self.assertEqual(self.http.call_count, 1)
        self.launch.assert_not_called()
        self.assertEqual(self.receipt()["status"], "failed")

    def test_secondary_receipt_failure_preserves_original_setup_error(self):
        self.http.side_effect = TimeoutError("original download deadline")
        original_open = Path.open

        def open_file(path, *args, **kwargs):
            if path == self.destination / "setup.json":
                raise OSError("receipt unavailable")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=open_file):
            with self.assertRaisesRegex(TimeoutError, "original download deadline") as caught:
                self.main()
        self.assertIn(
            "Native setup receipt also failed: receipt unavailable", caught.exception.__notes__
        )
        self.launch.assert_not_called()

    def test_redirect_response_is_rejected_before_read_or_extraction(self):
        response = self.download("https://other.example/package.deb", 30)
        self.http.side_effect = None
        self.http.return_value = response
        with self.assertRaisesRegex(ValueError, "redirects are not permitted"):
            self.main()
        response.read.assert_not_called()
        self.launch.assert_not_called()

    def test_extraction_failure_preserves_diagnostic_and_stops(self):
        failure = subprocess.CalledProcessError(2, ["dpkg-deb"], stderr="original extract failure")
        self.launch.side_effect = failure
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.main()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.launch.call_count, 1)
        self.assertEqual(self.receipt()["status"], "failed")

    def test_ldd_missing_library_preserves_diagnostic_and_failure_receipt(self):
        original = self.command

        def command(argv, **kwargs):
            if argv[0] == "ldd":
                return SimpleNamespace(returncode=0, stdout="libmissing.so => not found\n")
            return original(argv, **kwargs)

        self.launch.side_effect = command
        with self.assertRaisesRegex(RuntimeError, "libmissing.so => not found"):
            self.main()
        self.assertEqual(
            self.receipt()["ldd"],
            [
                {
                    "library": "libGL.so.1",
                    "stdout": "libmissing.so => not found\n",
                    "exit": 0,
                }
            ],
        )
        self.assertEqual(self.receipt()["status"], "failed")

    def test_nonzero_ldd_preserves_output_even_without_not_found(self):
        original = self.command

        def command(argv, **kwargs):
            if argv[0] == "ldd":
                return SimpleNamespace(returncode=7, stdout="original loader diagnostic\n")
            return original(argv, **kwargs)

        self.launch.side_effect = command
        with self.assertRaisesRegex(RuntimeError, "original loader diagnostic"):
            self.main()
        self.assertEqual(self.receipt()["ldd"][0]["exit"], 7)
        self.assertEqual(self.receipt()["status"], "failed")


if __name__ == "__main__":
    unittest.main()
