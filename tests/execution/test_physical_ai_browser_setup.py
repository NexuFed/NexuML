"""Hermetic setup checks: fake HTTP, archive metadata and subprocesses only."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import stat
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_browser_setup.py"


class PhysicalAIBrowserSetupTest(unittest.TestCase):
    """Exercise only private setup with inert archives and binaries."""

    def setUp(self):
        self.main = runpy.run_path(str(SCRIPT))["main"]
        self.g = self.main.__globals__
        self.pins = self.g["PACKAGES"]
        self.assets = self.g["ASSETS"]
        scratch = tempfile.TemporaryDirectory(prefix="browser ; $(literal)-")
        self.addCleanup(scratch.cleanup)
        self.base = Path(scratch.name)
        self.venv = self.base / ("a" * 32)
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin/python").touch()
        (self.venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
        self.native = self.venv / "native-runtime"
        self.native.mkdir()
        (self.native / "setup.json").write_text('{"status":"success"}\n')
        self.destination = self.venv / "browser-runtime"
        self.content = b"authenticated fake archive or deb"
        self.digest = hashlib.sha256(self.content).hexdigest()
        self.integrity = (
            "sha512-" + base64.b64encode(hashlib.sha512(self.content).digest()).decode()
        )
        self.packages = tuple(self.fake_pin(pin) for pin in self.pins)
        self.fake_assets = tuple(self.fake_pin(pin) for pin in self.assets)
        self.patch(
            mock.patch.dict(
                self.g, ENV_BASE=self.base, PACKAGES=self.packages, ASSETS=self.fake_assets
            )
        )
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
            mock.patch.object(self.g["platform"], "machine", return_value="x86_64")
        )
        self.which = self.patch(
            mock.patch.object(
                self.g["shutil"], "which", side_effect=lambda name: "/usr/bin/" + name
            )
        )
        self.http = mock.Mock(side_effect=self.download)
        self.opener = self.patch(
            mock.patch.object(
                self.g["urllib"].request,
                "build_opener",
                return_value=SimpleNamespace(open=self.http),
            )
        )
        self.launch = self.patch(
            mock.patch.object(self.g["subprocess"], "run", side_effect=self.command)
        )
        self.tar_open = self.patch(
            mock.patch.object(self.g["tarfile"], "open", side_effect=self.tar_source)
        )
        self.zip_open = self.patch(
            mock.patch.object(self.g["zipfile"], "ZipFile", side_effect=self.zip_source)
        )
        root = self.g["NODE_ROOT"]
        node = tarfile.TarInfo(root + "/bin/node")
        npm = tarfile.TarInfo(root + "/bin/npm")
        npm.type = tarfile.SYMTYPE
        npm.linkname = "../lib/node_modules/npm/bin/npm-cli.js"
        package_root = tarfile.TarInfo("package")
        package_root.type = tarfile.DIRTYPE
        self.tar_members = {
            self.assets[0]["filename"]: [node, npm],
            self.assets[1]["filename"]: [package_root, tarfile.TarInfo("package/index.js")],
            self.assets[2]["filename"]: [tarfile.TarInfo("package/index.js")],
        }
        self.zip_members = [zipfile.ZipInfo(self.g["CHROME_ROOT"] + "/chrome-headless-shell")]
        self.extracts = []
        self.output = io.StringIO()
        self.patch(contextlib.redirect_stdout(self.output))

    def patch(self, patch):
        value = patch.__enter__()
        self.addCleanup(patch.__exit__, None, None, None)
        return value

    def fake_pin(self, pin):
        result = {**pin, "size": len(self.content), "sha256": self.digest}
        if "integrity" in result:
            result["integrity"] = self.integrity
        return result

    def download(self, url, timeout):
        self.assertEqual(timeout, 30)
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.geturl.return_value = url
        response.read.side_effect = lambda size: self.content[:size]
        return response

    def all_verified(self):
        self.assertEqual(self.http.call_count, 31)
        cache = self.destination / "cache"
        self.assertEqual(len(list(cache.iterdir())), 31)
        for path in cache.iterdir():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.digest)

    def command(self, argv, **kwargs):
        self.all_verified()
        self.assertFalse(kwargs.get("shell", False))
        if argv[0] == "dpkg-deb":
            self.assertEqual(argv[1], "--extract")
            self.assertEqual(argv[3], str(self.destination / "clients"))
            self.assertEqual(kwargs, {"check": True, "timeout": 60})
            self.assertEqual(Path(argv[2]).read_bytes(), self.content)
            return SimpleNamespace(returncode=0)
        self.assertEqual(argv[0], "ldd")
        expected = [
            str(self.destination / "node" / self.g["NODE_ROOT"] / "bin/node"),
            str(self.destination / "chrome" / self.g["CHROME_ROOT"] / "chrome-headless-shell"),
        ]
        self.assertIn(argv[1], expected)
        self.assertEqual(len(argv), 2)
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["timeout"], 30)
        return SimpleNamespace(returncode=0, stdout="libc.so.6 => /lib/libc.so.6 (0x123)\n")

    def tar_source(self, archive):
        source = mock.MagicMock()
        source.__enter__.return_value = source
        source.getmembers.return_value = self.tar_members[Path(archive).name]

        def extract(destination, *, members, filter):
            self.all_verified()
            self.assertEqual(filter, "data")
            self.extracts.append(destination)
            for member in members:
                path = destination / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                if member.issym():
                    path.symlink_to(member.linkname)
                elif member.isdir():
                    path.mkdir(exist_ok=True)
                else:
                    path.write_bytes(b"\x7fELFfake verified binary or module")

        source.extractall.side_effect = extract
        return source

    def zip_source(self, archive):
        source = mock.MagicMock()
        source.__enter__.return_value = source
        source.infolist.return_value = self.zip_members

        def extract(destination):
            self.all_verified()
            self.extracts.append(destination)
            for member in self.zip_members:
                path = destination / member.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\x7fELFfake verified binary")

        source.extractall.side_effect = extract
        return source

    def receipt(self):
        return json.loads((self.destination / "setup.json").read_text())

    def test_exact_fixed_pins_and_asset_identities(self):
        self.assertEqual(self.base.parent, Path(tempfile.gettempdir()))
        self.assertEqual(len(self.pins), 27)
        self.assertEqual(len(self.assets), 4)
        for pins, expected in (
            (
                tuple(pin for pin in self.pins if pin["name"] != "libusb-1.0-0"),
                "d7bc53849f7b6bb6d78177ba6dc02045900a5dd9db0d2c50d2b0cc6a3f80e770",
            ),
            (self.assets, "1a933590929d6eb7f0bc8070fa323c1ebb14ca2e5dec67272e7bc6bb88227d0e"),
        ):
            self.assertEqual(
                hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest(), expected
            )
        self.assertEqual(self.g["CHROME_REVISION"], "1243")
        usb = next(pin for pin in self.pins if pin["name"] == "libusb-1.0-0")
        self.assertEqual(
            (usb["version"], usb["size"], usb["sha256"]),
            (
                "2:1.0.25-1ubuntu2",
                52688,
                "99b98c9cc9c88c4d7f50e106e9f0cbff868c5d14809f516b1947d3d2c4638aae",
            ),
        )
        for pin in self.pins:
            self.assertEqual(pin["url"], "https://archive.ubuntu.com/ubuntu/" + pin["filename"])
            self.assertRegex(pin["sha256"], r"^[a-f0-9]{64}$")
        self.http.assert_not_called()
        self.launch.assert_not_called()

    def test_tensorboard_open3d_usb_client_is_in_the_fixed_private_lock(self):
        self.assertIn("libusb-1.0-0", {pin["name"] for pin in self.pins})

    def test_installed_tensorboard_plugin_has_its_elf_closure_checked_without_import(self):
        plugin = (
            self.venv
            / "lib/python3.12/site-packages/open3d/cpu/pybind.cpython-312-x86_64-linux-gnu.so"
        )
        plugin.parent.mkdir(parents=True)
        plugin.write_bytes(b"\x7fELFfake locked plugin")
        original = self.command

        def command(argv, **kwargs):
            if argv == ["ldd", str(plugin)]:
                return SimpleNamespace(
                    returncode=0, stdout="all installed plugin clients resolved\n"
                )
            return original(argv, **kwargs)

        self.launch.side_effect = command
        self.main()
        self.assertIn(str(plugin), [item["binary"] for item in self.receipt()["ldd"]])

    def test_legacy_flat_open3d_plugin_from_actual_trace_is_checked(self):
        plugin = (
            self.venv / "lib/python3.12/site-packages/open3d/pybind.cpython-312-x86_64-linux-gnu.so"
        )
        plugin.parent.mkdir(parents=True)
        plugin.write_bytes(b"\x7fELFactual flat plugin layout")
        original = self.command

        def command(argv, **kwargs):
            if argv == ["ldd", str(plugin)]:
                return SimpleNamespace(returncode=0, stdout="libusb-1.0.so.0 => private clients\n")
            return original(argv, **kwargs)

        self.launch.side_effect = command
        self.main()
        self.assertIn(str(plugin), [item["binary"] for item in self.receipt()["ldd"]])
        self.assertEqual(
            self.receipt()["installed_plugin_sha256"],
            hashlib.sha256(plugin.read_bytes()).hexdigest(),
        )

    def test_installed_open3d_without_a_supported_plugin_cannot_skip_closure(self):
        (self.venv / "lib/python3.12/site-packages/open3d").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "installed TensorBoard Open3D plugin ABI"):
            self.main()

    def test_success_verifies_everything_first_and_keeps_parent_environment(self):
        before = dict(os.environ)
        native_receipt = (self.native / "setup.json").read_bytes()
        self.main()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual((self.native / "setup.json").read_bytes(), native_receipt)
        self.assertEqual(self.launch.call_count, 29)
        self.assertEqual(len(self.extracts), 4)
        record = self.receipt()
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["packages"], list(self.packages))
        self.assertEqual(record["assets"], list(self.fake_assets))
        self.assertEqual(len(record["verified_sha256"]), 31)
        self.assertEqual(len(record["ldd"]), 2)
        self.assertEqual(
            record["paths"]["playwrightModulePath"],
            str(self.destination / "node_modules/playwright"),
        )
        for name in ("playwright", "playwright-core"):
            self.assertTrue((self.destination / "node_modules" / name / "index.js").is_file())
            self.assertFalse((self.destination / "node_modules" / name / "package").exists())
        self.assertTrue(Path(record["paths"]["chromeBinary"]).stat().st_mode & stat.S_IXUSR)
        for name in ("", "cache", "home", "fontcache"):
            self.assertEqual((self.destination / name).stat().st_mode & 0o777, 0o700)
        fonts = ET.parse(record["paths"]["fontsConf"]).getroot()
        self.assertEqual([element.tag for element in fonts], ["dir", "cachedir"])
        self.assertEqual(fonts.findtext("dir"), str(self.destination / "clients/usr/share/fonts"))
        self.assertEqual(fonts.findtext("cachedir"), str(self.destination / "fontcache"))
        prefix = (
            str(self.destination / "clients/usr/lib/x86_64-linux-gnu")
            + ":"
            + str(self.destination / "clients/lib/x86_64-linux-gnu")
            + ":"
            + str(self.native / "usr/lib/x86_64-linux-gnu")
        )
        self.assertEqual(record["paths"]["libraryPrefix"], prefix)
        for call in self.launch.call_args_list[-2:]:
            self.assertEqual(
                call.kwargs["env"]["LD_LIBRARY_PATH"], prefix + ":" + before["LD_LIBRARY_PATH"]
            )
        self.assertEqual(
            json.loads(self.output.getvalue().removeprefix("NEXUML_BROWSER_RUNTIME_SETUP=")), record
        )
        self.assertNotIn("never-emit-this", self.output.getvalue())
        with self.assertRaises(FileExistsError):
            self.main()
        self.assertEqual(self.http.call_count, 31)
        self.assertEqual(self.receipt(), record)

    def test_empty_library_path_has_no_empty_search_component(self):
        os.environ.pop("LD_LIBRARY_PATH", None)
        self.main()
        record = self.receipt()
        self.assertEqual(record["paths"]["ldLibraryPath"], record["paths"]["libraryPrefix"])
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)

    def test_private_split_lib_and_usr_lib_layout_closes_dbus_dependency(self):
        original = self.command
        required = self.destination / "clients/lib/x86_64-linux-gnu"

        def command(argv, **kwargs):
            if argv[0] == "ldd" and str(required) not in kwargs["env"]["LD_LIBRARY_PATH"].split(
                ":"
            ):
                return SimpleNamespace(returncode=0, stdout="libdbus-1.so.3 => not found\n")
            return original(argv, **kwargs)

        self.launch.side_effect = command
        self.main()
        self.assertEqual(self.receipt()["status"], "success")
        self.assertIn(str(required), self.receipt()["paths"]["libraryPrefix"].split(":"))

    def test_platform_rejected_before_creating_runtime(self):
        for release, machine, system in (
            ({"ID": "debian", "VERSION_ID": "22.04"}, "x86_64", "linux"),
            ({"ID": "ubuntu", "VERSION_ID": "24.04"}, "x86_64", "linux"),
            ({"ID": "ubuntu", "VERSION_ID": "22.04"}, "aarch64", "linux"),
            ({"ID": "ubuntu", "VERSION_ID": "22.04"}, "x86_64", "darwin"),
        ):
            with self.subTest(release=release, machine=machine, system=system):
                self.release.return_value = release
                self.machine.return_value = machine
                with mock.patch.object(self.g["sys"], "platform", system):
                    with self.assertRaisesRegex(RuntimeError, "Ubuntu 22.04 amd64"):
                        self.main()
        self.http.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_no_arguments_or_mutable_lock_overrides(self):
        with mock.patch.object(self.g["sys"], "argv", [str(SCRIPT), "--url", "https://other"]):
            with self.assertRaisesRegex(ValueError, "no arguments"):
                self.main()
        for pins in (
            self.packages[:-1],
            (*self.packages[:-1], self.packages[0]),
            (*self.packages[:-1], {**self.packages[-1], "filename": self.packages[0]["filename"]}),
        ):
            with mock.patch.dict(self.g, PACKAGES=pins):
                with self.assertRaisesRegex(ValueError, "27 unique"):
                    self.main()
        self.http.assert_not_called()

    def test_private_real_owned_venv_required(self):
        for value in ("relative", str(self.base), str(self.base / "not-a-run-id")):
            with mock.patch.dict(os.environ, UV_PROJECT_ENVIRONMENT=value):
                with self.assertRaises(ValueError):
                    self.main()
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
        with mock.patch.object(self.g["os"], "getuid", return_value=self.venv.stat().st_uid + 1):
            with self.assertRaises(ValueError):
                self.main()
        for path in (self.venv / "pyvenv.cfg", self.venv / "bin/python"):
            path.unlink()
            with self.assertRaises(ValueError):
                self.main()
            path.touch()
        self.http.assert_not_called()

    def test_native_success_receipt_is_required_not_recreated(self):
        receipt = self.native / "setup.json"
        receipt.write_text('{"status":"failed"}')
        with self.assertRaisesRegex(ValueError, "successful private native"):
            self.main()
        receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            self.main()
        receipt.symlink_to(self.venv / "pyvenv.cfg")
        with self.assertRaises(ValueError):
            self.main()
        self.http.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_missing_tools_and_existing_foreign_destination_are_untouched(self):
        self.which.side_effect = None
        self.which.return_value = None
        with self.assertRaisesRegex(RuntimeError, "already be installed"):
            self.main()
        self.which.return_value = "/usr/bin/tool"
        foreign = self.base / "unrelated"
        foreign.mkdir()
        self.destination.symlink_to(foreign, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.main()
        self.assertEqual(list(foreign.iterdir()), [])
        self.http.assert_not_called()

    def test_last_bad_hash_stops_before_any_extraction(self):
        pins = (*self.packages[:-1], {**self.packages[-1], "sha256": "0" * 64})
        with mock.patch.dict(self.g, PACKAGES=pins):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch: libusb-1.0-0"):
                self.main()
        self.assertEqual(self.http.call_count, 31)
        self.launch.assert_not_called()
        self.tar_open.assert_not_called()
        self.zip_open.assert_not_called()
        self.assertEqual(len(self.receipt()["verified_sha256"]), 30)
        self.assertEqual(self.receipt()["status"], "failed")

    def test_last_bad_size_stops_before_any_archive_or_deb(self):
        pins = (*self.packages[:-1], {**self.packages[-1], "size": len(self.content) - 1})
        with mock.patch.dict(self.g, PACKAGES=pins):
            with self.assertRaisesRegex(ValueError, "size mismatch: libusb-1.0-0"):
                self.main()
        self.launch.assert_not_called()
        self.tar_open.assert_not_called()
        self.zip_open.assert_not_called()
        self.assertEqual(len(self.receipt()["verified_sha256"]), 30)

    def test_native_runtime_symlink_is_not_trusted(self):
        foreign = self.base / "foreign-native"
        self.native.rename(foreign)
        self.native.symlink_to(foreign, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "successful private native"):
            self.main()
        self.http.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_bad_size_and_npm_integrity_never_write_unverified_bytes(self):
        cache = self.base / "download-cache"
        cache.mkdir()
        for pin, message in (
            ({**self.fake_assets[0], "size": len(self.content) - 1}, "size mismatch"),
            ({**self.fake_assets[0], "size": len(self.content) + 1}, "size mismatch"),
            ({**self.fake_assets[0], "sha256": "0" * 64}, "SHA256 mismatch"),
            ({**self.fake_assets[1], "integrity": "sha512-bad"}, "SHA512 mismatch"),
        ):
            with self.subTest(message=message, pin=pin):
                with self.assertRaisesRegex(ValueError, message):
                    self.g["_fetch"](pin, cache)
                self.assertEqual(list(cache.iterdir()), [])
        self.launch.assert_not_called()

    def test_http_is_bounded_https_only_and_checks_final_url(self):
        cache = self.base / "download-cache"
        cache.mkdir()
        pin = self.fake_assets[0]
        with self.assertRaisesRegex(ValueError, "requires HTTPS"):
            self.g["_fetch"]({**pin, "url": "http://invalid"}, cache)
        self.http.assert_not_called()
        response = self.download("http://downgrade.invalid/archive", 30)
        self.http.side_effect = None
        self.http.return_value = response
        with self.assertRaisesRegex(ValueError, "outside HTTPS"):
            self.g["_fetch"](pin, cache)
        response.read.assert_not_called()
        response.geturl.return_value = "https://cdn.example/verified-redirect"
        self.g["_fetch"](pin, cache)
        response.read.assert_called_once_with(pin["size"] + 1)
        self.http.assert_called_with(pin["url"], timeout=30)

    def test_downgrade_redirect_is_rejected_before_following_it(self):
        handler = self.g["_HTTPSRedirect"]()
        with self.assertRaisesRegex(ValueError, "outside HTTPS"):
            handler.redirect_request(None, None, 302, "Found", {}, "http://downgrade.invalid/")
        self.http.assert_not_called()

    def test_https_redirect_preserves_standard_handler_behavior(self):
        request = self.g["urllib"].request.Request("https://origin.example/archive")
        redirect = self.g["_HTTPSRedirect"]().redirect_request(
            request, None, 302, "Found", {}, "https://cdn.example/archive"
        )
        self.assertEqual(redirect.full_url, "https://cdn.example/archive")

    def test_timeout_not_retried_and_partial_receipt_retained(self):
        self.http.side_effect = TimeoutError("original download deadline")
        with self.assertRaisesRegex(TimeoutError, "original download deadline"):
            self.main()
        self.assertEqual(self.http.call_count, 1)
        self.assertEqual(self.receipt()["status"], "failed")
        self.assertIn("original download deadline", self.receipt()["error"])
        self.launch.assert_not_called()

    def test_tar_unsafe_paths_and_types_rejected_before_extraction(self):
        root = self.g["NODE_ROOT"]
        for name, kind, link in (
            ("/absolute", tarfile.REGTYPE, ""),
            (root + "/../escape", tarfile.REGTYPE, ""),
            ("different-root/file", tarfile.REGTYPE, ""),
            (root + "/bad\\path", tarfile.REGTYPE, ""),
            (root + "/hard", tarfile.LNKTYPE, root + "/bin/node"),
            (root + "/fifo", tarfile.FIFOTYPE, ""),
            (root + "/device", tarfile.CHRTYPE, ""),
            (root + "/link", tarfile.SYMTYPE, "/etc/passwd"),
            (root + "/link", tarfile.SYMTYPE, "../../escape"),
        ):
            with self.subTest(name=name, kind=kind, link=link):
                member = tarfile.TarInfo(name)
                member.type, member.linkname = kind, link
                self.tar_members[self.assets[0]["filename"]] = [member]
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    self.g["_extract_tar"](
                        Path(self.assets[0]["filename"]), self.base / "bad-tar", root
                    )
                self.assertFalse((self.base / "bad-tar").exists())

    def test_npm_strip_rejects_file_or_symlink_package_root(self):
        for kind in (tarfile.REGTYPE, tarfile.SYMTYPE):
            member = tarfile.TarInfo("package")
            member.type, member.linkname = kind, "package"
            self.tar_members[self.assets[1]["filename"]] = [member]
            with self.assertRaisesRegex(ValueError, "root"):
                self.g["_extract_tar"](
                    Path(self.assets[1]["filename"]),
                    self.base / "bad-package",
                    "package",
                    strip=True,
                )
        self.assertFalse((self.base / "bad-package").exists())

    def test_zip_traversal_symlinks_and_limits_rejected_before_extraction(self):
        root = self.g["CHROME_ROOT"]
        for name, mode, size, count in (
            ("/absolute", stat.S_IFREG, 0, 1),
            (root + "/../escape", stat.S_IFREG, 0, 1),
            (root + "/bad\\path", stat.S_IFREG, 0, 1),
            (root + "/link", stat.S_IFLNK, 0, 1),
            (root + "/file", stat.S_IFREG, 600 * 1024**2 + 1, 1),
            (root + "/file", stat.S_IFREG, 0, 1001),
        ):
            with self.subTest(name=name, mode=mode, size=size, count=count):
                member = zipfile.ZipInfo(name)
                member.external_attr = mode << 16
                member.file_size = size
                self.zip_members = [member] * count
                with self.assertRaisesRegex(ValueError, "unsafe|limits"):
                    self.g["_extract_chrome"](self.base / "archive.zip", self.base / "bad-zip")
                self.assertFalse((self.base / "bad-zip").exists())

    def test_dpkg_failure_preserves_primary_and_stops_before_other_archives(self):
        failure = subprocess.CalledProcessError(2, ["dpkg-deb"], stderr="original extract error")
        self.launch.side_effect = failure
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.main()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.launch.call_count, 1)
        self.tar_open.assert_not_called()
        self.assertEqual(self.receipt()["status"], "failed")

    def test_ldd_missing_library_and_nonzero_status_retain_raw_output(self):
        original = self.command

        def command(argv, **kwargs):
            if argv[0] == "ldd":
                return SimpleNamespace(
                    returncode=7, stdout="libmissing.so => not found\nraw error\n"
                )
            return original(argv, **kwargs)

        self.launch.side_effect = command
        before = dict(os.environ)
        with self.assertRaisesRegex(RuntimeError, "libmissing.so => not found"):
            self.main()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(self.launch.call_count, 28)
        self.assertEqual(self.receipt()["ldd"][0]["exit"], 7)
        self.assertEqual(
            self.receipt()["ldd"][0]["stdout"], "libmissing.so => not found\nraw error\n"
        )

    def test_zero_exit_missing_library_still_fails(self):
        original = self.command
        self.launch.side_effect = lambda argv, **kwargs: (
            SimpleNamespace(returncode=0, stdout="libmissing.so => not found\n")
            if argv[0] == "ldd"
            else original(argv, **kwargs)
        )
        with self.assertRaisesRegex(RuntimeError, "ldd failed"):
            self.main()
        self.assertEqual(self.receipt()["status"], "failed")

    def test_nonzero_exit_without_missing_library_still_fails(self):
        original = self.command
        self.launch.side_effect = lambda argv, **kwargs: (
            SimpleNamespace(returncode=7, stdout="original loader diagnostic\n")
            if argv[0] == "ldd"
            else original(argv, **kwargs)
        )
        with self.assertRaisesRegex(RuntimeError, "original loader diagnostic"):
            self.main()
        self.assertEqual(self.receipt()["ldd"][0]["exit"], 7)

    def test_non_elf_never_reaches_ldd(self):
        original = self.zip_source

        def zip_source(archive):
            source = original(archive)
            extract = source.extractall.side_effect

            def non_elf(destination):
                extract(destination)
                (destination / self.g["CHROME_ROOT"] / "chrome-headless-shell").write_bytes(b"bad")

            source.extractall.side_effect = non_elf
            return source

        self.zip_open.side_effect = zip_source
        with self.assertRaisesRegex(ValueError, "verified ELF"):
            self.main()
        self.assertEqual(len(self.receipt()["ldd"]), 1)

    def test_receipt_failure_preserves_original_error_and_prints_marker(self):
        self.http.side_effect = TimeoutError("original download deadline")
        original = Path.open

        def open_file(path, *args, **kwargs):
            if path == self.destination / "setup.json":
                raise OSError("receipt unavailable")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=open_file):
            with self.assertRaisesRegex(TimeoutError, "original download deadline") as caught:
                self.main()
        self.assertIn(
            "Browser setup receipt also failed: receipt unavailable", caught.exception.__notes__
        )
        self.assertIn("NEXUML_BROWSER_RUNTIME_SETUP=", self.output.getvalue())
        self.assertIn('"status": "failed"', self.output.getvalue())

    def test_receipt_failure_after_setup_is_not_success(self):
        original = Path.open

        def open_file(path, *args, **kwargs):
            if path == self.destination / "setup.json":
                raise OSError("receipt unavailable")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=open_file):
            with self.assertRaisesRegex(OSError, "receipt unavailable"):
                self.main()
        self.assertIn('"status": "failed"', self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
