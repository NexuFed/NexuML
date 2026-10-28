"""Fixed job-private browser dependencies, not browser or visual acceptance.

Ubuntu pins were selected from signed Jammy indexes. Node uses its official
SHASUMS, npm uses retained SHA512 integrity, Chrome uses an official TLS download
SHA256 (not a PGP signature). The caller applies the receipt's private paths.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import NotRequired, TypedDict
import urllib.request
from xml.sax.saxutils import escape
import zipfile


class _Pin(TypedDict):
    name: str
    version: str
    filename: str
    url: str
    sha256: str
    size: int
    integrity: NotRequired[str]


ENV_BASE = Path("/tmp/ray/nexuml-envs")
ARCHIVE = "https://archive.ubuntu.com/ubuntu/"
NODE_ROOT = "node-v22.23.4-linux-x64"
CHROME_ROOT = "chrome-headless-shell-linux64"
ASSETS: tuple[_Pin, ...] = (
    {
        "name": "node",
        "version": "22.23.4",
        "filename": "node-v22.23.4-linux-x64.tar.xz",
        "url": "https://nodejs.org/dist/v22.23.4/node-v22.23.4-linux-x64.tar.xz",
        "sha256": "1805bcd35e6815559bd2eb97db438d3fdb0eb3a5e6a977b7c1badca76337e9ce",
        "size": 31029996,
    },
    {
        "name": "playwright",
        "version": "1.63.0",
        "filename": "playwright-1.63.0.tgz",
        "url": "https://registry.npmjs.org/playwright/-/playwright-1.63.0.tgz",
        "sha256": "195a5ee9bfed7c6e9c03965950e32e5e4dbedaa02e740e5a530eb4b87dae050c",
        "integrity": (
            "sha512-+7ziBLidS4NaNCdt57SUDT+wYmmd5fmiQejUic/kb+YsYSCPyOOE9sebzMjNmQrs"
            "nNpDJqd4WHvV/8lfKfUDUg=="
        ),
        "size": 899597,
    },
    {
        "name": "playwright-core",
        "version": "1.63.0",
        "filename": "playwright-core-1.63.0.tgz",
        "url": "https://registry.npmjs.org/playwright-core/-/playwright-core-1.63.0.tgz",
        "sha256": "208593d4e1bcd8f8fe5f869cad1cc332dc7f1d70dc1d58c102dc3ac36e30f26c",
        "integrity": (
            "sha512-rYCsBF/M5HjUch52bbtVONEFjv6Xu8sm8h72dNlR5bzIE1fvC/bxgspzkjSfU+Mw"
            "eEMmPM8KJebG6nnyxo5mCg=="
        ),
        "size": 3123044,
    },
    {
        "name": "chrome-headless-shell",
        "version": "153.0.8010.12",
        "filename": "chrome-headless-shell-linux64.zip",
        "url": (
            "https://cdn.playwright.dev/builds/cft/153.0.8010.12/"
            "linux64/chrome-headless-shell-linux64.zip"
        ),
        "sha256": "a9da028861a0cf789ff25c2fed45f5f1aaf969ed9247835b6a7821a4f7af9d1d",
        "size": 119809080,
    },
)
CHROME_REVISION = "1243"
PACKAGES: tuple[_Pin, ...] = tuple(
    {
        "name": name,
        "version": version,
        "filename": filename,
        "url": ARCHIVE + filename,
        "sha256": sha256,
        "size": size,
    }
    for name, version, filename, sha256, size in (
        (
            "libglib2.0-0",
            "2.72.4-0ubuntu2.10",
            "pool/main/g/glib2.0/libglib2.0-0_2.72.4-0ubuntu2.10_amd64.deb",
            "b82973c5d9698e48297000e74d2dc7431e726db8f0f948ea56be7f9e4c9e66fb",
            1468764,
        ),
        (
            "libnspr4",
            "2:4.35-0ubuntu0.22.04.1",
            "pool/main/n/nspr/libnspr4_4.35-0ubuntu0.22.04.1_amd64.deb",
            "b3c96e4a61675c87f8d9655109346748847d859abc95f20493159d06b5aa30ef",
            119208,
        ),
        (
            "libnss3",
            "2:3.98-0ubuntu0.22.04.5",
            "pool/main/n/nss/libnss3_3.98-0ubuntu0.22.04.5_amd64.deb",
            "75f3879ab9969a930b5bf4b9229345d57e2ebe310ad8ce58c9d91393d823b236",
            1345528,
        ),
        (
            "libatk1.0-0",
            "2.36.0-3build1",
            "pool/main/a/atk1.0/libatk1.0-0_2.36.0-3build1_amd64.deb",
            "598b522b64a20c9a3ea44fa816b539976c39d41bf2bccd104fed49954638f0f9",
            51938,
        ),
        (
            "libatk-bridge2.0-0",
            "2.38.0-3",
            "pool/main/a/at-spi2-atk/libatk-bridge2.0-0_2.38.0-3_amd64.deb",
            "98924a5b376da9c075663ea61721fb0d91a55be3473c4643286491247f229b48",
            66556,
        ),
        (
            "libatspi2.0-0",
            "2.44.0-3",
            "pool/main/a/at-spi2-core/libatspi2.0-0_2.44.0-3_amd64.deb",
            "93b243e9d8fb3de9ba78251cf3dec2c6b66836ea7329ae4026981ce8dda62de4",
            80886,
        ),
        (
            "libdbus-1-3",
            "1.12.20-2ubuntu4.1",
            "pool/main/d/dbus/libdbus-1-3_1.12.20-2ubuntu4.1_amd64.deb",
            "6a5171696c3b05aca273d637d9c0501c8033fda64a971e09fd545bc1ab72ca1e",
            188562,
        ),
        (
            "libxcomposite1",
            "1:0.4.5-1build2",
            "pool/main/libx/libxcomposite/libxcomposite1_0.4.5-1build2_amd64.deb",
            "1e3d7e7b53149f6fff92e1fbf4203b61301ca4e251b2949a34bf788dd999621f",
            7192,
        ),
        (
            "libxdamage1",
            "1:1.1.5-2build2",
            "pool/main/libx/libxdamage/libxdamage1_1.1.5-2build2_amd64.deb",
            "a705af86334b9c219e5f50e21e5bc51bfbf8b8189ca88d4fcafff734de81a503",
            7154,
        ),
        (
            "libgbm1",
            "23.2.1-1ubuntu3.1~22.04.4",
            "pool/main/m/mesa/libgbm1_23.2.1-1ubuntu3.1~22.04.4_amd64.deb",
            "10d37ffd63f8f98fc9a1ad5b32abc7a9da760f4ed85c806d98f10a61c17f8ef5",
            33678,
        ),
        (
            "libxkbcommon0",
            "1.4.0-1",
            "pool/main/libx/libxkbcommon/libxkbcommon0_1.4.0-1_amd64.deb",
            "a0470edb237d52b71ad41f755c043164abed01423585ed5ea19525b3baaf84f7",
            124684,
        ),
        (
            "libasound2",
            "1.2.6.1-1ubuntu1.2",
            "pool/main/a/alsa-lib/libasound2_1.2.6.1-1ubuntu1.2_amd64.deb",
            "dce4ce1043cde35f4bc375a4b1bd84badd5dec7e656ea2a8849edf9215bc8c33",
            390788,
        ),
        (
            "libasound2-data",
            "1.2.6.1-1ubuntu1.2",
            "pool/main/a/alsa-lib/libasound2-data_1.2.6.1-1ubuntu1.2_all.deb",
            "e175a0617ae4a35667a9cf159ee5ee226d09ad43ed7719b721a7ea797974f65c",
            19318,
        ),
        (
            "libdrm2",
            "2.4.113-2~ubuntu0.22.04.1",
            "pool/main/libd/libdrm/libdrm2_2.4.113-2~ubuntu0.22.04.1_amd64.deb",
            "555925f1caedfd98d89bae3bc11b4fe3401c2c4817dbbafcab25e1144582e946",
            38106,
        ),
        (
            "libdrm-common",
            "2.4.113-2~ubuntu0.22.04.1",
            "pool/main/libd/libdrm/libdrm-common_2.4.113-2~ubuntu0.22.04.1_all.deb",
            "35a306712d8b15b30c42ecd73ec087813eb01c0b3125dc8f7ca2b5134e133522",
            5450,
        ),
        (
            "libwayland-server0",
            "1.20.0-1ubuntu0.1",
            "pool/main/w/wayland/libwayland-server0_1.20.0-1ubuntu0.1_amd64.deb",
            "b5d514dc17edac6b93ba2476d90806337ca4afcc914952a7e78af2e5cd0c4a7b",
            34268,
        ),
        (
            "libxcb-randr0",
            "1.14-3ubuntu3",
            "pool/main/libx/libxcb/libxcb-randr0_1.14-3ubuntu3_amd64.deb",
            "39dc0f73fa3bd43f3f7755cb7d12bfd1cd59dd8966f747d66fba47dfc8652f62",
            18300,
        ),
        (
            "libxi6",
            "2:1.8-1build1",
            "pool/main/libx/libxi/libxi6_1.8-1build1_amd64.deb",
            "47662736cca4be456fd42a5f0a9d3bed8a5cfad4ee08b809a95dde6004daf275",
            32572,
        ),
        (
            "libatk1.0-data",
            "2.36.0-3build1",
            "pool/main/a/atk1.0/libatk1.0-data_2.36.0-3build1_all.deb",
            "a023521fb4930ba6a1f0c5a72a0d1c575b881ce1e117893621cab74e4bd0a60e",
            2824,
        ),
        (
            "xkb-data",
            "2.33-1",
            "pool/main/x/xkeyboard-config/xkb-data_2.33-1_all.deb",
            "26f74d6ac0caf3f7b2926d9c1fca85bc868531c8159f943fb78fbb4b6c5a8c80",
            394460,
        ),
        (
            "libfontconfig1",
            "2.13.1-4.2ubuntu5",
            "pool/main/f/fontconfig/libfontconfig1_2.13.1-4.2ubuntu5_amd64.deb",
            "6be0c141aed23d9eb801658d248fd667bea2b8a9ccab4db7b596a69ffe3393fd",
            131434,
        ),
        (
            "fontconfig-config",
            "2.13.1-4.2ubuntu5",
            "pool/main/f/fontconfig/fontconfig-config_2.13.1-4.2ubuntu5_all.deb",
            "c40405fa036d1732e269fabd4c5b40076b973a226da1df6921d4b7e7a1e2ff72",
            29096,
        ),
        (
            "libfreetype6",
            "2.11.1+dfsg-1ubuntu0.4",
            "pool/main/f/freetype/libfreetype6_2.11.1+dfsg-1ubuntu0.4_amd64.deb",
            "3728210ef44a7a51b3824d5c499f6fbd428584a42a65aee15794a4264c8c173a",
            388210,
        ),
        (
            "libpng16-16",
            "1.6.37-3ubuntu0.7",
            "pool/main/libp/libpng1.6/libpng16-16_1.6.37-3ubuntu0.7_amd64.deb",
            "f0e18c46205e6b1bf9eb20ed97037c128f03cf27c1ff818f442a9cc173f8a002",
            192392,
        ),
        (
            "libbrotli1",
            "1.0.9-2build6",
            "pool/main/b/brotli/libbrotli1_1.0.9-2build6_amd64.deb",
            "8e6aa2dfd44df287b9fb227862f3b6062ec074dd39a247874edfe12e1dcc3841",
            315164,
        ),
        (
            "fonts-liberation",
            "1:1.07.4-11",
            "pool/main/f/fonts-liberation/fonts-liberation_1.07.4-11_all.deb",
            "d359cba9c3ac8a40fb57773881e20827845e9fab1b2d6fd25cde4b7bade7d57f",
            821564,
        ),
        (
            "libusb-1.0-0",
            "2:1.0.25-1ubuntu2",
            "pool/main/libu/libusb-1.0/libusb-1.0-0_1.0.25-1ubuntu2_amd64.deb",
            "99b98c9cc9c88c4d7f50e106e9f0cbff868c5d14809f516b1947d3d2c4638aae",
            52688,
        ),
    )
)
# libgbm1 is a GBM client from Mesa source, not a DRI/ICD renderer/driver.
# Extract selected clients/data only; this is deliberately not an APT closure.


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise ValueError("download redirected outside HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(pin: _Pin, cache: Path) -> str:
    url = pin["url"]
    if not url.startswith("https://"):
        raise ValueError("download requires HTTPS")
    opener = urllib.request.build_opener(_HTTPSRedirect())
    with opener.open(url, timeout=30) as response:
        if not response.geturl().startswith("https://"):
            raise ValueError("download redirected outside HTTPS")
        content = response.read(pin["size"] + 1)
    if len(content) != pin["size"]:
        raise ValueError(f"size mismatch: {pin['name']}")
    digest = hashlib.sha256(content).hexdigest()
    if digest != pin["sha256"]:
        raise ValueError(f"SHA256 mismatch: {pin['name']}")
    if "integrity" in pin:
        integrity = "sha512-" + base64.b64encode(hashlib.sha512(content).digest()).decode()
        if integrity != pin["integrity"]:
            raise ValueError(f"SHA512 mismatch: {pin['name']}")
    with (cache / Path(pin["filename"]).name).open("xb") as output:
        output.write(content)
    return digest


def _archive_path(name: str, root: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or not path.parts
        or path.parts[0] != root
    ):
        raise ValueError(f"unsafe archive path: {name}")
    return path


def _extract_tar(archive: Path, destination: Path, root: str, *, strip: bool = False):
    with tarfile.open(archive) as source:
        members = source.getmembers()
        selected = []
        for member in members:
            path = _archive_path(member.name, root)
            if not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError(f"unsafe tar entry type: {member.name}")
            if member.issym():
                target = posixpath.normpath(str(path.parent / member.linkname))
                _archive_path(target, root)
                if strip and path == PurePosixPath(root):
                    raise ValueError("archive root cannot be a symlink")
            if strip:
                if len(path.parts) == 1:
                    if not member.isdir():
                        raise ValueError("archive root must be a directory")
                    continue
                member = copy.copy(member)
                member.name = str(PurePosixPath(*path.parts[1:]))
            selected.append(member)
        destination.mkdir(mode=0o700)
        source.extractall(destination, members=selected, filter="data")


def _extract_chrome(archive: Path, destination: Path):
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) > 1000 or sum(item.file_size for item in members) > 600 * 1024**2:
            raise ValueError("Chrome zip exceeds extraction limits")
        for member in members:
            _archive_path(member.filename, CHROME_ROOT)
            kind = stat.S_IFMT(member.external_attr >> 16)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"unsafe zip entry type: {member.filename}")
        destination.mkdir(mode=0o700)
        source.extractall(destination)
    binary = destination / CHROME_ROOT / "chrome-headless-shell"
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)


def main():
    """Prepare fixed dependencies exclusively inside the job's existing venv.

    Raises:
        ValueError: A private path, fixed lock, asset or ELF validation fails.
        RuntimeError: The worker prerequisites or actual ELF closure are missing.
    """
    destination = None
    record = {
        "status": "failed",
        "packages": PACKAGES,
        "assets": ASSETS,
        "chrome_revision": CHROME_REVISION,
        "verified_sha256": {},
        "ldd": [],
        "paths": {},
    }
    try:
        if len(sys.argv) != 1:
            raise ValueError("browser setup accepts no arguments")
        release = platform.freedesktop_os_release()
        if (
            sys.platform != "linux"
            or release.get("ID") != "ubuntu"
            or release.get("VERSION_ID") != "22.04"
            or platform.machine() != "x86_64"
        ):
            raise RuntimeError("browser setup requires Ubuntu 22.04 amd64")
        venv = Path(os.environ["UV_PROJECT_ENVIRONMENT"])
        if (
            not venv.is_absolute()
            or venv.parent != ENV_BASE
            or re.fullmatch(r"[0-9a-f]{32}", venv.name) is None
            or venv.resolve(strict=True) != venv
            or not venv.is_dir()
            or venv.stat().st_uid != os.getuid()
            or not (venv / "pyvenv.cfg").is_file()
            or not (venv / "bin/python").is_file()
        ):
            raise ValueError("UV_PROJECT_ENVIRONMENT must be an owned real private SDK venv")
        native = venv / "native-runtime"
        receipt = native / "setup.json"
        if (
            native.resolve(strict=True) != native
            or native.stat().st_uid != os.getuid()
            or receipt.is_symlink()
            or json.loads(receipt.read_text()).get("status") != "success"
        ):
            raise ValueError("successful private native-runtime/setup.json is required")
        pins = (*ASSETS, *PACKAGES)
        if (
            len(ASSETS) != 4
            or len(PACKAGES) != 27
            or len({pin["name"] for pin in pins}) != 31
            or len({Path(pin["filename"]).name for pin in pins}) != 31
        ):
            raise ValueError("browser lock must contain 4 assets and 27 unique packages/files")
        if shutil.which("dpkg-deb") is None or shutil.which("ldd") is None:
            raise RuntimeError("dpkg-deb and ldd must already be installed")
        owned = venv / "browser-runtime"
        owned.mkdir(mode=0o700)
        destination = owned  # Never write receipts into an existing/foreign destination.
        record["directory"] = str(owned)
        cache = owned / "cache"
        cache.mkdir(mode=0o700)
        for pin in pins:
            record["verified_sha256"][pin["name"]] = _fetch(pin, cache)
        # Nothing is extracted until every downloaded asset and deb has verified.
        clients = owned / "clients"
        clients.mkdir(mode=0o700)
        for pin in PACKAGES:
            subprocess.run(
                ["dpkg-deb", "--extract", str(cache / Path(pin["filename"]).name), str(clients)],
                check=True,
                timeout=60,
            )
        _extract_tar(cache / ASSETS[0]["filename"], owned / "node", NODE_ROOT)
        modules = owned / "node_modules"
        modules.mkdir(mode=0o700)
        for pin in ASSETS[1:3]:
            _extract_tar(cache / pin["filename"], modules / pin["name"], "package", strip=True)
        _extract_chrome(cache / ASSETS[3]["filename"], owned / "chrome")
        node = owned / "node" / NODE_ROOT / "bin/node"
        chrome = owned / "chrome" / CHROME_ROOT / "chrome-headless-shell"
        libraries = clients / "usr/lib/x86_64-linux-gnu"
        # Jammy clients use both layouts: libdbus is shipped below /lib, not /usr/lib.
        prefix = ":".join(
            map(
                str,
                (libraries, clients / "lib/x86_64-linux-gnu", native / "usr/lib/x86_64-linux-gnu"),
            )
        )
        home = owned / "home"
        home.mkdir(mode=0o700)
        fontcache = owned / "fontcache"
        fontcache.mkdir(mode=0o700)
        fonts = owned / "fonts.conf"
        fonts.write_text(
            '<?xml version="1.0"?>\n<fontconfig><dir>'
            + escape(str(clients / "usr/share/fonts"))
            + "</dir><cachedir>"
            + escape(str(fontcache))
            + "</cachedir></fontconfig>\n"
        )
        environment = os.environ.copy()
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = prefix + (":" + previous if previous else "")
        record["paths"] = {
            "node": str(node),
            "playwrightModulePath": str(modules / "playwright"),
            "chromeBinary": str(chrome),
            "libraryPrefix": prefix,
            "ldLibraryPath": environment["LD_LIBRARY_PATH"],
            "fontsConf": str(fonts),
            "fontCache": str(fontcache),
            "homePath": str(home),
        }
        binaries = [(node, owned), (chrome, owned)]
        plugin_roots = sorted(venv.glob("lib/python*/site-packages/open3d"))
        plugins = sorted(
            binary
            for package in plugin_roots
            for layout in ("pybind*.so", "cpu/pybind*.so")
            for binary in package.glob(layout)
        )
        if plugin_roots and len(plugins) != 1:
            raise ValueError("missing or ambiguous installed TensorBoard Open3D plugin ABI")
        if plugins:
            binaries.append((plugins[0], venv))
        for binary, boundary in binaries:
            if not binary.resolve(strict=True).is_relative_to(boundary) or not binary.is_file():
                raise ValueError("executable escapes private browser runtime")
            with binary.open("rb") as source:
                if source.read(4) != b"\x7fELF":
                    raise ValueError("ldd requires a verified ELF binary")
            if binary in plugins:
                with binary.open("rb") as source:
                    record["installed_plugin_sha256"] = hashlib.file_digest(
                        source, "sha256"
                    ).hexdigest()
            result = subprocess.run(
                ["ldd", str(binary)],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            record["ldd"].append(
                {"binary": str(binary), "stdout": result.stdout, "exit": result.returncode}
            )
            if result.returncode != 0 or "not found" in result.stdout:
                raise RuntimeError(f"ldd failed ({result.returncode}): {result.stdout}")
        record["status"] = "success"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        primary = sys.exception()
        try:
            if destination is not None:
                with (destination / "setup.json").open("x") as output:
                    json.dump(record, output, indent=2)
                    output.write("\n")
        except Exception as error:
            record["status"] = "failed"
            record["receipt_error"] = f"{type(error).__name__}: {error}"
            if primary is None:
                raise
            primary.add_note(f"Browser setup receipt also failed: {error}")
        finally:
            print("NEXUML_BROWSER_RUNTIME_SETUP=" + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
