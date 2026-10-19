"""Verified official Zenodo archive download for MIMII acoustic machine sounds."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nexuml_library.data.maintenance.bearings import _safe_extract

_RECORD_URL = "https://zenodo.org/records/3384388"
_MD5: dict[str, str] = {
    "-6_dB_fan.zip": "f02ae808a58d84b6815b7ec38ff30879",
    "-6_dB_pump.zip": "d20b783a0ff9c93d58f452f98c37b112",
    "-6_dB_slider.zip": "49913eda7d37f182cbf8ed5c984140e0",
    "-6_dB_valve.zip": "fdfaf185fea61b21e11952a070a4ada7",
    "0_dB_fan.zip": "6354d1cc2165c52168f9ef1bcd9c7c52",
    "0_dB_pump.zip": "488748295c3f60b25de07b58fe75b049",
    "0_dB_slider.zip": "4d674c21474f0646ecd75546db6c0c4e",
    "0_dB_valve.zip": "178478eb0d11c79080a35562bfdeee71",
    "6_dB_fan.zip": "0890f7d3c2fd8448634e69ff1d66dd47",
    "6_dB_pump.zip": "a09ba6060c10fc09cd4c8770213b0b9f",
    "6_dB_slider.zip": "838c2b3441858359c4704ef13a1b27ff",
    "6_dB_valve.zip": "fe5fb7c337cd701b1d31dc641e621892",
}


def _verify_md5(archive: Path, expected: str) -> bool:
    """Check Zenodo's published checksum without loading the archive in memory.

    Returns:
        True when the local archive exactly matches its published checksum.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def _download_archive(url: str, archive: Path, expected: str) -> None:
    """Download one large official archive with an atomic cache update.

    Returns:
        None after checking the downloaded archive's Zenodo checksum.

    Raises:
        RuntimeError: If the download fails or its checksum is invalid.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_file():
        if _verify_md5(archive, expected):
            return
        raise RuntimeError(
            f"MIMII cached archive has an invalid MD5: {archive}. "
            "Delete the damaged archive and retry."
        )

    partial = archive.with_name(f"{archive.name}.part")
    if partial.exists():
        partial.unlink()
    digest = hashlib.md5(usedforsecurity=False)
    request = Request(url, headers={"User-Agent": "NexuML-MIMII/1.0"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as target:
            while chunk := response.read(8 * 1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(
                f"MIMII archive checksum mismatch for {archive.name}: "
                f"expected {expected}, received {digest.hexdigest()}"
            )
        partial.replace(archive)
    except (OSError, HTTPError, URLError, RuntimeError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Cannot download verified MIMII archive from {url}: {exc}. "
            "You can download the same ZIP from Zenodo and place it at "
            f"{archive}, or set download=False and use extracted recordings."
        ) from exc


def _has_machine_ids(root: Path, machine_type: str) -> bool:
    """Identify the expected source layout without scanning file contents.

    Returns:
        True if a MIMII machine directory with IDs already exists.
    """
    return any(candidate.is_dir() and any(candidate.glob("id_*")) for candidate in
               (root, root / machine_type))


def ensure_mimii(
    root: Path,
    machine_type: str,
    noise_db: int,
    download: bool,
) -> None:
    """Auto-download and safely extract a verified MIMII source archive.

    Existing extracted recordings always take precedence. For a missing
    source, download only the selected machine/noise combination, not the
    entire MIMII corpus. Stream to a checksum-verified cache, extract into a
    separate directory, then publish the completed directory atomically.

    Raises:
        FileNotFoundError: If recordings are missing and download is disabled.
        RuntimeError: If the verified download or extraction cannot complete.
        ValueError: If the archive contains unsafe entries.
    """
    if _has_machine_ids(root, machine_type):
        return
    if not download:
        raise FileNotFoundError(
            f"No MIMII {machine_type} recordings under {root}. "
            f"Set download=True to fetch them from {_RECORD_URL}."
        )

    filename = f"{noise_db}_dB_{machine_type}.zip"
    checksum = _MD5[filename]
    archive = root.parent / filename
    url = f"{_RECORD_URL}/files/{filename}?download=1"
    _download_archive(url, archive, checksum)

    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"Cannot safely auto-extract MIMII into non-empty {root} because "
            "no valid machine-ID layout was found. Check or move that directory."
        )

    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}-extract-", dir=root.parent
    ) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        try:
            _safe_extract(archive, staging)
        except (zipfile.BadZipFile, OSError) as exc:
            raise RuntimeError(f"Failed to extract MIMII archive {archive}: {exc}") from exc
        if not _has_machine_ids(staging, machine_type):
            raise RuntimeError(
                f"Verified MIMII archive {archive.name} does not contain "
                f"{machine_type}/id_* directories; inspect the archive layout."
            )
        if root.is_dir():
            root.rmdir()
        staging.rename(root)
