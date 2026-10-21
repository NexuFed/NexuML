"""Official NASA C-MAPSS retrieval for NexuML's native RUL data source.

Download the public PCoE archive on demand. Only known C-MAPSS sensor tables
are extracted; the archive is never treated as Python code.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

CMAPSS_ARCHIVE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)


def _required_files(subset: str) -> tuple[str, str, str]:
    """Name the three official tables required to train and evaluate a subset.

    Returns:
        Official NASA C-MAPSS filename triple.

    Raises:
        ValueError: If the subset is not FD001 through FD004.
    """
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unsupported C-MAPSS subset: {subset}")
    return f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"


def _locate_tables(root: Path, subset: str) -> Path | None:
    """Find the three tables in one directory, including existing nested layouts.

    Returns:
        Existing directory containing the selected subset or None.
    """
    names = _required_files(subset)
    if all((root / name).is_file() for name in names):
        return root
    if not root.is_dir():
        return None
    for candidate in sorted(root.rglob(names[0])):
        if all((candidate.parent / name).is_file() for name in names):
            return candidate.parent
    return None


def _download_archive(destination: Path) -> None:
    """Download the official ZIP to an atomic reusable local cache.

    Raises:
        RuntimeError: If the NASA download fails or the archive is invalid.
    """
    if destination.is_file():
        if zipfile.is_zipfile(destination):
            return
        raise RuntimeError(f"Invalid cached NASA C-MAPSS ZIP: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    try:
        request = Request(CMAPSS_ARCHIVE_URL, headers={"User-Agent": "NexuML-RUL/1.0"})
        with urlopen(request, timeout=120) as source, partial.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        if not zipfile.is_zipfile(partial):
            raise RuntimeError("NASA response was not a ZIP archive")
        partial.replace(destination)
    except (OSError, URLError, RuntimeError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Cannot retrieve the official NASA C-MAPSS archive: {CMAPSS_ARCHIVE_URL}"
        ) from exc


def _extract_tables(archive: Path, destination: Path, subset: str) -> None:
    """Extract only named sensor tables, supporting an optional nested ZIP.

    Ignore unrelated entries, disallow symlinks and enforce per-file bounds
    to avoid extracting arbitrary paths or oversized ZIP members.

    Raises:
        ValueError: If the official archive lacks required tables or is unsafe.
    """
    required = set(_required_files(subset))
    found: set[str] = set()

    def scan(file: Path, depth: int) -> None:
        with zipfile.ZipFile(file) as zipped:
            for member in zipped.infolist():
                if member.is_dir():
                    continue
                filename = Path(member.filename).name
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError(f"Archive symlinks are not permitted: {member.filename}")
                if filename in required:
                    if filename in found:
                        raise ValueError(f"Duplicate C-MAPSS table: {filename}")
                    if member.file_size > 256 * 1024 * 1024:
                        raise ValueError(f"Oversized C-MAPSS table: {filename}")
                    with zipped.open(member) as source, (destination / filename).open("wb") as out:
                        shutil.copyfileobj(source, out, length=1024 * 1024)
                    found.add(filename)
                elif depth == 0 and filename.lower().endswith(".zip"):
                    if member.file_size > 512 * 1024 * 1024:
                        raise ValueError("Oversized nested C-MAPSS archive")
                    nested = destination / "nested.zip"
                    with zipped.open(member) as source, nested.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    try:
                        scan(nested, 1)
                    finally:
                        nested.unlink(missing_ok=True)

    scan(archive, 0)
    if found != required:
        raise ValueError(f"NASA archive is missing tables: {sorted(required - found)}")


def ensure_cmapss(root: str | Path, subset: str, *, download: bool = True) -> Path:
    """Find or automatically retrieve official C-MAPSS tables.

    Reuse existing extracted files and the cached ZIP. Publish only validated
    required tables, without overwriting existing user data.

    Args:
        root: Target dataset cache directory.
        subset: One of FD001, FD002, FD003 and FD004.
        download: Automatically retrieve the official archive when needed.

    Returns:
        Directory containing the selected official source tables.

    Raises:
        FileNotFoundError: If source files are missing and download is disabled.
        RuntimeError: If the official download fails.
        ValueError: If the downloaded source archive is unsafe or incomplete.
    """
    _required_files(subset)
    root = Path(root).expanduser().resolve()
    available = _locate_tables(root, subset)
    if available is not None:
        return available
    if not download:
        raise FileNotFoundError(
            f"C-MAPSS {subset} is missing under {root}. "
            f"Enable download or obtain it from {CMAPSS_ARCHIVE_URL}"
        )

    archive = root.parent / "cmapss-nasa-official.zip"
    _download_archive(archive)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cmapss-prepare-", dir=root.parent) as tmp:
        staging = Path(tmp)
        try:
            _extract_tables(archive, staging, subset)
        except zipfile.BadZipFile as exc:
            raise ValueError("Invalid NASA C-MAPSS archive structure") from exc
        for filename in _required_files(subset):
            target = root / filename
            if not target.exists():
                (staging / filename).replace(target)
    return root


def cmapss_source_sha256(root: Path) -> str | None:
    """Return the cached archive digest for reproducibility metadata.

    Returns:
        SHA256 digest of the NASA ZIP, or None for preexisting local tables.
    """
    archive = root.parent / "cmapss-nasa-official.zip"
    if not archive.is_file():
        return None
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
