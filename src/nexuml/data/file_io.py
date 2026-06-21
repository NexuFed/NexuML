"""File download utilities for NexuML datasets."""

import requests
import os
from pathlib import Path
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from loguru import logger
import zipfile
import time


def _acquire_download_lock(lock_path: Path):
    while True:
        try:
            return lock_path.open("x")
        except FileExistsError:
            time.sleep(1)


def _download_file(url: str, dest: Path) -> None:
    """Download a file from a URL to the destination path.

    Raises:
        ValueError: If the server returns a non-200 status code.
        zipfile.BadZipFile: If the downloaded zip archive has a bad CRC entry.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest.with_suffix(dest.suffix + ".lock")
    tmp_dest = dest.with_name(f"{dest.name}.{os.getpid()}.{time.time_ns()}.part")
    lock = _acquire_download_lock(lock_path)
    lock.close()
    try:
        if dest.exists():
            logger.info(f"Using existing {dest}")
            return
        with requests.get(url, stream=True) as response:
            if response.status_code == 200:
                total = int(response.headers.get("content-length") or 0)
                with open(tmp_dest, "wb") as f:
                    with Progress(
                        TextColumn("{task.description}"),
                        BarColumn(),
                        DownloadColumn(),
                        TransferSpeedColumn(),
                        TimeRemainingColumn(),
                    ) as progress:
                        task = progress.add_task(f"Downloading {url}", total=total)
                        for chunk in response.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
                if dest.suffix == ".zip":
                    with zipfile.ZipFile(tmp_dest, "r") as zip_ref:
                        bad_member = zip_ref.testzip()
                    if bad_member is not None:
                        raise zipfile.BadZipFile(f"Bad CRC-32 for file {bad_member!r}")
                tmp_dest.replace(dest)
                logger.info(f"Downloaded {url} to {dest}")
            else:
                raise ValueError(f"Failed to download {url}: Status code {response.status_code}")
    finally:
        tmp_dest.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def _unzip_file(zip_path: Path, extract_to: Path) -> None:
    """Unzip a file to the specified directory."""
    extract_to.mkdir(parents=True, exist_ok=True)

    # Assuming that the zip file is not too large, we can use the built-in
    # zipfile module. For larger files, consider using a streaming unzip approach.
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Extracted {zip_path} to {extract_to}")
