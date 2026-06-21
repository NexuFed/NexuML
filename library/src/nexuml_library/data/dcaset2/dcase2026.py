"""DCASE 2026 dataset sources for Tasks 1, 2, and 7.

All datasets are metadata-backed and fake-fixture friendly: they accept
pre-built DataFrames and tiny local audio files for testing without
requiring network downloads.
"""

from __future__ import annotations

from nexuml.core.discovery import data_source

import hashlib
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Literal, Union
from urllib.parse import urlparse
from urllib.request import urlopen
import wave

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
from tensordict import TensorDict

from nexuml.data.dataset import NexuDataset
from nexuml.data.file_io import _download_file, _unzip_file

logger = logging.getLogger(__name__)

DownloadMode = Literal["disabled", "dry_run", "manual", "already_downloaded", "download"]


# ---------------------------------------------------------------------------
# Legacy multi-year DCASE Task 2 dataset (2020-2025) and shared helpers
# ---------------------------------------------------------------------------


def _default_download_manifest() -> Path:
    return Path(__file__).resolve().parent / "dcase_zenodo.yaml"


def _parse_DCASE_identifier(identifier: str) -> dict:
    """Parse DCASE identifier string into components.

    Returns:
        dict: Parsed components with ``task`` and ``year`` keys.

    Raises:
        ValueError: If the identifier does not match the expected pattern.
    """
    match = re.match(
        r"([A-Za-z]+)(\d+)(.*)",
        identifier,
    )
    if match is None:
        raise ValueError(f"Invalid DCASE identifier: {identifier}")
    match = match.groups()
    return {"task": match[2], "year": match[1]}


def _resolve_dataset_dir(root: Path, dataset_name: str) -> Path:
    dataset_dir = root / dataset_name.lower()
    if dataset_dir.exists():
        return dataset_dir
    if root.name.lower() == dataset_name.lower():
        return root
    return dataset_dir


def _load_waveform(file_path: Path) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(file_path), normalize=True)
        return waveform, sample_rate
    except Exception:
        pass

    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(file_path), always_2d=True, dtype="float32")
        return torch.from_numpy(data.T.copy()), int(sample_rate)
    except Exception:
        pass

    with wave.open(str(file_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    dtype_map = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    waveform = np.frombuffer(frames, dtype=dtype_map[sample_width])
    if n_channels > 1:
        waveform = waveform.reshape(-1, n_channels).T
    else:
        waveform = waveform.reshape(1, -1)

    if sample_width == 1:
        waveform = (waveform.astype(np.float32) - 128.0) / 128.0
    else:
        waveform = waveform.astype(np.float32) / float(2 ** (8 * sample_width - 1))

    return torch.from_numpy(waveform.copy()), int(sample_rate)


def _valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


@data_source("DCASET2Dataset")
class DCASET2Dataset(NexuDataset):
    """DCASE Task 2 anomalous sound detection dataset.

    Builds a metadata DataFrame from the DCASE directory structure and
    lazily loads audio/features in load_item().

    Directory layout expected:
      <root>/<dataset_name>/<data_type>_data/raw/<machine_type>/train/
      <root>/<dataset_name>/<data_type>_data/raw/<machine_type>/test/

    Label columns: machine (int index), y_true (0=normal,1=anomalous),
                   condition (int), target (0/1).

    Args:
        root: Path to the dataset root directory.
        dataset_name: e.g. "DCASE2024T2"
        machine_type: e.g. "ToyCar", "fan", etc.
        section_keyword: prefix for section names, e.g. "section"
        section_ids: list of section id strings, e.g. ["00", "01"]
        data_type: "dev" or "eval"
        train: if True load train split, else test split
        sample_rate: target sample rate for audio loading
        machine_types: list of all machine types (for machine index mapping)
    """

    LABEL_NAMES = ["y_true", "machine", "target", "section"]

    def __init__(
        self,
        root: Union[str, Path] | None = None,
        data_root: Union[str, Path] | None = None,
        dataset_name: str = "DCASE2024T2",
        machine_type: str = "ToyCar",
        section_keyword: str = "section",
        section_ids: list[str] | None = None,
        data_type: str = "dev",
        train: bool = True,
        sample_rate: int = 16000,
        clip_num_samples: int | None = None,
        machine_types: list[str] | None = None,
        machine_id: int | None = None,
        machine_label: str | None = None,
        download: bool = False,
        download_manifest: Path | None = None,
    ):
        root = data_root if data_root is not None else root
        if root is None:
            raise ValueError("Either 'root' or 'data_root' must be provided.")

        self.root_dir = Path(root)
        self.machine_type = machine_type
        self.machine_label = machine_label or f"{dataset_name}:{data_type}:{machine_type}"
        self.train = train
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate
        self.clip_num_samples = clip_num_samples
        self.dali_x_keys = ["waveform"]
        self.dali_layout = "T"
        self.dali_sequence_length = clip_num_samples
        self.download_enabled = download
        self.download_manifest = (
            Path(download_manifest)
            if download_manifest is not None
            else _default_download_manifest()
        )

        if section_ids is None:
            section_ids = ["00"]

        all_machine_types = machine_types or [self.machine_label]
        self.machine_index = (
            machine_id
            if machine_id is not None
            else (
                all_machine_types.index(self.machine_label)
                if self.machine_label in all_machine_types
                else all_machine_types.index(machine_type)
                if machine_type in all_machine_types
                else 0
            )
        )

        dataset_dir = _resolve_dataset_dir(self.root_dir, dataset_name)
        if not download and not (dataset_dir / f"{data_type}_data" / "raw").exists():
            if (dataset_dir / "dev_data" / "raw").exists():
                data_type = "dev"
            elif (dataset_dir / "eval_data" / "raw").exists():
                data_type = "eval"
        self.data_type = data_type

        data_path = dataset_dir / f"{data_type}_data"
        target_dir = data_path / "raw" / machine_type

        if self.download_enabled:
            self.download()
            dataset_dir = _resolve_dataset_dir(self.root_dir, dataset_name)
            data_path = dataset_dir / f"{self.data_type}_data"
            target_dir = data_path / "raw" / machine_type

        if train:
            pattern_dirs = [target_dir / "train"]
        elif (target_dir / "test_rename").exists():
            pattern_dirs = [target_dir / "test_rename"]
        elif (target_dir / "source_test").exists() or (target_dir / "target_test").exists():
            pattern_dirs = [
                path
                for path in (target_dir / "source_test", target_dir / "target_test")
                if path.exists()
            ]
        else:
            pattern_dirs = [target_dir / "test"]

        section_names = [f"{section_keyword}_{sid}" for sid in section_ids]
        meta_rows = []

        for section_name in np.unique(section_names):
            wav_files = []
            for pattern_dir in pattern_dirs:
                if not pattern_dir.exists():
                    logger.warning(f"Directory not found: {pattern_dir}")
                    continue
                wav_files.extend(sorted(pattern_dir.glob(f"*{section_name}*.wav")))
            for wav_file in wav_files:
                name = wav_file.name
                y_true = 1 if "anomaly" in name else 0
                condition_str = (
                    [p for p in name.split("_") if p.startswith("id")][0]
                    if any(p.startswith("id") for p in name.split("_"))
                    else "id00"
                )
                try:
                    condition = int(condition_str.replace("id", ""))
                except ValueError:
                    condition = 0
                target = 1 if "target" in name or "target" in wav_file.parent.name else 0

                meta_rows.append(
                    {
                        "file": str(wav_file),
                        "basename": wav_file.name,
                        "machine": self.machine_index,
                        "machine_label": self.machine_label,
                        "machine_name": machine_type,
                        "dataset_name": dataset_name,
                        "data_type": data_type,
                        "y_true": y_true,
                        "condition": condition,
                        "section": int(section_name.split("_")[-1])
                        if section_name.split("_")[-1].isdigit()
                        else 0,
                        "target": target,
                        "split": "train" if train else "test",
                    }
                )

        if not meta_rows:
            logger.warning(
                f"No files found for {machine_type}/{[p.name for p in pattern_dirs]}. "
                "Dataset will be empty. Check root path and dataset layout."
            )
            meta_rows = []  # Empty but valid

        meta_columns = [
            "file",
            "basename",
            "machine",
            "machine_label",
            "machine_name",
            "dataset_name",
            "data_type",
            "y_true",
            "condition",
            "section",
            "target",
            "split",
        ]
        meta = (
            pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame(columns=meta_columns)  # ty: ignore[invalid-argument-type]
        )

        # Add anomaly column as alias for y_true (Task 4.1: ensure non-NaN anomaly metadata)
        if "y_true" in meta.columns and "anomaly" not in meta.columns:
            meta["anomaly"] = meta["y_true"]

        super().__init__(
            meta=meta,
            label_names=self.LABEL_NAMES,
            do_split=False,
        )

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Load raw audio waveform from file path. No channel operations.

        Returns:
            TensorDict: Loaded waveform under the ``waveform`` key,
                or an empty TensorDict if the file is missing.
        """
        file_path = row.get("file", "")
        if not file_path or not Path(file_path).exists():
            return TensorDict({}, batch_size=[])

        try:
            waveform, sr = _load_waveform(Path(file_path))
            if sr != self.sample_rate:
                import torchaudio

                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # waveform is [C, T]; clip on time axis
            if self.clip_num_samples is not None:
                t = waveform.shape[-1]
                if t < self.clip_num_samples:
                    pad = self.clip_num_samples - t
                    waveform = F.pad(waveform, (0, pad))
                elif t > self.clip_num_samples:
                    waveform = waveform[..., : self.clip_num_samples]

            return TensorDict({"waveform": waveform}, batch_size=[])
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return TensorDict({}, batch_size=[])

    def download(self) -> None:
        """Download the raw waveforms from the download manifest.

        Depends on dataset_name and machine_type.

        Raises:
            ValueError: If no download manifest path has been provided.
        """
        if self.download_manifest is None:
            raise ValueError("Download manifest path must be provided for downloading.")

        identifier_info = _parse_DCASE_identifier(self.dataset_name)
        task = identifier_info.get("task")
        year = identifier_info.get("year")

        yaml = YAML(typ="safe")
        with open(self.download_manifest, "r") as f:
            manifest = yaml.load(f)

        if self.data_type == "dev":
            archive_types = ["dev"]
        elif self.train:
            archive_types = ["additional"]
        else:
            archive_types = ["eval"]

        dataset_dir = _resolve_dataset_dir(self.root_dir, self.dataset_name)
        raw_root = dataset_dir / f"{self.data_type}_data" / "raw"
        target_dir = raw_root / self.machine_type
        expected_dir = target_dir / ("train" if self.train else "test")
        if expected_dir.exists() or (
            not self.train
            and (
                (target_dir / "test_rename").exists()
                or (target_dir / "source_test").exists()
                or (target_dir / "target_test").exists()
            )
        ):
            logger.info("DCASE data already present at %s; skipping download.", target_dir)
            return

        manifest_root = manifest.get(task, {}).get(year, {})

        download_links = [
            manifest_root.get(archive_type, {}).get("downloads", {}).get(self.machine_type, "")
            for archive_type in archive_types
        ]

        destinations = []
        for link in download_links:
            if link:
                parsed = urlparse(link)
                filename = Path(parsed.path).name
                destinations.append(raw_root / filename)
            else:
                destinations.append(None)

        for link, destination in zip(download_links, destinations, strict=False):
            if link:
                if "zenodo.org" not in link:
                    raise ValueError(f"Only zenodo.org download links are supported, got: {link}")
                if destination is None:
                    logger.error("Could not resolve download destination for %s", link)
                    continue
                if destination.exists() and _valid_zip(destination):
                    logger.info("Using existing archive %s", destination)
                else:
                    destination.unlink(missing_ok=True)
                    logger.info(f"Downloading from {link}...")
                    _download_file(link, destination)
            else:
                logger.error(
                    f"No download link found for {self.machine_type} in "
                    f"{task} {year} {archive_types}. Skipping download."
                )

        for dest in destinations:
            if dest is not None and dest.exists():
                logger.info(f"Unpacking {dest}...")
                _unzip_file(dest, raw_root)
            else:
                logger.error(f"Expected downloaded file not found: {dest}. Skipping unpacking.")

    @classmethod
    def from_config(
        cls,
        root: str | None = None,
        data_root: str | None = None,
        dataset_name: str = "DCASE2024T2",
        machine_types: list[str] | None = None,
        section_ids: list[str] | None = None,
        data_type: str = "dev",
        train: bool = True,
        sample_rate: int = 16000,
        clip_num_samples: int | None = None,
        download: bool = False,
        download_manifest: Path | None = None,
    ) -> list[DCASET2Dataset]:
        """Create one DCASET2Dataset per machine type.

        Returns:
            list[DCASET2Dataset]: One dataset instance per machine type.

        Raises:
            ValueError: If neither ``root`` nor ``data_root`` is provided.
        """
        root = data_root or root
        if root is None:
            raise ValueError("Either 'root' or 'data_root' must be provided.")
        machine_types = machine_types or ["ToyCar"]
        return [
            cls(
                root=root,
                dataset_name=dataset_name,
                machine_type=mt,
                section_ids=section_ids or ["00"],
                data_type=data_type,
                train=train,
                sample_rate=sample_rate,
                clip_num_samples=clip_num_samples,
                machine_types=machine_types,
                download=download,
                download_manifest=download_manifest,
            )
            for mt in machine_types
        ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_checksum(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_checksum(path: Path, expected: str | None, algorithm: str = "sha256") -> bool:
    if expected is None:
        return True
    return _compute_checksum(path, algorithm) == expected


def _write_dummy_wav(
    path: Path,
    duration_sec: float = 0.1,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> None:
    """Write a minimal WAV for fixture use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * sample_rate)
    data = np.random.randint(-32768, 32767, size=(n_samples, num_channels), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def _resolve_download_mode(mode: DownloadMode, dataset_dir: Path) -> DownloadMode:
    if mode == "already_downloaded":
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_dir}. "
                "Place files manually or use a different download mode."
            )
        return "disabled"
    if mode == "manual":
        logger.info("Manual download mode: ensure dataset is present at %s", dataset_dir)
        return "disabled"
    return mode


def _zenodo_files(record_id: int) -> list[dict]:
    with urlopen(f"https://zenodo.org/api/records/{record_id}", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return list(payload.get("files", []))


def _download_zenodo_files(
    record_id: int,
    destination: Path,
    *,
    keys: set[str] | None = None,
) -> list[Path]:
    """Download selected Zenodo record files into ``destination``.

    ``keys`` matches exact file names from Zenodo. Downloaded zip files are
    extracted into ``destination``. Existing valid archives are reused by the
    shared downloader.

    Returns:
        list[Path]: Paths to downloaded archive/files; zip archives are extracted as a
            side effect.
    """
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for file_info in _zenodo_files(record_id):
        key = str(file_info.get("key", ""))
        if keys is not None and key not in keys:
            continue
        link = file_info.get("links", {}).get("self")
        if not key or not link:
            continue
        archive_path = destination / key
        logger.info("Downloading Zenodo %s file %s", record_id, key)
        _download_file(link, archive_path)
        downloaded.append(archive_path)
        if archive_path.suffix == ".zip":
            _unzip_file(archive_path, destination)
    return downloaded


# ---------------------------------------------------------------------------
# Task 1 metadata helpers
# ---------------------------------------------------------------------------

# Official public dataset records.
DCASE2026_T1_BSD10K_RECORD = 17233905
DCASE2026_T1_OFFICIAL_URL = "https://zenodo.org/records/17233905"


def _download_task1(root_dir: Path) -> None:
    """Download the public Task 1 BSD10k files from Zenodo.

    The official metadata lives in ``metadata.zip`` as ``BSD10k_metadata.csv``.
    The audio archive is large (~8 GB), but is required for real training, so
    ``download_mode='download'`` fetches both metadata and audio by design.
    """
    if (root_dir / "audio").exists() and any(root_dir.glob("*metadata*.csv")):
        logger.info("Task 1 data already present at %s; skipping download", root_dir)
        return
    _download_zenodo_files(
        DCASE2026_T1_BSD10K_RECORD,
        root_dir,
        keys={"metadata.zip", "audio.zip"},
    )


def _normalise_task1_metadata(meta: pd.DataFrame, root_dir: Path) -> pd.DataFrame:
    col_map = {
        "filename": "file",
        "audio_file": "file",
        "filepath": "file",
        "path": "file",
        "id": "sound_id",
        "top_level": "class_top",
        "top": "class_top",
        "second_level": "class_second",
        "second": "class_second",
        "bst_second": "class_second",
        "class": "class_second",
    }
    meta = meta.rename(columns={k: v for k, v in col_map.items() if k in meta.columns})
    if "sound_id" not in meta.columns and "file" in meta.columns:
        meta["sound_id"] = meta["file"].map(lambda p: Path(str(p)).stem)
    if "file" not in meta.columns and "sound_id" in meta.columns:
        meta["file"] = meta["sound_id"].map(lambda sid: f"audio/{sid}.wav")
    if "class_idx" in meta.columns:
        meta["class_second"] = (
            pd.to_numeric(meta["class_idx"], errors="coerce").fillna(0).astype(int)
        )
    for req in ("file", "sound_id", "class_top", "class_second", "confidence"):
        if req not in meta.columns:
            meta[req] = 1.0 if req == "confidence" else "unknown"
    meta["file"] = meta["file"].apply(
        lambda p: str(root_dir / str(p)) if not Path(str(p)).is_absolute() else str(p)
    )
    if "class_top_name" not in meta.columns:
        meta["class_top_name"] = meta["class_top"].astype(str)
    if "class_second_name" not in meta.columns:
        meta["class_second_name"] = meta["class_second"].astype(str)
    meta["class_top"] = pd.Categorical(meta["class_top_name"]).codes.astype(float)
    if "class_idx" not in meta.columns:
        meta["class_second"] = pd.Categorical(meta["class_second_name"]).codes.astype(float)
    else:
        meta["class_second"] = (
            pd.to_numeric(meta["class_second"], errors="coerce").fillna(0).astype(float)
        )
    return meta


def _build_task1_metadata(root_dir: Path) -> pd.DataFrame | None:
    """Build metadata DataFrame from raw Task 1 audio files and optional labels.

    Scans ``root_dir`` for audio files (.wav, .mp3, .flac) and attempts to
    locate an accompanying labels file (``labels.csv``, ``ground_truth.csv``,
    ``metadata.csv``).  If no labels file is found, a stub metadata frame is
    returned with file paths and placeholder labels so the dataset can still
    be instantiated for inspection or pre-processing.

    Returns:
        pd.DataFrame | None: Metadata frame, or None if no audio files found.
    """
    audio_exts = {".wav", "*.mp3", "*.flac", "*.ogg"}
    audio_files: list[Path] = []
    for ext in audio_exts:
        audio_files.extend(sorted(root_dir.rglob(ext.replace("*.", "*."))))
    if not audio_files:
        audio_files = sorted(root_dir.rglob("*.wav"))
    if not audio_files:
        return None

    # Look for an official labels / metadata file
    label_candidates = [
        root_dir / "labels.csv",
        root_dir / "ground_truth.csv",
        root_dir / "metadata.csv",
        root_dir / "meta.csv",
        root_dir / "BSD10k_metadata.csv",
        root_dir / "metadata" / "BSD10k_metadata.csv",
    ]
    labels_path: Path | None = None
    for cand in label_candidates:
        if cand.exists():
            labels_path = cand
            break

    if labels_path is not None:
        try:
            return _normalise_task1_metadata(pd.read_csv(labels_path), root_dir)
        except Exception as exc:
            logger.warning("Failed to parse Task 1 labels file %s: %s", labels_path, exc)

    # Fallback: stub metadata from audio file names
    rows = []
    for af in audio_files:
        stem = af.stem
        rows.append(
            {
                "file": str(af),
                "sound_id": stem,
                "class_top": "unknown",
                "class_second": "unknown",
                "confidence": 1.0,
                "split": "fit",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 1: Hierarchical sound classification
# ---------------------------------------------------------------------------


@data_source("DCASE2026T1Dataset")
class DCASE2026T1Dataset(NexuDataset):
    """DCASE 2026 Task 1 dataset for BSD10k-v1.2 / BSD35k-CS.

    Expects a metadata CSV with columns including at least:
    ``file``, ``sound_id``, ``class_top``, ``class_second``.
    Optional columns: ``confidence``, ``uploader``, ``license``, ``title``,
    ``tags``, ``description``, ``clap_embedding``.

    Args:
        root: Dataset root directory.
        metadata_csv: Path to metadata CSV relative to root.
        download_mode: One of the DownloadMode literals.
        checksum: Optional expected checksum of the metadata CSV.
        clap_embeddings_dir: Optional directory with pre-computed CLAP .npy files.
        fold_seed: Seed for reproducible baseline 5-fold splits.
        fold_id: If provided, filter to that fold (0-4).
        split: ``"fit"``, ``"val"``, ``"test"``, or ``"eval"``.
        sample_rate: Target audio sample rate.
    """

    LABEL_NAMES = ["class_top", "class_second", "confidence"]

    def __init__(
        self,
        root: str | Path,
        metadata_csv: str = "metadata.csv",
        download_mode: DownloadMode = "disabled",
        checksum: str | None = None,
        clap_embeddings_dir: str | None = None,
        fold_seed: int = 42,
        fold_id: int | None = None,
        split: str = "fit",
        sample_rate: int = 16000,
        clip_num_samples: int | None = None,
        **_: object,
    ):
        self.root_dir = Path(root)
        self.download_mode = _resolve_download_mode(download_mode, self.root_dir)
        self.sample_rate = sample_rate
        self.clip_num_samples = clip_num_samples
        self.clap_embeddings_dir = Path(clap_embeddings_dir) if clap_embeddings_dir else None

        meta_path = self.root_dir / metadata_csv
        if self.download_mode == "download":
            _download_task1(self.root_dir)
        if self.download_mode == "dry_run":
            logger.info("[dry_run] Task1 dataset would read metadata from %s", meta_path)
            meta = pd.DataFrame(
                columns=["file", "sound_id", "class_top", "class_second", "confidence", "split"]  # ty: ignore[invalid-argument-type]
            )
            super().__init__(meta=meta, label_names=self.LABEL_NAMES, do_split=False)
            return

        if not meta_path.exists():
            # Attempt to build metadata from raw audio / official labels
            built = _build_task1_metadata(self.root_dir)
            if built is not None:
                logger.info("Built Task 1 metadata with %d rows from %s", len(built), self.root_dir)
                meta = built
            else:
                # Fake-fixture friendly: allow empty dataset with warning
                logger.warning(
                    "Task1 metadata not found at %s and no audio files discovered; "
                    "creating empty dataset.",
                    meta_path,
                )
                meta = pd.DataFrame(
                    columns=["file", "sound_id", "class_top", "class_second", "confidence", "split"]  # ty: ignore[invalid-argument-type]
                )
            super().__init__(meta=meta, label_names=self.LABEL_NAMES, do_split=False)
            return

        if checksum and not _validate_checksum(meta_path, checksum):
            raise ValueError(f"Checksum mismatch for {meta_path}")

        if not meta_path.exists():
            built = _build_task1_metadata(self.root_dir)
            if built is None:
                raise FileNotFoundError(
                    f"Task1 metadata not found at {meta_path} after download/search. "
                    "Expected official BSD metadata such as BSD10k_metadata.csv."
                )
            meta = built
        else:
            meta = _normalise_task1_metadata(pd.read_csv(meta_path), self.root_dir)
        required = {"file", "sound_id", "class_top", "class_second"}
        missing = required - set(meta.columns)
        if missing:
            raise ValueError(f"Task1 metadata missing columns: {missing}")

        # Folds
        if "fold" not in meta.columns and fold_id is not None:
            rng = np.random.default_rng(fold_seed)
            meta["fold"] = rng.integers(0, 5, size=len(meta))
        if fold_id is not None:
            meta = meta[meta["fold"] == fold_id].reset_index(drop=True)

        if "split" not in meta.columns:
            meta["split"] = "fit"
        meta = meta[meta["split"] == split].reset_index(drop=True)

        super().__init__(meta=meta, label_names=self.LABEL_NAMES, do_split=False)

    @staticmethod
    def _label_to_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, np.ndarray)):
            return torch.tensor(value, dtype=torch.float32)
        # Handle string labels by mapping to hash-based integer (deterministic for tests)
        if isinstance(value, str):
            return torch.tensor(float(hash(value) % 2**31), dtype=torch.float32)
        return torch.tensor(float(value), dtype=torch.float32)

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        file_path = row.get("file", "")
        if self.clap_embeddings_dir and "sound_id" in row:
            emb_path = self.clap_embeddings_dir / f"{row['sound_id']}.npy"
            if emb_path.exists():
                emb = torch.from_numpy(np.load(emb_path)).float()
                return TensorDict({"clap_embedding": emb}, batch_size=[])

        if not file_path or not Path(file_path).exists():
            return TensorDict({}, batch_size=[])

        try:
            waveform, sr = _load_waveform(Path(file_path))
            if sr != self.sample_rate:
                try:
                    import torchaudio

                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                except Exception:
                    import librosa

                    resampled = librosa.resample(
                        waveform.numpy(), orig_sr=sr, target_sr=self.sample_rate, axis=-1
                    )
                    waveform = torch.from_numpy(resampled.copy())
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            if self.clip_num_samples is not None:
                if waveform.shape[0] < self.clip_num_samples:
                    waveform = torch.nn.functional.pad(
                        waveform, (0, self.clip_num_samples - waveform.shape[0])
                    )
                elif waveform.shape[0] > self.clip_num_samples:
                    waveform = waveform[: self.clip_num_samples]
            return TensorDict({"waveform": waveform}, batch_size=[])
        except Exception as e:
            logger.debug("Failed to load %s: %s", file_path, e)
            return TensorDict({}, batch_size=[])


# ---------------------------------------------------------------------------
# Task 2 helpers
# ---------------------------------------------------------------------------

# Official DCASE 2026 Task 2 dataset page (for documentation; no auto-download)
DCASE2026_T2_OFFICIAL_URL = "https://dcase.community/challenge2026/"


def _resolve_task2_dataset_dir(root: Path, dataset_name: str) -> Path:
    """Resolve Task 2 dataset directory, trying common naming variants.

    In addition to the parent's ``_resolve_dataset_dir``, this tries
    common 2026-specific folder names (e.g. ``dcase2026_task2``,
    ``DCASE2026_T2``, ``dcase2026-t2``).

    Returns:
        Path: Resolved dataset directory path.
    """
    base = _resolve_dataset_dir(root, dataset_name)
    if base.exists():
        return base

    variants = [
        dataset_name.lower().replace("t2", "_task2"),
        dataset_name.lower().replace("t2", "-task2"),
        dataset_name.replace("T2", "_T2"),
        dataset_name.replace("T2", "-T2"),
        "dcase2026_task2",
        "DCASE2026_T2",
    ]
    for variant in variants:
        cand = root / variant
        if cand.exists():
            return cand
    return base


# ---------------------------------------------------------------------------
# Task 2: Anomalous sound detection (2026 extensions)
# ---------------------------------------------------------------------------


@data_source("DCASE2026T2Dataset")
class DCASE2026T2Dataset(DCASET2Dataset):
    """DCASE 2026 Task 2 dataset — metadata index with domain/attribute enrichment.

    Extends ``DCASET2Dataset`` with 2026-specific metadata:
    - ``domain`` inferred from filename source/target markers
    - ``attributes`` merged from the official attributes CSV when available

    Audio is 2-channel (stereo). Channel selection happens in pipeline layers,
    not here. DALI loads raw stereo audio as [C, T] via ``dali_layout="CT"``.
    """

    LABEL_NAMES = ["y_true", "machine", "target", "section"]

    def __init__(
        self,
        root: str | Path | None = None,
        data_root: str | Path | None = None,
        dataset_name: str = "DCASE2026T2",
        machine_type: str = "ToyCar",
        section_keyword: str = "section",
        section_ids: list[str] | None = None,
        data_type: str = "dev",
        train: bool = True,
        sample_rate: int = 16000,
        clip_num_samples: int | None = None,
        machine_types: list[str] | None = None,
        machine_id: int | None = None,
        machine_label: str | None = None,
        download: bool = False,
        download_manifest: Path | None = None,
        attributes_csv: str | None = None,
        **_ignored: object,
    ):
        self.attributes_csv = attributes_csv

        resolved_root = data_root if data_root is not None else root
        if resolved_root is not None:
            resolved_root = _resolve_task2_dataset_dir(Path(resolved_root), dataset_name)

        super().__init__(
            root=str(resolved_root) if resolved_root is not None else None,
            data_root=None,
            dataset_name=dataset_name,
            machine_type=machine_type,
            section_keyword=section_keyword,
            section_ids=section_ids,
            data_type=data_type,
            train=train,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            machine_types=machine_types,
            machine_id=machine_id,
            machine_label=machine_label,
            download=download,
            download_manifest=download_manifest,
        )

        # DALI outputs stereo [C, T] for 2026 two-channel WAVs
        self.dali_layout = "CT"

        # Official DCASE 2026 Task 2 eval/test labels are hidden; mark them
        # missing so metric algorithms skip them, while export still sees rows.
        if self.data_type == "eval" and not self.train and self.meta is not None:
            if "y_true" in self.meta.columns:
                self.meta["y_true"] = float("nan")
            if "anomaly" in self.meta.columns:
                self.meta["anomaly"] = float("nan")

        if self.meta is not None and not self.meta.empty:
            targets = (
                self.meta["target"].tolist()
                if "target" in self.meta.columns
                else [None] * len(self.meta)
            )
            self.meta["domain"] = [
                self._infer_domain(b, t)
                for b, t in zip(self.meta["basename"], targets, strict=False)
            ]
            if "y_true" in self.meta.columns and "anomaly" not in self.meta.columns:
                self.meta["anomaly"] = self.meta["y_true"]
            if attributes_csv:
                attr_path = (
                    Path(attributes_csv)
                    if Path(attributes_csv).is_absolute()
                    else self.root_dir / attributes_csv
                )
                if attr_path.exists():
                    attr_df = pd.read_csv(attr_path)
                    self.meta = self.meta.merge(attr_df, on="basename", how="left")

    def _infer_domain(self, basename: str, target: int | None = None) -> str:
        if "_source_" in basename:
            return "source"
        if "_target_" in basename:
            return "target"
        if target is not None:
            return "target" if target else "source"
        return "unknown"


# ---------------------------------------------------------------------------
# Task 7 metadata helpers
# ---------------------------------------------------------------------------

DCASE2026_T7_DEV_RECORD = 19335184
DCASE2026_T7_OFFICIAL_URL = "https://zenodo.org/records/19335184"


def _download_task7(root_dir: Path, domains: list[str]) -> None:
    keys = {f"DIL-DCASE26-Dev-{domain}.zip" for domain in domains}
    if all(_resolve_task7_domain_dir(root_dir, domain).exists() for domain in domains):
        logger.info("Task 7 data already present at %s; skipping download", root_dir)
        return
    _download_zenodo_files(DCASE2026_T7_DEV_RECORD, root_dir, keys=keys)


def _resolve_task7_domain_dir(root_dir: Path, domain: str) -> Path:
    candidates = [
        root_dir / domain,
        root_dir / f"DIL-DCASE26-Dev-{domain}",
        root_dir / "DIL-DCASE26-Dev" / domain,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root_dir.glob(f"*{domain}*"))
    for match in matches:
        if match.is_dir():
            return match
    return root_dir / domain


def _build_task7_domain_metadata(domain_dir: Path, domain: str) -> pd.DataFrame | None:
    """Build metadata for a single Task 7 domain from raw audio files.

    Scans ``domain_dir`` for audio files.  If an ``evaluation_setup/``
    directory exists, train/test split files are parsed.  Otherwise all
    discovered files are assigned to the ``fit`` split and class labels are
    inferred from parent folder names (e.g. ``class_0/``, ``class_1/``) or
    left as ``-1`` for unlabelled evaluation data.

    Returns:
        pd.DataFrame | None: Metadata frame, or None if no audio files found.
    """
    audio_exts = (".wav", ".mp3", ".flac", ".ogg")
    audio_files: list[Path] = []
    for ext in audio_exts:
        audio_files.extend(sorted(domain_dir.rglob(f"*{ext}")))
    if not audio_files:
        return None

    # Check for evaluation setup files (common DCASE pattern)
    setup_dirs = [domain_dir / "evaluation_setup", domain_dir.parent / "evaluation_setup"]
    split_by_name: dict[str, str] = {}
    class_by_name: dict[str, int] = {}
    for setup_dir in setup_dirs:
        if not setup_dir.exists():
            continue
        for setup_file in setup_dir.glob("*.txt"):
            try:
                with open(setup_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if not parts:
                            continue
                        # Baseline lines are commonly:
                        # <relative_path> <class_name> <domain> <class_index>
                        rel_path = parts[0]
                        fname = Path(rel_path).name
                        split_by_name[fname] = (
                            "test" if "test" in setup_file.name.lower() else "fit"
                        )
                        for token in reversed(parts[1:]):
                            try:
                                class_by_name[fname] = int(token)
                                break
                            except ValueError:
                                continue
            except Exception as exc:
                logger.debug("Could not parse setup file %s: %s", setup_file, exc)

    rows: list[dict] = []
    for af in audio_files:
        # Try to infer class from parent directory name
        parent_name = af.parent.name
        fname = af.name
        if fname in class_by_name:
            cls = class_by_name[fname]
        else:
            # Check for common patterns like "class_0", "label_1", etc.
            try:
                cls = int(parent_name)
            except ValueError:
                for prefix in ("class_", "label_", "c", "cls"):
                    if parent_name.startswith(prefix):
                        try:
                            cls = int(parent_name[len(prefix) :])
                            break
                        except ValueError:
                            continue
                else:
                    cls = -1  # unknown / evaluation

        split = split_by_name.get(fname, "fit")

        rows.append(
            {
                "file": str(af),
                "basename": af.name,
                "class": cls,
                "domain": domain,
                "split": split,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 7: Domain-incremental learning
# ---------------------------------------------------------------------------


@data_source("DCASE2026T7Dataset")
class DCASE2026T7Dataset(NexuDataset):
    """DCASE 2026 Task 7 DIL-DCASE26 dataset.

    Expects a root directory with subfolders ``D2/`` and ``D3/`` (and
    optionally ``D1/`` for baseline reference).  Each domain folder may
    contain:

    - ``metadata.csv`` with columns ``file``, ``class``, ``domain``
    - Raw audio files (optionally with ``evaluation_setup/`` split files)

    The dataset exposes a sequence-of-increments interface via
    ``get_increment()``.
    """

    LABEL_NAMES = ["class", "domain"]

    def __init__(
        self,
        root: str | Path,
        domains: list[str] | None = None,
        download_mode: DownloadMode = "disabled",
        checksum: str | None = None,
        d1_baseline_dir: str | None = None,
        sample_rate: int = 16000,
        clip_num_samples: int | None = None,
    ):
        self.root_dir = Path(root)
        self.download_mode = _resolve_download_mode(download_mode, self.root_dir)
        self.sample_rate = sample_rate
        self.clip_num_samples = clip_num_samples
        self.d1_baseline_dir = Path(d1_baseline_dir) if d1_baseline_dir else None

        if self.download_mode == "dry_run":
            logger.info("[dry_run] Task7 dataset would read from %s", self.root_dir)
            meta = pd.DataFrame(
                columns=["file", "class", "domain", "split"]  # ty: ignore[invalid-argument-type]
            )
            super().__init__(meta=meta, label_names=self.LABEL_NAMES, do_split=False)
            return

        domains = domains or ["D2", "D3"]
        if self.download_mode == "download":
            _download_task7(self.root_dir, domains)
        rows: list[pd.DataFrame] = []
        root_setup = self.root_dir / "evaluation_setup"
        if root_setup.exists():
            split_frames = [
                self._task7_metadata_from_setup_file(path, domains)
                for path in sorted(root_setup.glob("*.txt"))
            ]
            split_frames = [
                frame for frame in split_frames if frame is not None and not frame.empty
            ]
            if split_frames:
                meta = pd.concat(split_frames, ignore_index=True)
                super().__init__(meta=meta, label_names=self.LABEL_NAMES, do_split=False)
                return
        for domain in domains:
            domain_dir = _resolve_task7_domain_dir(self.root_dir, domain)
            meta_path = domain_dir / "metadata.csv"
            if meta_path.exists():
                df = pd.read_csv(meta_path)
                df["domain"] = domain
                df["file"] = df["file"].apply(
                    lambda p: str(domain_dir / p) if not Path(p).is_absolute() else p
                )
                rows.append(df)
            else:
                # Try to build metadata from raw audio / evaluation setup
                built = _build_task7_domain_metadata(domain_dir, domain)
                if built is not None:
                    logger.info(
                        "Built Task 7 domain %s metadata with %d rows from %s",
                        domain,
                        len(built),
                        domain_dir,
                    )
                    rows.append(built)
                else:
                    logger.warning(
                        "Task7 metadata not found for domain %s and no audio files discovered.",
                        domain,
                    )

        if not rows:
            meta = pd.DataFrame(
                columns=["file", "class", "domain", "split"]  # ty: ignore[invalid-argument-type]
            )
        else:
            meta = pd.concat(rows, ignore_index=True)
            if "split" not in meta.columns:
                meta["split"] = "fit"

        super().__init__(meta=meta, label_names=self.LABEL_NAMES, do_split=False)

    def _task7_metadata_from_setup_file(
        self,
        setup_file: Path,
        domains: list[str],
    ) -> pd.DataFrame | None:
        rows: list[dict] = []
        split = "test" if "test" in setup_file.name.lower() else "fit"
        with open(setup_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or line.startswith("#"):
                    continue
                rel_path = parts[0]
                domain = parts[2] if len(parts) >= 3 and parts[2].startswith("D") else ""
                if not domain:
                    domain = next(
                        (d for d in domains if d in rel_path or d in setup_file.name), domains[0]
                    )
                if domain not in domains:
                    continue
                try:
                    cls = int(parts[-1])
                except ValueError:
                    cls = -1
                file_path = self.root_dir / rel_path
                if not file_path.exists():
                    file_path = self.root_dir / domain / rel_path
                rows.append(
                    {
                        "file": str(file_path),
                        "basename": Path(rel_path).name,
                        "class": cls,
                        "domain": domain,
                        "split": split,
                    }
                )
        return pd.DataFrame(rows)

    def get_increment(self, domain: str) -> "DCASE2026T7Dataset":
        """Return a view containing only the requested domain.

        Raises:
            ValueError: If no metadata has been loaded.
        """
        if self.meta is None:
            raise ValueError("No metadata available")
        mask = self.meta["domain"] == domain
        return self.clone_with_meta(self.meta[mask].reset_index(drop=True))

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        file_path = row.get("file", "")
        if not file_path or not Path(file_path).exists():
            return TensorDict({}, batch_size=[])

        try:
            waveform, sr = _load_waveform(Path(file_path))
            if sr != self.sample_rate:
                try:
                    import torchaudio

                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                except Exception:
                    import librosa

                    resampled = librosa.resample(
                        waveform.numpy(), orig_sr=sr, target_sr=self.sample_rate, axis=-1
                    )
                    waveform = torch.from_numpy(resampled.copy())
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            if self.clip_num_samples is not None:
                if waveform.shape[0] < self.clip_num_samples:
                    waveform = torch.nn.functional.pad(
                        waveform, (0, self.clip_num_samples - waveform.shape[0])
                    )
                elif waveform.shape[0] > self.clip_num_samples:
                    waveform = waveform[: self.clip_num_samples]
            return TensorDict({"waveform": waveform}, batch_size=[])
        except Exception as e:
            logger.debug("Failed to load %s: %s", file_path, e)
            return TensorDict({}, batch_size=[])
