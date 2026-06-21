"""AudioSet dataset source for NexuML."""

from __future__ import annotations
from nexuml.core.discovery import data_source

import logging
import urllib.request
import wave
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from nexuml.data.dataset import NexuDataset

logger = logging.getLogger(__name__)
_LABEL_COLUMNS = pd.Index(["index", "mid", "display_name"])
_META_COLUMNS = pd.Index(["file", "basename", "class", "class_logits", "split"])


def _parse_label_index(labels_df: pd.DataFrame, label_str: str) -> int:
    """Return the first numeric index for a label string like 'Dog_Cat'."""
    parts = label_str.split("_")
    rows = labels_df[labels_df["display_name"].isin(parts)]
    if len(rows) == 0:
        return 0
    return int(rows.index[0])


def _parse_label_logits(labels_df: pd.DataFrame, label_str: str) -> list[int]:
    """Return a multi-hot integer vector for a label string."""
    parts = label_str.split("_")
    rows = labels_df[labels_df["display_name"].isin(parts)]
    hot = torch.zeros(len(labels_df), dtype=torch.long)
    hot[rows.index] = 1
    return hot.tolist()


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


@data_source("AudiosetDataset")
class AudiosetDataset(NexuDataset):
    """AudioSet dataset source.

    Scans <data_dir> for .wav or .npy files. File names must encode labels
    via the pattern ``<ytid>__<label_str>.<ext>`` where label_str is
    underscore-joined AudioSet display names.

    A cached metadata CSV is read/written to ``<meta_dir>/<data_dir.name>_meta.csv``
    so subsequent loads are fast.

    Label columns: class (int index), class_logits (multi-hot list).

    Args:
        data_dir: Directory containing audio/feature files.
        meta_dir: Directory for class_labels_indices.csv and the cached meta CSV.
        perform_checks: Whether to verify each file is loadable (slow on first run).
        len_seconds: Minimum audio duration in seconds to include a file.
        sample_rate: Target sample rate for audio loading.
    """

    LABEL_NAMES = ["class", "class_logits"]

    CLASS_LABELS_URL = (
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
    )

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/audioset/eval",
        meta_dir: Union[str, Path] = "data/AudioSet/meta",
        perform_checks: bool = True,
        len_seconds: int = 5,
        sample_rate: int = 16000,
    ):
        self.data_dir = Path(data_dir)
        self.meta_dir = Path(meta_dir)
        self.perform_checks = perform_checks
        self.len_seconds = len_seconds
        self.sample_rate = sample_rate
        self.clip_num_samples = int(len_seconds * sample_rate)
        self.dali_x_keys = ["waveform"]
        self.dali_layout = "T"
        self.dali_sequence_length = self.clip_num_samples

        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # Ensure class labels index exists
        class_labels_path = self.meta_dir / "class_labels_indices.csv"
        if not class_labels_path.exists():
            logger.info(f"Downloading class labels to {class_labels_path}")
            try:
                urllib.request.urlretrieve(self.CLASS_LABELS_URL, class_labels_path)
            except Exception as e:
                logger.warning(f"Could not download class labels: {e}")
                labels_df = pd.DataFrame(columns=_LABEL_COLUMNS)
                labels_df.to_csv(class_labels_path, index=False)

        self.labels_df = pd.read_csv(class_labels_path)

        meta_cache = self.meta_dir / f"{self.data_dir.name}_meta.csv"
        if meta_cache.exists():
            try:
                meta = self._load_cached_meta(meta_cache)
            except Exception as exc:
                logger.warning("Could not load AudioSet metadata cache %s: %s", meta_cache, exc)
                meta = pd.DataFrame(columns=_META_COLUMNS)
        else:
            meta = self._build_meta(meta_cache)

        super().__init__(
            meta=meta,
            label_names=self.LABEL_NAMES,
            do_split=False,
        )

    def _load_cached_meta(self, path: Path) -> pd.DataFrame:
        meta = pd.read_csv(path)
        meta["class"] = meta["class"].astype(int)
        if "class_logits" in meta.columns:
            meta["class_logits"] = meta["class_logits"].apply(
                lambda x: list(map(float, str(x).strip("[]").split(",")))
            )
        meta["file"] = meta["file"].astype(str)
        meta["basename"] = meta["basename"].astype(str)
        meta = self._remap_cached_paths(meta)
        existing_mask = meta["file"].map(lambda value: Path(str(value)).exists())
        if not bool(existing_mask.all()):
            dropped = int((~existing_mask).sum())
            logger.warning(
                "Dropping %d cached AudioSet rows with missing files under data_dir=%s",
                dropped,
                self.data_dir,
            )
            meta = meta[existing_mask].reset_index(drop=True)
        if "split" not in meta.columns:
            meta["split"] = "fit"
        return meta

    def _remap_cached_paths(self, meta: pd.DataFrame) -> pd.DataFrame:
        """Re-root cached absolute paths when data has moved to a new mount.

        Returns:
            pd.DataFrame: The metadata with remapped file paths.
        """
        if "file" not in meta.columns or "basename" not in meta.columns:
            return meta

        missing_mask = ~meta["file"].map(lambda value: Path(str(value)).exists())
        if not bool(missing_mask.any()):
            return meta

        remapped = 0
        for idx in meta[missing_mask].index.tolist():
            candidate = self.data_dir / str(meta.at[idx, "basename"])
            if candidate.exists():
                meta.at[idx, "file"] = str(candidate)
                remapped += 1

        if remapped > 0:
            logger.info(
                "Remapped %d cached AudioSet file paths to data_dir=%s",
                remapped,
                self.data_dir,
            )
        return meta

    def _build_meta(self, cache_path: Path) -> pd.DataFrame:
        if not self.data_dir.exists():
            logger.warning(f"AudioSet data directory not found: {self.data_dir}")
            return pd.DataFrame(columns=_META_COLUMNS)

        wav_files = list(self.data_dir.rglob("*.wav"))
        npy_files = list(self.data_dir.rglob("*.npy"))
        files = npy_files if len(npy_files) > len(wav_files) else wav_files

        if not files:
            logger.warning(f"No audio files found in {self.data_dir}")
            return pd.DataFrame(columns=_META_COLUMNS)

        logger.info(f"Building AudioSet metadata for {len(files)} files...")

        rows = []
        for file_path in files:
            label_str = file_path.stem.split("__")[-1]
            if self.perform_checks and not self._check_file(file_path):
                continue
            rows.append(
                {
                    "file": str(file_path),
                    "basename": file_path.name,
                    "class": _parse_label_index(self.labels_df, label_str),
                    "class_logits": _parse_label_logits(self.labels_df, label_str),
                    "split": "fit",
                }
            )

        meta = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_META_COLUMNS)
        meta.to_csv(cache_path, index=False)
        logger.info(f"Saved metadata to {cache_path} ({len(meta)} samples)")
        return meta

    def _check_file(self, file_path: Path) -> bool:
        """Return True if the file is valid and long enough."""
        try:
            if file_path.suffix == ".npy":
                vec = np.load(file_path)
                return vec.shape[-1] / 16000 >= self.len_seconds
            else:
                waveform, sample_rate = _load_waveform(file_path)
                return waveform.shape[-1] / float(sample_rate) >= self.len_seconds
        except Exception:
            return False

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Load audio waveform or precomputed features from file.

        Returns:
            TensorDict: Loaded audio data with ``waveform`` or ``features`` key,
                or an empty TensorDict if the file is missing.
        """
        file_path = Path(str(row.get("file", "")))
        if not file_path.exists():
            return TensorDict({}, batch_size=[])

        try:
            if file_path.suffix == ".npy":
                arr = np.load(file_path)
                return TensorDict({"features": torch.from_numpy(arr).float()}, batch_size=[])
            else:
                waveform, sr = _load_waveform(file_path)
                if sr != self.sample_rate:
                    try:
                        import torchaudio

                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    except Exception:
                        new_len = max(
                            1, int(round(waveform.shape[-1] * self.sample_rate / float(sr)))
                        )
                        waveform = F.interpolate(
                            waveform.unsqueeze(0),
                            size=new_len,
                            mode="linear",
                            align_corners=False,
                        ).squeeze(0)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze(0)
                if waveform.shape[0] < self.clip_num_samples:
                    waveform = torch.nn.functional.pad(
                        waveform,
                        (0, self.clip_num_samples - waveform.shape[0]),
                    )
                elif waveform.shape[0] > self.clip_num_samples:
                    waveform = waveform[: self.clip_num_samples]
                return TensorDict({"waveform": waveform}, batch_size=[])
        except Exception as e:
            logger.debug(f"Failed to load {file_path}: {e}")
            return TensorDict({}, batch_size=[])
