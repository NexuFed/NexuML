"""Paired healthy/faulty acoustic recordings for unconditional Flow Matching."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset

_SPLITS = ("train", "val", "test")
_CONDITIONS = ("healthy", "faulty")


def _group_id(file: Path, directory: Path) -> str:
    """Resolve a recording's grouping identity from its relative directory.

    Returns:
        Group identity derived from the machine subdirectory or filename.
    """
    relative = file.relative_to(directory)
    return relative.parts[0] if len(relative.parts) > 1 else file.stem


def _scan_split(root: Path, split: str) -> dict[str, list[Path]]:
    """Find WAV recordings for both endpoint classes in an explicit split.

    Returns:
        Healthy and faulty audio file lists for the requested split.

    Raises:
        FileNotFoundError: If either endpoint condition has no recordings.
    """
    sources: dict[str, list[Path]] = {}
    for condition in _CONDITIONS:
        folder = root / split / condition
        sources[condition] = sorted(folder.rglob("*.wav")) if folder.is_dir() else []
        if not sources[condition]:
            raise FileNotFoundError(
                f"No {condition} .wav recordings in {folder}. "
                "Supply both endpoint classes for every configured split."
            )
    return sources


def _pair_metadata(root: Path, seed: int, include_test: bool) -> pd.DataFrame:
    """Build deterministic independent endpoint pairings within each split.

    Returns:
        Metadata with file and grouping provenance for each endpoint.

    Raises:
        ValueError: If a recording group appears in multiple splits.
    """
    splits = _SPLITS if include_test else _SPLITS[:2]
    recordings = {split: _scan_split(root, split) for split in splits}
    group_splits: dict[str, str] = {}
    for split, sources in recordings.items():
        for condition, files in sources.items():
            directory = root / split / condition
            for file in files:
                group = _group_id(file, directory)
                previous = group_splits.setdefault(group, split)
                if previous != split:
                    raise ValueError(
                        f"Audio recording group {group!r} spans {previous} and {split}; "
                        "train/validation/test groups must be disjoint."
                    )

    rows: list[dict[str, str | int]] = []
    for index, (split, sources) in enumerate(recordings.items()):
        healthy = sources["healthy"]
        faulty = sources["faulty"]
        rng = np.random.default_rng(seed + index)
        healthy_order = rng.permutation(len(healthy))
        faulty_order = rng.permutation(len(faulty))
        count = max(len(healthy), len(faulty))
        for pair_index in range(count):
            healthy_file = healthy[int(healthy_order[pair_index % len(healthy)])]
            faulty_file = faulty[int(faulty_order[pair_index % len(faulty)])]
            rows.append(
                {
                    "healthy_file": str(healthy_file),
                    "faulty_file": str(faulty_file),
                    "healthy_group": _group_id(healthy_file, root / split / "healthy"),
                    "faulty_group": _group_id(faulty_file, root / split / "faulty"),
                    "pair_id": f"{split}:{pair_index}",
                    "source_dataset": str(root),
                    "paired": 1,
                    "split": split,
                }
            )
    return pd.DataFrame.from_records(rows)


@data_source("AudioEndpointPairs")
class AudioEndpointPairs(DataSourceDefinition):
    """Audio endpoint distributions with explicit group-disjoint partitions.

    Recordings must be stored under train/healthy, train/faulty, val/healthy,
    and val/faulty. Optional test partitions are excluded unless explicitly
    enabled; do not populate these partitions using official anomaly-test
    recordings for training.

    Attributes:
        root: Root directory containing explicit acoustic endpoint partitions.
        sample_rate: Required source waveform sampling rate.
        clip_num_samples: Fixed waveform length after truncation/zero padding.
        seed: Deterministic independent source/target pairing seed.
        include_test: Explicitly include held-out test recordings.
    """

    root: str | Path
    sample_rate: int = Field(default=16000, gt=0)
    clip_num_samples: int = Field(default=2048, gt=0)
    seed: int = 42
    include_test: bool = False

    def build(self) -> NexuDataset:
        """Create the lazily loaded acoustic-pair dataset.

        Returns:
            NexuDataset with healthy_waveform and faulty_waveform feature keys.
        """
        return _AudioEndpointPairsRuntime(**self.model_dump())


class _AudioEndpointPairsRuntime(NexuDataset):
    """Private mutable paired-waveform runtime."""

    def __init__(
        self,
        root: str | Path,
        sample_rate: int,
        clip_num_samples: int,
        seed: int,
        include_test: bool,
    ) -> None:
        self.sample_rate = sample_rate
        self.clip_num_samples = clip_num_samples
        root_path = Path(root).expanduser()
        metadata = _pair_metadata(root_path, seed, include_test)
        super().__init__(
            meta=metadata,
            label_names=["paired"],
            do_split=False,
            modality="audio",
        )

    def _waveform(self, file: Path) -> torch.Tensor:
        """Read a PCM WAV recording as a fixed-length mono waveform.

        Only the required number of frames is decoded. This deliberately uses
        Python's PCM WAV reader rather than torchaudio.load, whose TorchCodec
        decoder requires separate compatible FFmpeg shared libraries.

        Returns:
            Normalized float32 waveform of shape (clip_num_samples,).

        Raises:
            ValueError: If the sample rate, PCM encoding or bit depth is unsupported.
        """
        with wave.open(str(file), "rb") as recording:
            sample_rate = recording.getframerate()
            channels = recording.getnchannels()
            sample_width = recording.getsampwidth()
            if recording.getcomptype() != "NONE" or sample_width not in (1, 2, 3, 4):
                raise ValueError(f"Unsupported PCM WAV encoding in {file}")
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"Sample-rate mismatch in {file}: {sample_rate} Hz, "
                    f"expected {self.sample_rate} Hz."
                )
            raw = recording.readframes(self.clip_num_samples)

        if sample_width == 1:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 3:
            data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            values = (
                data[:, 0].astype(np.int32)
                | (data[:, 1].astype(np.int32) << 8)
                | (data[:, 2].astype(np.int32) << 16)
            )
            signed = (values ^ (1 << 23)) - (1 << 23)
            samples = signed.astype(np.float32) / float(1 << 23)
        else:
            dtype = np.dtype("<i2" if sample_width == 2 else "<i4")
            samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / float(
                1 << (8 * sample_width - 1)
            )

        mono = torch.from_numpy(samples.reshape(-1, channels).mean(axis=1))
        length = self.clip_num_samples
        if mono.numel() >= length:
            return mono[:length].contiguous()
        return torch.nn.functional.pad(mono, (0, length - mono.numel()))

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Load the two real acoustic endpoint recordings.

        Returns:
            TensorDict with independently sampled healthy and faulty waveforms.
        """
        return TensorDict(
            {
                "healthy_waveform": self._waveform(Path(str(row["healthy_file"]))),
                "faulty_waveform": self._waveform(Path(str(row["faulty_file"]))),
            },
            batch_size=[],
        )
