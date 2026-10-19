"""MIMII normal/abnormal acoustic endpoint data for supervised flow experiments.

This is a custom supervised transport split of the original MIMII recordings,
not the official unsupervised anomaly-detection benchmark protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset
from nexuml_library.data.maintenance.audio_endpoints import _AudioEndpointPairsRuntime
from nexuml_library.data.maintenance.mimii_download import ensure_mimii

_MIMII_URL = "https://zenodo.org/records/3384388"
_MACHINE_TYPES = ("fan", "pump", "slider", "valve")


def _machine_directory(root: Path, machine_type: str) -> Path:
    """Locate a locally extracted MIMII machine directory.

    Returns:
        Directory containing the requested machine-ID subdirectories.

    Raises:
        FileNotFoundError: If the extracted MIMII machine type is absent.
    """
    for candidate in (root, root / machine_type):
        if any(candidate.glob("id_*")):
            return candidate
    raise FileNotFoundError(
        f"No MIMII {machine_type} machine IDs under {root}. "
        f"Extract the corresponding archive from {_MIMII_URL} and pass its "
        "machine-type directory as root."
    )


def _recordings(machine_dir: Path, machine_id: str, condition: str) -> list[Path]:
    """Collect one machine's original acoustic recordings.

    Returns:
        Stable path-sorted list of WAV recordings.

    Raises:
        FileNotFoundError: If no recordings exist for that machine and condition.
    """
    folder = machine_dir / machine_id / condition
    files = sorted(folder.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No MIMII {condition} recordings in {folder}")
    return files


def _mimii_metadata(
    *,
    root: Path,
    machine_type: str,
    train_machine_ids: list[str],
    val_machine_ids: list[str],
    test_machine_ids: list[str],
    include_test: bool,
    max_pairs_per_machine: int,
    seed: int,
) -> pd.DataFrame:
    """Pair original MIMII recordings within machine-disjoint partitions.

    Returns:
        Per-pair metadata with machine, source file and split provenance.
    """
    machine_dir = _machine_directory(root, machine_type)
    splits = {"train": train_machine_ids, "val": val_machine_ids}
    if include_test:
        splits["test"] = test_machine_ids

    rows: list[dict[str, str | int]] = []
    for split_index, (split, machine_ids) in enumerate(splits.items()):
        for id_index, machine_id in enumerate(machine_ids):
            normal = _recordings(machine_dir, machine_id, "normal")
            abnormal = _recordings(machine_dir, machine_id, "abnormal")
            rng = np.random.default_rng(seed + 1000 * split_index + id_index)
            normal_order = rng.permutation(len(normal))
            abnormal_order = rng.permutation(len(abnormal))
            count = min(len(normal), len(abnormal), max_pairs_per_machine)

            for pair_index in range(count):
                healthy_file = normal[int(normal_order[pair_index % len(normal)])]
                faulty_file = abnormal[int(abnormal_order[pair_index % len(abnormal)])]
                rows.append(
                    {
                        "healthy_file": str(healthy_file),
                        "faulty_file": str(faulty_file),
                        "healthy_group": f"{machine_type}:{machine_id}",
                        "faulty_group": f"{machine_type}:{machine_id}",
                        "group_id": f"{machine_type}:{machine_id}",
                        "machine_type": machine_type,
                        "machine_id": machine_id,
                        "dataset_name": "MIMII",
                        "source_url": _MIMII_URL,
                        "source_dataset": str(machine_dir),
                        "pair_id": f"{machine_type}:{machine_id}:{pair_index}",
                        "paired": 1,
                        "split": split,
                    }
                )
    return pd.DataFrame.from_records(rows)


@data_source("MIMIIAcousticEndpointPairs")
class MIMIIAcousticEndpointPairs(DataSourceDefinition):
    """Locally extracted MIMII machine sounds with disjoint machine-ID splits.

    These data are used in a custom supervised normal-to-abnormal transport
    experiment. They must not be reported as an official unsupervised MIMII
    anomaly-detection benchmark result.

    Attributes:
        root: Extracted MIMII machine-type root or its parent.
        download: Automatically download missing recordings from official Zenodo.
        noise_db: Noise level of the selected official MIMII recording archive.
        machine_type: Selected machine family (one family per experiment).
        train_machine_ids: Machines used to fit the flow.
        val_machine_ids: Machines used for checkpoint selection.
        test_machine_ids: Machines held out of training and validation.
        include_test: Index held-out test machines without using them for fit.
        max_pairs_per_machine: Maximum normal/abnormal pairs per machine.
        sample_rate: Required input audio sampling rate.
        clip_num_samples: Fixed audio clip size after cropping/padding.
        seed: Reproducible independent endpoint pairing.
    """

    root: str | Path = "data/mimii/6_dB_fan"
    download: bool = True
    noise_db: Literal[-6, 0, 6] = 6
    machine_type: Literal["fan", "pump", "slider", "valve"] = "fan"
    train_machine_ids: list[str] = Field(default_factory=lambda: ["id_00", "id_02"])
    val_machine_ids: list[str] = Field(default_factory=lambda: ["id_04"])
    test_machine_ids: list[str] = Field(default_factory=lambda: ["id_06"])
    include_test: bool = True
    max_pairs_per_machine: int = Field(default=128, gt=0)
    sample_rate: int = Field(default=16000, gt=0)
    clip_num_samples: int = Field(default=16384, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def validate_machine_splits(self) -> "MIMIIAcousticEndpointPairs":
        """Reject machine identity overlap across experiment partitions.

        Returns:
            Validated definition.

        Raises:
            ValueError: If a split is empty, contains duplicates or shares IDs.
        """
        if not self.train_machine_ids or not self.val_machine_ids:
            raise ValueError("MIMII requires nonempty train and val machine-ID splits")
        selected = {
            "train": self.train_machine_ids,
            "val": self.val_machine_ids,
            "test": self.test_machine_ids,
        }
        if self.include_test and not self.test_machine_ids:
            raise ValueError("MIMII include_test=True requires held-out test IDs")
        seen: set[str] = set()
        for split, machine_ids in selected.items():
            for machine_id in machine_ids:
                if not machine_id.startswith("id_") or not machine_id[3:].isdigit():
                    raise ValueError(f"Invalid MIMII machine ID: {machine_id}")
                if machine_id in seen:
                    raise ValueError(f"MIMII machine ID {machine_id} overlaps split {split}")
                seen.add(machine_id)
        return self

    def build(self) -> NexuDataset:
        """Build a lazy dataset over original MIMII multichannel WAV files.

        Returns:
            NexuDataset with healthy/faulty waveform endpoints.
        """
        return _MIMIIAcousticEndpointRuntime(**self.model_dump())


class _MIMIIAcousticEndpointRuntime(_AudioEndpointPairsRuntime):
    """Reuse the shared PCM WAV decoder with direct MIMII metadata indexing."""

    def __init__(
        self,
        root: str | Path,
        download: bool,
        noise_db: int,
        machine_type: str,
        train_machine_ids: list[str],
        val_machine_ids: list[str],
        test_machine_ids: list[str],
        include_test: bool,
        max_pairs_per_machine: int,
        sample_rate: int,
        clip_num_samples: int,
        seed: int,
    ) -> None:
        if machine_type not in _MACHINE_TYPES:
            raise ValueError(f"Unknown MIMII machine type: {machine_type}")
        self.sample_rate = sample_rate
        self.clip_num_samples = clip_num_samples
        root_path = Path(root).expanduser()
        ensure_mimii(root_path, machine_type, noise_db, download)
        metadata = _mimii_metadata(
            root=root_path,
            machine_type=machine_type,
            train_machine_ids=train_machine_ids,
            val_machine_ids=val_machine_ids,
            test_machine_ids=test_machine_ids,
            include_test=include_test,
            max_pairs_per_machine=max_pairs_per_machine,
            seed=seed,
        )
        NexuDataset.__init__(
            self,
            meta=metadata,
            label_names=["paired"],
            do_split=False,
            modality="audio",
        )
