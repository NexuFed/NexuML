"""Acoustic health/fault endpoint data fragments for Flow Matching."""

from __future__ import annotations

from pathlib import Path

from nexuml.core.types import DataSpec, DatasetSpec, LoaderSpec
from nexuml.data.loaders.definitions import TorchLoader
from nexuml_library.data.maintenance.audio_endpoints import AudioEndpointPairs


def audio_endpoint_data(
    data_roots: list[str | Path],
    sample_rate: int = 16000,
    clip_num_samples: int = 2048,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> DataSpec:
    """Compose acoustic-only source datasets with protected group-level splits.

    Args:
        data_roots: Roots containing train/val healthy and faulty WAV directories.
        sample_rate: Required acoustic sampling rate in Hz.
        clip_num_samples: Fixed waveform size for each source and target.
        batch_size: Torch DataLoader batch size.
        num_workers: Number of worker processes.
        seed: Deterministic independent endpoint-pair sampling seed.

    Returns:
        DataSpec containing only acoustic endpoint pairs.

    Raises:
        ValueError: If no data root is configured.
    """
    if not data_roots:
        raise ValueError("Specify at least one acoustic endpoint dataset root")

    return DataSpec(
        datasets=[
            DatasetSpec(
                source=AudioEndpointPairs(
                    root=str(root),
                    sample_rate=sample_rate,
                    clip_num_samples=clip_num_samples,
                    seed=seed,
                ),
                modality="audio",
                split_type="keep",
            )
            for root in data_roots
        ],
        loader=LoaderSpec(
            backend=TorchLoader(),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
        input_shapes={
            "healthy_waveform": [clip_num_samples],
            "faulty_waveform": [clip_num_samples],
        },
    )
