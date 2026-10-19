"""Reproducible real-machine acoustic Flow Matching experiment using MIMII."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario as register_scenario
from nexuml.core.types import DatasetSpec, ScenarioSpec
from nexuml_library.data.maintenance.mimii import MIMIIAcousticEndpointPairs
from nexuml_library.scenarios.flow.audio_health_to_fault import audio_health_to_fault_flow
from nexuml_library.scenarios.training.defaults import default_logging


@register_scenario("mimii-acoustic-flow")
def mimii_acoustic_flow(
    root: str | Path | None = None,
    machine_type: Literal["fan", "pump", "slider", "valve"] = "fan",
    train_machine_ids: list[str] | None = None,
    val_machine_ids: list[str] | None = None,
    test_machine_ids: list[str] | None = None,
    max_pairs_per_machine: int = 128,
    clip_num_samples: int = 16384,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 32,
    hidden_dims: list[int] | None = None,
    num_steps: int = 50,
    lr: float = 3e-4,
    batch_size: int = 16,
    max_epochs: int = 30,
    num_workers: int = 2,
    seed: int = 42,
    checkpoint_dir: str = ".experiments/checkpoints/mimii-acoustic-flow",
) -> ScenarioSpec:
    """Compose real-audio Flow Matching with held-out validation and checkpoints.

    Source recordings remain in their original MIMII directories. Each
    machine ID belongs exclusively to train, val or test. The transport
    objective is supervised and not the MIMII unsupervised ASD protocol.

    Args:
        root: Original MIMII machine directory; defaults to NEXUML_MIMII_ROOT.
        machine_type: Single industrial machine family.
        train_machine_ids: Training IDs (default 00 and 02).
        val_machine_ids: Validation ID (default 04).
        test_machine_ids: Independent test ID (default 06).
        max_pairs_per_machine: Balanced capped sampling per machine ID.
        clip_num_samples: Audio length in samples at 16 kHz.
        n_fft: Size of the log-mel transform's FFT.
        hop_length: Log-mel hop size.
        n_mels: Number of mel bands.
        hidden_dims: Existing NEX-196 Flow model hidden widths.
        num_steps: Number of Euler integration steps.
        lr: Learning rate.
        batch_size: Torch DataLoader mini-batch size.
        max_epochs: Maximum number of training epochs.
        num_workers: DataLoader workers.
        seed: Independent endpoint-pairing RNG seed.
        checkpoint_dir: Directory for monitored best/last trainer checkpoints.

    Returns:
        Typed NexuML real-data Flow Matching experiment.
    """
    resolved_root = str(
        root if root is not None else os.environ.get("NEXUML_MIMII_ROOT", "data/mimii/6_dB_fan")
    )
    baseline = audio_health_to_fault_flow(
        data_roots=[resolved_root],
        sample_rate=16000,
        clip_num_samples=clip_num_samples,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        hidden_dims=hidden_dims,
        num_steps=num_steps,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
        seed=seed,
    )
    source = MIMIIAcousticEndpointPairs(
        root=resolved_root,
        machine_type=machine_type,
        train_machine_ids=(
            train_machine_ids if train_machine_ids is not None else ["id_00", "id_02"]
        ),
        val_machine_ids=val_machine_ids if val_machine_ids is not None else ["id_04"],
        test_machine_ids=test_machine_ids if test_machine_ids is not None else ["id_06"],
        include_test=True,
        max_pairs_per_machine=max_pairs_per_machine,
        sample_rate=16000,
        clip_num_samples=clip_num_samples,
        seed=seed,
    )
    data = baseline.data.model_copy(
        update={
            "datasets": [
                DatasetSpec(source=source, modality="audio", split_type="keep")
            ]
        }
    )
    return baseline.model_copy(
        update={
            "name": "mimii_acoustic_flow",
            "data": data,
            "callbacks": [
                callback(
                    ModelCheckpoint,
                    dirpath=checkpoint_dir,
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    filename="epoch={epoch:02d}-step={step:06d}",
                ),
                callback(EarlyStopping, monitor="val/loss", mode="min", patience=8),
            ],
            "logging": default_logging(
                experiment_name="NEX-233 MIMII acoustic health-to-fault transport",
                run_name=f"mimii-{machine_type}-seed-{seed}",
                use_mlflow=False,
            ),
        }
    )


def scenario() -> ScenarioSpec:
    """Expose a reproducibly seeded trusted NexuML scenario-file entrypoint.

    Returns:
        Default MIMII acoustic experiment configured by NEXUML_MIMII_ROOT.
    """
    import lightning as L

    L.seed_everything(42, workers=True)
    return mimii_acoustic_flow()
