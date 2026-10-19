"""Unconditional acoustic health-to-fault Flow Matching reference scenario."""

from __future__ import annotations

from pathlib import Path

from nexuml.core.discovery import scenario
from nexuml.core.types import (
    EvaluationSpec,
    LayerSpec,
    PipelineSpec,
    ScenarioSpec,
)
from nexuml_library.layers.feature.lmbe import LMBE
from nexuml_library.layers.generative.flow.paired_path import PairedLinearFlowPath
from nexuml_library.layers.loss.flow_matching_loss import FlowMatchingLoss
from nexuml_library.layers.model.flow import Flow
from nexuml_library.scenarios.data.audio_endpoints import audio_endpoint_data
from nexuml_library.scenarios.training.defaults import default_logging, default_training


@scenario("audio-health-to-fault-flow")
def audio_health_to_fault_flow(
    data_roots: list[str | Path] | None = None,
    sample_rate: int = 16000,
    clip_num_samples: int = 2048,
    n_fft: int = 256,
    hop_length: int = 128,
    n_mels: int = 16,
    hidden_dims: list[int] | None = None,
    num_steps: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 20,
    num_workers: int = 0,
    seed: int = 42,
) -> ScenarioSpec:
    """Train an unconditional flow from healthy to faulty acoustic recordings.

    Each endpoint is transformed using the same fixed log-mel configuration.
    PairedLinearFlowPath then samples a linear transport between independent
    healthy/faulty spectral observations. Neither physical degradation time
    nor remaining useful life is estimated by this scenario.

    Args:
        data_roots: One or more roots with explicit train/val audio endpoints.
        sample_rate: Required waveform sampling rate.
        clip_num_samples: Fixed number of audio samples per recording.
        n_fft: FFT length for the reused LMBE feature layer.
        hop_length: Log-mel frame hop length.
        n_mels: Number of mel bands.
        hidden_dims: Hidden widths of the existing MLP vector field.
        num_steps: Euler integration steps for sampling from healthy endpoints.
        lr: Optimizer learning rate.
        batch_size: Mini-batch size.
        max_epochs: Maximum training epochs.
        num_workers: Native Torch DataLoader workers.
        seed: Seed for deterministic independent endpoint pairing.

    Returns:
        Typed NexuML scenario with log-mel audio transport and Flow Matching loss.

    Raises:
        ValueError: If the audio clip is shorter than the FFT window.
    """
    if clip_num_samples < n_fft:
        raise ValueError("clip_num_samples must be at least n_fft")
    roots = data_roots if data_roots is not None else ["data/maintenance/audio_endpoints"]
    mel = {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "fmax": sample_rate // 2,
    }

    return ScenarioSpec(
        name="audio_health_to_fault_flow",
        pipeline=PipelineSpec(
            stages={
                "AudioFeatures": [
                    LayerSpec(
                        component=LMBE(**mel),
                        keys_in=["healthy_waveform"],
                        keys_out=["healthy_spectrum"],
                    ),
                    LayerSpec(
                        component=LMBE(**mel),
                        keys_in=["faulty_waveform"],
                        keys_out=["faulty_spectrum"],
                    ),
                ],
                "FlowPath": [
                    LayerSpec(
                        component=PairedLinearFlowPath(),
                        keys_in=["healthy_spectrum", "faulty_spectrum"],
                        keys_out=["flow_state", "flow_time", "flow_target_velocity"],
                    )
                ],
                "Flow": [
                    LayerSpec(
                        component=Flow(
                            hidden_dims=hidden_dims if hidden_dims is not None else [128, 128],
                            num_steps=num_steps,
                        ),
                        keys_in=["flow_state", "flow_time"],
                        keys_out=["predicted_velocity"],
                    )
                ],
                "Loss": [
                    LayerSpec(
                        component=FlowMatchingLoss(),
                        keys_in=["predicted_velocity", "flow_target_velocity"],
                        keys_out=["flow_matching_loss"],
                    )
                ],
            }
        ),
        data=audio_endpoint_data(
            data_roots=roots,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        ),
        evaluation=EvaluationSpec(metrics=[], algorithms=[]),
        logging=default_logging(
            experiment_name="Predictive Maintenance - Audio Flow Matching",
            run_name="audio-health-to-fault-flow",
        ),
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"flow_matching_loss": 1.0},
        ),
    )
