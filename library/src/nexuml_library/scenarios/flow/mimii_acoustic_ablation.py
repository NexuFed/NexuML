"""Explicit prepared-feature MIMII flow factory and scenario-file entry point.

The factory requires a prepared artifact, so it is intentionally not exposed
through zero-argument global scenario discovery. Use the scenario-file entry
point or the validation-only ablation trainer.
"""

from __future__ import annotations

import os
from pathlib import Path

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.types import DatasetSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.prepared_endpoints import (
    PreparedMaintenanceEndpointPairs,
    read_prepared_artifact,
)
from nexuml_library.scenarios.flow.audio_health_to_fault import audio_health_to_fault_flow
from nexuml_library.scenarios.training.defaults import default_logging


def mimii_acoustic_ablation(
    prepared_file: str | Path,
    *,
    hidden_dims: list[int] | None = None,
    num_steps: int = 50,
    lr: float = 3e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 25,
    seed: int = 42,
    checkpoint_root: str | Path = ".experiments/checkpoints/nex233-ablation",
) -> ScenarioSpec:
    """Train the existing MLP flow on externally frozen prepared embeddings.

    The upstream FISHER frontend, independent/OT coupling and train-only
    normalization are recorded in the immutable prepared dataset. No weights
    of either audio feature extractor are trained in this scenario.

    Args:
        prepared_file: Validated preparation artifact with original splits.
        hidden_dims: Vector-field widths; use [512,512] for capacity experiment.
        num_steps: Number of Euler steps for post-training transport.
        lr: Identical optimizer learning rate for controlled comparisons.
        batch_size: Mini-batch size.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience on validation loss.
        seed: Training random seed; must match recorded pairing seed.
        checkpoint_root: Parent directory for uniquely named run artifacts.

    Returns:
        Serialisable pipeline using the existing flow/path/loss definitions.

    Raises:
        ValueError: If the supplied seed or optimizer configuration is invalid.
    """
    payload = read_prepared_artifact(prepared_file)
    settings = payload["settings"]
    if int(settings["seed"]) != seed:
        raise ValueError("Training seed must equal the prepared endpoint pairing seed")
    if not (lr > 0 and batch_size > 0 and max_epochs > 0 and patience > 0):
        raise ValueError("Learning rate and training controls must be positive")
    dimensions = hidden_dims if hidden_dims is not None else [128, 128]
    if not dimensions or any(width <= 0 for width in dimensions):
        raise ValueError("Flow hidden dimensions must be positive")
    feature_dim = int(payload["features"].shape[-1])
    variant = Path(prepared_file).stem
    run_id = f"{variant}-mlp-{'x'.join(str(d) for d in dimensions)}"
    checkpoint_dir = Path(checkpoint_root) / run_id

    baseline = audio_health_to_fault_flow(
        data_roots=["unused-prepared-features"],
        clip_num_samples=int(settings.get("clip_num_samples", 16384)),
        hidden_dims=dimensions,
        num_steps=num_steps,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=0,
        seed=seed,
    )
    flow_stages = {
        name: layers
        for name, layers in baseline.pipeline.stages.items()
        if name != "AudioFeatures"
    }
    prepared = PreparedMaintenanceEndpointPairs(path=str(Path(prepared_file).resolve()))
    data = baseline.data.model_copy(
        update={
            "datasets": [DatasetSpec(source=prepared, modality="audio", split_type="keep")],
            "input_shapes": {
                "healthy_spectrum": [feature_dim],
                "faulty_spectrum": [feature_dim],
            },
        }
    )
    return baseline.model_copy(
        update={
            "name": f"mimii_acoustic_ablation_{run_id.replace('-', '_')}",
            "pipeline": PipelineSpec(stages=flow_stages),
            "data": data,
            "callbacks": [
                callback(
                    ModelCheckpoint,
                    dirpath=str(checkpoint_dir),
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    filename="epoch={epoch:02d}-step={step:06d}",
                ),
                callback(EarlyStopping, monitor="val/loss", mode="min", patience=patience),
            ],
            "logging": default_logging(
                experiment_name="NEX-233 feature-space FM ablation",
                run_name=run_id,
                use_mlflow=False,
            ),
        }
    )


def scenario() -> ScenarioSpec:
    """Load one prepared variant from environment for the NexuML scenario CLI.

    Returns:
        Configured experiment using independent, disjoint checkpoint storage.

    Raises:
        ValueError: If no prepared-file environment setting is provided.
    """
    import lightning as L

    path = os.environ.get("NEX233_PREPARED_FILE")
    if not path:
        raise ValueError("Set NEX233_PREPARED_FILE to a prepared feature artifact")
    seed = int(os.environ.get("NEX233_SEED", "42"))
    L.seed_everything(seed, workers=True)
    width = int(os.environ.get("NEX233_FLOW_WIDTH", "128"))
    return mimii_acoustic_ablation(
        path,
        hidden_dims=[width, width],
        lr=float(os.environ.get("NEX233_LR", "0.0003")),
        max_epochs=int(os.environ.get("NEX233_MAX_EPOCHS", "300")),
        seed=seed,
        checkpoint_root=os.environ.get(
            "NEX233_CHECKPOINT_ROOT", ".experiments/checkpoints/nex233-ablation"
        ),
    )
