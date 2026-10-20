"""NexuML scenario factory for healthy-reference-conditioned NEX-233 flows.

Reuses the existing PairedLinearFlowPath, FlowMatchingLoss, optimizer
and data/pipeline specifications. Only the library Flow layer is replaced.
"""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.types import DatasetSpec, LayerSpec, PipelineSpec, ScenarioSpec
from nexuml_library.data.maintenance.healthy_reference import (
    HealthyReferenceMaintenancePairs,
    read_healthy_reference_training,
)
from nexuml_library.evaluation.maintenance.ablation_checkpoint import SaveLatestEpoch
from nexuml_library.evaluation.maintenance.reference_conditioned_validation import (
    HealthyReferenceTransportValidation,
)
from nexuml_library.layers.model.healthy_reference_flow import (
    HealthyReferenceConditionedFlow,
)
from nexuml_library.scenarios.flow.audio_health_to_fault import audio_health_to_fault_flow
from nexuml_library.scenarios.training.defaults import default_logging


def healthy_reference_flow_scenario(
    training_file: str | Path,
    *,
    width: int = 128,
    context_dim: int = 32,
    encoder_width: int = 128,
    context_mode: str = "reference",
    num_steps: int = 50,
    lr: float = 3e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 25,
    seed: int = 42,
    checkpoint_root: str | Path = ".experiments/checkpoints/nex233-reference",
) -> ScenarioSpec:
    """Build a normal NexuML training scenario with healthy set conditioning.

    Args:
        training_file: Prepared training-only fold (never an assessment path).
        width: Vector-field hidden width for both MLP layers.
        context_dim: Continuous DeepSets context dimensionality.
        encoder_width: DeepSets pointwise encoder hidden width.
        context_mode: Reference-conditioned or constant-context ablation.
        num_steps: Euler steps during validation and evaluation.
        lr: Optimizer learning rate.
        batch_size: Training mini-batch size.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience using mean machine validation SWD.
        seed: Training and preparation seed.
        checkpoint_root: Unique checkpoint directory root.

    Returns:
        Typed pipeline preserving the base architecture and loss machinery.

    Raises:
        ValueError: For invalid configuration or mismatched fold provenance.
    """
    prepared = read_healthy_reference_training(training_file)
    settings = prepared["settings"]
    if int(settings["seed"]) != seed:
        raise ValueError("Training seed must match the prepared fold seed")
    if not all(value > 0 for value in (
        width, context_dim, encoder_width, num_steps, batch_size, max_epochs, patience
    )) or lr <= 0:
        raise ValueError("Healthy-reference flow hyperparameters must be positive")
    if context_mode not in {"reference", "constant"}:
        raise ValueError("Context mode must be reference or constant")

    feature_dim = int(prepared["features"].shape[-1])
    reference_size = int(prepared["references"].shape[1])
    folder = Path(training_file).stem
    run_name = (
        f"{folder}-{context_mode}-mlp-{width}x{width}"
        f"-context-{context_dim}-seed{seed}"
    )
    checkpoints = Path(checkpoint_root) / run_name
    baseline = audio_health_to_fault_flow(
        data_roots=["unused-healthy-reference-prepared-features"],
        hidden_dims=[width, width],
        num_steps=num_steps,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=0,
        seed=seed,
    )
    stages = {
        stage: layers
        for stage, layers in baseline.pipeline.stages.items()
        if stage != "AudioFeatures"
    }
    stages["Flow"] = [
        LayerSpec(
            component=HealthyReferenceConditionedFlow(
                hidden_dims=[width, width],
                context_dim=context_dim,
                encoder_hidden_dim=encoder_width,
                num_steps=num_steps,
                context_mode=context_mode,
            ),
            keys_in=["flow_state", "flow_time", "healthy_reference"],
            keys_out=["predicted_velocity"],
        )
    ]
    source = HealthyReferenceMaintenancePairs(path=str(Path(training_file).resolve()))
    data = baseline.data.model_copy(
        update={
            "datasets": [DatasetSpec(source=source, modality="audio", split_type="keep")],
            "input_shapes": {
                "healthy_spectrum": [feature_dim],
                "faulty_spectrum": [feature_dim],
                "healthy_reference": [reference_size, feature_dim],
            },
        }
    )
    return baseline.model_copy(
        update={
            "name": f"healthy_reference_flow_{run_name.replace('-', '_')}",
            "pipeline": PipelineSpec(stages=stages),
            "data": data,
            "callbacks": [
                callback(
                    HealthyReferenceTransportValidation,
                    training_file=str(Path(training_file).resolve()),
                    batch_size=batch_size,
                    projections=128,
                    seed=seed,
                ),
                callback(
                    ModelCheckpoint,
                    dirpath=str(checkpoints),
                    monitor="val/transport_swd",
                    mode="min",
                    save_top_k=1,
                    save_last=False,
                    filename="{epoch:02d}-{step:06d}",
                ),
                callback(SaveLatestEpoch, dirpath=str(checkpoints)),
                callback(
                    EarlyStopping,
                    monitor="val/transport_swd",
                    mode="min",
                    patience=patience,
                ),
            ],
            "logging": default_logging(
                experiment_name="NEX-233 healthy-reference conditional Flow Matching",
                run_name=run_name,
                use_mlflow=False,
            ),
        }
    )
