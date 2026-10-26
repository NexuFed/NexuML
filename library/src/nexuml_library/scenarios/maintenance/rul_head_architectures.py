"""Fixed-upstream RUL readout architecture comparison on FD001.

Only the temporal readout changes: Transformer, Mamba-style selective SSM,
or CfC. PatchFormer and CfC Flow stay frozen at upstream seed 42.
Both feature arms use the Gen2 validation-chosen optimizer/protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from nexuml.core.discovery import scenario
from nexuml.core.types import PipelineSpec, ScenarioSpec
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_gen2 import (
    cfc_rul_optimization_study,
)

HeadArchitecture = Literal["transformer", "mamba", "cfc"]
FeatureArm = Literal["encoder", "flow"]


def cfc_rul_head_architecture_study(
    architecture: HeadArchitecture,
    arm: FeatureArm,
    head_seed: int = 42,
) -> ScenarioSpec:
    """Compare three readout backbones under one locked Gen2 protocol.

    Args:
        architecture: Transformer, unfused Mamba-style SSM, or CfC.
        arm: Original frozen DPAN or DPAN plus frozen CfC displacement.
        head_seed: RUL readout and data-loader seed, 42, 43 or 44.

    Returns:
        Registered-compatible native NexuML readout benchmark.

    Raises:
        ValueError: If the head seed is not part of the preset study.
    """
    if head_seed not in (42, 43, 44):
        raise ValueError("RUL head comparison requires seed 42, 43 or 44")
    base = cfc_rul_optimization_study(
        arm, "d010", "adam_1e3", "window", head_seed
    )
    run = f"rul-FD001-cfc-head-{architecture}-{arm}-s{head_seed}"
    stages = dict(base.pipeline.stages)
    stages["RULModel"] = [
        stages["RULModel"][0].model_copy(
            update={
                "component": PatchFormerSequenceFlowRULHead(
                    architecture=architecture,
                    hidden_dim=64,
                    attention_heads=4,
                    dropout=0.1,
                    initialization_seed=head_seed,
                )
            }
        )
    ]
    source = base.data.datasets[0].source
    data = base.data.model_copy(
        update={
            "datasets": [
                entry.model_copy(
                    update={
                        "source": source.model_copy(
                            update={
                                "diagnostics_file": (
                                    f".experiments/rul/audits/{run}.geometry.json"
                                )
                            }
                        )
                    }
                )
                for entry in base.data.datasets
            ]
        }
    )
    callbacks = list(base.callbacks)
    callbacks[1] = callbacks[1].model_copy(
        update={
            "kwargs": {
                "output_file": f".experiments/rul/head-architectures/{run}.curves.csv"
            }
        }
    )
    callbacks[2] = callbacks[2].model_copy(
        update={
            "kwargs": {
                "dirpath": str(
                    Path(".experiments/rul/checkpoints").resolve() / run
                ),
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": True,
            }
        }
    )
    callbacks[-1] = callbacks[-1].model_copy(
        update={
            "kwargs": {
                "output_file": f".experiments/rul/audits/{run}.checkpoint.json"
            }
        }
    )
    audits = list(base.evaluation.algorithms)
    audits[-1] = audits[-1].model_copy(
        update={
            "algorithm": audits[-1].algorithm.model_copy(
                update={
                    "output_file": f".experiments/rul/audits/{run}.csv"
                }
            )
        }
    )
    return base.model_copy(
        update={
            "name": run,
            "data": data,
            "pipeline": PipelineSpec(stages=stages),
            "evaluation": base.evaluation.model_copy(
                update={"algorithms": audits}
            ),
            "logging": (
                base.logging.model_copy(update={"run_name": run})
                if base.logging is not None else None
            ),
            "callbacks": callbacks,
        }
    )


@scenario("maintenance-rul-cfc-head-transformer-encoder-s42")
def cfc_head_transformer_encoder_s42() -> ScenarioSpec:
    """Train the transformer RUL readout with encoder features, seed 42.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("transformer", "encoder", 42)


@scenario("maintenance-rul-cfc-head-transformer-flow-s42")
def cfc_head_transformer_flow_s42() -> ScenarioSpec:
    """Train the transformer RUL readout with flow features, seed 42.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("transformer", "flow", 42)


@scenario("maintenance-rul-cfc-head-mamba-encoder-s42")
def cfc_head_mamba_encoder_s42() -> ScenarioSpec:
    """Train the mamba RUL readout with encoder features, seed 42.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("mamba", "encoder", 42)


@scenario("maintenance-rul-cfc-head-mamba-flow-s42")
def cfc_head_mamba_flow_s42() -> ScenarioSpec:
    """Train the mamba RUL readout with flow features, seed 42.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("mamba", "flow", 42)


@scenario("maintenance-rul-cfc-head-cfc-encoder-s42")
def cfc_head_cfc_encoder_s42() -> ScenarioSpec:
    """Train the cfc RUL readout with encoder features, seed 42.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("cfc", "encoder", 42)


@scenario("maintenance-rul-cfc-head-cfc-flow-s42")
def cfc_head_cfc_flow_s42() -> ScenarioSpec:
    """Train the cfc RUL readout with flow features, seed 42.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("cfc", "flow", 42)


@scenario("maintenance-rul-cfc-head-transformer-encoder-s43")
def cfc_head_transformer_encoder_s43() -> ScenarioSpec:
    """Train the transformer RUL readout with encoder features, seed 43.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("transformer", "encoder", 43)


@scenario("maintenance-rul-cfc-head-transformer-flow-s43")
def cfc_head_transformer_flow_s43() -> ScenarioSpec:
    """Train the transformer RUL readout with flow features, seed 43.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("transformer", "flow", 43)


@scenario("maintenance-rul-cfc-head-mamba-encoder-s43")
def cfc_head_mamba_encoder_s43() -> ScenarioSpec:
    """Train the mamba RUL readout with encoder features, seed 43.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("mamba", "encoder", 43)


@scenario("maintenance-rul-cfc-head-mamba-flow-s43")
def cfc_head_mamba_flow_s43() -> ScenarioSpec:
    """Train the mamba RUL readout with flow features, seed 43.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("mamba", "flow", 43)


@scenario("maintenance-rul-cfc-head-cfc-encoder-s43")
def cfc_head_cfc_encoder_s43() -> ScenarioSpec:
    """Train the cfc RUL readout with encoder features, seed 43.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("cfc", "encoder", 43)


@scenario("maintenance-rul-cfc-head-cfc-flow-s43")
def cfc_head_cfc_flow_s43() -> ScenarioSpec:
    """Train the cfc RUL readout with flow features, seed 43.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("cfc", "flow", 43)


@scenario("maintenance-rul-cfc-head-transformer-encoder-s44")
def cfc_head_transformer_encoder_s44() -> ScenarioSpec:
    """Train the transformer RUL readout with encoder features, seed 44.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("transformer", "encoder", 44)


@scenario("maintenance-rul-cfc-head-transformer-flow-s44")
def cfc_head_transformer_flow_s44() -> ScenarioSpec:
    """Train the transformer RUL readout with flow features, seed 44.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("transformer", "flow", 44)


@scenario("maintenance-rul-cfc-head-mamba-encoder-s44")
def cfc_head_mamba_encoder_s44() -> ScenarioSpec:
    """Train the mamba RUL readout with encoder features, seed 44.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("mamba", "encoder", 44)


@scenario("maintenance-rul-cfc-head-mamba-flow-s44")
def cfc_head_mamba_flow_s44() -> ScenarioSpec:
    """Train the mamba RUL readout with flow features, seed 44.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("mamba", "flow", 44)


@scenario("maintenance-rul-cfc-head-cfc-encoder-s44")
def cfc_head_cfc_encoder_s44() -> ScenarioSpec:
    """Train the cfc RUL readout with encoder features, seed 44.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("cfc", "encoder", 44)


@scenario("maintenance-rul-cfc-head-cfc-flow-s44")
def cfc_head_cfc_flow_s44() -> ScenarioSpec:
    """Train the cfc RUL readout with flow features, seed 44.

    Returns:
        Matched frozen-upstream RUL head comparison scenario.
    """
    return cfc_rul_head_architecture_study("cfc", "flow", 44)


