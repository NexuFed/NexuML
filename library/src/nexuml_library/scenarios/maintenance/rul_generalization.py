"""Controlled CfC readout overfitting experiment, keeping upstream models fixed.

Factorial two-by-two comparison of head width and dropout, repeated across
readout seeds. Both arms reuse the same pretrained FD001 seed-42 DPAN/Flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import PipelineSpec, ScenarioSpec
from nexuml_library.evaluation.maintenance.rul_generalization_diagnostics import (
    SequenceRULGeneralizationDiagnostics,
)
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_sequence_flow import (
    patchformer_sequence_flow_rul,
)

ReadoutVariant = Literal["w128_d010", "w128_d030", "w064_d010", "w064_d030"]

_VARIANTS: dict[ReadoutVariant, tuple[int, float]] = {
    "w128_d010": (128, 0.1),
    "w128_d030": (128, 0.3),
    "w064_d010": (64, 0.1),
    "w064_d030": (64, 0.3),
}


def cfc_generalization_study(
    arm: Literal["encoder", "flow"],
    variant: ReadoutVariant,
    head_seed: int = 42,
) -> ScenarioSpec:
    """Re-train identical CfC RUL controls across width, dropout and readout RNG.

    Args:
        arm: Matched frozen DPAN only or DPAN + CfC-flow displacement.
        variant: Factorial readout width and dropout setting.
        head_seed: New head initialization and data-loader seed (42-44).
            The frozen upstream encoder/flow remains FD001 seed 42.

    Returns:
        Native NexuML scenario with comparable per-epoch diagnostic callbacks.

    Raises:
        ValueError: If head seed is outside the predefined study schedule.
    """
    if head_seed not in (42, 43, 44):
        raise ValueError("CfC generalization head seed must be 42, 43 or 44")
    width, dropout = _VARIANTS[variant]
    base = patchformer_sequence_flow_rul(
        "FD001", 42, "cfc", arm == "flow"
    )
    run = f"rul-FD001-cfc-generalization-{arm}-{variant}-head-s{head_seed}"
    layer = base.pipeline.stages["RULModel"][0].model_copy(
        update={
            "component": PatchFormerSequenceFlowRULHead(
                hidden_dim=width,
                attention_heads=4,
                dropout=dropout,
                initialization_seed=head_seed,
            )
        }
    )
    stages = dict(base.pipeline.stages)
    stages["RULModel"] = [layer]
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
    checkpoint_dir = Path(".experiments/rul/checkpoints").resolve() / run
    callback_specs = list(base.callbacks)
    callback_specs[0] = callback_specs[0].model_copy(
        update={"kwargs": {"seed": head_seed}}
    )
    callback_specs[1] = callback_specs[1].model_copy(
        update={
            "kwargs": {
                "dirpath": str(checkpoint_dir),
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": True,
            }
        }
    )
    callback_specs[-1] = callback_specs[-1].model_copy(
        update={
            "kwargs": {
                "output_file": f".experiments/rul/audits/{run}.checkpoint.json"
            }
        }
    )
    callback_specs.insert(
        1,
        callback(
            SequenceRULGeneralizationDiagnostics,
            output_file=f".experiments/rul/generalization/{run}.curves.csv",
        ),
    )
    audits = list(base.evaluation.algorithms)
    audits[-1] = audits[-1].model_copy(
        update={
            "algorithm": audits[-1].algorithm.model_copy(
                update={"output_file": f".experiments/rul/audits/{run}.csv"}
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
            "callbacks": callback_specs,
        }
    )


@scenario("maintenance-rul-cfc-generalization-encoder-w128_d010-head-s42")
def cfc_generalization_encoder_w128_d010_head_s42() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w128_d010, head seed 42.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w128_d010", 42)


@scenario("maintenance-rul-cfc-generalization-encoder-w128_d030-head-s42")
def cfc_generalization_encoder_w128_d030_head_s42() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w128_d030, head seed 42.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w128_d030", 42)


@scenario("maintenance-rul-cfc-generalization-encoder-w064_d010-head-s42")
def cfc_generalization_encoder_w064_d010_head_s42() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w064_d010, head seed 42.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w064_d010", 42)


@scenario("maintenance-rul-cfc-generalization-encoder-w064_d030-head-s42")
def cfc_generalization_encoder_w064_d030_head_s42() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w064_d030, head seed 42.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w064_d030", 42)


@scenario("maintenance-rul-cfc-generalization-flow-w128_d010-head-s42")
def cfc_generalization_flow_w128_d010_head_s42() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w128_d010, head seed 42.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w128_d010", 42)


@scenario("maintenance-rul-cfc-generalization-flow-w128_d030-head-s42")
def cfc_generalization_flow_w128_d030_head_s42() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w128_d030, head seed 42.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w128_d030", 42)


@scenario("maintenance-rul-cfc-generalization-flow-w064_d010-head-s42")
def cfc_generalization_flow_w064_d010_head_s42() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w064_d010, head seed 42.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w064_d010", 42)


@scenario("maintenance-rul-cfc-generalization-flow-w064_d030-head-s42")
def cfc_generalization_flow_w064_d030_head_s42() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w064_d030, head seed 42.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w064_d030", 42)


@scenario("maintenance-rul-cfc-generalization-encoder-w128_d010-head-s43")
def cfc_generalization_encoder_w128_d010_head_s43() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w128_d010, head seed 43.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w128_d010", 43)


@scenario("maintenance-rul-cfc-generalization-encoder-w128_d030-head-s43")
def cfc_generalization_encoder_w128_d030_head_s43() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w128_d030, head seed 43.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w128_d030", 43)


@scenario("maintenance-rul-cfc-generalization-encoder-w064_d010-head-s43")
def cfc_generalization_encoder_w064_d010_head_s43() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w064_d010, head seed 43.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w064_d010", 43)


@scenario("maintenance-rul-cfc-generalization-encoder-w064_d030-head-s43")
def cfc_generalization_encoder_w064_d030_head_s43() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w064_d030, head seed 43.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w064_d030", 43)


@scenario("maintenance-rul-cfc-generalization-flow-w128_d010-head-s43")
def cfc_generalization_flow_w128_d010_head_s43() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w128_d010, head seed 43.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w128_d010", 43)


@scenario("maintenance-rul-cfc-generalization-flow-w128_d030-head-s43")
def cfc_generalization_flow_w128_d030_head_s43() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w128_d030, head seed 43.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w128_d030", 43)


@scenario("maintenance-rul-cfc-generalization-flow-w064_d010-head-s43")
def cfc_generalization_flow_w064_d010_head_s43() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w064_d010, head seed 43.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w064_d010", 43)


@scenario("maintenance-rul-cfc-generalization-flow-w064_d030-head-s43")
def cfc_generalization_flow_w064_d030_head_s43() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w064_d030, head seed 43.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w064_d030", 43)


@scenario("maintenance-rul-cfc-generalization-encoder-w128_d010-head-s44")
def cfc_generalization_encoder_w128_d010_head_s44() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w128_d010, head seed 44.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w128_d010", 44)


@scenario("maintenance-rul-cfc-generalization-encoder-w128_d030-head-s44")
def cfc_generalization_encoder_w128_d030_head_s44() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w128_d030, head seed 44.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w128_d030", 44)


@scenario("maintenance-rul-cfc-generalization-encoder-w064_d010-head-s44")
def cfc_generalization_encoder_w064_d010_head_s44() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w064_d010, head seed 44.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w064_d010", 44)


@scenario("maintenance-rul-cfc-generalization-encoder-w064_d030-head-s44")
def cfc_generalization_encoder_w064_d030_head_s44() -> ScenarioSpec:
    """Study CfC encoder RUL generalization: w064_d030, head seed 44.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("encoder", "w064_d030", 44)


@scenario("maintenance-rul-cfc-generalization-flow-w128_d010-head-s44")
def cfc_generalization_flow_w128_d010_head_s44() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w128_d010, head seed 44.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w128_d010", 44)


@scenario("maintenance-rul-cfc-generalization-flow-w128_d030-head-s44")
def cfc_generalization_flow_w128_d030_head_s44() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w128_d030, head seed 44.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w128_d030", 44)


@scenario("maintenance-rul-cfc-generalization-flow-w064_d010-head-s44")
def cfc_generalization_flow_w064_d010_head_s44() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w064_d010, head seed 44.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w064_d010", 44)


@scenario("maintenance-rul-cfc-generalization-flow-w064_d030-head-s44")
def cfc_generalization_flow_w064_d030_head_s44() -> ScenarioSpec:
    """Study CfC flow RUL generalization: w064_d030, head seed 44.

    Returns:
        Native NexuML paired fixed-upstream overfitting ablation.
    """
    return cfc_generalization_study("flow", "w064_d030", 44)
