"""Matched CfC RUL optimization and engine-balanced supervision ablations.

All experiments share the same FD001 seed-42 frozen PatchFormer/CfC weights,
the exact original engine-disjoint split and a 64-wide attention readout.
Independent head seeds alter readout initialization and batch RNG only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from nexuml import callback, optimizer
from nexuml.core.discovery import scenario
from nexuml.core.types import LayerSpec, PipelineSpec, ScenarioSpec
from nexuml_library.evaluation.maintenance.rul_generalization_diagnostics import (
    SequenceRULGeneralizationDiagnostics,
)
from nexuml_library.layers.model.rul_engine_loss import (
    RULInverseEngineFrequencyLoss,
)
from nexuml_library.scenarios.maintenance.rul_generalization import (
    cfc_generalization_study,
)

OptimizerChoice = Literal["adam_1e3", "adamw_3e4", "adamw_1e4"]
WeightingChoice = Literal["window", "engine"]
DropoutChoice = Literal["d010", "d030"]

_OPTIMIZERS: dict[OptimizerChoice, tuple[float, float]] = {
    "adam_1e3": (1e-3, 0.0),
    "adamw_3e4": (3e-4, 1e-2),
    "adamw_1e4": (1e-4, 1e-2),
}


def cfc_rul_optimization_study(
    arm: Literal["encoder", "flow"],
    dropout: DropoutChoice,
    optimizer_choice: OptimizerChoice,
    weighting: WeightingChoice,
    head_seed: int = 42,
) -> ScenarioSpec:
    """Compare optimizer, engine weighting and regularization without leakage.

    Args:
        arm: Frozen encoder-only or frozen encoder plus CfC-flow features.
        dropout: Width-64 temporal-head dropout of 0.1 or 0.3.
        optimizer_choice: Adam reference or a predeclared AdamW configuration.
        weighting: Plain window MSE or inverse-engine-window-frequency MSE.
        head_seed: Independent readout-only RNG seed in 42, 43, 44.

    Returns:
        Native NexuML RUL training, diagnostics and official audit scenario.
    """
    base = cfc_generalization_study(
        arm, "w064_d010" if dropout == "d010" else "w064_d030",
        head_seed,
    )
    lr, decay = _OPTIMIZERS[optimizer_choice]
    run = (
        f"rul-FD001-cfc-gen2-{arm}-{dropout}-"
        f"{optimizer_choice}-{weighting}-head-s{head_seed}"
    )
    source = base.data.datasets[0].source
    updated_data = base.data.model_copy(
        update={
            "datasets": [
                entry.model_copy(
                    update={
                        "source": source.model_copy(
                            update={
                                "engine_balanced": weighting == "engine",
                                "diagnostics_file": (
                                    f".experiments/rul/audits/{run}.geometry.json"
                                ),
                            }
                        )
                    }
                )
                for entry in base.data.datasets
            ]
        }
    )
    stages = dict(base.pipeline.stages)
    if weighting == "engine":
        stages["RULLoss"] = [
            LayerSpec(
                component=RULInverseEngineFrequencyLoss(),
                keys_in=["predicted_rul"],
                keys_out=["rul_loss"],
            ),
            *stages["RULLoss"][1:],
        ]
    if optimizer_choice == "adam_1e3":
        optimizer_spec = optimizer(torch.optim.Adam, lr=lr)
    else:
        optimizer_spec = optimizer(
            torch.optim.AdamW, lr=lr, weight_decay=decay
        )
    callbacks = list(base.callbacks)
    callbacks[1] = callback(
        SequenceRULGeneralizationDiagnostics,
        output_file=f".experiments/rul/gen2/{run}.curves.csv",
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
                update={"output_file": f".experiments/rul/audits/{run}.csv"}
            )
        }
    )
    return base.model_copy(
        update={
            "name": run,
            "data": updated_data,
            "pipeline": PipelineSpec(stages=stages),
            "training": base.training.model_copy(
                update={"optimizer": optimizer_spec, "lr": lr}
            ),
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


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adam_1e3-window-head-s42")
def cfc_gen2_encoder_d010_adam_1e3_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adam_1e3, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adam_1e3", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adam_1e3-engine-head-s42")
def cfc_gen2_encoder_d010_adam_1e3_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adam_1e3, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adam_1e3", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_3e4-window-head-s42")
def cfc_gen2_encoder_d010_adamw_3e4_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_3e4, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_3e4", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_3e4-engine-head-s42")
def cfc_gen2_encoder_d010_adamw_3e4_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_3e4, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_3e4", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_1e4-window-head-s42")
def cfc_gen2_encoder_d010_adamw_1e4_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_1e4, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_1e4", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_1e4-engine-head-s42")
def cfc_gen2_encoder_d010_adamw_1e4_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_1e4, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_1e4", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adam_1e3-window-head-s42")
def cfc_gen2_encoder_d030_adam_1e3_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adam_1e3, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adam_1e3", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adam_1e3-engine-head-s42")
def cfc_gen2_encoder_d030_adam_1e3_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adam_1e3, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adam_1e3", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_3e4-window-head-s42")
def cfc_gen2_encoder_d030_adamw_3e4_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_3e4, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_3e4", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_3e4-engine-head-s42")
def cfc_gen2_encoder_d030_adamw_3e4_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_3e4, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_3e4", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_1e4-window-head-s42")
def cfc_gen2_encoder_d030_adamw_1e4_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_1e4, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_1e4", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_1e4-engine-head-s42")
def cfc_gen2_encoder_d030_adamw_1e4_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_1e4, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_1e4", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adam_1e3-window-head-s42")
def cfc_gen2_flow_d010_adam_1e3_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adam_1e3, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adam_1e3", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adam_1e3-engine-head-s42")
def cfc_gen2_flow_d010_adam_1e3_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adam_1e3, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adam_1e3", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_3e4-window-head-s42")
def cfc_gen2_flow_d010_adamw_3e4_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_3e4, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_3e4", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_3e4-engine-head-s42")
def cfc_gen2_flow_d010_adamw_3e4_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_3e4, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_3e4", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_1e4-window-head-s42")
def cfc_gen2_flow_d010_adamw_1e4_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_1e4, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_1e4", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_1e4-engine-head-s42")
def cfc_gen2_flow_d010_adamw_1e4_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_1e4, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_1e4", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adam_1e3-window-head-s42")
def cfc_gen2_flow_d030_adam_1e3_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adam_1e3, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adam_1e3", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adam_1e3-engine-head-s42")
def cfc_gen2_flow_d030_adam_1e3_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adam_1e3, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adam_1e3", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_3e4-window-head-s42")
def cfc_gen2_flow_d030_adamw_3e4_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_3e4, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_3e4", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_3e4-engine-head-s42")
def cfc_gen2_flow_d030_adamw_3e4_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_3e4, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_3e4", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_1e4-window-head-s42")
def cfc_gen2_flow_d030_adamw_1e4_window_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_1e4, window, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_1e4", "window", 42
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_1e4-engine-head-s42")
def cfc_gen2_flow_d030_adamw_1e4_engine_head_s42() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_1e4, engine, 42.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_1e4", "engine", 42
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adam_1e3-window-head-s43")
def cfc_gen2_encoder_d010_adam_1e3_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adam_1e3, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adam_1e3", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adam_1e3-engine-head-s43")
def cfc_gen2_encoder_d010_adam_1e3_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adam_1e3, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adam_1e3", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_3e4-window-head-s43")
def cfc_gen2_encoder_d010_adamw_3e4_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_3e4, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_3e4", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_3e4-engine-head-s43")
def cfc_gen2_encoder_d010_adamw_3e4_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_3e4, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_3e4", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_1e4-window-head-s43")
def cfc_gen2_encoder_d010_adamw_1e4_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_1e4, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_1e4", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_1e4-engine-head-s43")
def cfc_gen2_encoder_d010_adamw_1e4_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_1e4, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_1e4", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adam_1e3-window-head-s43")
def cfc_gen2_encoder_d030_adam_1e3_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adam_1e3, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adam_1e3", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adam_1e3-engine-head-s43")
def cfc_gen2_encoder_d030_adam_1e3_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adam_1e3, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adam_1e3", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_3e4-window-head-s43")
def cfc_gen2_encoder_d030_adamw_3e4_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_3e4, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_3e4", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_3e4-engine-head-s43")
def cfc_gen2_encoder_d030_adamw_3e4_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_3e4, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_3e4", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_1e4-window-head-s43")
def cfc_gen2_encoder_d030_adamw_1e4_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_1e4, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_1e4", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_1e4-engine-head-s43")
def cfc_gen2_encoder_d030_adamw_1e4_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_1e4, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_1e4", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adam_1e3-window-head-s43")
def cfc_gen2_flow_d010_adam_1e3_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adam_1e3, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adam_1e3", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adam_1e3-engine-head-s43")
def cfc_gen2_flow_d010_adam_1e3_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adam_1e3, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adam_1e3", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_3e4-window-head-s43")
def cfc_gen2_flow_d010_adamw_3e4_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_3e4, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_3e4", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_3e4-engine-head-s43")
def cfc_gen2_flow_d010_adamw_3e4_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_3e4, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_3e4", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_1e4-window-head-s43")
def cfc_gen2_flow_d010_adamw_1e4_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_1e4, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_1e4", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_1e4-engine-head-s43")
def cfc_gen2_flow_d010_adamw_1e4_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_1e4, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_1e4", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adam_1e3-window-head-s43")
def cfc_gen2_flow_d030_adam_1e3_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adam_1e3, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adam_1e3", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adam_1e3-engine-head-s43")
def cfc_gen2_flow_d030_adam_1e3_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adam_1e3, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adam_1e3", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_3e4-window-head-s43")
def cfc_gen2_flow_d030_adamw_3e4_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_3e4, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_3e4", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_3e4-engine-head-s43")
def cfc_gen2_flow_d030_adamw_3e4_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_3e4, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_3e4", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_1e4-window-head-s43")
def cfc_gen2_flow_d030_adamw_1e4_window_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_1e4, window, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_1e4", "window", 43
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_1e4-engine-head-s43")
def cfc_gen2_flow_d030_adamw_1e4_engine_head_s43() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_1e4, engine, 43.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_1e4", "engine", 43
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adam_1e3-window-head-s44")
def cfc_gen2_encoder_d010_adam_1e3_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adam_1e3, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adam_1e3", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adam_1e3-engine-head-s44")
def cfc_gen2_encoder_d010_adam_1e3_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adam_1e3, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adam_1e3", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_3e4-window-head-s44")
def cfc_gen2_encoder_d010_adamw_3e4_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_3e4, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_3e4", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_3e4-engine-head-s44")
def cfc_gen2_encoder_d010_adamw_3e4_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_3e4, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_3e4", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_1e4-window-head-s44")
def cfc_gen2_encoder_d010_adamw_1e4_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_1e4, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_1e4", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d010-adamw_1e4-engine-head-s44")
def cfc_gen2_encoder_d010_adamw_1e4_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d010, adamw_1e4, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d010", "adamw_1e4", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adam_1e3-window-head-s44")
def cfc_gen2_encoder_d030_adam_1e3_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adam_1e3, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adam_1e3", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adam_1e3-engine-head-s44")
def cfc_gen2_encoder_d030_adam_1e3_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adam_1e3, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adam_1e3", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_3e4-window-head-s44")
def cfc_gen2_encoder_d030_adamw_3e4_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_3e4, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_3e4", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_3e4-engine-head-s44")
def cfc_gen2_encoder_d030_adamw_3e4_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_3e4, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_3e4", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_1e4-window-head-s44")
def cfc_gen2_encoder_d030_adamw_1e4_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_1e4, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_1e4", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-encoder-d030-adamw_1e4-engine-head-s44")
def cfc_gen2_encoder_d030_adamw_1e4_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for encoder, d030, adamw_1e4, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "encoder", "d030", "adamw_1e4", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adam_1e3-window-head-s44")
def cfc_gen2_flow_d010_adam_1e3_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adam_1e3, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adam_1e3", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adam_1e3-engine-head-s44")
def cfc_gen2_flow_d010_adam_1e3_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adam_1e3, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adam_1e3", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_3e4-window-head-s44")
def cfc_gen2_flow_d010_adamw_3e4_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_3e4, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_3e4", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_3e4-engine-head-s44")
def cfc_gen2_flow_d010_adamw_3e4_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_3e4, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_3e4", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_1e4-window-head-s44")
def cfc_gen2_flow_d010_adamw_1e4_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_1e4, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_1e4", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d010-adamw_1e4-engine-head-s44")
def cfc_gen2_flow_d010_adamw_1e4_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d010, adamw_1e4, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d010", "adamw_1e4", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adam_1e3-window-head-s44")
def cfc_gen2_flow_d030_adam_1e3_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adam_1e3, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adam_1e3", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adam_1e3-engine-head-s44")
def cfc_gen2_flow_d030_adam_1e3_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adam_1e3, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adam_1e3", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_3e4-window-head-s44")
def cfc_gen2_flow_d030_adamw_3e4_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_3e4, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_3e4", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_3e4-engine-head-s44")
def cfc_gen2_flow_d030_adamw_3e4_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_3e4, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_3e4", "engine", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_1e4-window-head-s44")
def cfc_gen2_flow_d030_adamw_1e4_window_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_1e4, window, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_1e4", "window", 44
    )


@scenario("maintenance-rul-cfc-gen2-flow-d030-adamw_1e4-engine-head-s44")
def cfc_gen2_flow_d030_adamw_1e4_engine_head_s44() -> ScenarioSpec:
    """Run controlled CfC RUL generalization for flow, d030, adamw_1e4, engine, 44.

    Returns:
        Native NexuML scenario with matched frozen upstream RUL features.
    """
    return cfc_rul_optimization_study(
        "flow", "d030", "adamw_1e4", "engine", 44
    )
