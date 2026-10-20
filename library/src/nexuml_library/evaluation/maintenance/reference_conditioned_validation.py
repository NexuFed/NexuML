"""Validation-first machine-wise SWD for healthy-reference Flow Matching.

Checkpoint selection uses only validation recordings of *training* machines.
Neither unseen-machine healthy support nor its faulty assessment data are
loaded by the Lightning callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from nexuml.training.lightning import NexuLightningModule
from nexuml_library.data.maintenance.healthy_reference import (
    read_healthy_reference_training,
)
from nexuml_library.evaluation.maintenance.flow_transport import sliced_wasserstein
from nexuml_library.layers.model.healthy_reference_flow import (
    _HealthyReferenceConditionedFlowRuntime,
)


@torch.no_grad()
def evaluate_reference_transport(
    flow: _HealthyReferenceConditionedFlowRuntime,
    healthy: Tensor,
    faulty: Tensor,
    reference: Tensor,
    *,
    device: torch.device,
    batch_size: int = 64,
    projections: int = 128,
    seed: int = 42,
) -> dict[str, float]:
    """Measure real-healthy-initialized transport against faulty marginals.

    Returns:
        Source and transported sliced W1 and relative reduction.

    Raises:
        ValueError: If observations or healthy reference context are invalid.
    """
    if (
        healthy.ndim != 2
        or healthy.shape != faulty.shape
        or len(healthy) == 0
        or reference.ndim != 2
        or reference.shape[1] != healthy.shape[1]
        or len(reference) == 0
        or batch_size <= 0
        or projections <= 0
    ):
        raise ValueError("Need nonempty compatible healthy, faulty, reference tensors")
    source = healthy.cpu()
    target = faulty.cpu()
    transported_parts: list[Tensor] = []
    support = reference.to(device)
    for batch in source.split(batch_size):
        transported_parts.append(flow.sample(batch.to(device), support).detach().cpu())
    transported = torch.cat(transported_parts, dim=0)
    initial = sliced_wasserstein(source, target, projections=projections, seed=seed)
    final = sliced_wasserstein(transported, target, projections=projections, seed=seed)
    if not initial > 0:
        raise ValueError("Zero source-to-faulty SWD cannot be normalized")
    return {
        "initial_swd": initial,
        "transported_swd": final,
        "swd_relative_reduction": 1.0 - final / initial,
    }


@torch.no_grad()
def evaluate_training_machine_validation(
    flow: _HealthyReferenceConditionedFlowRuntime,
    training_file: str | Path,
    *,
    device: torch.device,
    projections: int = 128,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate each training machine separately to avoid pooled SWD bias.

    Returns:
        Per-machine validation SWD and unweighted mean across machine IDs.
    """
    artifact = read_healthy_reference_training(training_file)
    features = cast(Tensor, artifact["features"])
    references = cast(Tensor, artifact["references"])
    rows = cast(list[dict[str, Any]], artifact["rows"])
    machines = cast(list[str], artifact["machine_groups"])
    per_machine: dict[str, dict[str, float]] = {}
    for index, name in enumerate(machines):
        mask = [i for i, row in enumerate(rows) if row["split"] == "val" and row["group_id"] == name]
        if not mask:
            raise ValueError(f"Missing validation data for training machine {name}")
        per_machine[name] = evaluate_reference_transport(
            flow, features[mask, 0], features[mask, 1], references[index],
            device=device, projections=projections, batch_size=batch_size, seed=seed
        )
    return {
        "by_machine": per_machine,
        "mean_initial_swd": sum(row["initial_swd"] for row in per_machine.values())
        / len(per_machine),
        "mean_transported_swd": sum(row["transported_swd"] for row in per_machine.values())
        / len(per_machine),
        "mean_relative_reduction": sum(
            row["swd_relative_reduction"] for row in per_machine.values()
        ) / len(per_machine),
    }


class HealthyReferenceTransportValidation(Callback):
    """Checkpoint-select on mean training-machine validation transport SWD."""

    def __init__(
        self,
        training_file: str,
        *,
        projections: int = 128,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.training_file = training_file
        self.projections = projections
        self.batch_size = batch_size
        self.seed = seed

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log the macro-averaged conditional transport SWD for callbacks.

        Raises:
            ValueError: If the expected trained conditional layer is missing.
        """
        if not isinstance(pl_module, NexuLightningModule):
            raise TypeError("Healthy-reference validation requires NexuLightningModule")
        matches = [
            layer
            for stage, _, layer in pl_module.pipeline.iter_layers()
            if stage == "Flow"
        ]
        if len(matches) != 1 or not isinstance(
            matches[0], _HealthyReferenceConditionedFlowRuntime
        ):
            raise ValueError("Expected one HealthyReferenceConditionedFlow runtime")
        scores = evaluate_training_machine_validation(
            matches[0],
            self.training_file,
            device=pl_module.device,
            projections=self.projections,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        for name, value in {
            "val/transport_swd": scores["mean_transported_swd"],
            "val/healthy_to_faulty_swd": scores["mean_initial_swd"],
            "val/swd_relative_reduction": scores["mean_relative_reduction"],
        }.items():
            pl_module.log(
                name,
                torch.tensor(value, device=pl_module.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
