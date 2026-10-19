"""Pairing-invariant distributional validation for prepared NEX-233 flows."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from nexuml.training.lightning import NexuLightningModule
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.evaluation.maintenance.flow_transport import sliced_wasserstein
from nexuml_library.layers.model.flow import _FlowRuntime


@torch.no_grad()
def score_distributional_transport(
    flow: _FlowRuntime,
    healthy: Tensor,
    faulty: Tensor,
    *,
    device: torch.device,
    batch_size: int = 64,
    projections: int = 128,
    seed: int = 42,
) -> dict[str, float]:
    """Measure complete validation-distribution transport, ignoring pair assignments.

    Source and target are compared as unordered samples using fixed random
    projections. Integration uses the same trained flow and Euler steps as
    post-training evaluation. The sealed test split is never accessed.

    Returns:
        Sliced W1 before and after transport, and relative reduction.

    Raises:
        ValueError: If feature sets or transport predictions are invalid.
    """
    if healthy.ndim != 2 or healthy.shape != faulty.shape or healthy.shape[0] == 0:
        raise ValueError("Validation requires nonempty equal-shaped endpoint distributions")
    if batch_size <= 0 or projections <= 0:
        raise ValueError("Validation batch size and projection count must be positive")

    transported_parts: list[Tensor] = []
    for subset in healthy.split(batch_size):
        transported = flow.sample(subset.to(device)).detach().cpu()
        if transported.shape != subset.shape:
            raise ValueError("Transported feature shape must match the healthy distribution")
        transported_parts.append(transported)

    transported_all = torch.cat(transported_parts)
    initial_swd = sliced_wasserstein(healthy, faulty, projections=projections, seed=seed)
    final_swd = sliced_wasserstein(
        transported_all, faulty, projections=projections, seed=seed
    )
    if not initial_swd > 0:
        raise ValueError("Cannot compute relative SWD improvement with zero endpoint distance")
    return {
        "val/healthy_to_faulty_swd": initial_swd,
        "val/transport_swd": final_swd,
        "val/swd_relative_reduction": 1.0 - final_swd / initial_swd,
    }


class DistributionalTransportValidation(Callback):
    """Log paired-independent validation SWD for checkpointing and early stopping."""

    def __init__(
        self,
        prepared_file: str | Path,
        *,
        batch_size: int = 64,
        projections: int = 128,
        seed: int = 42,
    ) -> None:
        """Configure deterministic complete-validation-distribution scoring."""
        super().__init__()
        if batch_size <= 0 or projections <= 0:
            raise ValueError("Batch size and projections must be positive")
        self.prepared_file = Path(prepared_file)
        self.batch_size = batch_size
        self.projections = projections
        self.seed = seed
        self._validation: tuple[Tensor, Tensor] | None = None

    def _get_validation(self) -> tuple[Tensor, Tensor]:
        """Load and cache only validation observations from the frozen artifact."""
        if self._validation is None:
            prepared = read_prepared_artifact(self.prepared_file)
            features = cast(Tensor, prepared["features"])
            rows = cast(list[dict[str, object]], prepared["rows"])
            indices = [i for i, row in enumerate(rows) if row["split"] == "val"]
            if not indices:
                raise ValueError("Distributional validation requires a nonempty val split")
            self._validation = (features[indices, 0], features[indices, 1])
        return self._validation

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log SWD before checkpoint-selection and early-stopping callbacks run."""
        if not isinstance(pl_module, NexuLightningModule):
            raise TypeError("NEX-233 validation requires a NexuLightningModule")
        flows = [
            layer
            for stage, _, layer in pl_module.pipeline.iter_layers()
            if stage == "Flow"
        ]
        if len(flows) != 1 or not isinstance(flows[0], _FlowRuntime):
            raise ValueError("Distributional validation expects exactly one Flow layer")
        healthy, faulty = self._get_validation()
        scores = score_distributional_transport(
            flows[0],
            healthy,
            faulty,
            device=pl_module.device,
            batch_size=self.batch_size,
            projections=self.projections,
            seed=self.seed,
        )
        for name, value in scores.items():
            pl_module.log(
                name,
                torch.tensor(value, device=pl_module.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
