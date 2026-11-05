"""Cluster-conditioned CfC RUL regression with deployment-safe soft membership.

No oracle working/failure cluster ID ever enters the prediction graph.
Two learned gates map the CURRENT 30-cycle observations to probabilities;
optional auxiliary pseudo-label supervision is handled in the loss layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext
from nexuml.core.discovery import layer
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul_cluster_conditions import (
    read_cluster_artifact,
)
from nexuml_library.evaluation.maintenance.rul_sequence_flow_checkpoint import (
    sequence_flow_run,
)
from nexuml_library.layers.model.rul_enhanced_cfc import (
    EnhancedCfCRULHead,
    _EnhancedCfCRuntime,
)

ConditionKind = Literal["working", "failure", "both"]


@layer("ClusterConditionedCfCRULHead")
class ClusterConditionedCfCRULHead(EnhancedCfCRULHead):
    """Add predicted working/failure cluster distributions to the CfC head."""

    condition: ConditionKind = "both"
    cluster_artifact: str = (
        ".experiments/rul/prepared/cmapss-FD003-seed42-cfc-conditional-clusters-v1.pt"
    )
    working_clusters: int = Field(default=3, ge=2, le=12)
    failure_clusters: int = Field(default=2, ge=2, le=12)
    temperature: float = Field(default=1.0, gt=0.01, le=10.0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build the native CfC with learned, inference-available soft gates.

        Returns:
            CfC layer producing RUL and optional training-only gate logits.

        Raises:
            ValueError: If this is not a compatible FD003 RUL scenario.
        """
        if self.subset != "FD003" or self.upstream_seed != 42:
            raise ValueError("Conditional CfC currently supports FD003 encoder seed 42")
        if (
            context.input_sizes.get("rul_history") != (30, 15)
            or context.input_sizes.get("previous_rul_history") != (30, 15)
            or context.input_sizes.get("pair_mask") != (1,)
            or context.keys_in != ["rul_history"]
            or len(context.keys_out) != 2
            or context.keys_out[0] == context.keys_out[1]
        ):
            raise ValueError("Conditional CfC requires existing [30,15] paired RUL routing")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            return _ClusterConditionedCfCRuntime(
                subset=self.subset,
                upstream_seed=self.upstream_seed,
                initialization_seed=self.initialization_seed,
                pretrained_encoder=self.pretrained_encoder,
                train_encoder=self.train_encoder,
                sensor_graph=self.sensor_graph,
                trend=self.trend,
                multiscale=self.multiscale,
                segmentation=self.segmentation,
                trajectory=self.trajectory,
                dropout=self.dropout,
                condition=self.condition,
                cluster_artifact=self.cluster_artifact,
                working_clusters=self.working_clusters,
                failure_clusters=self.failure_clusters,
                temperature=self.temperature,
                **context.runtime_kwargs(),
            )


class _ClusterConditionedCfCRuntime(_EnhancedCfCRuntime):
    """Inject inferred cluster probabilities before the common CfC backbone."""

    def __init__(
        self,
        *,
        condition: ConditionKind,
        cluster_artifact: str,
        working_clusters: int,
        failure_clusters: int,
        temperature: float,
        initialization_seed: int,
        upstream_seed: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            initialization_seed=initialization_seed,
            upstream_seed=upstream_seed,
            **kwargs,
        )
        endpoint = Path(
            f".experiments/rul/prepared/"
            f"{sequence_flow_run('FD003', upstream_seed, 'cfc')}-endpoints.pt"
        )
        # The parent has already validated this endpoint against the true
        # frozen PatchFormer weights. Bind cluster centroids to the same hash.
        settings = read_prepared_artifact(endpoint)["settings"]
        clusters = read_cluster_artifact(
            cluster_artifact,
            subset="FD003",
            seed=upstream_seed,
            working_clusters=working_clusters,
            failure_clusters=failure_clusters,
            encoder_sha256=str(settings["frozen_patchformer_checkpoint_sha256"]),
        )
        self.condition = condition
        self.temperature = temperature
        self.working_clusters = working_clusters
        self.failure_clusters = failure_clusters
        self.working_centroids: Tensor
        self.working_mean: Tensor
        self.working_std: Tensor
        self.failure_centroids: Tensor
        self.failure_mean: Tensor
        self.failure_std: Tensor
        for prefix in ("working", "failure"):
            for suffix in ("centroids", "mean", "std"):
                self.register_buffer(
                    f"{prefix}_{suffix}",
                    cast(Tensor, clusters[f"{prefix}_{suffix}"]),
                    persistent=True,
                )
        self.working_gate: nn.Module | None = None
        self.working_projection: nn.Linear | None = None
        self.failure_gate: nn.Module | None = None
        self.failure_projection: nn.Linear | None = None
        if condition in ("working", "both"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed + 2001)
                self.working_gate = nn.Sequential(
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, working_clusters),
                )
                self.working_projection = nn.Linear(working_clusters, 64, bias=False)
                nn.init.zeros_(self.working_projection.weight)
        if condition in ("failure", "both"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed + 2002)
                self.failure_gate = nn.Sequential(
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, failure_clusters),
                )
                self.failure_projection = nn.Linear(failure_clusters, 64, bias=False)
                nn.init.zeros_(self.failure_projection.weight)

    def _membership(self, embedding: Tensor, kind: Literal["working", "failure"]) -> Tensor:
        """Predict cluster logits from current observables and frozen centroids.

        Returns:
            Differentiable logits of shape [batch, cluster_count].

        Raises:
            ValueError: If a requested cluster gate is disabled.
        """
        if kind == "working":
            mean, std, centers, gate = (
                self.working_mean,
                self.working_std,
                self.working_centroids,
                self.working_gate,
            )
        else:
            mean, std, centers, gate = (
                self.failure_mean,
                self.failure_std,
                self.failure_centroids,
                self.failure_gate,
            )
        if gate is None:
            raise ValueError("Requested cluster gate is disabled")
        standardized = (embedding - mean) / std
        distance = (standardized.unsqueeze(1) - centers.unsqueeze(0)).square().mean(-1)
        return gate(embedding) - distance / self.temperature

    def _predict(self, history: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Return RUL and predicted memberships from current sensor history.

        Returns:
            Capped RUL [B,1] and enabled cluster logits.

        Raises:
            ValueError: If the sensor history has an incompatible shape.
        """
        if history.ndim != 3 or history.shape[1:] != (30, 15):
            raise ValueError("Conditional CfC expects [batch,30,15] sensor windows")
        raw = self.sensor_graph(history) if self.sensor_graph is not None else history
        features = self.encoder.forward_features(raw)
        normalized = (features - self.reference_mean) / self.reference_std
        encoded = (
            self.input_projection(torch.cat((normalized, torch.zeros_like(normalized)), dim=-1))
            + self.positions
        )
        if self.trend_projection is not None:
            velocity = torch.cat(
                (torch.zeros_like(normalized[:, :1]), normalized[:, 1:] - normalized[:, :-1]),
                dim=1,
            )
            encoded = encoded + self.trend_projection(velocity)
        pooled_embedding = normalized.mean(dim=1)
        logits: dict[str, Tensor] = {}
        if self.working_gate is not None and self.working_projection is not None:
            working = self._membership(pooled_embedding, "working")
            logits["working_cluster_logits"] = working
            encoded = encoded + self.working_projection(working.softmax(-1)).unsqueeze(1)
        if self.failure_gate is not None and self.failure_projection is not None:
            failure = self._membership(pooled_embedding, "failure")
            logits["failure_cluster_logits"] = failure
            encoded = encoded + self.failure_projection(failure.softmax(-1)).unsqueeze(1)
        if self.multiscale is not None:
            encoded = self.multiscale(encoded)
        if self.segmentation is not None:
            encoded = self.segmentation(encoded)
        tokens = self.norm(self.temporal(encoded))
        attention = torch.softmax(self.token_scores(tokens), dim=1)
        prediction = 130.0 * torch.sigmoid(self.regressor((tokens * attention).sum(dim=1)))
        return prediction, logits

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Predict RUL without requiring any cluster labels at inference.

        Returns:
            RUL [B,1] computed using internally predicted memberships.
        """
        return self._predict(x)[0]

    def forward(
        self,
        x: TensorDict | Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Expose only predicted memberships, never training oracle IDs.

        Returns:
            Original TensorDict with RUL, predecessors and soft-gate logits.

        Raises:
            TypeError: If the native paired data source was not used.
        """
        if not isinstance(x, TensorDict):
            raise TypeError("Cluster-conditioned CfC expects TensorDict")
        prediction, memberships = self._predict(cast(Tensor, x["rul_history"]))
        current_key, previous_key = self.keys_out
        x[current_key] = prediction
        for key, logits in memberships.items():
            x[key] = logits
        if self.trajectory and self.training and y is not None:
            predecessor = cast(Tensor, x["previous_rul_history"])
            x[previous_key] = self._predict(predecessor)[0]
        else:
            x[previous_key] = prediction.detach()
        return x, y
