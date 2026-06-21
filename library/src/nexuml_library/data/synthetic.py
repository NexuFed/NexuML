"""Synthetic dataset for testing architectures without real data."""

from __future__ import annotations
from nexuml.core.discovery import data_source

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict

from nexuml.data.dataset import NexuDataset


@dataclass
class TargetConfig:
    """Configuration for a synthetic target."""

    type: str  # "multiclass", "multilabel", "regression"
    key: str
    num_classes: int | None = None
    num_outputs: int | None = None
    label_density: float = 0.3  # for multilabel
    positive_fraction: float | None = None


@data_source("synthetic")
class SyntheticDataset(NexuDataset):
    """Universal synthetic tensor dataset for architecture testing.

    Generates arbitrary feature tensors with optional targets:
    - reconstruction (features are their own target)
    - multiclass labels (from cluster assignments)
    - multilabel labels (random binary vectors)
    - multi-output regression targets (linear combination + noise)
    """

    def __init__(
        self,
        feature_shape: tuple[int, ...] = (128,),
        num_samples: int = 1000,
        noise_type: str = "gaussian",
        num_clusters: int | None = None,
        targets: Sequence[TargetConfig | dict[str, Any]] | None = None,
        seed: int = 42,
        feature_key: str = "features",
    ):
        self.feature_shape = feature_shape
        self.num_samples = num_samples
        self.noise_type = noise_type
        self.num_clusters = num_clusters
        targets_list = targets if targets is not None else []
        self.targets_config = [
            t if isinstance(t, TargetConfig) else TargetConfig(**t) for t in targets_list
        ]
        self.feature_key = feature_key

        label_names = [tc.key for tc in self.targets_config]
        if "reconstruction_target" not in label_names:
            label_names.append("reconstruction_target")
        super().__init__(label_names=label_names)

        generator = torch.Generator().manual_seed(seed)

        # Generate features
        flat_dim = math.prod(feature_shape)

        if num_clusters and num_clusters > 1:
            # Clustered data: generate cluster centers, assign samples
            self._cluster_ids = torch.randint(0, num_clusters, (num_samples,), generator=generator)
            centers = torch.randn(num_clusters, flat_dim, generator=generator) * 3.0
            noise = torch.randn(num_samples, flat_dim, generator=generator) * 0.5
            self._features = centers[self._cluster_ids] + noise
        elif noise_type == "uniform":
            self._features = torch.rand(num_samples, flat_dim, generator=generator) * 2 - 1
            self._cluster_ids = torch.zeros(num_samples, dtype=torch.long)
        else:
            # Gaussian
            self._features = torch.randn(num_samples, flat_dim, generator=generator)
            self._cluster_ids = torch.zeros(num_samples, dtype=torch.long)

        # Reshape to target feature_shape
        self._features = self._features.view(num_samples, *feature_shape)

        # Generate targets
        self._targets: dict[str, torch.Tensor] = {}
        for tc in self.targets_config:
            if tc.type == "multiclass":
                n_cls = tc.num_classes or (num_clusters if num_clusters else 5)
                if num_clusters and num_clusters > 1:
                    # Map cluster IDs to class labels
                    self._targets[tc.key] = self._cluster_ids % n_cls
                else:
                    self._targets[tc.key] = torch.randint(
                        0, n_cls, (num_samples,), generator=generator
                    )
            elif tc.type == "multilabel":
                n_cls = tc.num_classes or 5
                probs = torch.full((num_samples, n_cls), tc.label_density)
                self._targets[tc.key] = torch.bernoulli(probs, generator=generator).float()
            elif tc.type == "regression":
                n_out = tc.num_outputs or 1
                # Linear combination of flattened features + noise
                flat = self._features.view(num_samples, -1)
                weight = torch.randn(flat.shape[1], n_out, generator=generator) * 0.1
                noise = torch.randn(num_samples, n_out, generator=generator) * 0.1
                self._targets[tc.key] = flat @ weight + noise
            elif tc.type == "anomaly":
                positive_fraction = (
                    tc.positive_fraction if tc.positive_fraction is not None else 0.2
                )
                positive_count = max(1, int(num_samples * positive_fraction))
                labels = torch.zeros(num_samples, dtype=torch.float32)
                positive_idx = torch.randperm(num_samples, generator=generator)[:positive_count]
                labels[positive_idx] = 1.0
                flat_features = self._features.view(num_samples, -1)
                flat_features[positive_idx] += 3.0
                self._features = flat_features.view(num_samples, *feature_shape)
                self._targets[tc.key] = labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[TensorDict, TensorDict | None]:
        x = TensorDict(
            {self.feature_key: self._features[index]},
            batch_size=[],
        )

        if self._targets:
            y_data = {key: val[index] for key, val in self._targets.items()}
            # Always include reconstruction target
            y_data["reconstruction_target"] = self._features[index]
            y = TensorDict(y_data, batch_size=[])  # ty: ignore[invalid-argument-type]
        else:
            y = TensorDict(
                {"reconstruction_target": self._features[index]},
                batch_size=[],
            )

        return x, y
