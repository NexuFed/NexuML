"""Synthetic data scenario fragments."""

from __future__ import annotations

from nexuml.core.types import DataSpec, DatasetSpec, TargetSpec
from nexuml_library.data.synthetic import SyntheticDataset


def synthetic_vector_data(
    feature_shape: tuple[int, ...] = (128,),
    num_samples: int = 1000,
    noise_type: str = "gaussian",
    num_clusters: int | None = None,
    targets: list[TargetSpec] | None = None,
    seed: int = 42,
    feature_key: str = "features",
) -> DataSpec:
    """Create a DataSpec for synthetic vector data.

    Returns:
        DataSpec: Synthetic dataset specification.
    """
    target_dicts = []
    for t in targets or []:
        d = {
            "type": t.type,
            "key": t.key,
            "num_classes": t.num_classes,
            "num_outputs": t.num_outputs,
            "positive_fraction": t.positive_fraction,
            "label_density": 0.3,
        }
        target_dicts.append(d)

    return DataSpec(
        targets=targets or [],
        num_classes=next(
            (target.num_classes for target in targets or [] if target.num_classes), None
        ),
        feature_key=feature_key,
        input_shapes={feature_key: list(feature_shape)},
        datasets=[
            DatasetSpec(
                source=SyntheticDataset(
                    feature_shape=feature_shape,
                    num_samples=num_samples,
                    noise_type=noise_type,
                    num_clusters=num_clusters,
                    seed=seed,
                    targets=target_dicts,
                    feature_key=feature_key,
                ),
            )
        ],
    )
