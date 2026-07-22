"""MNIST ResNet classification using tensor-shard data loading."""

from __future__ import annotations

from nexuml.core.discovery import scenario
from nexuml.core.types import LoaderSpec, PreprocessingSpec, ScenarioSpec
from nexuml_library.scenarios.vision.mnist_resnet import mnist_resnet


@scenario("mnist-resnet-shards")
def mnist_resnet_shards(
    download: bool = True,
    resnet_type: str = "resnet18",
    pretrained: bool = False,
    cifar_stem: bool = True,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
    samples_per_shard: int = 4096,
    shards_per_window: int = 6,
    prefetch_windows: int = 2,
    prefetch_workers: int = 2,
    shuffle_shards: bool = True,
    shuffle_samples: bool = True,
    pin_memory: bool = False,
    overwrite_shards: bool = False,
) -> ScenarioSpec:
    """MNIST ResNet using materialized tensor shards."""
    base = mnist_resnet(
        download=download,
        resnet_type=resnet_type,
        pretrained=pretrained,
        cifar_stem=cifar_stem,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
    )

    sharded_data = base.data.model_copy(
        update={
            "loader": LoaderSpec(
                backend="tensor_shards",
                batch_size=batch_size,
                num_workers=0,
                shuffle_train=True,
                params={
                    "shards_per_window": shards_per_window,
                    "prefetch_windows": prefetch_windows,
                    "prefetch_workers": prefetch_workers,
                    "shuffle_shards": shuffle_shards,
                    "shuffle_samples": shuffle_samples,
                    "pin_memory": pin_memory,
                    "drop_last": False,
                    "seed": 42,
                },
            ),
            "preprocessing": PreprocessingSpec(
                enabled=True,
                source_view="raw",
                target_view="prepared",
                writer="tensor_shards",
                writer_params={
                    "samples_per_shard": samples_per_shard,
                },
                overwrite=overwrite_shards,
            ),
        }
    )

    return base.model_copy(
        update={
            "name": "mnist_resnet_shards",
            "data": sharded_data,
        }
    )
