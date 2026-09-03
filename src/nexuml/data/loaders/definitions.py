"""Typed definitions for built-in loader backends."""

from __future__ import annotations

from nexuml.core.components import LoaderBackendDefinition
from nexuml.core.discovery import loader_backend


@loader_backend("torch")
class TorchLoader(LoaderBackendDefinition):
    """Use the standard PyTorch data loader backend."""

    def build(self):
        from nexuml.data.loaders.torch_backend import TorchLoaderBackend

        return TorchLoaderBackend()


@loader_backend("dali")
class DaliLoader(LoaderBackendDefinition):
    """Use the NVIDIA DALI data loader backend."""

    def build(self):
        from nexuml.data.loaders.dali_backend import DaliLoaderBackend

        return DaliLoaderBackend()


@loader_backend("tensor_shards")
class TensorShardsLoader(LoaderBackendDefinition):
    """Configure the windowed tensor-shard loader backend."""

    shards_per_window: int = 6
    prefetch_windows: int = 2
    prefetch_workers: int = 2
    shuffle_shards: bool | None = None
    shuffle_samples: bool | None = None
    pin_memory: bool = False
    drop_last: bool = False
    seed: int | None = None

    def build(self):
        from nexuml.data.loaders.tensor_shards_backend import TensorShardsLoaderBackend

        return TensorShardsLoaderBackend(self)
