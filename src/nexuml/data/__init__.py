"""Data loading primitives for NexuML."""

from typing import TYPE_CHECKING

from nexuml.data.dataset import NexuDataset
from nexuml.data.loaders.definitions import DaliLoader, TensorShardsLoader, TorchLoader

if TYPE_CHECKING:
    from nexuml.data.creator import NexuDataCreator
    from nexuml.data.module import NexuDataModule


def __getattr__(name):
    # Policy config restoration needs loader definitions, not Lightning's image dependencies.
    if name == "NexuDataCreator":
        from nexuml.data.creator import NexuDataCreator

        return NexuDataCreator
    if name == "NexuDataModule":
        from nexuml.data.module import NexuDataModule

        return NexuDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DaliLoader",
    "NexuDataCreator",
    "NexuDataModule",
    "NexuDataset",
    "TensorShardsLoader",
    "TorchLoader",
]
