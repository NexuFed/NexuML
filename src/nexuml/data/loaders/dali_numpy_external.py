"""DALI external source pipeline for NumPy datasets."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import nvidia.dali.fn as fn
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from tensordict import TensorDict


class ExternalInputIterator:
    """Iterator that loads NumPy files on demand for DALI external_source."""

    def __init__(self, files: list[str], num_shards: int, shard_id: int, shuffle: bool):
        self.files = list(enumerate(files))
        self.shuffle = shuffle
        self.data_set_len = len(self.files)

        self.files = self.files[
            self.data_set_len * shard_id // num_shards : self.data_set_len
            * (shard_id + 1)
            // num_shards
        ]

        self.n = len(self.files)
        random.seed(42)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            random.shuffle(self.files)
        return self

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()

        label, file = self.files[self.i]
        data = np.load(file)
        label = torch.tensor([label])
        self.i += 1
        return (data, label)


class PyTorchIterator(DALIGenericIterator):
    """DALIGenericIterator with multi-label support for external source pipelines."""

    def __init__(self, labels: list[list[int]] | TensorDict, device: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(labels, TensorDict):
            self.labels = labels
        else:
            self.labels = [torch.tensor(label).squeeze(-1).to(device) for label in labels]

    def __next__(self):
        out = super().__next__()
        out = out[0]

        if isinstance(self.labels, TensorDict):
            labels = self.labels[out["label"]]
        else:
            labels = [self.labels[i][out["label"]].squeeze(1) for i in range(len(self.labels))]

        return out["feature"], labels


@pipeline_def
def external_source_pipeline(
    external_source: Iterable,
    device: str = "cpu",
):
    """DALI pipeline using external source for flexible data loading.

    Returns:
        Tuple of data tensor and label tensor from the external source.
    """
    data, label = fn.external_source(
        source=external_source,
        num_outputs=2,
        device="cpu",
        batch=False,
        parallel=False,
        cycle="quiet",
    )

    return data, label


def DaliNumpyExternalPipeline(
    files: list[str],
    labels: list[list[int]] | TensorDict,
    batch_size: int,
    num_threads: int = -1,
    prefetch_factor: int = 2,
    shuffle: bool = False,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    **kwargs,
) -> PyTorchIterator:
    """Build and return a DALI external source pipeline as a PyTorchIterator.

    Returns:
        A ``PyTorchIterator`` over the built DALI pipeline.
    """
    num_threads = num_threads if num_threads > 0 else torch.multiprocessing.cpu_count()
    prefetch_factor = prefetch_factor if prefetch_factor is not None else 2
    device_id = local_rank
    shard_id = global_rank
    num_shards = world_size

    device = "gpu" if device_id >= 0 else "cpu"

    external_source = ExternalInputIterator(
        files=files,
        num_shards=num_shards,
        shard_id=shard_id,
        shuffle=shuffle,
    )

    pipeline = external_source_pipeline(
        external_source=external_source,
        batch_size=batch_size,
        num_threads=num_threads,
        device=device,
        device_id=device_id,
        prefetch_queue_depth=prefetch_factor,
    )
    pipeline.build()

    torch_device = f"cuda:{device_id}" if device_id >= 0 else "cpu"
    return PyTorchIterator(
        pipelines=[pipeline],
        labels=labels,
        device=torch_device,
        output_map=["feature", "label"],
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        size=len(files),
        prepare_first_batch=True,
    )
