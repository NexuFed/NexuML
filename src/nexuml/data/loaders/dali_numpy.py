"""DALI NumPy pipeline for file-backed numpy datasets."""

from __future__ import annotations

import logging

import nvidia.dali.fn as fn
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

logger = logging.getLogger(__name__)


class PyTorchIterator(DALIGenericIterator):
    """DALIGenericIterator with filename-based label lookup for NumPy datasets."""

    def __init__(self, labels: dict[int, torch.Tensor], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels

    def __next__(self):
        out = super().__next__()
        out = out[0]

        idcs = [int("".join(map(chr, row))) for row in out["label"]]

        label_tensors = [self.labels[idx] for idx in idcs]
        labels = torch.cat(
            (
                torch.stack(label_tensors, dim=0).to(out["audio"].device),
                torch.tensor(idcs).unsqueeze(1).to(out["audio"].device),
            ),
            dim=-1,
        ).to(out["audio"].device)

        return out["audio"], labels


@pipeline_def
def numpy_data_pipeline(
    files: list[str],
    filename_len: int,
    target_sr: int = 16000,
    target_length: int = 10,
    shuffle: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
    device: str = "cpu",
    direct_store: bool = False,
    mono: bool = True,
    rnd_crop_size: float | None = None,
    start_sec: float | None = None,
):
    """Load pre-encoded NumPy arrays with DALI native reader.

    Returns:
        Tuple of audio tensor and integer file-index label.
    """
    if rnd_crop_size is not None:
        rnd_choice_list = list(
            range(int(target_length * target_sr) - int(rnd_crop_size * target_sr))
        )
        if len(rnd_choice_list) == 0:
            rnd_choice_list = [0]
        start = fn.random.choice(rnd_choice_list, shape=[1])
        end = start + int(rnd_crop_size * target_sr)
    elif start_sec is not None:
        start = int(start_sec * target_sr)
        end = start + int(target_length * target_sr)
    else:
        start = 0
        end = target_length * target_sr

    audio = fn.readers.numpy(
        files=files,
        cache_header_information=True,
        out_of_bounds_policy="pad",
        fill_value=0.0,
        roi_start=start,
        roi_end=end,
        roi_axes=[1],
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        device=device if direct_store else "cpu",
        seed=42,
        name="Reader",
    )

    label = fn.get_property(audio, key="source_info", name="Label")
    label = label[-filename_len:-4]

    if mono:
        audio = fn.reductions.mean(audio, axes=[-2], keep_dims=True)

    return audio, label


def DaliNumpyPipeline(
    files: list[str],
    labels: list[list[int]],
    batch_size: int,
    target_sr: int = 16000,
    target_length: int = 10,
    num_threads: int = -1,
    prefetch_factor: int = 2,
    shuffle: bool = False,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    mono: bool = True,
    random_crop_size: float | None = None,
    start_sec: float | None = None,
    direct_store: bool = False,
    **kwargs,
) -> PyTorchIterator:
    """Build and return a DALI numpy pipeline as a PyTorchIterator.

    Returns:
        A ``PyTorchIterator`` over the built DALI pipeline.
    """
    num_threads = num_threads if num_threads > 0 else torch.multiprocessing.cpu_count()
    device_id = local_rank
    shard_id = global_rank
    num_shards = world_size

    device = "gpu" if device_id >= 0 else "cpu"

    filename_len = len(files[0].split("/")[-1])

    labels_dict = {}
    for idx, file in enumerate(files):
        labels_dict[int(file[-filename_len:-4])] = torch.tensor(labels)[:, idx]

    pipeline = numpy_data_pipeline(
        files=files,
        filename_len=filename_len,
        target_sr=target_sr,
        target_length=target_length,
        batch_size=batch_size,
        num_threads=num_threads,
        shuffle=shuffle,
        device=device,
        device_id=device_id,
        shard_id=shard_id,
        num_shards=num_shards,
        direct_store=direct_store,
        mono=mono,
        rnd_crop_size=random_crop_size,
        start_sec=start_sec,
        **kwargs,
    )
    pipeline.build()

    return PyTorchIterator(
        pipelines=[pipeline],
        labels=labels_dict,
        output_map=["audio", "label"],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        reader_name="Reader",
        prepare_first_batch=True,
    )
