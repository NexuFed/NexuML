"""DALI WAV audio pipeline for file-backed audio datasets."""

from __future__ import annotations

import logging
from typing import Any

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from tensordict import TensorDict

logger = logging.getLogger(__name__)


def _dali_type(name: str) -> Any:
    """Return a DALI type constant by name despite incomplete DALI stubs."""
    return getattr(types, name)


class PyTorchIterator(DALIGenericIterator):
    """DALIGenericIterator with multi-label support via TensorDict lookup."""

    def __init__(self, labels: list[list[int]] | TensorDict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(labels, TensorDict):
            self.labels = labels
        else:
            self.labels = [torch.tensor(label).squeeze(-1) for label in labels]
            logger.warning("Providing labels as list is deprecated. Use TensorDict instead.")

    def __next__(self):
        out = super().__next__()
        out = out[0]

        if isinstance(self.labels, TensorDict):
            labels = self.labels[out["label"]]
        else:
            labels = [
                self.labels[i][out["label"]].squeeze(1).to(out["audio"].device)
                for i in range(len(self.labels))
            ]

        return out["audio"].movedim(-2, -1), labels


@pipeline_def
def wav_data_pipeline(
    files: list[str],
    target_sr: int = 16000,
    target_length: float | None = 10,
    mono: bool = True,
    shuffle: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
    device: str = "cpu",
    dont_use_mmap: bool = True,
    rnd_crop_size: float | None = None,
    start_sec: float | None = None,
):
    """Load WAV files with DALI native readers and return audio + file index label.

    Returns:
        Tuple of audio tensor and integer file-index label.

    Raises:
        ValueError: If a crop/start offset is requested without a target length.
    """
    if rnd_crop_size is not None:
        if target_length is None:
            raise ValueError("target_length is required when rnd_crop_size is set")
        rnd_choice_list = list(
            range(int(target_length * target_sr) - int(rnd_crop_size * target_sr))
        )
        if len(rnd_choice_list) == 0:
            rnd_choice_list = [0]
        start = fn.random.choice(rnd_choice_list, shape=[1])
        end = start + int(rnd_crop_size * target_sr)
    elif start_sec is not None:
        if target_length is None:
            raise ValueError("target_length is required when start_sec is set")
        start = int(start_sec * target_sr)
        end = start + int(target_length * target_sr)
    elif target_length is None:
        start = 0
        end = None
    else:
        start = 0
        end = target_length * target_sr

    encoded, label = fn.readers.file(
        files=files,
        labels=list(torch.arange(len(files))),
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        device="cpu",
        seed=42,
        name="Reader",
        dont_use_mmap=dont_use_mmap,
        read_ahead=True,
    )

    audio, sr = fn.decoders.audio(
        encoded,
        dtype=_dali_type("FLOAT"),
        downmix=False,
        sample_rate=target_sr,
        device="cpu",
    )

    if mono:
        audio = fn.reductions.mean(audio, axes=[-1], keep_dims=True)

    audio = fn.copy(audio, device=device)

    audio = fn.slice(
        audio,
        start=start,
        end=end,
        axes=[0],
        out_of_bounds_policy="pad",
        device=device,
    )

    audio = fn.transpose(audio, perm=[1, 0])

    return audio, label


def DaliAudioPipeline(
    files: list[str],
    labels: list[list[int]] | TensorDict,
    batch_size: int,
    target_sr: int = 16000,
    target_length: float = 10,
    mono: bool = True,
    num_threads: int = -1,
    prefetch_factor: int = 2,
    shuffle: bool = False,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    random_crop_size: float | None = None,
    start_sec: float | None = None,
    **kwargs,
) -> PyTorchIterator:
    """Build and return a DALI audio pipeline as a PyTorchIterator.

    Returns:
        A ``PyTorchIterator`` over the built DALI pipeline.
    """
    if num_threads == 0:
        num_threads = 1
    if num_threads < 0:
        num_threads = torch.multiprocessing.cpu_count()

    device_id = local_rank
    shard_id = global_rank
    num_shards = world_size

    device = "gpu" if device_id >= 0 else "cpu"

    pipeline = wav_data_pipeline(
        files=files,
        target_sr=target_sr,
        target_length=target_length,
        mono=mono,
        batch_size=batch_size,
        num_threads=num_threads,
        shuffle=shuffle,
        device=device,
        device_id=device_id,
        shard_id=shard_id,
        num_shards=num_shards,
        rnd_crop_size=random_crop_size,
        start_sec=start_sec,
        prefetch_queue_depth=prefetch_factor if prefetch_factor is not None else 1,
        **kwargs,
    )
    pipeline.build()

    return PyTorchIterator(
        pipelines=[pipeline],
        labels=labels,
        output_map=["audio", "label"],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        reader_name="Reader",
        prepare_first_batch=True,
    )
