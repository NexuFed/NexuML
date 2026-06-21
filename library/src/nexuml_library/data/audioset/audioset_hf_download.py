"""Hugging Face-only AudioSet downloader."""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_ID = "agkphysics/AudioSet"
CONFIG_NAME = "full"
SPLITS = ("bal_train", "eval")  # "unbal_train"


def is_hf_audioset_complete(root: str | Path) -> bool:
    """Return whether the Hugging Face AudioSet layout exists."""
    root_path = Path(root)
    return (
        all((root_path / split).is_dir() for split in SPLITS) and (root_path / "metadata").is_dir()
    )


def expected_hf_audioset_layout(root: str | Path) -> str:
    """Return the expected HuggingFace AudioSet directory layout path."""
    root_path = Path(root)
    return (
        f"{root_path}/{{bal_train,eval,unbal_train,metadata}} "
        f"from Hugging Face dataset {DATASET_ID!r} config {CONFIG_NAME!r}"
    )


def download_hf_audioset(root: str | Path, decode: bool = True) -> None:
    """Download AudioSet from Hugging Face into ``root``.

    Uses only ``agkphysics/AudioSet`` config ``full`` and splits
    ``bal_train``, ``eval``, and ``unbal_train``. No YouTube or yt-dlp code is
    imported or invoked.

    Raises:
        RuntimeError: If required packages (datasets, torchaudio, tqdm)
            are not installed.
    """
    try:
        import tqdm
        import torchaudio
        from datasets import Audio, load_dataset, load_dataset_builder
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "AudioSet Hugging Face download requires optional packages: "
            "datasets, torchaudio, and tqdm."
        ) from exc

    root_path = Path(root)
    (root_path / "metadata").mkdir(parents=True, exist_ok=True)
    builder = load_dataset_builder(DATASET_ID, CONFIG_NAME)

    for split in SPLITS:
        out_dir = root_path / split
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(DATASET_ID, CONFIG_NAME, split=split, streaming=True)
        dataset = dataset.cast_column("audio", Audio(decode=decode))
        splits = builder.info.splits
        total = splits[split].num_examples if splits is not None else None
        progress = tqdm.tqdm(
            dataset, desc=f"AudioSet {CONFIG_NAME}/{split}", unit="ex", total=total
        )
        for index, example in enumerate(progress):
            video_id = str(example.get("video_id", index))
            labels = example.get("human_labels") or ["no_label"]
            label_str = "_".join(str(label).replace("/", "_") for label in labels)
            audio = example["audio"]
            suffix = ".wav" if decode else Path(str(audio.get("path") or "")).suffix or ".wav"

            target = out_dir / f"{index:07d}__{video_id}__{label_str}{suffix}"

            if target.exists():
                continue

            if not decode:
                payload = audio.get("bytes")
                if payload is None:
                    raise RuntimeError(
                        f"AudioSet example {split}/{index} did not include raw audio bytes"
                    )
                target.write_bytes(payload)
            else:
                samples = audio.get_all_samples()
                torchaudio.save_with_torchcodec(
                    str(target),
                    src=samples.data,
                    sample_rate=audio.metadata.sample_rate,
                    channels_first=True,
                )
            if (index + 1) % 1000 == 0:
                rate = progress.format_dict.get("rate") or 0
                remaining_total = total or index + 1
                remaining = (remaining_total - index - 1) / rate if rate else 0
                eta = _dt.datetime.now() + _dt.timedelta(seconds=remaining)
                logger.info(
                    "Saved %d/%s AudioSet %s examples; ETA %s",
                    index + 1,
                    total if total is not None else "?",
                    split,
                    eta,
                )
