"""AudioSet data specification builders."""

from __future__ import annotations

from typing import Literal

from nexuml.core.types import DatasetSpec, DataSpec, LoaderSpec
from nexuml_library.data.audioset.audioset_hf_download import (
    download_hf_audioset,
    expected_hf_audioset_layout,
    is_hf_audioset_complete,
)
from nexuml_library.scenarios.data.roots import resolve_data_root


def audioset_data(
    data_root: str = "audioset_hf/full",
    train_subset: Literal["bal_train", "unbal_train"] = "bal_train",
    download: bool = False,
    sample_rate: int = 16000,
    clip_num_samples: int = 160000,
    batch_size: int = 128,
    num_workers: int = 4,
    num_classes: int = 527,
    perform_checks: bool = False,
    validate_layout: bool = True,
    max_samples: int | None = None,
) -> DataSpec:
    """AudioSet balanced-train + eval data spec.

    Returns:
        DataSpec: Audio dataset specification with fit and test splits.

    Raises:
        FileNotFoundError: If AudioSet data is missing or incomplete after
            validation.
    """
    root = resolve_data_root(data_root)
    if download and not is_hf_audioset_complete(root):
        download_hf_audioset(root)
    if validate_layout and not is_hf_audioset_complete(root):
        if not is_hf_audioset_complete(root):
            raise FileNotFoundError(
                "AudioSet data is missing or incomplete. "
                f"Resolved root: {root}. Expected layout: {expected_hf_audioset_layout(root)}. "
                "Set NEXUML_DATA_ROOT, pass data_root, or call audioset_data(download=True)."
            )
    len_seconds = clip_num_samples / float(sample_rate)
    datasets = [
        DatasetSpec(
            type_key="AudiosetDataset",
            params={
                "data_dir": str(root / train_subset),
                "meta_dir": str(root / "metadata"),
                "perform_checks": perform_checks,
                "len_seconds": len_seconds,
                "sample_rate": sample_rate,
            },
            modality="audio",
            split_type="fit",
            max_samples=max_samples,
        ),
        DatasetSpec(
            type_key="AudiosetDataset",
            params={
                "data_dir": str(root / "eval"),
                "meta_dir": str(root / "metadata"),
                "perform_checks": perform_checks,
                "len_seconds": len_seconds,
                "sample_rate": sample_rate,
            },
            modality="audio",
            split_type="test",
            max_samples=max_samples,
        ),
    ]

    return DataSpec(
        source_type="audioset",
        datasets=datasets,
        loader=LoaderSpec(
            backend="dali",
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
        params={
            "data_root": str(root),
            "sample_rate": sample_rate,
            "download": download,
            "expected_layout": expected_hf_audioset_layout(root),
        },
        feature_key="waveform",
        input_shapes={"waveform": [clip_num_samples]},
        num_classes=num_classes,
    )
