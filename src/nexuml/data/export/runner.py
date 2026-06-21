"""Loader-driven dataset export utilities."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TypeAlias

import pandas as pd
import torch
import yaml
from tensordict import TensorDict
from torch.utils.data import Subset
from tqdm.auto import tqdm

from nexuml.data.dataset import NexuDataset
from nexuml.data.export.backend import ExportConfig, get_export_backend
from nexuml.data.module import NexuDataModule

logger = logging.getLogger(__name__)

BatchTransform: TypeAlias = Callable[
    [TensorDict, TensorDict | None],
    tuple[TensorDict, TensorDict | None],
]


def export_data_module(
    data_module: NexuDataModule,
    path: str | Path,
    *,
    backend: str = "numpy",
    splits: Sequence[str] | None = None,
    transform: BatchTransform | None = None,
    x_keys: Sequence[str] | None = None,
    y_keys: Sequence[str] | None = None,
    include_labels: bool = True,
    label_prefix: str = "label__",
    dtype: object | None = None,
    device: torch.device | str | None = None,
    **backend_kwargs,
) -> Path:
    """Export the data exactly as seen through a configured data module.

    Returns:
        Path to the export directory.

    Raises:
        ValueError: If there are no samples to export.
    """
    data_module.setup()
    export_dir = Path(path)
    export_dir.mkdir(parents=True, exist_ok=True)

    split_datasets = _resolve_data_module_splits(data_module, splits)
    num_samples = sum(len(dataset_split) for _, dataset_split in split_datasets)
    if num_samples == 0:
        raise ValueError(
            f"No samples to export from data module "
            f"(splits={list(splits) if splits is not None else None})"
        )

    metadata = pd.concat(
        [
            _split_metadata(dataset_split, split_name)
            for split_name, dataset_split in split_datasets
        ],
        ignore_index=True,
    )

    return _export_batches(
        export_dir=export_dir,
        backend=backend,
        num_samples=num_samples,
        metadata=metadata,
        modality=getattr(data_module.dataset, "modality", "audio"),
        label_names=getattr(data_module.dataset, "label_names", []),
        num_classes=getattr(data_module.dataset, "num_classes", {}),
        source_datasets=list(
            getattr(getattr(data_module, "dataset", None), "meta_data_list", {}).keys()
        ),
        batch_iterables=[
            (
                split_name,
                data_module._loader(dataset_split, split=split_name, shuffle=False),
            )
            for split_name, dataset_split in split_datasets
        ],
        transform=transform,
        x_keys=x_keys,
        y_keys=y_keys,
        include_labels=include_labels,
        label_prefix=label_prefix,
        dtype=dtype,
        device=device,
        backend_kwargs=backend_kwargs,
    )


def _export_batches(
    *,
    export_dir: Path,
    backend: str,
    num_samples: int,
    metadata: pd.DataFrame,
    modality: str,
    label_names: Sequence[str],
    num_classes: dict[str, int],
    source_datasets: Sequence[str],
    batch_iterables: Sequence[tuple[str, Iterable[tuple[TensorDict, TensorDict | None]]]],
    transform: BatchTransform | None,
    x_keys: Sequence[str] | None,
    y_keys: Sequence[str] | None,
    include_labels: bool,
    label_prefix: str,
    dtype: object | None,
    device: torch.device | str | None,
    backend_kwargs: dict[str, object],
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    if num_samples == 0:
        raise ValueError("No samples to export")

    backend_instance = None
    stored_feature_dtypes: dict[str, str] | None = None
    stored_feature_shapes: dict[str, tuple[int, ...]] | None = None
    stored_keys: list[str] = []
    resolved_x_keys: list[str] = []
    resolved_y_keys: list[str] = []
    export_index = 0

    with _prepared_transform(transform, device) as transform_device:
        progress = tqdm(
            total=num_samples,
            desc="Exporting",
            unit="sample",
            dynamic_ncols=True,
        )
        with torch.no_grad():
            with progress:
                for _split_name, loader in batch_iterables:
                    for batch in loader:
                        x_batch, y_batch = batch
                        if transform is not None:
                            x_batch = x_batch.to(transform_device)
                            y_batch = y_batch.to(transform_device) if y_batch is not None else None
                            x_batch, y_batch = transform(x_batch, y_batch)

                        payload = _batch_payload(
                            x_batch,
                            y_batch,
                            x_keys=x_keys,
                            y_keys=y_keys,
                            include_labels=include_labels,
                            label_prefix=label_prefix,
                        )
                        current_batch_size = _batch_size(payload)

                        if stored_feature_shapes is None:
                            stored_feature_shapes = {
                                key: tuple(tensor.shape[1:]) for key, tensor in payload.items()
                            }
                            stored_feature_dtypes = {
                                key: _tensor_dtype_name(tensor) for key, tensor in payload.items()
                            }
                            stored_keys = list(payload.keys())
                            resolved_x_keys = [
                                key for key in stored_keys if not key.startswith(label_prefix)
                            ]
                            resolved_y_keys = [
                                key[len(label_prefix) :]
                                for key in stored_keys
                                if key.startswith(label_prefix)
                            ]
                            backend_cls = get_export_backend(backend)
                            backend_instance = backend_cls(
                                modality=modality,
                                x_keys=resolved_x_keys,
                                y_keys=resolved_y_keys,
                                transform_applied=transform is not None,
                                label_prefix=label_prefix,
                                **backend_kwargs,
                            )
                            backend_instance.initialize(
                                export_dir,
                                num_samples,
                                stored_feature_shapes,
                                dtype=dtype,
                            )

                        assert backend_instance is not None
                        backend_instance.save_batch(export_index, payload)
                        export_index += current_batch_size
                        progress.update(current_batch_size)

                        if (
                            export_index % max(1, num_samples // 10) == 0
                            or export_index == num_samples
                        ):
                            logger.info("Exported %d / %d samples", export_index, num_samples)

    if stored_feature_shapes is None:
        raise ValueError("No batches were produced during export")
    if stored_feature_dtypes is None:
        raise ValueError("No feature dtypes were captured during export")
    if backend_instance is None:
        raise ValueError("Export backend was not initialized")

    metadata = metadata.reset_index(drop=True).copy()
    metadata["export_index"] = list(range(num_samples))

    backend_meta = backend_instance.finalize()
    metadata_path, metadata_format = _write_metadata(metadata, export_dir)
    key_specs = _build_key_specs(
        stored_feature_shapes=stored_feature_shapes,
        stored_feature_dtypes=stored_feature_dtypes,
        resolved_x_keys=resolved_x_keys,
        resolved_y_keys=resolved_y_keys,
        label_prefix=label_prefix,
        backend_key_specs=backend_meta.get("key_specs", {}),
    )

    config = ExportConfig(
        format_version=2,
        backend=backend,
        writer=backend,
        num_samples=num_samples,
        label_names=list(label_names),
        num_classes=dict(num_classes),
        modality=modality,
        x_keys=resolved_x_keys,
        y_keys=resolved_y_keys,
        label_prefix=label_prefix,
        feature_shapes={key: list(shape) for key, shape in stored_feature_shapes.items()},
        key_specs=key_specs,
        source_datasets=list(source_datasets),
        extra={
            **backend_meta,
            "export_mode": "loader",
            "metadata_file": metadata_path.name,
            "metadata_format": metadata_format,
            "stored_keys": stored_keys,
            "labels_included": include_labels,
            "transform_applied": transform is not None,
            "dtype": None if dtype is None else str(dtype),
        },
    )

    with open(export_dir / "config.yaml", "w") as handle:
        yaml.dump(dataclasses.asdict(config), handle, default_flow_style=False, sort_keys=False)

    logger.info("Export complete: %s (backend=%s, samples=%d)", export_dir, backend, num_samples)
    return export_dir


def _resolve_data_module_splits(
    data_module: NexuDataModule,
    splits: Sequence[str] | None,
) -> list[tuple[str, NexuDataset | Subset]]:
    requested = list(splits) if splits is not None else ["train", "val", "test"]
    split_map = {
        "train": data_module._train_ds,
        "val": data_module._val_ds,
        "test": data_module._test_ds,
        "predict": data_module.dataset,
    }

    resolved: list[tuple[str, NexuDataset | Subset]] = []
    for split_name in requested:
        if split_name not in split_map:
            raise ValueError(
                f"Unknown export split '{split_name}'. Expected train/val/test/predict."
            )
        dataset_split = split_map[split_name]
        if dataset_split is None:
            continue
        resolved.append((split_name, dataset_split))

    return resolved


def _split_metadata(dataset_split, split_name: str) -> pd.DataFrame:
    if hasattr(dataset_split, "meta") and getattr(dataset_split, "meta", None) is not None:
        meta = dataset_split.meta.reset_index(drop=True).copy()
    elif isinstance(dataset_split, Subset):
        root_meta = getattr(dataset_split.dataset, "meta", None)
        if root_meta is not None:
            meta = root_meta.iloc[list(dataset_split.indices)].reset_index(drop=True).copy()
        else:
            meta = pd.DataFrame({"export_index": list(range(len(dataset_split)))})
    else:
        meta = pd.DataFrame({"export_index": list(range(len(dataset_split)))})

    meta["split"] = split_name
    return meta


def _write_metadata(metadata: pd.DataFrame, export_dir: Path) -> tuple[Path, str]:
    parquet_path = export_dir / "metadata.parquet"
    try:
        metadata.to_parquet(parquet_path, index=False)
        return parquet_path, "parquet"
    except ImportError:
        csv_path = export_dir / "metadata.csv"
        metadata.to_csv(csv_path, index=False)
        logger.warning("Parquet engine unavailable; wrote metadata as %s instead", csv_path.name)
        return csv_path, "csv"


@contextmanager
def _prepared_transform(
    transform: BatchTransform | None,
    device: torch.device | str | None,
):
    transform_device = _resolve_device(transform, device)
    restore_training = None
    predict_started = False

    if isinstance(transform, torch.nn.Module):
        restore_training = transform.training
        transform.eval()
        if device is not None:
            transform.to(transform_device)

    on_predict_start = (
        getattr(transform, "on_predict_start", None) if transform is not None else None
    )
    if callable(on_predict_start):
        on_predict_start()
        predict_started = True

    try:
        yield transform_device
    finally:
        on_predict_end = (
            getattr(transform, "on_predict_end", None) if transform is not None else None
        )
        if predict_started and callable(on_predict_end):
            on_predict_end()
        if isinstance(transform, torch.nn.Module) and restore_training is not None:
            transform.train(restore_training)


def _resolve_device(
    transform: BatchTransform | None,
    device: torch.device | str | None,
) -> torch.device:
    if device is not None:
        return torch.device(device)

    if isinstance(transform, torch.nn.Module):
        first_parameter = next(transform.parameters(), None)
        if first_parameter is not None:
            return first_parameter.device

    return torch.device("cpu")


def _batch_payload(
    x_batch: TensorDict,
    y_batch: TensorDict | None,
    *,
    x_keys: Sequence[str] | None,
    y_keys: Sequence[str] | None,
    include_labels: bool,
    label_prefix: str,
) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}

    for key in _selected_keys(x_batch, x_keys, kind="x"):
        tensor = _as_tensor(x_batch[key])
        if tensor is not None:
            payload[key] = tensor

    if include_labels and y_batch is not None:
        for key in _selected_keys(y_batch, y_keys, kind="y"):
            tensor = _as_tensor(y_batch[key])
            if tensor is not None:
                payload[f"{label_prefix}{key}"] = tensor

    if not payload:
        raise ValueError("Export payload is empty; no x/y keys were selected")

    return payload


def _selected_keys(
    batch: TensorDict,
    requested: Sequence[str] | None,
    *,
    kind: str,
) -> list[str]:
    available = [key for key in batch.keys() if isinstance(key, str)]
    if requested is None:
        return available

    missing = [key for key in requested if key not in batch.keys()]
    if missing:
        raise KeyError(f"Missing {kind} keys for export: {missing}. Available: {available}")
    return list(requested)


def _as_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError):
        return None


def _batch_size(payload: dict[str, torch.Tensor]) -> int:
    batch_sizes = {int(tensor.shape[0]) for tensor in payload.values()}
    if len(batch_sizes) != 1:
        raise ValueError(f"Export payload batch size mismatch: {sorted(batch_sizes)}")
    return next(iter(batch_sizes))


def _tensor_dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).split(".")[-1]


def _build_key_specs(
    *,
    stored_feature_shapes: dict[str, tuple[int, ...]],
    stored_feature_dtypes: dict[str, str],
    resolved_x_keys: Sequence[str],
    resolved_y_keys: Sequence[str],
    label_prefix: str,
    backend_key_specs: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    key_specs: dict[str, dict[str, object]] = {}
    x_key_set = set(resolved_x_keys)
    y_key_set = set(resolved_y_keys)

    for key, shape in stored_feature_shapes.items():
        logical_key = key[len(label_prefix) :] if key.startswith(label_prefix) else key
        role = "y" if logical_key in y_key_set and key.startswith(label_prefix) else "x"
        if role == "x" and logical_key not in x_key_set:
            role = "y" if logical_key in y_key_set else "x"

        key_specs[key] = {
            "key": logical_key,
            "role": role,
            "shape": list(shape),
            "dtype": stored_feature_dtypes[key],
            "layout": None,
            "encoding": "tensor",
            "storage": {},
        }

        backend_spec = backend_key_specs.get(key)
        if backend_spec is not None:
            key_specs[key].update(backend_spec)

    return key_specs
