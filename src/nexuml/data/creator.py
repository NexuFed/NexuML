"""Unified dataset and datamodule creation flow."""

from __future__ import annotations

import logging

from nexuml.core.types import AutoBatchSizeSpec, DatasetSpec, DataSpec, LoaderSpec
from nexuml.data.dataset import NexuDataset
from nexuml.data.loaders import get_loader_backend
from nexuml.data.module import NexuDataModule
from nexuml.data.registry import DatasetRegistry, get_dataset_registry
from nexuml.data.super_dataset import SuperDataset

logger = logging.getLogger("lightning.pytorch")


class NexuDataCreator:
    """Build datasets and datamodules from a ``DataSpec``."""

    def __init__(self, registry: DatasetRegistry | None = None):
        self.registry = registry or get_dataset_registry()

    def build_dataset(self, spec: DataSpec) -> NexuDataset:
        # Normalize synthetic data specs that don't have datasets
        if not spec.datasets and spec.source_type == "synthetic":
            target_dicts = []
            for t in spec.targets:
                d = {
                    "type": t.type,
                    "key": t.key,
                    "num_classes": t.num_classes,
                    "num_outputs": t.num_outputs,
                    "positive_fraction": t.positive_fraction,
                    "label_density": 0.3,
                }
                target_dicts.append(d)

            spec = spec.model_copy(
                update={
                    "datasets": [
                        DatasetSpec(
                            type_key="synthetic",
                            params={
                                "feature_shape": spec.params.get("feature_shape", (128,)),
                                "num_samples": spec.params.get("num_samples", 1000),
                                "noise_type": spec.params.get("noise_type", "gaussian"),
                                "num_clusters": spec.params.get("num_clusters"),
                                "seed": spec.params.get("seed", 42),
                                "targets": target_dicts or None,
                                "feature_key": spec.feature_key,
                            },
                        )
                    ]
                }
            )

        if not spec.datasets:
            raise ValueError("DataSpec.datasets is empty. Add at least one DatasetSpec.")

        source_datasets: dict[str, NexuDataset] = {}
        seen_types: dict[str, int] = {}
        all_label_names: list[str] = []

        for ds_spec in spec.datasets:
            dataset = self.registry.instantiate(ds_spec.type_key, **ds_spec.params)
            if not isinstance(dataset, NexuDataset):
                raise TypeError(
                    f"Dataset '{ds_spec.type_key}' must be a NexuDataset subclass, "
                    f"got {type(dataset).__name__}"
                )

            prepared = self._prepare_dataset(dataset, ds_spec)
            source_name = self._make_source_name(ds_spec.type_key, seen_types)
            setattr(prepared, "source_type", ds_spec.type_key)
            source_datasets[source_name] = prepared

            # Collect label names from all datasets
            for label in prepared.label_names:
                if label not in all_label_names:
                    all_label_names.append(label)

        if len(source_datasets) == 1:
            merged_dataset = next(iter(source_datasets.values()))
            if spec.merge_labels:
                # Single dataset with merge_labels: wrap in SuperDataset
                merged_dataset = SuperDataset(
                    meta_data_list=source_datasets,
                    label_names=all_label_names,
                    merge_labels=spec.merge_labels,
                )
                for label_name in spec.merge_labels:
                    if label_name not in all_label_names:
                        all_label_names.append(label_name)
                merged_dataset.label_names = all_label_names
                merged_dataset._update_num_classes()
        else:
            merged_dataset = SuperDataset(
                meta_data_list=source_datasets,
                label_names=all_label_names,
                merge_labels=spec.merge_labels,
            )
            if spec.merge_labels:
                for label_name in spec.merge_labels:
                    if label_name not in all_label_names:
                        all_label_names.append(label_name)
                merged_dataset.label_names = all_label_names
                merged_dataset._update_num_classes()

        if len(merged_dataset) == 0:
            dataset_summaries = []
            for ds_spec in spec.datasets:
                root = (
                    ds_spec.params.get("data_root")
                    or ds_spec.params.get("root")
                    or ds_spec.params.get("data_dir")
                )
                machine_type = ds_spec.params.get("machine_type")
                summary = ds_spec.type_key
                if root is not None:
                    summary += f"(root={root})"
                meta_dir = ds_spec.params.get("meta_dir")
                if meta_dir is not None:
                    summary += f"[meta_dir={meta_dir}]"
                if machine_type is not None:
                    summary += f"[machine_type={machine_type}]"
                dataset_summaries.append(summary)
            details = ", ".join(dataset_summaries)
            recovery = ""
            if spec.params.get("download") is False:
                recovery = (
                    " Enable dataset download if supported or set NEXUML_DATA_ROOT/data_root."
                )
            raise ValueError(
                "No samples found while building the dataset. "
                f"Check the configured data roots and filters: {details}.{recovery}"
            )

        merged_dataset.split_meta([spec.train_split, spec.val_split, spec.test_split])
        return merged_dataset

    def build(
        self,
        spec: DataSpec,
        default_batch_size: int | AutoBatchSizeSpec = 64,
    ) -> NexuDataModule:
        dataset = self.build_dataset(spec)
        loader_spec = self._resolve_loader_spec(spec.loader, default_batch_size)

        return NexuDataModule(
            dataset=dataset,
            loader_spec=loader_spec,
            train_split=spec.train_split,
            val_split=spec.val_split,
            test_split=spec.test_split,
            split_by_column=getattr(dataset, "meta", None) is not None,
        )

    def _prepare_dataset(self, dataset: NexuDataset, ds_spec: DatasetSpec) -> NexuDataset:
        meta = getattr(dataset, "meta", None)
        if (
            meta is not None
            and ds_spec.max_samples is not None
            and len(dataset) > ds_spec.max_samples
        ):
            sample_idx = meta.sample(n=ds_spec.max_samples).index.tolist()
            dataset = dataset.take(sample_idx)
            meta = getattr(dataset, "meta", None)
            logger.info("[%s] Sampled to %d rows", ds_spec.type_key, ds_spec.max_samples)

        if meta is not None:
            if ds_spec.split_type != "keep" or "split" not in meta.columns:
                meta["split"] = ds_spec.split_type

            meta["split"] = meta["split"].apply(
                lambda split_name: "val" if "val" in str(split_name) else split_name
            )
            meta["modality"] = ds_spec.modality

        dataset.modality = ds_spec.modality
        return dataset

    def _resolve_loader_spec(
        self,
        loader_spec: LoaderSpec,
        default_batch_size: int | AutoBatchSizeSpec,
    ) -> LoaderSpec:
        batch_size = loader_spec.batch_size or default_batch_size
        if isinstance(batch_size, AutoBatchSizeSpec):
            batch_size: int = batch_size.min

        # Validate the backend exists
        backend_name = loader_spec.backend
        get_loader_backend(backend_name)

        return loader_spec.model_copy(update={"batch_size": batch_size})

    @staticmethod
    def _make_source_name(type_key: str, seen_types: dict[str, int]) -> str:
        occurrence = seen_types.get(type_key, 0)
        seen_types[type_key] = occurrence + 1
        return type_key if occurrence == 0 else f"{type_key}#{occurrence}"
