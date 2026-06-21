"""SuperDataset: merges multiple datasets into one unified dataset."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tensordict import TensorDict

from nexuml.data.dataset import NexuDataset

logger = logging.getLogger(__name__)


class SuperDataset(NexuDataset):
    """Merges multiple datasets (keyed by name) into one dataset.

    Adds a "dataset" column to the merged DataFrame to track origin.
    Supports label merging and dataset subdivision for federated/multi-client scenarios.
    """

    def __init__(
        self,
        meta_data_list: dict[str, NexuDataset],
        label_names: list[str] | None = None,
        merge_labels: dict[str, dict] | None = None,
        **kwargs,
    ):
        self.meta_data_list = dict(meta_data_list)

        # Build merged DataFrame
        frames = []
        for name, ds in self.meta_data_list.items():
            dataset_meta = ds.meta
            if dataset_meta is None:
                raise ValueError("SuperDataset requires datasets with meta information")

            frame = dataset_meta.copy()
            frame["dataset"] = name
            frame["source_type"] = getattr(ds, "source_type", ds.__class__.__name__)
            frame["source_dataset"] = name
            frame["source_index"] = np.arange(len(frame))
            frames.append(frame)

        merged = pd.concat(frames, ignore_index=True)

        if merge_labels is not None:
            merged = self._apply_merge_labels(merged, meta_data_list, merge_labels)

        super().__init__(meta=merged, label_names=label_names, **kwargs)

    def _apply_merge_labels(
        self,
        meta: pd.DataFrame,
        meta_data_list: dict[str, NexuDataset],
        merge_labels: dict[str, dict],
    ) -> pd.DataFrame:
        for label_name, spec in merge_labels.items():
            logits = spec.get("logits", False)
            columns = spec.get("columns", [])
            include_dataset = spec.get("include_dataset", True)
            dataset_names = meta["dataset"].unique().tolist()

            if not logits:
                # Categorical merge: combine source columns into a single string label
                merged_labels = pd.Series(index=meta.index, dtype="object")
                for dataset in dataset_names:
                    idx = meta[meta["dataset"] == dataset].index
                    prefix = f"{dataset}_" if include_dataset else ""
                    merged_labels.loc[idx] = meta.loc[idx, columns].apply(
                        lambda row: prefix + "_".join(row.values.astype(str)),
                        axis=1,
                    )
                meta[label_name] = pd.Categorical(merged_labels).codes.astype(float)
                logger.info(f"Merged labels {columns} → '{label_name}' (categorical)")
            else:
                # Logits merge: create a one-hot/logit vector across all labels
                conversion_dict: dict[str, int] = {}
                for dataset_name in dataset_names:
                    for label in columns:
                        col = meta.loc[meta["dataset"] == dataset_name, label]
                        if col.empty:
                            continue
                        first = col.iloc[0]
                        if isinstance(first, (list, np.ndarray)):
                            for i in range(len(first)):
                                key = f"{label}_{i}"
                                if key not in conversion_dict:
                                    conversion_dict[key] = len(conversion_dict)
                        else:
                            for val in col.unique():
                                key = f"{label}_{float(val)}"
                                if key not in conversion_dict:
                                    conversion_dict[key] = len(conversion_dict)

                n = len(conversion_dict)
                meta[label_name] = [np.zeros(n).tolist() for _ in range(len(meta))]

                for dataset_name in dataset_names:
                    for label in columns:
                        ds_idx = meta[meta["dataset"] == dataset_name].index
                        for i in ds_idx:
                            val = meta.at[i, label]
                            vec = list(meta.at[i, label_name])
                            if isinstance(val, (list, np.ndarray)):
                                for j, v in enumerate(val):
                                    key = f"{label}_{j}"
                                    if key in conversion_dict:
                                        vec[conversion_dict[key]] += float(v)
                            else:
                                key = f"{label}_{float(val)}"
                                if key in conversion_dict:
                                    vec[conversion_dict[key]] += 1.0
                            meta.at[i, label_name] = vec

                logger.info(f"Merged labels {columns} → '{label_name}' (logits, dim={n})")

        return meta

    def divide_dataset(
        self,
        num_splits: int,
        key: str | None = None,
        alpha_in: float = 5.0,
        alpha_out: float = 0.0,
        labels_per_cluster: int = 3,
        num_clusters: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Assign samples to splits via Dirichlet cluster skew distribution.

        Raises:
            ValueError: If ``key`` is ``None`` or metadata is missing.
        """
        if key is None:
            raise ValueError("key must be provided for divide_dataset")
        meta = self.meta
        if meta is None:
            raise ValueError("divide_dataset() requires meta information")

        num_classes = int(meta[key].nunique())
        if num_clusters is None:
            num_clusters = max(1, num_splits // 3 + 1)

        proportions = self._create_cluster_skew(
            num_clients=num_splits,
            num_labels=num_classes,
            num_clusters=num_clusters,
            labels_per_cluster=min(labels_per_cluster, num_classes),
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            seed=seed,
        )
        proportions = proportions / (proportions.sum(axis=0, keepdims=True) + 1e-9)
        self._assign_subset(proportions, key=key)

    def divide_dataset_by_key(self, key: str) -> int:
        """Assign each unique key value to its own split index.

        Returns:
            Number of unique splits created.

        Raises:
            ValueError: If metadata is missing.
        """
        meta = self.meta
        if meta is None:
            raise ValueError("divide_dataset_by_key() requires meta information")

        for split_idx, val in enumerate(meta[key].unique()):
            meta.loc[meta[key] == val, "assignment"] = split_idx
        self.meta = meta
        return int(meta["assignment"].nunique())

    def _assign_subset(self, dataset_probability: np.ndarray, key: str = "dataset") -> None:
        meta = self.meta
        if meta is None:
            raise ValueError("_assign_subset() requires meta information")

        num_splits = dataset_probability.shape[0]
        meta["assignment"] = -1
        for ds_idx, ds in enumerate(meta[key].unique()):
            idx = meta[meta[key] == ds].index
            probs = dataset_probability[:, ds_idx]
            assignments = np.random.choice(num_splits, size=len(idx), p=probs)
            meta.loc[idx, "assignment"] = assignments
        self.meta = meta

    def create_subset_dataset(self, idx: int, do_split: bool = True) -> NexuDataset:
        """Return a dataset for a single assignment subset.

        Returns:
            A :class:`NexuDataset` containing only samples assigned to ``idx``.

        Raises:
            ValueError: If metadata is missing.
        """
        meta = self.meta
        if meta is None:
            raise ValueError("create_subset_dataset() requires meta information")

        subset_idx = meta[meta["assignment"] == idx].index.tolist()
        subset_dataset = self.take(subset_idx)
        if do_split:
            subset_dataset.split_meta(subset_dataset.split_ratio)
        return subset_dataset

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        source_name = row.get("source_dataset")
        if source_name in self.meta_data_list:
            source_dataset = self.meta_data_list[source_name]
            raw_source_idx = row.get("source_index", idx)
            source_idx = idx if raw_source_idx is None else int(raw_source_idx)
            if source_dataset.meta is not None:
                source_row = source_dataset.meta.iloc[source_idx]
                return source_dataset.load_item(source_idx, source_row)
            x, _ = source_dataset[source_idx]
            return x

        return super().load_item(idx, row)

    @staticmethod
    def _create_cluster_skew(
        num_clients: int,
        num_labels: int,
        num_clusters: int,
        labels_per_cluster: int,
        alpha_in: float,
        alpha_out: float,
        seed: int | None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        base = num_clients // num_clusters
        rem = num_clients % num_clusters
        sizes = [base + (1 if c < rem else 0) for c in range(num_clusters)]

        client_to_cluster = np.empty(num_clients, dtype=int)
        idx = 0
        for c, size in enumerate(sizes):
            client_to_cluster[idx : idx + size] = c
            idx += size

        cluster_to_labels = {
            c: [(c * labels_per_cluster + i) % num_labels for i in range(labels_per_cluster)]
            for c in range(num_clusters)
        }

        proportions = np.zeros((num_clients, num_labels))
        for i in range(num_clients):
            c = client_to_cluster[i]
            conc = np.full(num_labels, max(alpha_out, 1e-9))
            conc[cluster_to_labels[c]] = alpha_in
            proportions[i] = rng.dirichlet(conc)

        return proportions
