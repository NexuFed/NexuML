"""Base class for post-training evaluation algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from tensordict import TensorDict


class ContractError(KeyError):
    """Raised when a declared key contract cannot be satisfied at runtime."""


def get_declared_tensor(x: TensorDict, key: str, *, algorithm: str = "") -> Any:
    """Return a declared tensor key from x; raise ContractError if absent.

    ``algorithm`` is included in diagnostics when provided.

    Returns:
        The tensor stored under *key*.

    Raises:
        ContractError: If *key* is not present in *x*.
    """
    if key not in x.keys():
        prefix = f"Algorithm '{algorithm}': " if algorithm else ""
        raise ContractError(
            f"{prefix}declared tensor key '{key}' not found in evaluation output. "
            f"Available keys: {sorted(x.keys())}"
        )
    return x[key]


def get_declared_axis(
    x: TensorDict,
    y: TensorDict | None,
    axis_spec: Any,
    *,
    metadata: Any = None,
    algorithm: str = "",
) -> Any:
    """Resolve a declared axis key from x, y, or metadata per its provenance.

    Enforces EXACT provenance — no silent fallback between sources.
    axis_spec may be a string (shorthand, defaults source=y) or an AxisKeySpec.
    ``metadata`` is an optional DataFrame-like with per-sample columns.

    Returns:
        The resolved axis value from the appropriate source.

    Raises:
        ValueError: If *axis_spec* references an unknown source.
        ContractError: If the declared key is missing from the resolved source.
    """
    if isinstance(axis_spec, str):
        key = axis_spec
        source = "y"
    else:
        key = axis_spec.key
        source = axis_spec.source

    prefix = f"Algorithm '{algorithm}': " if algorithm else ""

    if source == "x":
        if key not in x.keys():
            raise ContractError(
                f"{prefix}declared axis '{key}' (source=x) not found in evaluation output. "
                f"Available x keys: {sorted(x.keys())}"
            )
        return x[key]
    elif source == "y":
        if y is None or key not in y.keys():
            raise ContractError(
                f"{prefix}declared axis '{key}' (source=y) not found in labels. "
                f"Available y keys: {sorted(y.keys()) if y is not None else []}"
            )
        return y[key]
    elif source == "metadata":
        if metadata is None:
            raise ContractError(
                f"{prefix}declared axis '{key}' (source=metadata) but no metadata attached."
            )
        if hasattr(metadata, "columns") and key in metadata.columns:
            return metadata[key]
        raise ContractError(
            f"{prefix}declared axis '{key}' (source=metadata) not found in metadata. "
            f"Available columns: {sorted(metadata.columns) if hasattr(metadata, 'columns') else []}"
        )
    else:
        raise ValueError(f"Unknown axis source '{source}' for key '{key}'.")


def get_fit_mask(
    x: TensorDict,
    y: TensorDict | None,
    fit_mask_key: str | None,
    batch_size: int,
    *,
    algorithm: str = "",
) -> torch.Tensor:
    """Resolve fit mask to a bool tensor; returns all-True if fit_mask_key is None.

    Returns:
        Bool tensor of shape ``(batch_size,)``.

    Raises:
        ContractError: If *fit_mask_key* is not found in *y* or *x*.
    """
    if fit_mask_key is None:
        return torch.ones(batch_size, dtype=torch.bool)
    prefix = f"Algorithm '{algorithm}': " if algorithm else ""
    if y is not None and fit_mask_key in y.keys():
        mask = y[fit_mask_key].detach().cpu().reshape(-1).bool()
    elif fit_mask_key in x.keys():
        mask = x[fit_mask_key].detach().cpu().reshape(-1).bool()
    else:
        raise ContractError(
            f"{prefix}declared fit_mask key '{fit_mask_key}' not found in labels or output. "
            f"Available x keys: {sorted(x.keys())}, "
            f"y keys: {sorted(y.keys()) if y is not None else []}"
        )
    return mask


class EvalAlgorithm(ABC):
    """Base class for post-training evaluation algorithms.

    Flat batch lifecycle:
      - fit_batch(x, y) — accumulate statistics from one train batch (optional)
      - fit_end() — finalize fitting after all train batches
      - eval_batch(x, y) — process one test batch using fitted state
      - eval_end() — finalize evaluation after all test batches
      - results() — return computed metrics as a flat dict

    Evaluation algorithms are consumer-only: they read from the pipeline output
    TensorDict and never produce new score keys. Score-producing components
    belong in the pipeline as PostTrainFitLayer subclasses.
    """

    def fit_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        """Accumulate statistics from one train batch."""

    def fit_end(self) -> None:
        """Finalize fitting after all train batches. Default: no-op."""

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        """Score/process one test batch using fitted state."""

    def eval_end(self) -> None:
        """Finalize evaluation after all test batches."""

    def visualize(self, logger: Any) -> None:
        """Produce visualizations and log them (optional)."""

    @abstractmethod
    def results(self) -> dict[str, float]:
        """Return computed metrics as a flat dict."""


class DistanceEstimator(ABC):
    """Abstract base for streaming distance estimators.

    Lifecycle:
      1. fit_batch(features, labels=None) — accumulate train statistics batch-by-batch
      2. fit_end() — finalize (compute inverse covariance, fit GMM, etc.)
      3. score(features, labels=None) — return per-sample anomaly scores
    """

    @abstractmethod
    def fit_batch(self, features: torch.Tensor, labels: TensorDict | None = None) -> None:
        """Accumulate statistics from a batch of features."""

    @abstractmethod
    def fit_end(self) -> None:
        """Finalize the estimator after all batches."""

    @abstractmethod
    def score(self, features: torch.Tensor, labels: TensorDict | None = None) -> torch.Tensor:
        """Return per-sample anomaly scores (higher = more anomalous)."""
