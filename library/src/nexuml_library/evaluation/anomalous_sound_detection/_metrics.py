"""Shared anomaly detection metrics — single source of truth.

Used by AnomalyEvaluator, AnomalyScoreVisualizer, and DCASEDetectorSelector.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def harmonic_mean(values: Iterable[float]) -> float | None:
    """Compute a harmonic mean over positive finite values.

    Returns:
        Harmonic mean of positive finite values, or None if no valid values.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return None
    return float(array.size / np.sum(1.0 / array))


def _binary_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = preds.long().reshape(-1)
    labels = labels.long().reshape(-1)
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


def _optimal_f1(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.float().reshape(-1)
    labels = labels.long().reshape(-1)
    if scores.numel() == 0:
        return 0.0
    best = 0.0
    for threshold in torch.unique(scores):
        best = max(best, _binary_f1((scores >= threshold).long(), labels))
    return best


def compute_anomaly_metrics(
    scores: torch.Tensor, labels: torch.Tensor, max_fpr: float = 0.1
) -> dict[str, float]:
    """Compute AUC, pAUC, optimal F1, and score stats from raw scores and binary labels.

    Returns:
        Dict with keys ``auc``, ``pauc_<fpr>``, ``f1_optimal``, ``score_mean``,
        ``score_std``.  When sklearn is unavailable or labels are degenerate,
        only ``score_mean`` and ``score_std`` are returned.
    """
    try:
        from sklearn.metrics import roc_auc_score

        scores = scores.float().reshape(-1)
        labels = labels.float().reshape(-1)
        valid = torch.isfinite(labels) & ((labels == 0) | (labels == 1))
        if not valid.any():
            return {"score_mean": float(scores.mean()), "score_std": float(scores.std())}
        scores = scores[valid]
        labels = labels[valid]
        scores_np = scores.numpy()
        labels_np = labels.long().numpy()
        if labels_np.sum() == 0 or labels_np.sum() == len(labels_np):
            return {"score_mean": float(scores.mean()), "score_std": float(scores.std())}
        auc = float(roc_auc_score(labels_np, scores_np))
        pauc = float(roc_auc_score(labels_np, scores_np, max_fpr=max_fpr))
        return {
            "auc": auc,
            f"pauc_{int(max_fpr * 100)}": pauc,
            "f1_optimal": _optimal_f1(scores, labels.long()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
    except ImportError:
        return {"score_mean": float(scores.mean()), "score_std": float(scores.std())}
