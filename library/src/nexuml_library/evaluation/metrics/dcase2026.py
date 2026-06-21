"""DCASE 2026 evaluation metrics (pure functions).

All functions operate on torch Tensors or numpy arrays and do not require
sklearn unless noted.  They are designed to be used inside EvalAlgorithm
subclasses or standalone scripts.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 1: Hierarchical classification metrics
# ---------------------------------------------------------------------------


def task1_hierarchical_metrics(
    y_true_top: np.ndarray,
    y_true_second: np.ndarray,
    y_pred_top: np.ndarray,
    y_pred_second: np.ndarray,
    num_top_classes: int | None = None,
) -> dict[str, float]:
    """Compute Task 1 hierarchical precision, recall, and F-score.

    Returns macro-averaged hierarchical F-score (hF) over second-level classes
    plus top-level accuracy and per-class tables.

    Returns:
        Dict with keys ``top_level_accuracy``, ``macro_hierarchical_precision``,
        ``macro_hierarchical_recall``, ``macro_hierarchical_fscore``,
        ``num_second_classes``, ``num_top_classes``.
    """
    y_true_top = np.asarray(y_true_top)
    y_true_second = np.asarray(y_true_second)
    y_pred_top = np.asarray(y_pred_top)
    y_pred_second = np.asarray(y_pred_second)

    # Top-level accuracy
    top_acc = float(np.mean(y_true_top == y_pred_top))

    # Second-level metrics per class
    second_classes = np.unique(y_true_second)
    precisions: list[float] = []
    recalls: list[float] = []
    fscores: list[float] = []

    for cls in second_classes:
        tp = int(((y_pred_second == cls) & (y_true_second == cls)).sum())
        fp = int(((y_pred_second == cls) & (y_true_second != cls)).sum())
        fn = int(((y_pred_second != cls) & (y_true_second == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    macro_hp = float(np.mean(precisions)) if precisions else 0.0
    macro_hr = float(np.mean(recalls)) if recalls else 0.0
    macro_hf = float(np.mean(fscores)) if fscores else 0.0

    return {
        "top_level_accuracy": top_acc,
        "macro_hierarchical_precision": macro_hp,
        "macro_hierarchical_recall": macro_hr,
        "macro_hierarchical_fscore": macro_hf,
        "num_second_classes": len(second_classes),
        "num_top_classes": num_top_classes or len(np.unique(y_true_top)),
    }


def task1_top_level_confusion(y_true_top: np.ndarray, y_pred_top: np.ndarray) -> np.ndarray:
    """Return a top-level confusion matrix as a 2-D numpy array."""
    labels = sorted(set(np.unique(y_true_top)) | set(np.unique(y_pred_top)))
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true_top, y_pred_top):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


# ---------------------------------------------------------------------------
# Task 2: Anomaly detection metrics
# ---------------------------------------------------------------------------


def _safe_roc_auc(
    labels: np.ndarray, scores: np.ndarray, max_fpr: float | None = None
) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        if labels.size == 0 or np.unique(labels).size < 2:
            return None
        if max_fpr is not None:
            return float(roc_auc_score(labels, scores, max_fpr=max_fpr))
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def task2_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute AUC for Task 2.

    Returns:
        AUC score, or None if sklearn is unavailable or labels are degenerate.
    """
    return _safe_roc_auc(labels, scores)


def task2_pauc(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.1) -> float | None:
    """Compute partial AUC with ``max_fpr`` (default 0.1).

    Returns:
        Partial AUC score, or None if sklearn is unavailable or labels are degenerate.
    """
    return _safe_roc_auc(labels, scores, max_fpr=max_fpr)


def task2_harmonic_mean(values: Iterable[float], eps: float = 1e-12) -> float | None:
    """Official harmonic mean used for DCASE Task 2 ranking.

    Clips values to ``[eps, 1.0]`` following the official DCASE 2026 evaluation
    protocol where inputs are AUC/pAUC probabilities bounded in [0, 1].

    Intentionally differs from the generic ``_plotting.harmonic_mean``, which
    filters to positive finite values ``(0, ∞)`` and is designed for arbitrary
    positive quantities.  Do **not** unify these two functions.

    Returns:
        Clipped harmonic mean, or None if no finite values remain after clipping.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    arr = np.clip(arr, eps, 1.0)
    return float(arr.size / np.sum(1.0 / arr))


def _as_1d_array(values: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(values).reshape(-1)


def _format_group_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _domain_masks(domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(source_mask, target_mask)`` boolean arrays for a domain array.

    **Canonical domain normalization policy (internal):**
    Values are normalised to stripped lowercase strings and matched against
    known aliases.  The selector's ``target`` label is equivalent to the
    canonical ``domain`` label — both represent the source/target split.

    Recognised aliases (case-insensitive, stripped):
      - Source: ``"source"``, ``"src"``, ``"0"``, ``"false"``
      - Target: ``"target"``, ``"tgt"``, ``"1"``, ``"true"``

    Strict policy: when no aliases match, returns two all-False masks.
    Ambiguous/unrecognized domain values must be resolved explicitly by the
    caller. No silent guessing via sorted-unique fallback.
    """
    normalized = np.asarray([_format_group_value(d).strip().lower() for d in domain])
    source_names = {"source", "src", "0", "false"}
    target_names = {"target", "tgt", "1", "true"}
    source_mask = np.isin(normalized, list(source_names))
    target_mask = np.isin(normalized, list(target_names))
    # Return masks even if only one side matches (caller handles warnings)
    return source_mask, target_mask


def compute_dcase_task2_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    machine: np.ndarray,
    section: np.ndarray,
    domain: np.ndarray,
    max_fpr: float = 0.1,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute official DCASE Task 2 per-section metrics and Omega.

    AUC_source/target use domain-split normals and the same anomaly set pooled
    across domains. pAUC uses both domains pooled and sklearn's standardized
    ``roc_auc_score(..., max_fpr=max_fpr)`` result without extra scaling.

    Returns:
        Dict with keys ``per_machine``, ``omega``, ``mean_auc_source``,
        ``mean_auc_target``, ``mean_pauc``, ``hmean_auc_source``,
        ``hmean_auc_target``, ``hmean_pauc``, ``hmean_all``, ``warnings``.

    Raises:
        ValueError: If input arrays have mismatched lengths.
    """
    scores = _as_1d_array(scores).astype(float)
    labels = _as_1d_array(labels).astype(int)
    machine = _as_1d_array(machine)
    section = _as_1d_array(section)
    domain = _as_1d_array(domain)
    if not (scores.size == labels.size == machine.size == section.size == domain.size):
        raise ValueError("scores, labels, machine, section, and domain must have equal length")

    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    machine = machine[valid]
    section = section[valid]
    domain = domain[valid]

    warnings: list[str] = []

    # DCASE-specific label validation — fail loudly on structural issues.
    unique_labels = np.unique(labels)
    if unique_labels.size == 0:
        warnings.append("DCASE Task 2: no valid samples remaining after filtering")
    elif not np.isin(unique_labels, [0, 1]).all():
        warnings.append(
            f"DCASE Task 2: anomaly labels contain non-binary values {unique_labels.tolist()}; "
            "expected 0 (normal) and 1 (anomaly)"
        )
    if np.unique(machine).size == 0:
        warnings.append("DCASE Task 2: machine labels are empty or all-NaN")
    if np.unique(section).size == 0:
        warnings.append("DCASE Task 2: section labels are empty or all-NaN")
    # Domain may use canonical aliases or raw target values; check for ambiguity.
    _src_check, _tgt_check = _domain_masks(domain)
    if not _src_check.any() and not _tgt_check.any() and domain.size > 0:
        warnings.append(
            "DCASE Task 2: domain labels could not be resolved to source/target; "
            "expected values like 'source'/'target', 0/1, or 'src'/'tgt'"
        )

    per_machine: dict[str, dict[str, float | int | str | None]] = {}
    auc_source_values: list[float] = []
    auc_target_values: list[float] = []
    pauc_values: list[float] = []
    omega_values: list[float] = []

    group_values = sorted(
        {(m, s) for m, s in zip(machine, section, strict=False)},
        key=lambda x: (str(x[0]), str(x[1])),
    )
    for m, s in group_values:
        group_mask = (machine == m) & (section == s)
        source_domain, target_domain = _domain_masks(domain[group_mask])
        group_indices = np.flatnonzero(group_mask)
        source_mask = np.zeros_like(group_mask, dtype=bool)
        target_mask = np.zeros_like(group_mask, dtype=bool)
        source_mask[group_indices] = source_domain
        target_mask[group_indices] = target_domain

        normal_source = group_mask & source_mask & (labels == 0)
        normal_target = group_mask & target_mask & (labels == 0)
        anomaly = group_mask & (labels == 1)
        pooled = group_mask & ((labels == 0) | (labels == 1))
        key = f"m{_format_group_value(m)}_s{_format_group_value(s)}"
        metrics: dict[str, float | int | str | None] = {
            "machine": _format_group_value(m),
            "section": _format_group_value(s),
            "n_source_normal": int(normal_source.sum()),
            "n_target_normal": int(normal_target.sum()),
            "n_anomaly": int(anomaly.sum()),
        }

        for metric_name, normal_mask, dest in (
            ("auc_source", normal_source, auc_source_values),
            ("auc_target", normal_target, auc_target_values),
        ):
            metric_mask = normal_mask | anomaly
            if normal_mask.sum() == 0 or anomaly.sum() == 0:
                warnings.append(
                    f"Skipped {metric_name} for {key}: empty normal domain or anomaly set"
                )
                metrics[metric_name] = None
                continue
            value = task2_auc(scores[metric_mask], labels[metric_mask])
            if value is None:
                warnings.append(f"Skipped {metric_name} for {key}: single class present")
                metrics[metric_name] = None
                continue
            value = float(np.clip(value, eps, 1.0))
            metrics[metric_name] = value
            dest.append(value)
            omega_values.append(value)

        if normal_source.sum() == 0 or normal_target.sum() == 0 or anomaly.sum() == 0:
            warnings.append(f"Skipped pauc for {key}: empty domain or anomaly set")
            metrics["pauc"] = None
        else:
            value = task2_pauc(scores[pooled], labels[pooled], max_fpr=max_fpr)
            if value is None:
                warnings.append(f"Skipped pauc for {key}: single class present")
                metrics["pauc"] = None
            else:
                value = float(np.clip(value, eps, 1.0))
                metrics["pauc"] = value
                pauc_values.append(value)
                omega_values.append(value)

        hmean_values = [
            float(v)
            for v in (metrics.get("auc_source"), metrics.get("auc_target"), metrics.get("pauc"))
            if isinstance(v, (int, float))
        ]
        metrics["hmean"] = task2_harmonic_mean(hmean_values)
        per_machine[key] = metrics

    if not omega_values:
        warnings.append(
            "DCASE Task 2: zero valid metric groups — official Omega cannot be reported"
        )

    result: dict[str, Any] = {
        "per_machine": per_machine,
        "omega": task2_harmonic_mean(omega_values),
        "mean_auc_source": float(np.mean(auc_source_values)) if auc_source_values else None,
        "mean_auc_target": float(np.mean(auc_target_values)) if auc_target_values else None,
        "mean_pauc": float(np.mean(pauc_values)) if pauc_values else None,
        "hmean_auc_source": task2_harmonic_mean(auc_source_values),
        "hmean_auc_target": task2_harmonic_mean(auc_target_values),
        "hmean_pauc": task2_harmonic_mean(pauc_values),
        "hmean_all": task2_harmonic_mean(omega_values),
        "warnings": warnings,
    }
    for warning in warnings:
        logger.warning("DCASE Task 2 metric warning: %s", warning)
    return result


def task2_f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Compute F1 score at a fixed threshold.

    Returns:
        F1 score.
    """
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def task2_metrics_per_group(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    max_fpr: float = 0.1,
) -> dict[str, dict[str, float | None]]:
    """Return AUC, pAUC, and best-F1 per group.

    ``groups`` is an array of the same length as ``scores``/``labels``.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    result: dict[str, dict[str, float | None]] = {}
    for g in sorted(np.unique(groups)):
        mask = groups == g
        auc = task2_auc(scores[mask], labels[mask])
        pauc = task2_pauc(scores[mask], labels[mask], max_fpr=max_fpr)
        # Best F1 via threshold sweep
        best_f1 = 0.0
        if auc is not None:
            try:
                from sklearn.metrics import roc_curve

                fpr, tpr, thresholds = roc_curve(labels[mask], scores[mask])
                for thresh in thresholds:
                    f1 = task2_f1_at_threshold(scores[mask], labels[mask], thresh)
                    if f1 > best_f1:
                        best_f1 = f1
            except Exception:
                pass
        result[str(g)] = {
            "auc": auc,
            f"pauc_{int(max_fpr * 100)}": pauc,
            "f1_optimal": best_f1 if auc is not None else None,
        }
    return result


# ---------------------------------------------------------------------------
# Task 7: Domain-incremental metrics
# ---------------------------------------------------------------------------


def task7_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy (average of per-class recalls).

    Returns:
        Balanced accuracy score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    recalls: list[float] = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recall = float(np.mean(y_pred[mask] == cls))
        recalls.append(recall)
    return float(np.mean(recalls)) if recalls else 0.0


def task7_domain_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    domains: np.ndarray,
) -> dict[str, float]:
    """Compute per-domain balanced accuracy and final averaged score.

    Returns a dict with keys like ``domain_D2_balanced_accuracy`` and
    ``final_averaged_score``.

    Returns:
        Dict with per-domain balanced accuracy and averaged score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    domains = np.asarray(domains)
    result: dict[str, float] = {}
    domain_scores: list[float] = []
    for d in sorted(np.unique(domains)):
        mask = domains == d
        ba = task7_balanced_accuracy(y_true[mask], y_pred[mask])
        result[f"domain_{d}_balanced_accuracy"] = ba
        domain_scores.append(ba)
    result["final_averaged_score"] = float(np.mean(domain_scores)) if domain_scores else 0.0
    return result


def task7_incremental_table(
    results: list[dict[str, float]], domain_order: list[str] | None = None
) -> dict[str, list[float]]:
    """Build an incremental table from a list of per-step metric dicts.

    Each dict should contain keys like ``domain_D2_balanced_accuracy``.
    Returns a dict mapping metric names to lists of values across steps.

    Returns:
        Dict mapping metric names to lists of accumulated float values.
    """
    table: dict[str, list[float]] = {}
    for step_result in results:
        for key, value in step_result.items():
            if isinstance(value, (int, float, np.floating)):
                table.setdefault(key, []).append(float(value))
    return table
