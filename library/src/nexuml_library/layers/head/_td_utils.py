"""TensorDict helpers for pipeline layers (group keys, fit masks)."""

from __future__ import annotations

from numbers import Integral, Real
import re
from typing import Any, cast

import torch
from tensordict import NonTensorData, TensorDict

GroupKey = tuple[Any, ...]


def _canonical_group_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)(?:\.\d+)?", text):
            numeric = float(text)
            return int(numeric) if numeric.is_integer() else numeric
        return text
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    return value


def _axis_to_list(val: Any, n: int) -> list[Any]:
    """Coerce a TensorDict value (Tensor / NonTensorData / list) to a flat Python list.

    Returns:
        list[Any]: Flattened list of canonical values with length ``n``.
    """
    if isinstance(val, list):
        return [_canonical_group_value(v) for v in val]
    if isinstance(val, NonTensorData):
        return [_canonical_group_value(v) for v in val.data]
    flat = val.detach().cpu().reshape(-1)
    return [_canonical_group_value(v.item() if hasattr(v, "item") else v) for v in flat]


def get_group_keys_from_td(
    x: TensorDict,
    y: TensorDict | None,
    group_key_names: list[str],
    n: int,
) -> list[GroupKey]:
    """Build per-sample group key tuples from declared axis names in x/y.

    Returns:
        list[GroupKey]: One tuple per sample combining all group axis values.

    Raises:
        KeyError: If a declared group axis name is missing from both x and y.
    """
    cols: list[list[Any]] = []
    for name in group_key_names:
        val = None
        if y is not None and name in y.keys():
            val = y[name]
        elif name in x.keys():
            val = x[name]
        if val is None:
            raise KeyError(f"Group axis '{name}' not found in x or y.")
        col = _axis_to_list(val, n)
        cols.append(col)
    return [tuple(row) for row in zip(*cols)]


def get_fit_mask_from_td(
    x: TensorDict,
    y: TensorDict | None,
    fit_mask_key: str | None,
    fit_label_key: str | None,
    normal_label_value: int | float,
    n: int,
) -> torch.Tensor:
    """Return a boolean mask selecting samples to use for fitting.

    Raises:
        KeyError: If ``fit_mask_key`` is provided but not found in x or y.
    """
    if fit_mask_key is not None:
        if y is not None and fit_mask_key in y.keys():
            return y[fit_mask_key].detach().cpu().reshape(-1).bool()
        if fit_mask_key in x.keys():
            return x[fit_mask_key].detach().cpu().reshape(-1).bool()
        raise KeyError(f"fit_mask_key '{fit_mask_key}' not found in x or y.")
    if fit_label_key is not None:
        src = y if y is not None and fit_label_key in y.keys() else x
        if fit_label_key in src.keys():
            labels = src[fit_label_key].detach().cpu().reshape(-1)
            if labels.numel() == n:
                return cast(torch.Tensor, labels == normal_label_value)
    return torch.ones(n, dtype=torch.bool)
