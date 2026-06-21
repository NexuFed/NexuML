"""Opt-in real-data contract tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nexuml.data.registry import get_dataset_registry


@pytest.mark.requires_data
def test_real_dataset_loads():
    root = Path(os.environ["NEXUML_DATA_ROOT"])
    registry = get_dataset_registry()
    # Try to instantiate the first real source that accepts a root path.
    for key, cls in registry.list().items():
        if key == "synthetic":
            continue
        try:
            dataset = registry.instantiate(key, root=str(root))
        except Exception:
            continue
        assert len(dataset) > 0
        x, y = dataset[0]
        assert x is not None
        break
    else:
        pytest.skip("no real data source could be instantiated")
