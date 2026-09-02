"""Opt-in real-data contract tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nexuml_library.data.dcaset2.dcase2026 import DCASET2Dataset


@pytest.mark.requires_data
def test_real_dataset_loads():
    root = Path(os.environ["NEXUML_DATA_ROOT"])
    dataset = DCASET2Dataset(root=root).build()
    if not len(dataset):
        pytest.skip("no matching DCASE Task 2 samples found")

    x, y = dataset[0]
    assert x is not None
