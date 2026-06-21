"""Registry-driven contract tests for every discovered data source."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from nexuml.data.dataset import NexuDataset
from nexuml.data.registry import DatasetRegistry

# Data sources known to require real downloaded/exported data and therefore
# cannot be instantiated against synthetic-only test conditions. Any other
# discovered data source that raises fails its parameter case instead of skipping.
_SKIP_ALLOWLIST: dict[str, str] = {
    "AudiosetDataset": "requires a real Audioset data root",
    "CIFAR100Dataset": "requires download=True against a real data root",
    "CIFAR10Dataset": "requires download=True against a real data root",
    "DCASE2026T1Dataset": "requires a real DCASE2026 Task 1 data root",
    "DCASE2026T2Dataset": "requires a real DCASE2026 Task 2 data root",
    "DCASE2026T7Dataset": "requires a real DCASE2026 Task 7 data root",
    "DCASET2Dataset": "requires a real DCASE Task 2 data root",
    "ExportedDataset": "requires a real exported data module config.yaml",
    "FashionMNISTDataset": "requires download=True against a real data root",
    "MNISTDataset": "requires download=True against a real data root",
}


def _data_source_skip_or_fail(key: str, exc: Exception) -> None:
    """Skip allowlisted data sources; fail others with rich, actionable context."""
    if key not in _SKIP_ALLOWLIST:
        raise AssertionError(
            f"Conformance failure for data_source {key!r}: "
            f"{type(exc).__name__}: {exc}\n"
            f"Hint: add {key!r} to the data_source skip allowlist only if the failure "
            f"requires a dependency or real data that synthetic fixtures cannot provide."
        ) from exc
    pytest.skip(f"{_SKIP_ALLOWLIST[key]}: {exc}")


def _data_source_fail(key: str, detail: str) -> None:
    """Fail a data source contract case with a consistent, actionable message."""
    pytest.fail(
        f"Conformance failure for data_source {key!r}: {detail}\n"
        f"Hint: add {key!r} to the data_source skip allowlist only if the failure "
        f"requires a dependency or real data that synthetic fixtures cannot provide."
    )


def _minimal_params_for(data_key: str, cls: type) -> dict[str, object]:
    """Infer the smallest usable constructor kwargs for a dataset class."""
    import inspect

    sig = inspect.signature(cls.__init__)
    params: dict[str, object] = {}
    run_data_tests = os.environ.get("NEXUML_RUN_DATA_TESTS") == "1"
    data_root = Path(os.environ.get("NEXUML_DATA_ROOT", "/mnt/local"))
    root = data_root / data_key
    for name, param in sig.parameters.items():
        if name in ("self", "kwargs"):
            continue
        # Always wire the test data root, even when the constructor default is None.
        if name in ("root", "data_root"):
            params[name] = root
            continue
        if param.default is not inspect.Parameter.empty:
            if name == "download":
                params[name] = False
            elif name == "download_mode":
                params[name] = "disabled"
            continue
        if name in ("feature_shape",):
            params[name] = (16,)
        elif name in ("num_samples",):
            params[name] = 32
        elif name in ("path", "data_dir", "meta_dir"):
            params[name] = root
        elif name in ("download",):
            params[name] = run_data_tests
        elif name in ("download_mode",):
            params[name] = "download" if run_data_tests else "disabled"
        elif name in ("split",):
            params[name] = "train"
        else:
            params[name] = None
    return params


def _clean_stale_locks(root: Path) -> None:
    """Remove stale *.lock files under a dataset root before tests run."""
    if not root.exists():
        return
    for lock_file in root.rglob("*.lock"):
        lock_file.unlink(missing_ok=True)


@pytest.mark.conformance
def test_data_source_contract(
    data_key: str,
    dataset_registry: DatasetRegistry,
    discovered_data_source: type,
) -> None:
    """Every data source must yield (TensorDict, Optional[TensorDict]) samples."""
    params = _minimal_params_for(data_key, discovered_data_source)
    root = Path(os.environ.get("NEXUML_DATA_ROOT", "/mnt/local")) / data_key
    _clean_stale_locks(root)

    try:
        dataset = dataset_registry.instantiate(data_key, **params)
    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as exc:
        _data_source_skip_or_fail(data_key, exc)

    assert isinstance(dataset, NexuDataset)
    if len(dataset) == 0:
        if data_key not in _SKIP_ALLOWLIST:
            _data_source_fail(data_key, "has no samples")
        pytest.skip(f"{_SKIP_ALLOWLIST[data_key]}: has no samples under synthetic test conditions")

    x, y = dataset[0]
    assert isinstance(x, TensorDict)
    assert y is None or isinstance(y, TensorDict)
    assert len(x.keys()) > 0


@pytest.mark.conformance
def test_data_source_auto_batching(
    data_key: str,
    dataset_registry: DatasetRegistry,
    discovered_data_source: type,
) -> None:
    """Every data source must stack into a batched TensorDict."""
    params = _minimal_params_for(data_key, discovered_data_source)
    root = Path(os.environ.get("NEXUML_DATA_ROOT", "/mnt/local")) / data_key
    _clean_stale_locks(root)

    try:
        dataset = dataset_registry.instantiate(data_key, **params)
    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as exc:
        _data_source_skip_or_fail(data_key, exc)

    if len(dataset) < 2:
        if data_key not in _SKIP_ALLOWLIST:
            _data_source_fail(data_key, "has fewer than 2 samples")
        pytest.skip(
            f"{_SKIP_ALLOWLIST[data_key]}: has fewer than 2 samples under synthetic test conditions"
        )

    samples = [dataset[i] for i in range(2)]
    x_list = [s[0] for s in samples]
    y_list = [s[1] for s in samples]

    x_batch = torch.stack(x_list)  # ty: ignore[invalid-argument-type]
    assert isinstance(x_batch, TensorDict)
    assert x_batch.batch_size[0] == 2

    if any(y is not None for y in y_list):
        y_batch = torch.stack([y for y in y_list if y is not None])  # ty: ignore[invalid-argument-type]
        assert isinstance(y_batch, TensorDict)


def test_data_source_allowlist_is_self_auditing(dataset_registry) -> None:
    """Every entry in the data_source skip allowlist must still exist in the registry."""
    registered = set(dataset_registry.list().keys())
    stale = [key for key in _SKIP_ALLOWLIST if key not in registered]
    assert not stale, f"Stale data_source allowlist keys no longer in registry: {stale}"
