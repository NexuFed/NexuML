"""Registry-driven conformance parametrization.

Every contract test in this directory is parametrized over the corresponding
``scan_all()`` discovery results. Stable string ids keep failure reports readable.
"""

from __future__ import annotations

import pytest

from nexuml.core.discovery import scan_all


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize registry contract tests over discovered keys."""
    scanner = scan_all()

    if "layer_key" in metafunc.fixturenames:
        keys = sorted(item.key for item in scanner.by_kind("layer"))
        metafunc.parametrize("layer_key", keys, ids=keys)

    if "eval_key" in metafunc.fixturenames:
        keys = sorted(item.key for item in scanner.by_kind("eval_algorithm"))
        metafunc.parametrize("eval_key", keys, ids=keys)

    if "data_key" in metafunc.fixturenames:
        keys = sorted(item.key for item in scanner.by_kind("data_source"))
        metafunc.parametrize("data_key", keys, ids=keys)

    if "scenario_key" in metafunc.fixturenames:
        keys = sorted(item.key for item in scanner.by_kind("scenario"))
        metafunc.parametrize("scenario_key", keys, ids=keys)


@pytest.fixture
def discovered_layer(layer_key: str):
    """Lookup a discovered layer class by key."""
    from nexuml.core.registry import get_registry

    return get_registry().get(layer_key)


@pytest.fixture
def discovered_eval_algorithm(eval_key: str):
    """Lookup a discovered eval-algorithm class by key."""
    from nexuml.evaluation.registry import get_eval_registry

    return get_eval_registry().get(eval_key)


@pytest.fixture
def discovered_data_source(data_key: str):
    """Lookup a discovered dataset class by key."""
    from nexuml.data.registry import get_dataset_registry

    return get_dataset_registry().get(data_key)


@pytest.fixture
def discovered_scenario(scenario_key: str):
    """Lookup a discovered scenario function by key."""
    from nexuml.core.scenario_registry import get_scenario_registry

    return get_scenario_registry().get(scenario_key)
