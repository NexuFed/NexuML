"""Scenario conformance parametrization."""

from __future__ import annotations

import pytest

from nexuml.core.discovery import Scanner


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize registry contract tests over discovered keys."""
    scanner = Scanner()
    scanner.scan_package("nexuml_library")

    if "scenario_key" in metafunc.fixturenames:
        keys = sorted(item.key for item in scanner.by_kind("scenario"))
        metafunc.parametrize("scenario_key", keys, ids=keys)


@pytest.fixture
def discovered_scenario(scenario_key: str):
    """Lookup a discovered scenario function by key."""
    from nexuml.core.scenario_registry import get_scenario_registry

    return get_scenario_registry().get(scenario_key)
