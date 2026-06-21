"""Registry-driven contract tests for every discovered scenario."""

from __future__ import annotations

from typing import Callable

import pytest

from nexuml.core.compiler import compile
from nexuml.core.registry import get_registry
from nexuml.core.types import ScenarioSpec

# Scenarios known to be untestable with this generic contract.
# Any other discovered scenario that raises fails its parameter case instead of skipping.
_SKIP_ALLOWLIST: dict[str, str] = {
    "audioset-conv-ae-clshead": (
        "ReconstructionLoss input shape mismatch with default audio sample length "
        "(requires match_min_length=True or a matching sample length to compile)"
    ),
}


def _scenario_skip_or_fail(key: str, exc: Exception) -> None:
    """Skip allowlisted scenarios; fail others with rich, actionable context."""
    if key not in _SKIP_ALLOWLIST:
        raise AssertionError(
            f"Conformance failure for scenario {key!r}: "
            f"{type(exc).__name__}: {exc}\n"
            f"Hint: add {key!r} to the scenario skip allowlist only if the failure "
            f"requires a dependency or real data that synthetic fixtures cannot provide."
        ) from exc
    pytest.skip(f"{_SKIP_ALLOWLIST[key]}: {exc}")


@pytest.mark.conformance
def test_scenario_builds_spec(
    scenario_key: str,
    discovered_scenario: Callable[..., ScenarioSpec],
) -> None:
    """Every scenario function must build a valid ScenarioSpec with default args."""
    try:
        scenario = discovered_scenario()
    except TypeError as exc:
        _scenario_skip_or_fail(scenario_key, exc)

    assert isinstance(scenario, ScenarioSpec)
    assert scenario.name
    assert scenario.pipeline.stages


@pytest.mark.conformance
def test_scenario_compiles(
    scenario_key: str,
    discovered_scenario: Callable[..., ScenarioSpec],
) -> None:
    """Every scenario spec must pass compiler key-contract validation."""
    try:
        scenario = discovered_scenario()
    except TypeError as exc:
        _scenario_skip_or_fail(scenario_key, exc)

    registry = get_registry()
    try:
        compile(scenario, registry)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        _scenario_skip_or_fail(scenario_key, exc)


def test_scenario_allowlist_is_self_auditing(scenario_registry) -> None:
    """Every entry in the scenario skip allowlist must still exist in the registry."""
    registered = set(scenario_registry.list().keys())
    stale = [key for key in _SKIP_ALLOWLIST if key not in registered]
    assert not stale, f"Stale scenario allowlist keys no longer in registry: {stale}"
