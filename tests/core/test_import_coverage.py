"""Smoke-import tests for modules that are otherwise hard to exercise.

The goal is to ensure the full package imports cleanly and to raise statement
coverage for modules that need real optional resources (DALI, optuna, data
roots, etc.) to run meaningfully.
"""

import importlib
import importlib.util

import pytest


OPTIONAL_MODULES = [
    "nexuml.cli.overrides",
    "nexuml.core.provenance",
    "nexuml.core.schema",
    "nexuml.core.scenario_loader",
    "nexuml.core.storage",
    "nexuml.core.torch_adapter",
    "nexuml.evaluation.algorithm",
    "nexuml.evaluation.utils",
    "nexuml.tracking.logger",
]


@pytest.mark.parametrize("name", OPTIONAL_MODULES)
def test_module_imports(name: str) -> None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        pytest.skip(f"{name} is not installed/available")
    # Importing executes top-level definitions and is sufficient to surface
    # obvious breakage in modules that are otherwise hard to exercise.
    importlib.import_module(name)
