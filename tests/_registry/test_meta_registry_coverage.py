"""Meta-test: scan_all() discovery results must match registry contents.

Guards against silent drift between discovery (what scan_all() finds) and the
registries actually consumed by conformance parametrization (what got
registered) — e.g. a registration error that quietly drops a key.
"""

from __future__ import annotations

from nexuml.core.discovery import scan_all
from nexuml.core.registry import get_registry
from nexuml.core.scenario_registry import get_scenario_registry
from nexuml.data.registry import get_dataset_registry
from nexuml.evaluation.registry import get_eval_registry


def test_discovered_layer_keys_match_registry():
    scanner = scan_all()
    discovered = {item.key for item in scanner.by_kind("layer")}
    registered = set(get_registry().list().keys())
    assert discovered == registered


def test_discovered_eval_algorithm_keys_match_registry():
    scanner = scan_all()
    discovered = {item.key for item in scanner.by_kind("eval_algorithm")}
    registered = set(get_eval_registry().list().keys())
    assert discovered == registered


def test_discovered_data_source_keys_match_registry():
    scanner = scan_all()
    discovered = {item.key for item in scanner.by_kind("data_source")}
    registered = set(get_dataset_registry().list().keys())
    assert discovered == registered


def test_discovered_scenario_keys_match_registry():
    scanner = scan_all()
    discovered = {item.key for item in scanner.by_kind("scenario")}
    registered = set(get_scenario_registry().list().keys())
    assert discovered == registered
