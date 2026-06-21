"""Pure-logic tests for nexuml.cli.overrides."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexuml.cli.overrides import _coerce_value, apply_overrides


def test_coerce_value_parses_int():
    assert _coerce_value("42") == 42


def test_coerce_value_parses_float():
    assert _coerce_value("3.14") == 3.14


def test_coerce_value_parses_bool():
    assert _coerce_value("true") is True
    assert _coerce_value("False") is False


def test_coerce_value_parses_json_list():
    assert _coerce_value("[1, 2, 3]") == [1, 2, 3]


def test_coerce_value_parses_json_dict():
    assert _coerce_value('{"a": 1}') == {"a": 1}


def test_coerce_value_falls_back_to_string():
    assert _coerce_value("hello") == "hello"


def test_apply_overrides_on_dict(capsys):
    scenario = {"training": {"max_epochs": 1, "lr": 0.01}}
    apply_overrides(scenario, ["training.max_epochs=10", "training.lr=0.001"])
    assert scenario["training"]["max_epochs"] == 10
    assert scenario["training"]["lr"] == 0.001


def test_apply_overrides_on_object(capsys):
    scenario = SimpleNamespace(name="test", training=SimpleNamespace(lr=0.01))
    apply_overrides(scenario, ["training.lr=0.005"])
    assert scenario.training.lr == 0.005


def test_apply_overrides_invalid_format():
    with pytest.raises(ValueError, match="Invalid override"):
        apply_overrides({}, ["no_equals"])


def test_apply_overrides_empty_segment():
    with pytest.raises(ValueError, match="empty segment"):
        apply_overrides({}, ["..foo=1"])


def test_apply_overrides_missing_path():
    with pytest.raises(KeyError, match="Override path not found"):
        apply_overrides({}, ["missing.key=1"])


def test_apply_overrides_no_overrides_returns_input():
    scenario = {"a": 1}
    assert apply_overrides(scenario, []) is scenario
