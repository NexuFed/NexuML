"""Pure-logic tests for nexuml.training.callbacks."""

from __future__ import annotations

from types import SimpleNamespace

from nexuml.training.callbacks import (
    _resolve_callback_path_params,
    build_callbacks,
    get_callback_path,
    list_callbacks,
    register_callback,
)


def test_get_callback_path_known_alias():
    assert get_callback_path("checkpoint") == "lightning.pytorch.callbacks.ModelCheckpoint"


def test_get_callback_path_missing_alias():
    assert get_callback_path("not_registered") is None


def test_list_callbacks_returns_copy():
    registry = list_callbacks()
    assert isinstance(registry, dict)
    assert "checkpoint" in registry
    # Mutating the returned dict must not affect the registry.
    registry["new"] = "value"
    assert get_callback_path("new") is None


def test_register_callback():
    register_callback("my_callback", "some.module.Class")
    assert get_callback_path("my_callback") == "some.module.Class"


def test_resolve_callback_path_params(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUML_LOGS_ROOT", str(tmp_path))
    params = {"dirpath": "checkpoints", "monitor": "val_loss"}
    resolved = _resolve_callback_path_params(params)
    assert resolved["monitor"] == "val_loss"
    assert str(resolved["dirpath"]).startswith(str(tmp_path))


def test_build_callbacks_with_lightning_alias():
    specs = [SimpleNamespace(type="lr_monitor", params={})]
    callbacks = build_callbacks(specs)
    assert len(callbacks) == 1
    assert type(callbacks[0]).__name__ == "LearningRateMonitor"


def test_build_callbacks_unknown_type_logs_warning(caplog):
    specs = [SimpleNamespace(type="not.a.real.Class", params={})]
    callbacks = build_callbacks(specs)
    assert callbacks == []


def test_build_callbacks_empty_list():
    assert build_callbacks([]) == []
