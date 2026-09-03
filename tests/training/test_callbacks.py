"""Pure-logic tests for NexuML callback configuration."""

from __future__ import annotations

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from nexuml_library.scenarios.training.defaults import default_callbacks

from nexuml import callback
from nexuml.core.types import CallbackSpec
from nexuml.training.callbacks import _resolve_callback_path_params, build_callbacks


def test_callback_helper_captures_importable_factory() -> None:
    spec = callback(ModelCheckpoint, monitor="val/loss", save_top_k=1)

    assert spec.factory == "lightning.pytorch.callbacks.model_checkpoint:ModelCheckpoint"
    assert spec.kwargs == {"monitor": "val/loss", "save_top_k": 1}


def test_resolve_callback_path_params(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUML_LOGS_ROOT", str(tmp_path))
    params = {"dirpath": "checkpoints", "monitor": "val_loss"}
    resolved = _resolve_callback_path_params(params)
    assert resolved["monitor"] == "val_loss"
    assert str(resolved["dirpath"]).startswith(str(tmp_path))


def test_build_callbacks_from_typed_factory():
    callbacks = build_callbacks([callback(LearningRateMonitor)])

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], LearningRateMonitor)


def test_build_callbacks_invalid_factory_logs_warning(caplog):
    specs = [CallbackSpec(factory="not.a.real:Class")]

    assert build_callbacks(specs) == []
    assert "Could not build callback" in caplog.text


def test_build_callbacks_empty_list():
    assert build_callbacks([]) == []


def test_default_checkpoint_callback_defers_path_to_lightning():
    spec = next(spec for spec in default_callbacks() if spec.factory.endswith(":ModelCheckpoint"))
    assert "dirpath" not in spec.kwargs

    built = build_callbacks([spec])[0]
    assert isinstance(built, ModelCheckpoint)
    assert built.dirpath is None
