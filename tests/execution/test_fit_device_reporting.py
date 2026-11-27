"""Capture fit-device metadata using fake hook owners; never construct a model."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("kind", ["supervised", "reinforcement"])
def test_fit_hook_captures_device_before_teardown_without_changing_state(kind, device, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    name = "lightning.py" if kind == "supervised" else "reinforcement.py"
    class_name = "NexuLightningModule" if kind == "supervised" else "NexuRLLightningModule"
    source = root / "src/nexuml/training" / name
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    hook = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "on_fit_start"
    )
    namespace = {}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(source), "exec"), namespace)
    owner = SimpleNamespace(device=device, fit_device=None)
    validate = Mock()
    if kind == "supervised":
        owner.pipeline = SimpleNamespace(call_layer_hook=Mock())
    else:
        from nexuml.core.types import RLEvaluationSpec
        from nexuml.tracking import logger

        monkeypatch.setattr(logger, "preflight_evaluation_media", Mock())
        owner.scenario = SimpleNamespace(
            interaction=SimpleNamespace(environment=SimpleNamespace(validate_training=validate))
        )
        owner.training_spec = SimpleNamespace(model_copy=lambda **kwargs: kwargs)
        owner.evaluation_spec = RLEvaluationSpec(episodes=0)
        owner._resume_world_size = None
        owner._rank_guard = lambda _phase: nullcontext()
        owner._fit_training = lambda: owner.training_spec
        owner.trainer = SimpleNamespace(loggers=[], world_size=1, is_global_zero=True)
    namespace["on_fit_start"](owner)
    owner.device = "cpu"  # Simulated Lightning teardown, not the recorded fit placement.
    assert owner.fit_device == device
    if kind == "supervised":
        owner.pipeline.call_layer_hook.assert_called_once_with("on_fit_start")
    else:
        assert validate.call_args.kwargs["device"] == device
