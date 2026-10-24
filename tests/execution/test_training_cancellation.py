"""Training termination semantics with a fake trainer, never a real fit."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("code,expected", [(None, 143), (143, 143), (7, 7)])
def test_sigterm_preserves_nonzero_status_and_closes_owned_collection(monkeypatch, code, expected):
    class SIGTERMException(SystemExit):
        pass

    monkeypatch.setitem(
        sys.modules,
        "lightning.pytorch.utilities.exceptions",
        SimpleNamespace(SIGTERMException=SIGTERMException),
    )
    source = Path(__file__).resolve().parents[2] / "src/nexuml/training/lightning.py"
    tree = ast.parse(source.read_text())
    session = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NexuSession")
    fit = next(n for n in session.body if isinstance(n, ast.FunctionDef) and n.name == "fit")

    class Module:
        def __init__(self):
            self.close_collection = Mock()

    module_type = Module
    module = module_type()
    namespace: dict[str, Any] = {"NexuRLLightningModule": module_type}
    exec(compile(ast.Module(body=[fit], type_ignores=[]), str(source), "exec"), namespace)
    error = SIGTERMException(code)
    owner = SimpleNamespace(
        lightning_module=module,
        trainer=SimpleNamespace(fit=Mock(side_effect=error)),
        trainer_checkpoint=None,
    )
    with pytest.raises(SIGTERMException) as caught:
        namespace["fit"](owner)
    assert caught.value is error and caught.value.code == expected
    module.close_collection.assert_called_once()
