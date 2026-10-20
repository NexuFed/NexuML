"""Help must work in a metadata-only installation, without ML import attempts."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from importlib import import_module
from pathlib import Path

import pytest


def test_recursive_help_without_runtime_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.abc
                import importlib.metadata
                import sys

                blocked = {
                    'torch', 'lightning', 'lightning_utilities', 'tensordict',
                    'numpy', 'pandas', 'torchvision', 'torchaudio', 'torchmetrics',
                    'ray', 'kubernetes', 'nvidia', 'cupy', 'cudf', 'cuml',
                    'mlflow', 'optuna', 'boto3', 'safetensors', 'tensorboard',
                    'onnx', 'onnxscript', 'PIL', 'soundfile', 'fastapi', 'uvicorn',
                    'websockets', 'scipy', 'sklearn', 'matplotlib', 'wandb',
                }

                class NoRuntimeImports(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        assert fullname.split('.')[0] not in blocked, fullname

                sys.meta_path.insert(0, NoRuntimeImports())
                import nexuml
                import nexuml.core
                from nexuml.core import factory
                from nexuml.cli.main import app
                from typer.main import get_command
                from typer.testing import CliRunner
                from typer import Context
                from typer.core import TyperGroup
                from docs.scripts.extract_cli import extract_cli

                assert nexuml.__version__ == importlib.metadata.version('nexuml')
                entries = importlib.metadata.distribution('nexuml').entry_points
                assert any(e.name == 'nexuml' and e.value == 'nexuml.cli.main:app'
                           for e in entries)
                for module in (nexuml, nexuml.core):
                    assert set(module.__all__) <= set(dir(module))
                    for name in ('callback', 'optimizer', 'scheduler', 'strategy', 'writer'):
                        assert getattr(module, name) is getattr(factory, name)
                    try:
                        getattr(module, 'not_an_export')
                    except AttributeError as error:
                        assert str(error) == (
                            f"module {module.__name__!r} has no attribute 'not_an_export'"
                        )
                    else:
                        raise AssertionError('Unknown attribute did not raise')

                runner = CliRunner()
                paths = []
                def visit(command, path):
                    paths.append(' '.join(['nexuml', *path]))
                    result = runner.invoke(app, [*path, '--help'], terminal_width=240)
                    assert result.exit_code == 0, (path, result.output, result.exception)
                    if isinstance(command, TyperGroup):
                        with Context(command) as ctx:
                            for name in command.list_commands(ctx):
                                child = command.get_command(ctx, name)
                                if not child.hidden:
                                    visit(child, [*path, name])

                visit(get_command(app), [])
                assert [command['path'] for command in extract_cli()] == paths
                assert not blocked.intersection(sys.modules)
                """
            ),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_export_identity():
    pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    import nexuml
    import nexuml.core

    for module in (nexuml, nexuml.core):
        for name in module.__all__:
            if name in {"PostTrainFitLayer", "PostTrainLayerNotFittedError"}:
                implementation = import_module("nexuml.core.post_train_layer")
            elif name in {"NnModuleLayer", "TorchModuleAdapter", "nn_module"}:
                implementation = import_module("nexuml.core.torch_adapter")
            else:
                implementation = import_module("nexuml.core.factory")
            assert getattr(module, name) is getattr(implementation, name)
            assert module.__dict__[name] is getattr(implementation, name)


def test_runtime_warning_suppression(monkeypatch):
    rank_zero = pytest.importorskip("lightning_utilities.core.rank_zero")
    from nexuml.cli import main
    from typer.testing import CliRunner

    calls = []
    monkeypatch.setattr(rank_zero, "rank_zero_info", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(main, "_runtime_warnings_configured", False)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = CliRunner().invoke(main.app, ["train"])
        assert result.exit_code == 1
        assert "Provide a scenario/config/scenario-file" in result.output
        warnings.warn("`isinstance(treespec, LeafSpec)` is deprecated", FutureWarning)
        warnings.warn("another warning", FutureWarning)
        assert [str(item.message) for item in recorded] == ["another warning"]

    filtered = rank_zero.rank_zero_info
    main._configure_runtime_warnings()
    assert rank_zero.rank_zero_info is filtered
    rank_zero.rank_zero_info("try litlogger")
    rank_zero.rank_zero_info("try litmodels")
    rank_zero.rank_zero_info("normal message", stacklevel=2)
    rank_zero.rank_zero_info(42)
    assert calls == [(("normal message",), {"stacklevel": 2}), ((42,), {})]
