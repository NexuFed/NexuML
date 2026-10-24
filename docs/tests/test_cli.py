"""Contract checks against real help, not a separately defined documentation CLI."""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path

import griffe
import pytest
import typer
from rich.text import Text
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from docs.scripts.extract_cli import extract_cli
from nexuml.cli.main import app

ROOT = Path(__file__).resolve().parents[2]


class FixtureMode(str, Enum):
    """Representative enum-valued option."""

    fast = "fast"
    safe = "safe"


def _unsafe_default():
    raise AssertionError("Extraction or help evaluated a dynamic default")


def _text(value):
    return re.sub(r"[\s│┃]+", "", Text.from_ansi(value).plain)


def _assert_matches_help():
    extracted = extract_cli()
    by_path = {command["path"]: command for command in extracted}
    visited = []
    runner = CliRunner()

    def visit(command, names, parent=None):
        if command.hidden:
            return
        with typer.Context(
            command, info_name=names[-1], parent=parent, **command.context_settings
        ) as ctx:
            path = " ".join(names)
            visited.append(path)
            actual = by_path[path]
            assert actual["help"] == inspect.cleandoc(command.help or "")
            result = runner.invoke(
                app, [*names[1:], "--help"], terminal_width=3000, env={"COLUMNS": "3000"}
            )
            assert result.exit_code == 0, (path, result.output, result.exception)
            text = _text(result.output)
            assert _text(Text.from_markup(actual["help"]).plain) in text

            parameters = [p for p in command.get_params(ctx) if not getattr(p, "hidden", False)]
            assert [p["name"] for p in actual["parameters"]] == [p.name for p in parameters]
            for metadata, parameter in zip(actual["parameters"], parameters, strict=True):
                assert metadata["kind"] == parameter.param_type_name
                for field in ("opts", "secondary_opts", "required", "multiple", "nargs"):
                    assert metadata[field] == getattr(parameter, field)
                assert metadata["type"] == (
                    parameter.type.get_metavar(param=parameter, ctx=ctx) or parameter.type.name
                )
                assert metadata["help"] == inspect.cleandoc(getattr(parameter, "help", None) or "")
                assert _text(Text.from_markup(metadata["help"]).plain) in text
                if metadata["kind"] == "option":
                    for option in metadata["opts"] + metadata["secondary_opts"]:
                        assert option in text, (path, option, result.output)
                else:
                    assert parameter.name.upper() in text.upper()
            assert any("--help" in p["opts"] for p in actual["parameters"])

            if isinstance(command, TyperGroup):
                for name in command.list_commands(ctx):
                    child = command.get_command(ctx, name)
                    if child is not None:
                        if not child.hidden:
                            assert name in result.output
                        visit(child, [*names, name], ctx)

    visit(get_command(app), ["nexuml"])
    assert list(by_path) == visited
    assert len(by_path) == len(extracted)
    return by_path


@pytest.mark.parametrize("force_color", [None, "1"])
def test_cli_matches_recursive_help(monkeypatch, force_color):
    """Every public path, option form, argument, and description matches real help."""
    if force_color is None:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
    else:
        monkeypatch.setenv("FORCE_COLOR", force_color)
    commands = _assert_matches_help()
    export = {p["name"]: p for p in commands["nexuml export-dataset"]["parameters"]}
    assert export["preprocess"]["secondary_opts"] == ["--no-preprocess"]
    assert export["include_labels"]["default"] is True
    assert export["split"]["multiple"] is True
    assert export["samples_per_shard"]["default"] == 256
    train = {p["name"]: p for p in commands["nexuml train"]["parameters"]}
    assert train["override"]["opts"] == ["--override", "-O"]
    assert train["override"]["multiple"] is True
    assert train["scenario_name"]["required"] is False
    assert train["scenario_name"]["default"] is None


def test_new_nested_commands_aliases_and_defaults(monkeypatch):
    """Registration automatically appears, with deterministic non-executed defaults."""
    monkeypatch.setattr(app, "registered_groups", list(app.registered_groups))
    fixture = typer.Typer(help="Fixture <text> & {not MDX}")
    nested = typer.Typer(help="Nested fixture")
    fixture.add_typer(nested, name="nested")
    app.add_typer(fixture, name="docs-fixture")

    @nested.command("sample")
    def sample(
        value: str = typer.Argument(help="Required fixture argument"),
        mode: FixtureMode = typer.Option(FixtureMode.fast, "--mode", help="Enum fixture"),
        output: Path = typer.Option(Path("a <b> & {c}"), "--output", "-o"),
        tag: list[str] = typer.Option(["one", "two"], "--tag", "-t", help="Repeat fixture"),
        pair: tuple[str, int] = typer.Option(("one", 2), "--pair"),
        enabled: bool = typer.Option(True, "--enabled/--disabled"),
        dynamic: str = typer.Option(_unsafe_default, "--dynamic"),
        secret: str = typer.Option("hidden", "--secret", hidden=True),
    ):
        """Fixture command with <unsafe-looking> & {literal} help text."""
        assert False, "Extraction ran a command callback"

    nested.command("alias")(sample)

    @nested.command("hidden-command", hidden=True)
    def hidden():
        raise AssertionError("Extraction ran a hidden callback")

    hidden_group = typer.Typer()
    hidden_group.command("child")(hidden)
    fixture.add_typer(hidden_group, name="hidden-group", hidden=True)

    commands = _assert_matches_help()
    assert "nexuml docs-fixture nested alias" in commands
    assert not any("hidden-" in path for path in commands)
    assert commands["nexuml docs-fixture"]["help"] == "Fixture <text> & {not MDX}"
    parameters = {p["name"]: p for p in commands["nexuml docs-fixture nested sample"]["parameters"]}
    assert "secret" not in parameters
    assert parameters["value"]["required"] is True
    assert parameters["mode"]["default"] == "fast"
    assert parameters["mode"]["type"] == "<fast|safe>"
    assert parameters["output"]["default"] == "a <b> & {c}"
    assert parameters["output"]["opts"] == ["--output", "-o"]
    assert parameters["tag"]["default"] == ["one", "two"]
    assert parameters["tag"]["multiple"] is True
    assert parameters["pair"]["default"] == ["one", 2]
    assert parameters["pair"]["nargs"] == 2
    assert parameters["enabled"]["secondary_opts"] == ["--disabled"]
    assert parameters["dynamic"]["default"].endswith("._unsafe_default>")
    assert extract_cli() == extract_cli()
    json.dumps(list(commands.values()), allow_nan=False)


def test_extraction_cli_requires_output_and_is_repeatable(tmp_path):
    """The script writes only explicitly selected JSON, reproducibly."""
    command = [sys.executable, str(ROOT / "docs/scripts/extract_cli.py")]
    missing = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, timeout=30)
    assert missing.returncode == 2
    assert "--output" in missing.stderr
    output = tmp_path / "generated" / "cli.json"
    for _ in range(2):
        result = subprocess.run(
            [*command, "--output", str(output)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(output.read_text(encoding="utf-8")) == extract_cli()
        contents = output.read_bytes()
        if _ == 0:
            first = contents
        else:
            assert contents == first


def test_lazy_exports_stay_visible_to_static_griffe():
    """TYPE_CHECKING imports retain resolvable public aliases without inspection."""
    loader = griffe.GriffeLoader(search_paths=[ROOT / "src"], allow_inspection=False)
    root = loader.load("nexuml")
    unresolved, _ = loader.resolve_aliases(external=False)
    assert not unresolved
    exports = {"callback", "optimizer", "scheduler", "strategy", "writer"}
    adapters = {"NnModuleLayer", "nn_module"}
    post_train = {"PostTrainFitLayer", "PostTrainLayerNotFittedError"}
    assert root.exports is not None
    assert set(root.exports) == exports | adapters
    core = root.members["core"]
    assert core.exports is not None
    assert set(core.exports) == exports | adapters | post_train | {"TorchModuleAdapter"}
    for module in (root, core):
        assert module.exports is not None
        for name in module.exports:
            assert isinstance(name, str)
            alias = module.members[name]
            assert isinstance(alias, griffe.Alias)
            assert alias.is_alias and alias.resolved
            assert alias.is_exported and alias.is_public
            if name in exports:
                source = "factory"
            elif name in post_train:
                source = "post_train_layer"
            else:
                source = "torch_adapter"
            assert alias.target_path == f"nexuml.core.{source}.{name}"
