"""Tests for nexuml.cli.main."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.discovery import discover_local_packages, scan_all
from nexuml.core.registry import LayerRegistry

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "resolve" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["resolve", "--help"],
        ["build", "--help"],
        ["train", "--help"],
        ["export-dataset", "--help"],
        ["export", "--help"],
        ["smoke", "--help"],
        ["tune", "--help"],
        ["registry", "list", "--help"],
        ["backend", "list", "--help"],
        ["library", "--help"],
    ],
)
def test_subcommand_help(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output


def test_registry_list_layers():
    result = runner.invoke(app, ["registry", "list", "layers"])
    assert result.exit_code == 0
    assert "LinearEncoder" in result.output


def test_backend_list():
    result = runner.invoke(app, ["backend", "list"])
    assert result.exit_code == 0
    assert "data-export" in result.output


def test_resolve_scenario(tmp_path):
    Path("configs").mkdir(exist_ok=True)
    result = runner.invoke(app, ["resolve", "synthetic-linear-ae-reconstruction"])
    assert result.exit_code == 0, result.output
    assert "Resolved config saved" in result.output


def test_library_list():
    result = runner.invoke(app, ["library", "list"])
    assert result.exit_code == 0


def test_library_add_persists_and_is_discoverable(isolated_library_config, minimal_local_library):
    root = minimal_local_library.root

    result = runner.invoke(app, ["library", "add", str(root)])
    assert result.exit_code == 0, result.output

    config_data = json.loads(isolated_library_config.read_text())
    assert str(root.resolve()) in config_data["roots"]

    packages = discover_local_packages()
    assert minimal_local_library.package_name in packages

    scanner = scan_all()
    assert any(item.key == minimal_local_library.layer_key for item in scanner.by_kind("layer"))
    assert any(
        item.key == minimal_local_library.scenario_key for item in scanner.by_kind("scenario")
    )
    assert any(
        item.key == minimal_local_library.dataset_key for item in scanner.by_kind("data_source")
    )
    assert any(
        item.key == minimal_local_library.eval_key for item in scanner.by_kind("eval_algorithm")
    )

    fresh_registry = LayerRegistry()
    fresh_registry.scan()
    assert minimal_local_library.layer_key in fresh_registry.list()


def test_library_delete_removes_root_and_registry_keys(
    isolated_library_config, minimal_local_library
):
    root = minimal_local_library.root
    add_result = runner.invoke(app, ["library", "add", str(root)])
    assert add_result.exit_code == 0, add_result.output

    delete_result = runner.invoke(app, ["library", "delete", str(root)])
    assert delete_result.exit_code == 0, delete_result.output

    config_data = json.loads(isolated_library_config.read_text())
    assert str(root.resolve()) not in config_data["roots"]

    packages = discover_local_packages()
    assert minimal_local_library.package_name not in packages


def test_library_delete_unconfigured_root_preserves_config(isolated_library_config, tmp_path):
    other_root = tmp_path / "other_root"
    other_root.mkdir()
    runner.invoke(app, ["library", "add", str(other_root)])
    before = isolated_library_config.read_text()

    result = runner.invoke(app, ["library", "delete", str(tmp_path / "not_configured")])
    assert result.exit_code != 0

    after = isolated_library_config.read_text()
    assert before == after


def test_library_add_missing_root_fails_clearly(isolated_library_config, tmp_path):
    missing_root = tmp_path / "does_not_exist"

    result = runner.invoke(app, ["library", "add", str(missing_root)])

    assert result.exit_code != 0
    assert not isolated_library_config.exists()
