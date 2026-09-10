"""Tests for nexuml.cli.main."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import discover_local_packages, scan_all
from nexuml.core.registry import ComponentRegistry
from nexuml.core.serialization import lower_component, restore_component

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "resolve" in result.output
    assert "export-model" in result.output
    assert "export-dataset" in result.output

    legacy_result = runner.invoke(app, ["export", "--help"])
    assert legacy_result.exit_code != 0


@pytest.mark.parametrize(
    "args",
    [
        ["resolve", "--help"],
        ["build", "--help"],
        ["train", "--help"],
        ["export-dataset", "--help"],
        ["export-model", "--help"],
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


def _write_scenario_file(path: Path) -> Path:
    path.write_text(
        "from nexuml.core.types import ScenarioSpec\n\n"
        "def scenario():\n"
        "    return ScenarioSpec(name='scenario_file_cli')\n"
    )
    return path


def test_export_cli_accepts_scenario_file_and_preserves_s3_output(tmp_path, monkeypatch):
    scenario_file = _write_scenario_file(tmp_path / "scenario.py")
    checkpoint = tmp_path / "local-last.ckpt"
    output = "s3://prisma/models/name"
    captured = {}

    lightning = importlib.import_module("nexuml.training.lightning")
    export_module = importlib.import_module("nexuml.core.export")

    class FakeSession:
        pipeline = object()
        lightning_module = object()
        trainer = object()

        @classmethod
        def from_scenario(cls, scenario):
            return cls()

        def setup(self):
            return self

    def capture_export(pipeline, path, **kwargs):
        captured["path"] = path

    monkeypatch.setattr(lightning, "NexuSession", FakeSession)
    monkeypatch.setattr(export_module, "export_package", capture_export)

    result = runner.invoke(
        app,
        [
            "export-model",
            "--scenario-file",
            str(scenario_file),
            "--checkpoint",
            str(checkpoint),
            "--output",
            output,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["path"] == output


def test_export_cli_rejects_multiple_scenario_selectors(tmp_path):
    scenario_file = _write_scenario_file(tmp_path / "scenario.py")

    result = runner.invoke(
        app,
        [
            "export-model",
            "synthetic-linear-ae-reconstruction",
            "--scenario-file",
            str(scenario_file),
        ],
    )

    assert result.exit_code != 0
    assert "Provide only one of scenario name, --config, or --scenario-file" in result.output


def test_export_dataset_cli_accepts_scenario_file(tmp_path, monkeypatch):
    scenario_file = _write_scenario_file(tmp_path / "scenario.py")
    data_export = importlib.import_module("nexuml.data.export")
    lightning = importlib.import_module("nexuml.training.lightning")
    captured = {}

    def fake_runtime(scenario):
        captured["scenario"] = scenario
        return SimpleNamespace(data_module=object(), lightning_module=object())

    def fake_export(data_module, output, **kwargs):
        captured["output"] = output
        return output

    monkeypatch.setattr(lightning, "create_runtime_artifacts", fake_runtime)
    monkeypatch.setattr(data_export, "export_data_module", fake_export)

    result = runner.invoke(
        app,
        ["export-dataset", "--scenario-file", str(scenario_file), "--output", "dataset"],
    )

    assert result.exit_code == 0, result.output
    assert captured["scenario"].name == "scenario_file_cli"
    assert captured["output"] == "dataset"


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


def test_library_add_persists_and_is_discoverable(
    isolated_library_config, minimal_local_library, monkeypatch
):
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

    fresh_registry = ComponentRegistry()
    fresh_registry.scan([minimal_local_library.package_name])
    definition_type = fresh_registry.get_type("layer", minimal_local_library.layer_key)
    definition = definition_type.model_validate({"scale": 2.0})

    monkeypatch.setattr("nexuml.core.serialization.get_component_registry", lambda: fresh_registry)
    lowered = lower_component(definition)
    restored = restore_component(kind="layer", value=lowered)
    assert isinstance(restored, LayerDefinition)
    runtime = restored.build(
        LayerBuildContext(
            input_sizes={"features": (2,)},
            keys_in=["features"],
            keys_out=["scaled"],
        )
    )

    assert runtime.scale == 2.0


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
