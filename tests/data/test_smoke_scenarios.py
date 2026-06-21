"""Opt-in smoke tests against real data / GPU."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from nexuml.cli.main import app

runner = CliRunner()


@pytest.mark.requires_data
@pytest.mark.requires_gpu
def test_smoke_synthetic_scenario():
    result = runner.invoke(
        app,
        ["smoke", "synthetic-linear-ae-reconstruction", "--max-epochs", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "Smoke test PASSED" in result.output
