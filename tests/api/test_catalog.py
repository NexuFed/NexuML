"""Catalog projections share one execution discovery result."""

from unittest.mock import Mock

from nexuml.api.worker import catalog


def test_execution_catalog_is_evaluated_once(monkeypatch):
    rows = [
        {
            "type": "local",
            "version": "1",
            "available": True,
            "capabilities": {"metrics": True, "resume": True},
        },
        {
            "type": "remote",
            "version": "1",
            "available": False,
            "capabilities": {"metrics": False, "resume": False},
        },
    ]
    execution_catalog = Mock(return_value=rows)
    monkeypatch.setattr("nexuml.execution.catalog", execution_catalog)

    result = catalog()

    execution_catalog.assert_called_once_with()
    assert set(result) == {
        "libraries",
        "components",
        "scenarios",
        "errors",
        "schema",
        "backend_descriptions",
        "execution_backends",
        "backends",
    }
    assert result["execution_backends"] == rows
    assert result["backends"] == {
        "local": {"available": True, "metrics": True, "resume": True},
        "remote": {"available": False, "metrics": False, "resume": False},
    }
