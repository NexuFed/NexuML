"""Real selected-runtime round-trips and transport file/discovery safety."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from nexuml.api.app import create_app
from nexuml.api.security import Settings
from nexuml.core.config import ResolvedConfig
from nexuml.core.backends import backend_rows
from nexuml.core.serialization import lower_model
from nexuml.core.types import LayerSpec, PipelineSpec, ScenarioSpec
from nexuml_library.layers.model.linear_encoder import LinearEncoder

TOKEN = "api-test-only-token-" * 4


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(Settings(tmp_path, "http://127.0.0.1:3000", TOKEN)),
        base_url="http://127.0.0.1:8000",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client


def payload():
    scenario = ScenarioSpec(
        name="round-trip",
        pipeline=PipelineSpec(
            stages={
                "10": [
                    LayerSpec(
                        component=LinearEncoder(output_dim=4),
                        keys_in={"input": "features"},
                        keys_out=["latent"],
                        meta_out={"output_dim": "width"},
                    )
                ],
                "2": [],
            }
        ),
    )
    return {"data": lower_model(scenario), "stage_order": ["10", "2"]}


def test_config_roundtrip_conflicts_and_invalid_input(client, tmp_path):
    body = payload()
    response = client.post("/api/v1/config/validate", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["stage_order"] == ["10", "2"]
    restored = ResolvedConfig.from_yaml(result["yaml"])
    assert list(restored.pipeline.stages) == ["10", "2"]
    assert restored.pipeline.stages["10"][0].keys_in == {"input": "features"}
    saved = client.post(
        "/api/v1/config/save", json={**body, "path": "config.yaml", "base_revision": None}
    )
    assert saved.status_code == 200, saved.text
    revision = saved.json()["base_revision"]
    loaded = client.post("/api/v1/config/load", json={"path": "config.yaml"}).json()
    assert loaded["data"] == result["data"]
    assert loaded["base_revision"] == revision
    original = (tmp_path / "config.yaml").read_text()
    bad = {**body["data"], "training": {"batch_size": 0}}
    invalid = client.post(
        "/api/v1/config/save", json={"data": bad, "path": "config.yaml", "base_revision": revision}
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["fields"][0]["loc"] == ["training", "batch_size"]
    assert (tmp_path / "config.yaml").read_text() == original
    (tmp_path / "config.yaml").write_text("# external edit\n" + original)
    assert (
        client.post(
            "/api/v1/config/save", json={**body, "path": "config.yaml", "base_revision": revision}
        ).status_code
        == 409
    )
    assert (
        client.post("/api/v1/config/validate", json={**body, "stage_order": ["2", "2"]}).status_code
        == 422
    )
    assert client.post("/api/v1/config/validate", json={"yaml": "[invalid yaml"}).status_code == 422
    body["data"]["pipeline"]["stages"]["10"][0]["component"]["params"]["output_dim"] = "wrong"
    typed_error = client.post("/api/v1/config/validate", json=body)
    assert typed_error.status_code == 422
    assert typed_error.json()["error"]["fields"][0]["loc"] == [
        "pipeline",
        "stages",
        "10",
        0,
        "component",
        "params",
        "output_dim",
    ]
    body["data"]["pipeline"]["stages"]["10"][0]["component"]["params"]["output_dim"] = 4
    body["data"]["pipeline"]["stages"]["10"][0]["keys_in"] = ["not_available"]
    # This would fail a dummy forward. Field checks must never compile.
    assert client.post("/api/v1/config/validate", json=body).status_code == 200


def test_paths_and_layout(client, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "secret.yaml").write_text("private")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    for path in ("../secret.yaml", str(outside / "secret.yaml"), "escape/secret.yaml"):
        assert client.post("/api/v1/config/load", json={"path": path}).status_code == 403
    layout = {
        "semantic_revision": "draft-revision",
        "layout": {
            "positions": {"stage:Encoder": {"x": 4, "y": 8}},
            "sizes": {"stage:Encoder": {"width": 800, "height": 900}},
            "ids": {"Encoder": ["node-1"]},
            "names": {"node-1": "Visual encoder"},
        },
        "path": "config.yaml",
        "base_revision": None,
    }
    saved = client.post("/api/v1/config/layout/save", json=layout)
    assert saved.status_code == 200
    loaded = client.post("/api/v1/config/layout/load", json={"path": "config.yaml"}).json()
    assert loaded["layout"] == layout["layout"]
    assert loaded["semantic_revision"] == "draft-revision"
    assert not (tmp_path / "config.yaml").exists()
    assert (
        json.loads((tmp_path / "config.yaml.studio.json").read_text())["layout"] == layout["layout"]
    )
    # Layout writes retain the existing conflict guard; no executable YAML is created.
    layout["base_revision"] = "external-change"
    assert client.post("/api/v1/config/layout/save", json=layout).status_code == 409
    assert (
        client.post("/api/v1/config/layout/load", json={"path": "config.yaml"}).json()["layout"]
        == loaded["layout"]
    )


def test_fresh_library_discovery(client, minimal_local_library):
    library = minimal_local_library
    assert (
        client.post("/api/v1/libraries", json={"path": str(library.root / "missing")}).status_code
        == 422
    )
    assert client.post("/api/v1/libraries", json={"path": str(library.root)}).status_code == 200
    (library.root / library.package_name / "broken.py").write_text(
        "raise RuntimeError('broken plugin')"
    )
    result = client.get("/api/v1/registry")
    assert result.status_code == 200, result.text
    catalog = result.json()
    component = next(item for item in catalog["components"] if item["name"] == library.layer_key)
    assert "scale" in component["schema"]["properties"]
    assert component["version"] == "1"
    assert component["import_target"].startswith(library.package_name)
    assert any(e["message"] == "broken plugin" for e in catalog["errors"])
    assert library.scenario_key in [s["name"] for s in catalog["scenarios"]]
    assert "nexuml_library" in catalog["libraries"]["packages"]
    assert [
        (e["category"], e["name"], e["import_target"]) for e in catalog["backend_descriptions"]
    ] == backend_rows()
    removed = client.request("DELETE", "/api/v1/libraries", json={"path": str(library.root)})
    assert removed.status_code == 200
    assert library.layer_key not in [
        item["name"] for item in client.get("/api/v1/registry").json()["components"]
    ]
