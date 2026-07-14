"""Tests for nexuml.core.export and data/export/runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.export import (
    export_onnx,
    export_package,
    export_safetensors,
    infer,
    load_package,
)
from nexuml.core.registry import LayerRegistry, get_registry
from nexuml.core.types import LayerSpec, PipelineSpec, ScenarioSpec, TrainingSpec
from nexuml.data.export.runner import export_data_module
from nexuml.training.lightning import create_data_module_from_spec
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data


def _make_simple_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="test_export",
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        type_key="LinearEncoder",
                        keys_in=["features"],
                        keys_out=["latent"],
                        params={"hidden_dims": [8], "output_dim": 4},
                    ),
                ],
                "decode": [
                    LayerSpec(
                        type_key="LinearEncoder",
                        keys_in=["latent"],
                        keys_out=["reconstructed"],
                        params={"hidden_dims": [8], "output_dim": 16},
                    ),
                ],
            }
        ),
        training=TrainingSpec(max_epochs=1, batch_size=4, loss_keys={"reconstruction_loss": 1.0}),
        data=synthetic_vector_data(feature_shape=(16,), num_samples=32),
    )


@pytest.fixture
def compiled_pipeline(tmp_path):
    scenario = _make_simple_scenario()
    return compile(scenario, get_registry())


def test_export_package_and_reload(compiled_pipeline, tmp_path):
    export_dir = tmp_path / "exported"
    export_package(compiled_pipeline, export_dir)
    assert (export_dir / "pipeline.package").exists()

    loaded, config, metadata = load_package(export_dir, get_registry())
    assert loaded is not None
    assert metadata.get("schema_version") == 2


def test_export_safetensors(compiled_pipeline, tmp_path):
    path = tmp_path / "model.safetensors"
    export_safetensors(compiled_pipeline, path)
    assert path.exists()


@pytest.mark.requires_optional("onnxscript")
def test_export_onnx(compiled_pipeline, tmp_path):
    path = tmp_path / "model.onnx"
    export_onnx(
        compiled_pipeline,
        path,
        input_key="features",
        output_key="reconstructed",
    )
    assert path.exists()


def test_infer(compiled_pipeline):
    x = TensorDict({"features": torch.randn(1, 16)}, batch_size=[1])
    x_out = infer(compiled_pipeline, x)
    assert "reconstructed" in x_out.keys()


def test_export_data_module_numpy(tmp_path):
    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="numpy",
        splits=["train"],
    )
    assert (export_dir / "config.yaml").exists()


def _expected_first_train_sample(data_module):
    data_module.setup()
    return data_module._train_ds[0]


def _assert_exported_labels_match(
    sample: dict[str, torch.Tensor],
    expected_y: TensorDict | None,
    label_prefix: str = "label__",
) -> None:
    """Assert every expected label is present in the exported sample under the prefixed key.

    Derives the exported label key from ``expected_y`` rather than hardcoding a
    specific target name, so the roundtrip assertion stays valid if the synthetic
    dataset's label set changes.
    """
    if expected_y is None:
        return
    for label_key in expected_y.keys():
        exported_key = f"{label_prefix}{label_key}"
        assert exported_key in sample, f"missing exported label key {exported_key!r}"
        expected = expected_y[label_key]
        assert isinstance(expected, torch.Tensor)
        assert torch.equal(sample[exported_key], expected)


def test_export_data_module_numpy_mmap_roundtrip(tmp_path):
    from nexuml.data.export.numpy_mmap import NumpyMmapBackend

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="numpy_mmap",
        splits=["train"],
    )

    sample = NumpyMmapBackend.load_sample(export_dir, 0)
    assert torch.equal(sample["features"], expected_x["features"])
    _assert_exported_labels_match(sample, expected_y)


def test_export_data_module_torch_roundtrip(tmp_path):
    from nexuml.data.export.torch_files import TorchBackend

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="torch",
        splits=["train"],
    )

    sample = TorchBackend.load_sample(export_dir, 0)
    assert torch.equal(sample["features"], expected_x["features"])
    _assert_exported_labels_match(sample, expected_y)


def test_export_data_module_tensordict_memmap_roundtrip(tmp_path):
    from nexuml.data.export.tensordict_memmap import TensorDictMemmapBackend

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="tensordict_memmap",
        splits=["train"],
    )

    sample = TensorDictMemmapBackend.load_sample(export_dir, 0)
    assert torch.equal(sample["features"], expected_x["features"])
    _assert_exported_labels_match(sample, expected_y)


def test_export_data_module_webdataset_roundtrip(tmp_path):
    from nexuml.data.export.webdataset import WebDatasetBackend

    scenario = _make_simple_scenario()
    data_module = create_data_module_from_spec(scenario)
    expected_x, expected_y = _expected_first_train_sample(data_module)

    export_dir = export_data_module(
        data_module,
        tmp_path / "exported_data",
        backend="webdataset",
        splits=["train"],
    )

    sample = WebDatasetBackend.load_sample(export_dir, 0)
    # The label key is stored losslessly (npy encoding); the "features" key goes
    # through lossy audio (wav) encoding since the dataset's default modality is
    # "audio", so only shape/finiteness is asserted for it.
    assert sample["features"].numel() == expected_x["features"].numel()
    assert torch.isfinite(sample["features"]).all()
    _assert_exported_labels_match(sample, expected_y)


# ---------------------------------------------------------------------------
# Clean-environment subprocess helpers
# ---------------------------------------------------------------------------


def _repo_pythonpath_roots() -> set[str]:
    workspace = Path(__file__).resolve().parents[2]
    return {
        str((workspace / "src").resolve()),
        str((workspace / "library" / "src").resolve()),
    }


def _clean_pythonpath() -> str:
    roots = _repo_pythonpath_roots()
    paths = [
        p
        for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if p and Path(p).resolve() not in roots
    ]
    return os.pathsep.join(paths)


def _run_in_clean_subprocess(script_code: str, cwd: Path, env_extra: dict | None = None):
    """Run a Python script in a subprocess without the NexuML repo on PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _clean_pythonpath()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_extra:
        env.update(env_extra)
    script_path = cwd / "_clean_load.py"
    script_path.write_text(script_code)
    return subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
    )


def _diagnose_package(package_path: Path) -> str:
    """Return the package file structure for debugging load failures."""
    from torch.package.package_importer import PackageImporter

    importer = PackageImporter(str(package_path))
    return str(importer.file_structure())


# ---------------------------------------------------------------------------
# Self-contained package policy tests
# ---------------------------------------------------------------------------


def test_export_package_external_dependency_manifest(compiled_pipeline, tmp_path):
    export_dir = tmp_path / "exported"
    export_package(compiled_pipeline, export_dir)

    assert (export_dir / "pipeline.package").exists()
    assert (export_dir / "requirements.txt").exists()

    metadata = json.loads((export_dir / "metadata.json").read_text())
    deps = metadata.get("external_dependencies", [])
    assert deps, "expected at least one external dependency entry"
    dists = {d["distribution"] for d in deps}
    assert "torch" in dists
    assert "nexuml" not in dists
    assert "nexuml_library" not in dists

    requirements = (export_dir / "requirements.txt").read_text()
    assert "torch" in requirements
    assert "nexuml" not in requirements
    assert "nexuml_library" not in requirements


def test_export_package_checkpoint_metadata_and_sidecar(compiled_pipeline, tmp_path):
    checkpoint_path = tmp_path / "fake.ckpt"
    torch.save(
        {
            "epoch": 5,
            "global_step": 42,
            "state_dict": {},
            "optimizer_states": [{}],
            "lr_schedulers": [{}],
            "callbacks": {
                "ModelCheckpoint": {
                    "best_model_path": str(tmp_path / "best.ckpt"),
                    "best_model_score": 0.123,
                    "monitor": "val/loss",
                    "mode": "min",
                }
            },
            "hyper_parameters": {"scenario": {"name": "test_export"}},
        },
        checkpoint_path,
    )

    export_dir = tmp_path / "exported"
    export_package(compiled_pipeline, export_dir, checkpoint_path=checkpoint_path)

    assert (export_dir / "lightning.ckpt").exists()
    metadata = json.loads((export_dir / "metadata.json").read_text())
    checkpoint_meta = metadata.get("checkpoint", {})
    assert checkpoint_meta.get("epoch") == 5
    assert checkpoint_meta.get("global_step") == 42
    assert checkpoint_meta.get("best_model_score") == 0.123
    assert checkpoint_meta.get("monitor") == "val/loss"


def test_custom_layer_package_export_and_clean_load(tmp_path):
    """A layer defined outside nexuml/nexuml_library must be packaged and loadable."""
    pkg_dir = tmp_path / "custom_test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "layers.py").write_text(
        "import torch\n"
        "from tensordict import TensorDict\n"
        "from nexuml.core.base_layer import PipelineLayer\n\n"
        "class CustomLinear(PipelineLayer):\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.linear = torch.nn.Linear(16, 4)\n"
        "    def forward_tensor(self, x, y=None):\n"
        "        return self.linear(x)\n"
    )

    sys.path.insert(0, str(tmp_path))
    try:
        from custom_test_pkg.layers import CustomLinear  # ty: ignore[unresolved-import]

        registry = LayerRegistry()
        registry.register("CustomLinear", CustomLinear)

        scenario = ScenarioSpec(
            name="test_custom_layer",
            pipeline=PipelineSpec(
                stages={
                    "encode": [
                        LayerSpec(
                            type_key="CustomLinear",
                            keys_in=["features"],
                            keys_out=["latent"],
                            params={},
                        ),
                    ]
                }
            ),
            training=TrainingSpec(max_epochs=1, batch_size=4, loss_keys={"latent_sum": 1.0}),
            data=synthetic_vector_data(feature_shape=(16,), num_samples=32),
        )
        pipeline = compile(scenario, registry)

        export_dir = tmp_path / "exported"
        export_package(pipeline, export_dir, include_modules=["custom_test_pkg.**"])

        assert (export_dir / "pipeline.package").exists()
        assert (export_dir / "requirements.txt").exists()

        script = f'''
import torch
from tensordict import TensorDict
from torch.package.package_importer import PackageImporter
pkg = PackageImporter("{export_dir / "pipeline.package"}")
payload = pkg.load_pickle("nexuml_export", "artifact.pkl")
pipeline = payload["pipeline"]
assert isinstance(pipeline, torch.nn.Module)
assert any(p.requires_grad for p in pipeline.parameters())
x = TensorDict({{"features": torch.randn(2, 16)}}, batch_size=[2])
x_out, _ = pipeline(x, None)
loss = x_out["latent"].sum()
opt = pipeline.create_optimizer()
opt.zero_grad()
loss.backward()
opt.step()
print("CUSTOM_OK")
'''
        result = _run_in_clean_subprocess(script, export_dir)
        if result.returncode != 0:
            raise AssertionError(
                f"Clean load failed:\n{result.stderr}\n"
                f"Package structure:\n{_diagnose_package(export_dir / 'pipeline.package')}"
            )
        assert "CUSTOM_OK" in result.stdout
    finally:
        sys.path.remove(str(tmp_path))


def test_explicit_include_modules_package_dynamic_import(tmp_path):
    """Explicit include patterns package modules hidden behind dynamic imports."""
    custom_dir = tmp_path / "custom_dynamic_pkg"
    helper_dir = tmp_path / "dynamic_helper"
    custom_dir.mkdir()
    helper_dir.mkdir()
    (custom_dir / "__init__.py").write_text("")
    (helper_dir / "__init__.py").write_text("")
    (helper_dir / "ops.py").write_text("def scale(x):\n    return x * 3.0\n")
    (custom_dir / "layers.py").write_text(
        "import torch\n"
        "from nexuml.core.base_layer import PipelineLayer\n\n"
        "class DynamicImportLayer(PipelineLayer):\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.linear = torch.nn.Linear(16, 4)\n"
        "    def forward_tensor(self, x, y=None):\n"
        "        from dynamic_helper import ops\n"
        "        return ops.scale(self.linear(x))\n"
    )

    sys.path.insert(0, str(tmp_path))
    try:
        from custom_dynamic_pkg.layers import DynamicImportLayer  # ty: ignore[unresolved-import]

        registry = LayerRegistry()
        registry.register("DynamicImportLayer", DynamicImportLayer)
        scenario = ScenarioSpec(
            name="test_dynamic_include",
            pipeline=PipelineSpec(
                stages={
                    "encode": [
                        LayerSpec(
                            type_key="DynamicImportLayer",
                            keys_in=["features"],
                            keys_out=["latent"],
                            params={},
                        ),
                    ]
                }
            ),
            training=TrainingSpec(max_epochs=1, batch_size=4, loss_keys={"latent_sum": 1.0}),
            data=synthetic_vector_data(feature_shape=(16,), num_samples=32),
        )
        pipeline = compile(scenario, registry)
        export_dir = tmp_path / "exported_dynamic"

        export_package(
            pipeline,
            export_dir,
            include_modules=["dynamic_helper.**"],
        )

        script = f'''
import torch
from tensordict import TensorDict
from torch.package.package_importer import PackageImporter
pkg = PackageImporter("{export_dir / "pipeline.package"}")
payload = pkg.load_pickle("nexuml_export", "artifact.pkl")
pipeline = payload["pipeline"]
x = TensorDict({{"features": torch.ones(2, 16)}}, batch_size=[2])
x_out, _ = pipeline(x, None)
assert "latent" in x_out.keys()
assert torch.isfinite(x_out["latent"]).all()
print("DYNAMIC_OK")
'''
        result = _run_in_clean_subprocess(script, export_dir)
        if result.returncode != 0:
            raise AssertionError(
                f"Dynamic clean load failed:\n{result.stderr}\n"
                f"Package structure:\n{_diagnose_package(export_dir / 'pipeline.package')}"
            )
        assert "DYNAMIC_OK" in result.stdout
    finally:
        sys.path.remove(str(tmp_path))


@pytest.fixture(scope="session")
def cifar_resnet_export_dir(tmp_path_factory):
    from nexuml_library.scenarios.vision.cifar_resnet import cifar_resnet
    from nexuml.training.lightning import NexuSession

    try:
        scenario = cifar_resnet(download=False, max_epochs=0)
    except Exception as exc:
        pytest.skip(f"CIFAR data not available: {exc}")

    try:
        session = NexuSession.from_scenario(scenario)
        session.setup()
    except Exception as exc:
        pytest.skip(f"CIFAR data not available: {exc}")

    export_dir = tmp_path_factory.mktemp("cifar_resnet_export")
    export_package(
        session.pipeline,
        export_dir,
        lightning_module=session.lightning_module,
        trainer=session.trainer,
    )
    return export_dir


@pytest.mark.slow
def test_cifar_resnet_package_export_and_clean_load(cifar_resnet_export_dir):
    assert (cifar_resnet_export_dir / "pipeline.package").exists()
    assert (cifar_resnet_export_dir / "requirements.txt").exists()

    metadata = json.loads((cifar_resnet_export_dir / "metadata.json").read_text())
    deps = metadata.get("external_dependencies", [])
    dists = {d["distribution"] for d in deps}
    assert "torch" in dists
    assert "torchvision" in dists
    assert "nexuml" not in dists
    assert "nexuml_library" not in dists

    script = f'''
import torch
from tensordict import TensorDict
from torch.package.package_importer import PackageImporter
pkg = PackageImporter("{cifar_resnet_export_dir / "pipeline.package"}")
payload = pkg.load_pickle("nexuml_export", "artifact.pkl")
for key in ("pipeline", "resolved_config", "metadata", "training_state"):
    assert key in payload, key
pipeline = payload["pipeline"]
assert isinstance(pipeline, torch.nn.Module)
assert any(p.requires_grad for p in pipeline.parameters())
x = TensorDict({{"features": torch.randn(2, 3, 32, 32)}}, batch_size=[2])
y = TensorDict({{"class_labels": torch.randint(0, 10, (2,))}}, batch_size=[2])
x_out, _ = pipeline(x, y)
assert "classification_loss" in x_out.keys()
loss = x_out["classification_loss"].mean()
assert loss.dim() == 0
opt = pipeline.create_optimizer()
opt.zero_grad()
loss.backward()
opt.step()
print("CIFAR_OK")
'''
    result = _run_in_clean_subprocess(script, cifar_resnet_export_dir)
    if result.returncode != 0:
        raise AssertionError(
            f"Clean CIFAR load failed:\n{result.stderr}\n"
            f"Package structure:\n{_diagnose_package(cifar_resnet_export_dir / 'pipeline.package')}"
        )
    assert "CIFAR_OK" in result.stdout
