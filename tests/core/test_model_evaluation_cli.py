"""Unified evaluation uses exported weights and only the declared dataset test split."""

from hashlib import sha256
from typing import Any, cast
import random
import pickle
import os
import signal
import subprocess
import sys

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch.loggers.logger import DummyLogger
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.export import export_package, load_inference_package
from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.types import EvaluationSpec, LoaderSpec
from nexuml.training.lightning import NexuSession, create_data_module_from_spec
from nexuml_library.scenarios.asd.synthetic_linear_ae import synthetic_linear_ae_reconstruction


runner = CliRunner()


def forbidden(*args, **kwargs):
    pytest.fail("evaluation must not train, recompile a model, or create an optimizer")


def model_scenario():
    scenario = synthetic_linear_ae_reconstruction(
        feature_shape=(4,),
        num_samples=32,
        hidden_dims=[8],
        latent_dim=2,
        max_epochs=1,
        batch_size=4,
    )
    scenario.training.accelerator = "cpu"
    scenario.training.max_steps = 2
    scenario.logging = None
    scenario.evaluation = EvaluationSpec()
    return scenario


@pytest.fixture(scope="module")
def model_package(tmp_path_factory):
    root = tmp_path_factory.mktemp("model_evaluation_package")
    session = NexuSession(
        model_scenario(), log_dir=root, enable_loggers=False, enable_progress_bar=False
    )
    session.fit()
    return export_package(session.pipeline, root / "model")


def forbid_training(monkeypatch):
    monkeypatch.setattr(L.Trainer, "fit", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr("nexuml.training.lightning.NexuSession", forbidden)
    monkeypatch.setattr("nexuml.core.compiler.compile", forbidden)
    monkeypatch.setattr("nexuml.core.export.export_package", forbidden)


def test_real_model_evaluation_runs_only_test_split_without_training(
    model_package,
    monkeypatch,
):
    loaded, config, metadata = load_inference_package(model_package)
    model = cast(CompiledPipeline, loaded)
    data = create_data_module_from_spec(config.to_scenario())
    data.setup("test")
    expected = torch.cat([x["features"] for x, _ in data.test_dataloader()])
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    content = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in model_package.iterdir()
        if path.is_file()
    }
    seen = []
    original = model.forward

    def forward(x, y=None):
        seen.append(x["features"].detach().cpu().clone())
        return original(x, y)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package", lambda _: (model, config, metadata)
    )
    forbid_training(monkeypatch)
    result = runner.invoke(app, ["evaluate", str(model_package)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Evaluation complete" in result.output and "test/reconstruction_loss" in result.output
    torch.testing.assert_close(torch.cat(seen), expected, rtol=0, atol=0)
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters())
    assert content == {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in model_package.iterdir()
        if path.is_file()
    }


@pytest.mark.parametrize(
    "override", ["data.loader.batch_size=2", "evaluation.test_result_metrics=all"]
)
def test_model_evaluation_accepts_data_and_reporting_overrides(
    model_package, monkeypatch, override
):
    forbid_training(monkeypatch)
    result = runner.invoke(app, ["evaluate", str(model_package), "-O", override])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "test/reconstruction_loss" in result.output


@pytest.mark.parametrize(
    "override",
    ["pipeline={}", "training.max_epochs=2", "exports=[]", "interaction.environment.task=Other"],
)
def test_model_evaluation_rejects_model_and_training_replacement_before_data(
    model_package,
    monkeypatch,
    override,
):
    monkeypatch.setattr("nexuml.training.lightning.create_data_module_from_spec", forbidden)
    result = runner.invoke(app, ["evaluate", str(model_package), "-O", override])
    assert result.exit_code != 0 and "not supported" in result.output


def test_model_evaluation_rejects_empty_test_split(model_package, monkeypatch):
    model, config, metadata = load_inference_package(model_package)
    config.data.train_split, config.data.val_split, config.data.test_split = 1.0, 0.0, 0.0
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package", lambda _: (model, config, metadata)
    )
    forbid_training(monkeypatch)
    result = runner.invoke(app, ["evaluate", str(model_package)])
    assert result.exit_code != 0 and "test split is empty" in result.output


def test_model_evaluation_never_builds_missing_preprocessing_cache(
    model_package, monkeypatch, tmp_path
):
    model, config, metadata = load_inference_package(model_package)
    config.data.preprocessing.enabled = True
    config.data.preprocessing.path = str(tmp_path / "missing-cache")
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package", lambda _: (model, config, metadata)
    )
    monkeypatch.setattr("nexuml.training.lightning.materialize_preprocessed_dataset", forbidden)
    forbid_training(monkeypatch)
    result = runner.invoke(app, ["evaluate", str(model_package)])
    assert result.exit_code != 0 and "preprocessed dataset" in result.output.lower()
    assert not (tmp_path / "missing-cache").exists()


@pytest.mark.parametrize("kind", ["model", "policy"])
def test_declarative_python_file_trains_exports_and_evaluates(tmp_path, monkeypatch, kind):
    package = tmp_path / kind
    source = tmp_path / "experiment.py"
    if kind == "model":
        factory = (
            "from nexuml_library.scenarios.asd.synthetic_linear_ae import (\n"
            "    synthetic_linear_ae_reconstruction,\n"
            ")\n"
            "def scenario():\n"
            "    s = synthetic_linear_ae_reconstruction(feature_shape=(4,), num_samples=32, "
            "hidden_dims=[8], latent_dim=2, max_epochs=1, batch_size=4)\n"
            "    s.training.accelerator = 'cpu'\n"
            "    s.training.max_steps = 2\n"
            "    s.evaluation = EvaluationSpec()\n"
        )
    else:
        pytest.importorskip("gymnasium")
        factory = (
            "from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo\n"
            "def scenario():\n"
            "    s = cartpole_ppo(total_frames=16, frames_per_batch=16)\n"
            "    s.training.algorithm = s.training.algorithm.model_copy("
            "update={'update_epochs': 1, 'minibatch_size': 8})\n"
        )
    source.write_text(
        "from nexuml.core.types import ExportSpec, EvaluationSpec\n"
        + factory
        + "    s.logging = None\n"
        + f"    s.exports = [ExportSpec(output={str(package)!r})]\n"
        + "    return s\n"
    )
    monkeypatch.setenv("NEXUML_LOGS_ROOT", str(tmp_path / "logs"))
    result = runner.invoke(app, ["train", "--scenario-file", str(source)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert (package / "pipeline.package").is_file()
    forbid_training(monkeypatch)
    command = ["evaluate", str(package)]
    if kind == "policy":
        command += [
            "-O",
            "training.evaluation.episodes=1",
            "-O",
            "training.evaluation.max_steps_per_episode=3",
        ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Evaluation complete" in result.output


def test_evaluate_help_is_not_policy_only():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0 and "model" in result.output.lower()


class RecordingLogger(DummyLogger):
    def __init__(self):
        super().__init__()
        self.metrics = []
        self.statuses = []

    def log_metrics(self, metrics, step=None):
        assert not self.statuses, "metrics must be written before logger finalization"
        self.metrics.append(dict(metrics))

    def finalize(self, status):
        self.statuses.append(status)


def test_dataset_logger_is_finalized_once_after_metrics(model_package, monkeypatch):
    logger = RecordingLogger()
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", lambda *a, **k: [logger])
    monkeypatch.setattr("nexuml.tracking.logger.log_text_artifact", lambda *a, **k: None)
    forbid_training(monkeypatch)
    result = runner.invoke(app, ["evaluate", str(model_package)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert any("test/reconstruction_loss" in metrics for metrics in logger.metrics)
    assert logger.statuses == ["success"]


@pytest.mark.parametrize("error", [RuntimeError("model evaluation failed"), SystemExit(143)])
def test_dataset_failure_finalizes_all_loggers(model_package, monkeypatch, error):
    loggers = [RecordingLogger(), RecordingLogger()]
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", lambda *a, **k: loggers)
    monkeypatch.setattr("nexuml.tracking.logger.log_text_artifact", lambda *a, **k: None)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("nexuml.evaluation.dataset.evaluate_model", fail)
    result = runner.invoke(app, ["evaluate", str(model_package)])
    assert result.exit_code == (143 if isinstance(error, SystemExit) else 1)
    assert all(logger.statuses == ["failed"] for logger in loggers)
    assert "Evaluation complete" not in result.output


def test_explicit_nexudataset_test_metadata_is_respected(model_package, monkeypatch):
    from nexuml.data.dataset import NexuDataset
    from nexuml.data.module import NexuDataModule
    from nexuml.evaluation.dataset import evaluate_model

    loaded, config, _ = load_inference_package(model_package)
    model = cast(CompiledPipeline, loaded)
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    data = NexuDataset(
        meta=pd.DataFrame({"split": ["train", "test", "val", "train", "test", "val"]}), data=values
    )
    module = NexuDataModule(data, loader_spec=LoaderSpec(batch_size=2), split_by_column=True)
    monkeypatch.setattr("nexuml.training.lightning.create_data_module_from_spec", lambda _: module)
    original = model.forward
    seen = []

    def forward(x, y=None):
        seen.append(x["features"].clone())
        return original(x, y)

    monkeypatch.setattr(model, "forward", forward)
    forbid_training(monkeypatch)
    metrics = evaluate_model(model, config.to_scenario())
    assert "test/reconstruction_loss" in metrics
    torch.testing.assert_close(torch.cat(seen), values[[1, 4]], rtol=0, atol=0)


def test_cached_preprocessing_is_read_without_overwrite_or_recompilation(
    model_package,
    tmp_path,
    monkeypatch,
):
    from nexuml.data.export import export_data_module
    from nexuml.evaluation.dataset import evaluate_model

    loaded, config, _ = load_inference_package(model_package)
    model = cast(CompiledPipeline, loaded)
    scenario = config.to_scenario()
    cache = export_data_module(
        create_data_module_from_spec(scenario), tmp_path / "cache", backend="numpy"
    )
    before = {
        str(p.relative_to(cache)): sha256(p.read_bytes()).hexdigest()
        for p in cache.rglob("*")
        if p.is_file()
    }
    scenario.data.preprocessing.enabled = True
    scenario.data.preprocessing.overwrite = True
    scenario.data.preprocessing.path = str(cache)
    forbid_training(monkeypatch)
    metrics = evaluate_model(model, scenario)
    assert "test/reconstruction_loss" in metrics
    assert scenario.data.preprocessing.overwrite is True
    assert before == {
        str(p.relative_to(cache)): sha256(p.read_bytes()).hexdigest()
        for p in cache.rglob("*")
        if p.is_file()
    }


def test_dataset_evaluation_preserves_rng_and_model_mode(model_package, monkeypatch):
    from nexuml.evaluation.dataset import evaluate_model

    loaded, config, _ = load_inference_package(model_package)
    model = cast(CompiledPipeline, loaded)
    scenario = config.to_scenario()
    scenario.training.seed = 13
    model.train()
    torch_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())
    forbid_training(monkeypatch)
    evaluate_model(model, scenario)
    assert model.training
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state


def test_packaged_fitted_state_and_evaluation_algorithms_are_reused(tmp_path, monkeypatch):
    from tensordict import TensorDict
    from nexuml.core.compiler import compile
    from nexuml.core.types import EvalAlgorithmSpec, LayerSpec
    from nexuml.evaluation.dataset import evaluate_model
    from nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator import AnomalyEvaluator
    from nexuml_library.layers.head.anomaly_score import AnomalyScore
    from nexuml_library.layers.head.decision_rule import DecisionRulePipelineLayer
    from nexuml_library.scenarios.data.synthetic import synthetic_vector_data

    scenario = model_scenario()
    scenario.data = synthetic_vector_data(feature_shape=(4,), num_samples=64)
    scenario.pipeline.stages["Scores"] = [
        LayerSpec(
            component=AnomalyScore(),
            keys_in=["features", "reconstructed"],
            keys_out=["anomaly_score"],
        ),
        LayerSpec(
            component=DecisionRulePipelineLayer(), keys_in=["anomaly_score"], keys_out=["decision"]
        ),
    ]
    scenario.evaluation = EvaluationSpec(
        algorithms=[
            EvalAlgorithmSpec(
                algorithm=AnomalyEvaluator(decision_key="decision"),
                label_key="anomaly",
                name="anomaly_report",
            )
        ],
        test_result_metrics="all",
    )
    pipeline = compile(scenario)
    layer = cast(
        Any, next(layer for _, _, layer in pipeline.iter_layers() if hasattr(layer, "_fitted"))
    )
    layer.collect_batch(
        TensorDict({"anomaly_score": torch.arange(1.0, 5.0)}, batch_size=(4,)), None
    )
    layer.finalize_fit()
    layer._fitted = True
    threshold = layer._rule.threshold
    package = export_package(pipeline, tmp_path / "fitted-model")
    loaded, config, _ = load_inference_package(package)
    model = cast(CompiledPipeline, loaded)
    fitted = cast(
        Any, next(layer for _, _, layer in model.iter_layers() if hasattr(layer, "_fitted"))
    )
    assert fitted._fitted and fitted._rule.threshold == threshold
    monkeypatch.setattr(fitted, "collect_batch", forbidden)
    monkeypatch.setattr(fitted, "finalize_fit", forbidden)
    forbid_training(monkeypatch)
    metrics = evaluate_model(model, config.to_scenario())
    assert "eval/anomaly_report/score_mean" in metrics
    assert "test/anomaly_report/score_mean" in metrics
    assert torch.isfinite(torch.tensor(metrics["eval/anomaly_report/score_mean"]))
    assert fitted._rule.threshold == threshold and fitted._fitted


@pytest.mark.parametrize("selector", ["--scenario-file", "--config"])
def test_model_scenario_file_selects_test_data_not_the_model(
    model_package, monkeypatch, tmp_path, selector
):
    source = tmp_path / "test_data.py"
    source.write_text(
        "from nexuml.core.types import ScenarioSpec\n"
        "from nexuml_library.scenarios.data.synthetic import synthetic_vector_data\n"
        "def scenario():\n"
        "    data = synthetic_vector_data(feature_shape=(4,), num_samples=8)\n"
        "    data.train_split, data.val_split, data.test_split = 0.0, 0.0, 1.0\n"
        "    return ScenarioSpec(name='data_only', data=data)\n"
    )
    if selector == "--config":
        import runpy
        from nexuml.core.config import ResolvedConfig

        selected = runpy.run_path(str(source))["scenario"]()
        source = tmp_path / "test_data.yaml"
        ResolvedConfig.from_scenario(selected).save(source)
    loaded, config, metadata = load_inference_package(model_package)
    model = cast(CompiledPipeline, loaded)
    original = model.forward
    counts = []

    def forward(x, y=None):
        counts.append(x.batch_size[0])
        return original(x, y)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package", lambda _: (model, config, metadata)
    )
    forbid_training(monkeypatch)
    result = runner.invoke(app, ["evaluate", str(model_package), selector, str(source)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert sum(counts) == 8 and "test/reconstruction_loss" in result.output


@pytest.mark.parametrize("selector", ["--scenario-file", "--config"])
def test_policy_scenario_file_selects_environment_not_the_policy(tmp_path, monkeypatch, selector):
    pytest.importorskip("gymnasium")
    from nexuml.core.compiler import compile
    from nexuml.core.policy import CompiledPolicy
    from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo

    scenario = cartpole_ppo()
    scenario.logging = None
    package = export_package(CompiledPolicy.from_pipeline(compile(scenario)), tmp_path / "policy")
    source = tmp_path / "replay.py"
    source.write_text(
        "from nexuml.core.types import PipelineSpec, PolicyEvaluationSpec\n"
        "from nexuml_library.environments.gymnasium import GymnasiumEnvironment\n"
        "from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo\n"
        "def scenario():\n"
        "    s = cartpole_ppo()\n"
        "    s.pipeline = PipelineSpec()\n"
        "    s.interaction.environment = GymnasiumEnvironment(env_name='CartPole-v1', "
        "kwargs={'max_episode_steps': 2})\n"
        "    s.policy_evaluation = PolicyEvaluationSpec(episodes=1, seed=13)\n"
        "    s.logging = None\n"
        "    return s\n"
    )
    if selector == "--config":
        import runpy
        from nexuml.core.config import ResolvedConfig

        selected = runpy.run_path(str(source))["scenario"]()
        source = tmp_path / "replay.yaml"
        ResolvedConfig.from_scenario(selected).save(source)
    forbid_training(monkeypatch)
    result = runner.invoke(app, ["evaluate", str(package), selector, str(source)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "eval/rl/length_mean" in result.output and "2.0" in result.output


def test_double_precision_trained_export_keeps_test_input_conversion(tmp_path, monkeypatch):
    scenario = model_scenario()
    scenario.training.precision = "64-true"
    session = NexuSession(
        scenario, log_dir=tmp_path, enable_loggers=False, enable_progress_bar=False
    )
    session.fit()
    package = export_package(session.pipeline, tmp_path / "double-model")
    model, config, metadata = load_inference_package(package)
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package", lambda _: (model, config, metadata)
    )
    forbid_training(monkeypatch)
    result = runner.invoke(app, ["evaluate", str(package)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "test/reconstruction_loss" in result.output


@pytest.mark.parametrize("error", [RuntimeError("primary test failure"), SystemExit(143)])
def test_real_test_loop_retains_primary_when_logger_cleanup_fails(
    model_package,
    monkeypatch,
    error,
):
    from nexuml.training.lightning import NexuLightningModule

    class BrokenLogger(RecordingLogger):
        def finalize(self, status):
            super().finalize(status)
            raise RuntimeError("logger cleanup failure")

    loggers = [BrokenLogger(), RecordingLogger()]
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", lambda *a, **k: loggers)
    monkeypatch.setattr("nexuml.tracking.logger.log_text_artifact", lambda *a, **k: None)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(NexuLightningModule, "test_step", fail)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        result = runner.invoke(app, ["evaluate", str(model_package)])
        assert result.exit_code == (143 if isinstance(error, SystemExit) else 1), result.output
        if isinstance(error, RuntimeError):
            assert "primary test failure" in result.output
        assert all("failed" in logger.statuses for logger in loggers)
        assert signal.getsignal(signal.SIGTERM) == previous
        assert "Evaluation complete" not in result.output
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_real_sigterm_in_test_loop_exits_143(model_package):
    code = """
import os, signal, sys
from typer.testing import CliRunner
from nexuml.cli.main import app
from nexuml.training.lightning import NexuLightningModule
original = NexuLightningModule.test_step
def interrupt(module, *args, **kwargs):
    os.kill(os.getpid(), signal.SIGTERM)
    return original(module, *args, **kwargs)
NexuLightningModule.test_step = interrupt
result = CliRunner().invoke(app, ['evaluate', sys.argv[1]])
print(result.output, flush=True)
raise SystemExit(result.exit_code)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(model_package)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 143, result.stdout + result.stderr
    assert "Evaluation complete" not in result.stdout


@pytest.mark.requires_gpu
def test_explicit_cuda_model_evaluation_seeds_and_restores_requested_device(
    model_package,
    monkeypatch,
):
    from nexuml.evaluation.dataset import evaluate_model

    loaded, config, _ = load_inference_package(model_package)
    model = cast(CompiledPipeline, loaded)
    scenario = config.to_scenario()
    scenario.training.seed = 13
    original = model.forward
    seen = []

    def forward(x, y=None):
        seen.append(torch.rand_like(x["features"]).detach().cpu())
        return original(x, y)

    monkeypatch.setattr(model, "forward", forward)
    saved = torch.cuda.get_rng_state(0).clone()
    forbid_training(monkeypatch)
    evaluate_model(model, scenario, device="cuda:0")
    assert torch.equal(torch.cuda.get_rng_state(0), saved)
    first = torch.cat(seen)
    seen.clear()
    evaluate_model(model, scenario, device="cuda:0")
    assert torch.equal(torch.cuda.get_rng_state(0), saved)
    torch.testing.assert_close(torch.cat(seen), first, rtol=0, atol=0)
