"""Small synthetic unit fixtures, never native SO-101 demonstration evidence."""

from dataclasses import asdict
import importlib
import json

import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.serialization import lower_model
from nexuml.core.types import (
    DataSpec,
    DatasetSpec,
    InteractionSpec,
    LayerSpec,
    PipelineSpec,
    PolicySpec,
    ScenarioSpec,
    TrainingSpec,
)
from nexuml.data.dataset import NexuDataset
from nexuml.training import local_fit, local_fit_manifest
from nexuml_library.action_adapters.adapters import DirectActionAdapter
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.layers.head.regression_head import LatentRegressionHead
from nexuml_library.layers.loss.regression_loss import RegressionLoss


@data_source("local-fit-unit-episodes")
class UnitEpisodes(DataSourceDefinition):
    root: str
    broken: str = ""
    robot: bool = False

    def build(self):
        return UnitDataset(self)


class UnitDataset(NexuDataset):
    def __init__(self, source):
        self.robot = source.robot
        self.broken = source.broken
        meta = pd.DataFrame(
            [
                {
                    "episode_index": e,
                    "frame_index": f,
                    "timestamp": f / 30,
                    "split": ("train", "train", "val", "test")[e],
                }
                for e in range(4)
                for f in range(4)
            ]
        )
        if source.broken == "split":
            meta.loc[0, "split"] = "test"
        if source.broken == "clock":
            meta.loc[1, "timestamp"] = 0
        self.info = {
            "provenance": {
                "interaction_contract": lower_model(
                    IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach").describe()
                )
            }
        }
        if source.broken == "physical":
            self.info["provenance"]["interaction_contract"]["control_period_s"] = 0.1
        super().__init__(meta=meta, label_names=[])

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        value = torch.tensor([float(row.episode_index) / 10, float(row.frame_index) / 10])
        if self.robot:
            value = torch.cat((value, torch.zeros(13)))
            target = torch.zeros(5)
        else:
            target = value * 0.3
        if self.broken == "nan":
            target[0] = float("nan")
        return TensorDict({"observation": value}, []), TensorDict({"action": target}, [])


def recipe(root, *, robot=False, broken=""):
    env = IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach")
    return ScenarioSpec(
        name="unit-local-fit",
        data=DataSpec(
            datasets=[
                DatasetSpec(
                    source=UnitEpisodes(root=str(root), robot=robot, broken=broken),
                    split_type="keep",
                )
            ],
            input_shapes={"observation": [15 if robot else 2]},
            input_contracts=env.describe().observations if robot else {},
        ),
        pipeline=PipelineSpec(
            stages={
                "actor": [
                    LayerSpec(
                        component=LatentRegressionHead(num_outputs=5 if robot else 2),
                        keys_in=["observation"],
                        keys_out=["prediction"],
                    )
                ],
                "loss": [
                    LayerSpec(
                        component=RegressionLoss(),
                        keys_in=["prediction"],
                        keys_out=["bc_loss"],
                        label_key="action",
                    )
                ],
            }
        ),
        training=TrainingSpec(batch_size=4, lr=0.01, loss_keys={"bc_loss": 1}),
        interaction=InteractionSpec(environment=env) if robot else None,
        policy=PolicySpec(action_adapter=DirectActionAdapter(source_key="prediction"))
        if robot
        else None,
    )


def request(root, **kwargs):
    scenario = recipe(root, **kwargs)
    state = {name: value.detach().clone() for name, value in compile(scenario).state_dict().items()}
    return dict(
        trusted_recipe=scenario,
        initial_tensors=state,
        model_manifest=local_fit_manifest(scenario, state),
        local_data=scenario.data,
        budget={"steps": 3},
        seed=7,
    )


def test_two_external_initial_states_are_used_repeatably_and_inputs_remain_private(
    tmp_path, monkeypatch
):
    from nexuml.training.lightning import NexuSession

    starts = []
    original_fit = NexuSession.fit

    def checked_fit(session):
        starts.append(
            {name: value.clone() for name, value in session.pipeline.state_dict().items()}
        )
        assert not session.trainer.optimizers
        return original_fit(session)

    monkeypatch.setattr(NexuSession, "fit", checked_fit)
    first = request(tmp_path)
    original = {name: value.clone() for name, value in first["initial_tensors"].items()}
    a = local_fit(**first)
    a_again = local_fit(**first)
    second = {**first, "initial_tensors": {name: value + 0.25 for name, value in original.items()}}
    second["model_manifest"] = local_fit_manifest(
        second["trusted_recipe"], second["initial_tensors"]
    )
    b = local_fit(**second)
    b_again = local_fit(**second)
    for actual, expected in zip(
        starts,
        [original, original, second["initial_tensors"], second["initial_tensors"]],
        strict=True,
    ):
        assert all(torch.equal(actual[name], expected[name]) for name in original)
    assert a.train_windows == b.train_windows == 8 and a.episodes == b.episodes == 2
    assert a.metrics["optimizer_steps"] == b.metrics["optimizer_steps"] == 3
    assert a.metrics["consumed_windows"] == 12
    assert (
        a.provenance["model_manifest"]["base_version"]
        != b.provenance["model_manifest"]["base_version"]
    )
    for name in original:
        assert torch.equal(first["initial_tensors"][name], original[name])
        assert torch.equal(a.updated_tensors[name], a_again.updated_tensors[name])
        assert torch.equal(b.updated_tensors[name], b_again.updated_tensors[name])
        assert not torch.equal(a.updated_tensors[name], original[name])
        assert not torch.equal(a.updated_tensors[name], b.updated_tensors[name])
        assert a.updated_tensors[name].device.type == "cpu"
        assert not a.updated_tensors[name].requires_grad
    public = asdict(a)
    del public["updated_tensors"]
    assert str(tmp_path) not in json.dumps(public)
    assert set(a.metrics) <= {"train/loss", "val/loss", "optimizer_steps", "consumed_windows"}


def test_epoch_budget_uses_existing_session(tmp_path):
    result = local_fit(**{**request(tmp_path), "budget": {"epochs": 2}})
    assert result.metrics["optimizer_steps"] == 4
    assert result.metrics["consumed_windows"] == 16


@pytest.mark.parametrize("bad", ["missing", "unexpected", "shape", "dtype", "nan", "pickle"])
def test_reject_invalid_tensor_state_before_dataset_build(tmp_path, monkeypatch, bad):
    args = request(tmp_path)
    state = dict(args["initial_tensors"])
    name = next(iter(state))
    if bad == "missing":
        state.pop(name)
    elif bad == "unexpected":
        state["unexpected"] = torch.zeros(1)
    elif bad == "shape":
        state[name] = state[name].reshape(-1)
    elif bad == "dtype":
        state[name] = state[name].double()
    elif bad == "nan":
        state[name] = state[name] * float("nan")
    else:
        state[name] = b"not a tensor or pickle update"
    monkeypatch.setattr(UnitEpisodes, "build", lambda *_: pytest.fail("unauthorized build"))
    with pytest.raises(ValueError, match="tensor|state"):
        local_fit(**{**args, "initial_tensors": state})


@pytest.mark.parametrize("change", ["manifest", "data", "optimizer", "remote_recipe", "seed"])
def test_reject_contract_source_or_state_requests_before_build(tmp_path, monkeypatch, change):
    args = request(tmp_path, robot=True)
    if change == "manifest":
        args["model_manifest"]["physical_schema_hash"] = "wrong-joint-order"
    elif change == "data":
        args["local_data"] = recipe(tmp_path / "unauthorized").data
    elif change == "optimizer":
        args["optimizer_mode"] = "continue"
    elif change == "seed":
        args["seed"] = True
    else:
        args["trusted_recipe"] = "https://untrusted/remote.py"
    monkeypatch.setattr(UnitEpisodes, "build", lambda *_: pytest.fail("unauthorized build"))
    with pytest.raises(ValueError):
        local_fit(**args)


@pytest.mark.parametrize(
    "budget",
    [{}, {"steps": 0}, {"steps": True}, {"epochs": -1}, {"steps": 1, "epochs": 2}, {"steps": 1.5}],
)
def test_reject_unbounded_or_ambiguous_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="budget"):
        local_fit(**{**request(tmp_path), "budget": budget})


@pytest.mark.parametrize("broken", ["split", "clock", "nan", "physical"])
def test_reject_invalid_episode_or_physical_data_before_fit(tmp_path, monkeypatch, broken):
    args = request(tmp_path, robot=True, broken=broken)
    monkeypatch.setattr(
        "nexuml.training.lightning.NexuSession.fit", lambda *_: pytest.fail("fit called")
    )
    with pytest.raises(ValueError):
        local_fit(**args)


def test_same_shapes_wrong_control_period_rejected(tmp_path):
    args = request(tmp_path, robot=True)
    other = json.loads(json.dumps(args["model_manifest"]))
    other["physical_schema_hash"] = "same-shapes-but-different-clock"
    with pytest.raises(ValueError, match="physical"):
        local_fit(**{**args, "model_manifest": other})


def test_manifest_preserves_executable_stage_order(tmp_path):
    from nexuml import nn_module

    scenario = recipe(tmp_path)
    scenario.pipeline = PipelineSpec(
        stages={
            "linear": [
                LayerSpec(
                    component=nn_module(torch.nn.Linear, 2, 2),
                    keys_in=["observation"],
                    keys_out=["observation"],
                )
            ],
            "relu": [
                LayerSpec(
                    component=nn_module(torch.nn.ReLU),
                    keys_in=["observation"],
                    keys_out=["observation"],
                )
            ],
        }
    )
    state = compile(scenario).state_dict()
    original = local_fit_manifest(scenario, state)
    scenario.pipeline.stages = dict(reversed(list(scenario.pipeline.stages.items())))
    reordered = local_fit_manifest(scenario, state)
    assert original["tensors"] == reordered["tensors"]
    assert original["base_version"] == reordered["base_version"]
    assert original["recipe_hash"] != reordered["recipe_hash"]
    with pytest.raises(ValueError, match="manifest"):
        local_fit(
            trusted_recipe=scenario,
            initial_tensors=state,
            model_manifest=original,
            local_data=scenario.data,
            budget={"steps": 1},
            seed=0,
        )


def test_bias_only_scope_is_applied_before_optimizer_and_repeats_exactly(tmp_path, monkeypatch):
    args = request(tmp_path)
    initial = args["initial_tensors"]
    scope = tuple(name for name in initial if name.endswith(".bias"))
    assert scope
    manifest = local_fit_manifest(args["trusted_recipe"], initial, trainable_names=scope)
    assert manifest["recipe_hash"] != args["model_manifest"]["recipe_hash"]
    original_optimizer = CompiledPipeline.create_optimizer
    optimizer_scopes = []

    def checked_optimizer(pipeline):
        selected = {name for name, param in pipeline.named_parameters() if param.requires_grad}
        assert selected == set(scope)
        optimizer_scopes.append(selected)
        return original_optimizer(pipeline)

    monkeypatch.setattr(CompiledPipeline, "create_optimizer", checked_optimizer)
    selected_args: dict[str, Any] = {**args, "trainable_names": scope, "model_manifest": manifest}
    result = local_fit(**selected_args)
    repeated = local_fit(**selected_args)
    assert len(optimizer_scopes) == 2
    assert result.train_windows == 8 and result.episodes == 2
    assert result.metrics["optimizer_steps"] == 3
    assert result.metrics["consumed_windows"] == 12
    assert result.provenance["model_manifest"] == manifest
    assert set(result.updated_tensors) == set(initial)
    for name, value in result.updated_tensors.items():
        assert manifest["tensors"][name]["role"] == ("trainable" if name in scope else "frozen")
        assert torch.equal(value, repeated.updated_tensors[name])
        assert torch.equal(value, initial[name]) == (name not in scope)
        assert value.device.type == "cpu" and not value.requires_grad


def test_manifest_distinguishes_explicit_scope_without_changing_default(tmp_path):
    args = request(tmp_path)
    scenario, state = args["trusted_recipe"], args["initial_tensors"]
    assert local_fit_manifest(scenario, state, trainable_names=None) == args["model_manifest"]
    scope = tuple(state)
    explicit = local_fit_manifest(scenario, state, trainable_names=scope)
    assert explicit["tensors"] == args["model_manifest"]["tensors"]
    assert explicit["base_version"] == args["model_manifest"]["base_version"]
    assert explicit["recipe_hash"] != args["model_manifest"]["recipe_hash"]
    assert explicit == local_fit_manifest(scenario, state, trainable_names=tuple(reversed(scope)))


@pytest.mark.parametrize("bad", ["empty", "string", "list", "duplicate", "unknown", "type", "glob"])
def test_reject_invalid_scope_before_dataset_build(tmp_path, monkeypatch, bad):
    args = request(tmp_path)
    name = next(iter(args["initial_tensors"]))
    scope: Any = {
        "empty": (),
        "string": name,
        "list": [name],
        "duplicate": (name, name),
        "unknown": ("not_a_parameter",),
        "type": (name, []),
        "glob": ("*.bias",),
    }[bad]
    monkeypatch.setattr(UnitEpisodes, "build", lambda *_: pytest.fail("unauthorized build"))
    with pytest.raises(ValueError, match="scope"):
        local_fit_manifest(args["trusted_recipe"], args["initial_tensors"], trainable_names=scope)
    with pytest.raises(ValueError, match="scope"):
        local_fit(**{**args, "trainable_names": scope})


@pytest.mark.parametrize("bad", ["buffer", "frozen"])
def test_reject_ineligible_tensor_scope_before_dataset_build(tmp_path, monkeypatch, bad):
    args = request(tmp_path)
    name = "unit_buffer" if bad == "buffer" else next(iter(args["initial_tensors"]))

    def ineligible_pipeline(scenario):
        pipeline = compile(scenario)
        if bad == "buffer":
            pipeline.register_buffer(name, torch.zeros(1))
        else:
            pipeline.get_parameter(name).requires_grad_(False)
        return pipeline

    if bad == "buffer":
        args["initial_tensors"][name] = torch.zeros(1)
    args["trainable_names"] = (name,)
    module = importlib.import_module("nexuml.training.local_fit")
    monkeypatch.setattr(module, "compile", ineligible_pipeline)
    monkeypatch.setattr(UnitEpisodes, "build", lambda *_: pytest.fail("unauthorized build"))
    with pytest.raises(ValueError, match="originally trainable parameter"):
        local_fit(**args)


def test_reject_scope_manifest_mismatch_before_dataset_build(tmp_path, monkeypatch):
    args = request(tmp_path)
    scope = tuple(name for name in args["initial_tensors"] if name.endswith(".bias"))
    args["model_manifest"] = local_fit_manifest(
        args["trusted_recipe"], args["initial_tensors"], trainable_names=scope
    )
    monkeypatch.setattr(UnitEpisodes, "build", lambda *_: pytest.fail("unauthorized build"))
    with pytest.raises(ValueError, match="manifest"):
        local_fit(**args)


@pytest.mark.parametrize(
    "device", [None, 0, "auto", "cpu:0", "cuda", "gpu", "mps", "cuda:-1", "cuda:0,1", "cuda:01"]
)
def test_reject_ambiguous_or_unsupported_device_before_dataset_build(tmp_path, monkeypatch, device):
    args = request(tmp_path)
    monkeypatch.setattr(UnitEpisodes, "build", lambda *_: pytest.fail("unauthorized build"))
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("ambiguous device must not query CUDA")
    )
    with pytest.raises(ValueError, match="device"):
        local_fit(**{**args, "device": device})


@pytest.mark.parametrize("available,count,device", [(False, 0, "cuda:0"), (True, 1, "cuda:1")])
def test_reject_unavailable_device_before_dataset_build(
    tmp_path, monkeypatch, available, count, device
):
    args = request(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(UnitEpisodes, "build", lambda *_: pytest.fail("unauthorized build"))
    with pytest.raises(ValueError, match="not available"):
        local_fit(**{**args, "device": device})


def test_episode_receipt_describes_physical_contract_once_without_changing_hash(
    tmp_path, monkeypatch
):
    from nexuml.training.local_fit import _episode_receipt

    scenario = recipe(tmp_path, robot=True)
    dataset = UnitDataset(scenario.data.datasets[0].source)
    expected = _episode_receipt(dataset, scenario)
    original_describe = IsaacLabEnvironment.describe
    calls = []

    def counted_describe(environment):
        calls.append(environment)
        return original_describe(environment)

    monkeypatch.setattr(IsaacLabEnvironment, "describe", counted_describe)
    assert _episode_receipt(dataset, scenario) == expected
    assert len(calls) == 1


def test_repeated_fit_revalidates_changed_private_windows(tmp_path, monkeypatch):
    args = request(tmp_path)
    local_fit(**args)
    original_getitem = UnitDataset.__getitem__

    def changed_window(dataset, index):
        x, y = original_getitem(dataset, index)
        y["action"][0] = float("nan")
        return x, y

    monkeypatch.setattr(UnitDataset, "__getitem__", changed_window)
    monkeypatch.setattr(
        "nexuml.training.lightning.NexuSession.fit", lambda *_: pytest.fail("fit called")
    )
    with pytest.raises(ValueError, match="finite"):
        local_fit(**args)


def test_reject_changes_outside_selected_scope(tmp_path, monkeypatch):
    from nexuml.training.lightning import NexuSession

    args = request(tmp_path)
    scope = tuple(name for name in args["initial_tensors"] if name.endswith(".bias"))
    frozen = next(name for name in args["initial_tensors"] if name not in scope)
    args["trainable_names"] = scope
    args["model_manifest"] = local_fit_manifest(
        args["trusted_recipe"], args["initial_tensors"], trainable_names=scope
    )
    original_fit = NexuSession.fit

    def changed_frozen(session):
        result = original_fit(session)
        with torch.no_grad():
            session.pipeline.get_parameter(frozen).add_(1)
        return result

    monkeypatch.setattr(NexuSession, "fit", changed_frozen)
    with pytest.raises(ValueError, match="outside the shared trainable scope"):
        local_fit(**args)


@pytest.mark.requires_gpu
def test_explicit_single_cuda_fit_updates_on_gpu_and_returns_cpu_tensors(tmp_path, monkeypatch):
    from nexuml.training.lightning import NexuSession
    from nexuml.training.local_fit import _hash

    args = request(tmp_path)
    steps = []
    config_hashes = []
    original_optimizer = CompiledPipeline.create_optimizer
    original_fit = NexuSession.fit

    def checked_step(optimizer, step_args, step_kwargs):
        parameters = [param for group in optimizer.param_groups for param in group["params"]]
        assert parameters and all(param.device == torch.device("cuda:0") for param in parameters)
        gradients = [param.grad for param in parameters if param.grad is not None]
        assert gradients and all(grad.device == torch.device("cuda:0") for grad in gradients)
        steps.append(1)

    def checked_optimizer(pipeline):
        optimizer = original_optimizer(pipeline)
        optimizer.register_step_post_hook(checked_step)
        return optimizer

    def checked_fit(session):
        training = session.scenario.training
        assert training.accelerator == session.accelerator == "gpu"
        assert training.devices == session.devices == "0,"
        assert training.precision == "32-true" and training.deterministic
        config_hashes.append(_hash(lower_model(training)))
        return original_fit(session)

    monkeypatch.setattr(CompiledPipeline, "create_optimizer", checked_optimizer)
    monkeypatch.setattr(NexuSession, "fit", checked_fit)
    args["device"] = "cuda:0"
    result = local_fit(**args)
    assert len(steps) == result.metrics["optimizer_steps"] == 3
    assert result.provenance["fit_config_hash"] == config_hashes[0]
    assert result.provenance["model_manifest"] == args["model_manifest"]
    assert result.train_windows == 8 and result.episodes == 2
    for name, value in result.updated_tensors.items():
        assert value.device.type == "cpu" and not value.requires_grad
        assert not torch.equal(value, args["initial_tensors"][name])
