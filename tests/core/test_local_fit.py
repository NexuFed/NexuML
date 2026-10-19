"""Small synthetic unit fixtures, never native SO-101 demonstration evidence."""

from dataclasses import asdict
import json

import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
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
