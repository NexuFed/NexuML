"""Synthetic CPU fixtures ONLY: schema/recipe tests, never native learning evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.config import ResolvedConfig
from nexuml.core.types import TrainingSpec
from nexuml.interaction.runtime import InteractionStep
from nexuml_library.data.lerobot import LeRobotDataSource
from nexuml_library.environments._isaaclab_so101 import GOALS, HOME, TARGET_SCALE
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.scenarios.imitation.so101 import (
    REPO_ID,
    TASK,
    _source_hashes,
    _split_episodes,
    _task_identity,
    so101_behavior_cloning,
    validate_so101_demonstrations,
)


@pytest.fixture(autouse=True)
def restore_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture(scope="module")
def script():
    return SimpleNamespace(
        **runpy.run_path(
            str(Path(__file__).resolve().parents[2] / "scripts/isaaclab/so101_imitation.py")
        )
    )


def _synthetic_receipt(episodes=6):
    # Deliberately synthetic; matching receipt fields are NOT proof of native execution.
    return {
        "schema_version": 1,
        **_task_identity(),
        "source_sha256": _source_hashes(),
        "seed": 20,
        "episode_seeds": list(range(20, 20 + episodes)),
        "split_episodes": _split_episodes(episodes),
        "packages": {
            name: {"version": "synthetic-test-only"}
            for name in ("nexuml", "nexuml-library", "torch", "lerobot", "isaaclab", "isaacsim")
        }
        | {
            "isaaclab": {
                "version": "synthetic-test-only",
                "revision": IsaacLabEnvironment(task=TASK).revision,
            }
        },
        "test_only": True,
    }


@pytest.fixture
def receipt_root(tmp_path):
    root = tmp_path / "receipt-only"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "total_episodes": 72,
                "provenance": _synthetic_receipt(72),
            }
        )
    )
    return root


def test_scenario_reuses_supervised_layers_and_explicit_actor_contract(receipt_root):
    scenario = so101_behavior_cloning(str(receipt_root))
    assert isinstance(scenario.training, TrainingSpec)
    assert scenario.training.lr == 0.01 and scenario.training.max_epochs == 100
    assert scenario.training.seed == 20 and scenario.training.deterministic
    assert scenario.data.input_shapes == {"observation": [15]}
    assert scenario.data.input_contracts == scenario.interaction.environment.describe().observations
    dataset = scenario.data.datasets[0]
    assert dataset.split_type == "keep"
    assert tuple(map(len, dataset.source.split_episodes.values())) == (48, 12, 12)
    assert dataset.source.input_keys == ("observation",)
    actor = scenario.pipeline.stages["Actor"][0]
    assert actor.component.num_outputs == 5 and actor.component.hidden_dims == []
    assert actor.keys_in == ["observation"] and actor.keys_out == ["predicted_action"]
    assert scenario.pipeline.stages["Loss"][0].label_key == "action"
    assert scenario.policy.action_adapter.source_key == "predicted_action"
    assert scenario.policy_evaluation.episodes == 60
    assert scenario.policy_evaluation.seed == 10000
    assert ResolvedConfig.from_scenario(scenario).to_scenario() == scenario


def test_authored_handoff_fixture_matches_the_compiled_reference(receipt_root):
    from nexuml.core.compiler import compile
    from nexuml.training import local_fit_manifest

    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures/so101_local_fit_contract.json").read_text()
    )
    recipe = so101_behavior_cloning(str(receipt_root))
    manifest = local_fit_manifest(recipe, compile(recipe).state_dict())
    contract = recipe.interaction.environment.describe()
    assert fixture["trainable_tensors"] == manifest["tensors"]
    assert fixture["action"]["joints"] == list(contract.actions["action"].physical.joints)
    assert fixture["action"]["control_period_s"] == contract.control_period_s
    assert fixture["action"]["scale_rad"] == TARGET_SCALE
    assert fixture["isaaclab_revision"] == recipe.interaction.environment.revision


def test_scenario_is_simulator_and_lerobot_import_free(receipt_root):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'isaacsim', 'isaaclab', 'lerobot'}:
            raise AssertionError('optional native/data backend imported')
sys.meta_path.insert(0, Forbidden())
from nexuml_library.scenarios.imitation.so101 import so101_behavior_cloning
recipe = so101_behavior_cloning(sys.argv[1])
assert recipe.data.input_shapes == {'observation': [15]}
""",
            str(receipt_root),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def synthetic_lerobot(tmp_path, script):
    """Real v3 files containing invented test transitions; never simulator demonstrations."""
    cls = pytest.importorskip("lerobot.datasets.lerobot_dataset").LeRobotDataset
    root = tmp_path / "synthetic-test-only"
    writer = cls.create(
        repo_id=REPO_ID,
        root=root,
        fps=30,
        features=script._features(),
        robot_type="synthetic-test-only",
        use_videos=False,
        video_backend="pyav",
    )
    try:
        for episode in range(6):
            goal = np.asarray(GOALS[episode % 3], dtype=np.float32)
            observation = np.array([*HOME, 0, 0, 0, 0, 0, *goal], dtype=np.float32)
            following = np.array([*goal, 0, 0, 0, 0, 0, *goal], dtype=np.float32)
            for frame in range(8):
                writer.add_frame(
                    {
                        "observation": observation.copy(),
                        "next.observation": following.copy(),
                        "action": goal / TARGET_SCALE,
                        "task": "synthetic-test-only",
                        "next.joint_error_rad": np.zeros(1, dtype=np.float32),
                        "next.success": np.array([frame == 7], dtype=np.int64),
                        "next.terminated": np.array([frame == 7], dtype=np.int64),
                        "next.truncated": np.zeros(1, dtype=np.int64),
                    }
                )
                observation = following.copy()
            writer.save_episode()
    finally:
        writer.finalize()
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["provenance"] = _synthetic_receipt()
    path.write_text(json.dumps(info))
    return root


def _source(root, **overrides):
    return LeRobotDataSource.model_validate(
        {
            "root": str(root),
            "repo_id": REPO_ID,
            "input_keys": ("observation",),
            "split_episodes": _split_episodes(6),
            **overrides,
        }
    )


def _change_info(root, modify):
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    modify(info)
    path.write_text(json.dumps(info))


def _change_frame(root, index, key, value):
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    path = next(root.glob("data/*/*.parquet"))
    table = parquet.read_table(path)
    records = table.to_pylist()
    records[index][key] = value(records[index][key]) if callable(value) else value
    parquet.write_table(arrow.Table.from_pylist(records, schema=table.schema), path)


def test_validator_counts_hashes_and_model_inputs(synthetic_lerobot):
    source = _source(synthetic_lerobot)
    first = validate_so101_demonstrations(source)
    assert first == validate_so101_demonstrations(source)
    assert first["episodes"] == 6 and first["frames"] == 48
    assert first["train_episodes"] == 4 and first["train_windows"] == 32
    assert first["splits"]["val"] == {"episodes": 1, "windows": 8}
    assert len(first["dataset_sha256"]) == len(first["contract_sha256"]) == 64
    assert str(synthetic_lerobot) not in json.dumps(first)
    x, y = source.build()[0]
    assert set(x.keys()) == {"observation"} and set(y.keys()) == {"action"}


@pytest.mark.parametrize(
    "key", ["interaction_contract", "processor", "environment", "source_sha256"]
)
def test_validator_rejects_changed_physical_identity(synthetic_lerobot, key):
    _change_info(synthetic_lerobot, lambda info: info["provenance"].update({key: {}}))
    with pytest.raises(ValueError, match="contract|hashes"):
        validate_so101_demonstrations(_source(synthetic_lerobot))


@pytest.mark.parametrize("change", ["units", "joints", "asset_id"])
def test_validator_rejects_same_shape_different_physical_semantics(synthetic_lerobot, change):
    def alter(info):
        contract = info["provenance"]["interaction_contract"]
        if change == "asset_id":
            contract["physical"][change] = "different-asset"
        else:
            contract["actions"]["action"]["physical"][change].reverse()
            if change == "units":
                contract["actions"]["action"]["physical"][change] = ["degree"]

    _change_info(synthetic_lerobot, alter)
    with pytest.raises(ValueError, match="exact task contract"):
        validate_so101_demonstrations(_source(synthetic_lerobot))


@pytest.mark.parametrize(
    "selection",
    [
        {"input_keys": ("observation", "next.success")},
        {"input_keys": ("next.observation",)},
        {"action_key": "next.success"},
        {"split_episodes": {"train": (0, 1, 2, 3), "val": (5,), "test": (4,)}},
        {"split_episodes": {"train": (0, 1, 2, 3)}},
    ],
)
def test_validator_rejects_privileged_inputs_and_changed_splits(synthetic_lerobot, selection):
    with pytest.raises(ValueError, match="actor inputs|splits"):
        validate_so101_demonstrations(_source(synthetic_lerobot, **selection))


@pytest.mark.parametrize(
    ("index", "key", "value", "message"),
    [
        (0, "observation", [float("nan")] * 15, "finite"),
        (0, "action", [1.01] * 5, "within"),
        (0, "action", [0] * 5, "scripted expert"),
        (1, "observation", lambda x: [x[0] + 0.1, *x[1:]], "continuity"),
        (0, "next.observation", lambda x: [3, *x[1:]], "joint limits"),
        (0, "next.observation", lambda x: [*x[:5], 31, *x[6:]], "velocity limits"),
        (0, "next.joint_error_rad", -1.0, "joint error"),
        (0, "next.success", 1, "held success"),
        (7, "next.truncated", 1, "held success"),
        (7, "next.success", 0, "held success"),
        (0, "timestamp", 0.01, "timestamps"),
        (1, "frame_index", 0, "frame indices"),
    ],
)
def test_validator_rejects_invalid_or_incomplete_transitions(
    synthetic_lerobot,
    index,
    key,
    value,
    message,
):
    _change_frame(synthetic_lerobot, index, key, value)
    with pytest.raises(ValueError, match=message):
        validate_so101_demonstrations(_source(synthetic_lerobot))


def test_validator_rejects_timeout_as_success(synthetic_lerobot):
    for key, value in (("next.success", 0), ("next.terminated", 0), ("next.truncated", 1)):
        _change_frame(synthetic_lerobot, 7, key, value)
    with pytest.raises(ValueError, match="never timeout"):
        validate_so101_demonstrations(_source(synthetic_lerobot))


def test_validator_rejects_relabelled_joint_order(synthetic_lerobot):
    _change_info(synthetic_lerobot, lambda info: info["features"]["action"]["names"].reverse())
    with pytest.raises(ValueError, match="feature joint ordering"):
        validate_so101_demonstrations(_source(synthetic_lerobot))


def test_synthetic_recipe_materializes_without_native_allocation(synthetic_lerobot, monkeypatch):
    from nexuml.training.lightning import create_runtime_artifacts

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic CPU recipe test attempted native allocation")

    monkeypatch.setattr(IsaacLabEnvironment, "build", forbidden)
    scenario = so101_behavior_cloning(str(synthetic_lerobot))
    runtime = create_runtime_artifacts(scenario)
    runtime.data_module.setup("fit")
    x, y = next(iter(runtime.data_module.train_dataloader()))
    output, _ = runtime.pipeline(x, y)
    assert output["predicted_action"].shape[-1] == 5
    assert torch.isfinite(output["imitation_loss"]).all()
    output["imitation_loss"].mean().backward()
    assert all(parameter.grad is not None for parameter in runtime.pipeline.parameters())
    assert runtime.policy.pipeline is runtime.pipeline


def test_episode_records_pre_step_and_true_terminal_observation(script):
    """A test double checks transition plumbing, not native correctness."""
    initial = torch.tensor([*HOME, 0, 0, 0, 0, 0, *GOALS[1]], dtype=torch.float32)
    final = torch.tensor([*GOALS[1], 0, 0, 0, 0, 0, *GOALS[1]], dtype=torch.float32)
    reset_calls = []

    def reset(*, seed):
        reset_calls.append(seed)
        return TensorDict({"observation": initial.clone()}, batch_size=())

    def step(action):
        torch.testing.assert_close(action["action"], initial[10:15] / TARGET_SCALE)
        return InteractionStep(
            observation=TensorDict({"observation": final.clone()}, batch_size=()),
            reward=torch.zeros(1),
            terminated=torch.ones(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            info={
                "task_success": torch.ones(1, dtype=torch.bool),
                "joint_error_rad": torch.zeros(1),
            },
        )

    frames, outcome = script._episode(
        SimpleNamespace(reset=reset, step=step), seed=123, capture=True
    )
    assert reset_calls == [123] and outcome["success"]
    np.testing.assert_array_equal(frames[0]["observation"], initial.numpy())
    np.testing.assert_array_equal(frames[0]["next.observation"], final.numpy())
    assert frames[0]["next.terminated"].item() == 1


def test_episode_records_policy_contract_failure(script):
    initial = torch.tensor([*HOME, 0, 0, 0, 0, 0, *GOALS[1]], dtype=torch.float32)
    runtime = SimpleNamespace(reset=lambda **kwargs: TensorDict({"observation": initial}, []))

    def invalid(*args, **kwargs):
        raise ValueError("Action violates bounds")

    frames, outcome = script._episode(runtime, seed=0, policy=invalid)
    assert not frames and outcome["action_violations"] == 1
    assert not outcome["success"] and outcome["failure"] == "policy_contract_violation"


def test_repeated_episodes_keep_native_mutable_counters_outside_inference_mode(script):
    value = torch.tensor([*HOME, 0, 0, 0, 0, 0, *GOALS[0]], dtype=torch.float32)
    state = {"counter": torch.zeros(1)}

    def reset(**kwargs):
        state["counter"].zero_()
        return TensorDict({"observation": value.clone()}, [])

    def step(action):
        state["counter"] = torch.ones(1)
        return InteractionStep(
            observation=TensorDict({"observation": value.clone()}, []),
            reward=torch.zeros(1),
            terminated=torch.ones(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            info={
                "task_success": torch.ones(1, dtype=torch.bool),
                "joint_error_rad": torch.zeros(1),
            },
        )

    runtime = SimpleNamespace(reset=reset, step=step)
    for seed in (1, 2):
        _, outcome = script._episode(runtime, seed=seed)
        assert outcome["success"]


def test_clean_package_replay_plumbing_with_synthetic_runtime(
    synthetic_lerobot,
    tmp_path,
    monkeypatch,
    script,
):
    """CPU-only export/reload test double; not evidence of native replay or learned success."""
    from nexuml.core.export import export_package
    from nexuml.training.lightning import create_runtime_artifacts

    runtime = create_runtime_artifacts(so101_behavior_cloning(str(synthetic_lerobot)))
    package = export_package(runtime.policy, tmp_path / "policy")
    calls = []
    count = 0
    value = torch.tensor([*GOALS[1], 0, 0, 0, 0, 0, *GOALS[1]], dtype=torch.float32)

    def reset(*, seed):
        nonlocal count
        count = 0
        calls.append(seed)
        return TensorDict({"observation": value.clone()}, [])

    def step(action):
        nonlocal count
        count += 1
        return InteractionStep(
            observation=TensorDict({"observation": value.clone()}, []),
            reward=torch.zeros(1),
            terminated=torch.tensor([count == 8]),
            truncated=torch.tensor([False]),
            info={"task_success": torch.tensor([count == 8]), "joint_error_rad": torch.zeros(1)},
        )

    fake = SimpleNamespace(reset=reset, step=step, close=lambda: calls.append("closed"))
    monkeypatch.setattr(IsaacLabEnvironment, "build", lambda *args: fake)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *args, **kwargs: self.cpu())
    monkeypatch.setitem(script.evaluate.__globals__, "_packages", lambda: {})
    result = script.evaluate(package, episodes=2, seed=10000, reference="package")
    assert calls == [10000, 10001, "closed"]
    assert len(result["trials"]) == 2 and result["summary"]["episodes"] == 2
    assert len(result["package_sha256"]) == 64
    assert str(synthetic_lerobot) not in json.dumps(result)


def test_complete_episode_policy_inference_needs_no_data_or_training_imports(
    synthetic_lerobot, tmp_path
):
    from nexuml.core.export import export_package
    from nexuml.training.lightning import create_runtime_artifacts

    runtime = create_runtime_artifacts(so101_behavior_cloning(str(synthetic_lerobot)))
    package = export_package(runtime.policy, tmp_path / "portable")
    code = """
import importlib.abc, sys, torch
from tensordict import TensorDict
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('nexuml.training') or fullname.split('.')[0] in {
            'lightning', 'torchrl', 'gymnasium', 'isaacsim', 'isaaclab',
            'omni', 'lerobot', 'datasets'
        }:
            raise AssertionError('unexpected inference dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from nexuml.core.export import load_inference_package
policy, config, metadata = load_inference_package(sys.argv[1])
result = policy(TensorDict({'observation': torch.zeros(15)}, []))
assert result['action'].shape == (5,) and torch.isfinite(result['action']).all()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(package)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
