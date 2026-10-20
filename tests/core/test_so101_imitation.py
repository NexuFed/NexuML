"""Synthetic CPU fixtures ONLY: schema/recipe tests, never native learning evidence."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
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
from nexuml_library.layers.head.regression_head import LatentRegressionHead
from nexuml_library.scenarios.imitation import so101 as so101_scenario
from nexuml_library.scenarios.imitation.so101 import (
    REPO_ID,
    SO101DemonstrationSource,
    TASK,
    _capture_provenance,
    _receipt,
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


@pytest.fixture(autouse=True)
def implementation_sources(monkeypatch):
    # Snapshot real files per test; parallel provider edits must not manufacture drift.
    current = _source_hashes()
    monkeypatch.setattr(so101_scenario, "_source_hashes", lambda: dict(current))
    return current


@pytest.fixture(scope="module")
def script():
    return SimpleNamespace(
        **runpy.run_path(
            str(Path(__file__).resolve().parents[2] / "scripts/isaaclab/so101_imitation.py")
        )
    )


def _synthetic_receipt(episodes=6):
    # Deliberately synthetic; matching receipt fields are NOT proof of native execution.
    packages = {
        name: {"version": "synthetic-test-only"}
        for name in ("nexuml", "nexuml-library", "torch", "lerobot", "isaaclab", "isaacsim")
    }
    packages["isaaclab"]["revision"] = IsaacLabEnvironment(task=TASK).revision
    return {
        "schema_version": 2,
        **_task_identity(),
        **_capture_provenance(packages, "synthetic-test-only"),
        "seed": 20,
        "episode_seeds": list(range(20, 20 + episodes)),
        "split_episodes": _split_episodes(episodes),
        "test_only": True,
    }


def _canonical_sha256(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _qualification(root, validation):
    receipt = json.loads((root / "meta/info.json").read_text())["provenance"]
    return {
        "schema_version": 1,
        "dataset_sha256": validation["dataset_sha256"],
        "capture_sha256": receipt["capture_sha256"],
        "semantic_sha256": _canonical_sha256(_task_identity()),
        "implementation_source_sha256": so101_scenario._source_hashes(),
        "isaaclab_revision": so101_scenario.IsaacLabEnvironment(task=TASK).revision,
        "checks": {
            name: {"passed": True, "evidence_sha256": _canonical_sha256(f"synthetic-only:{name}")}
            for name in ("runtime", "sdk", "native_control", "native_export_replay")
        },
    }


def _write_qualification(root, qualification):
    (root / "meta/runtime_qualification.json").write_text(json.dumps(qualification, indent=2))


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
    assert isinstance(dataset.source, SO101DemonstrationSource)
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
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())
    assert isinstance(restored.data.datasets[0].source, SO101DemonstrationSource)
    assert restored.data.datasets[0].source.component_name == "SO101Demonstrations"
    assert restored.data.datasets[0].source.component_version == "2"
    assert restored.to_scenario() == scenario


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
attempts = []
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'isaacsim', 'isaaclab', 'omni', 'pxr', 'lerobot', 'viser', 'torchrl'
        }:
            attempts.append(fullname)
            raise AssertionError('optional native/data backend imported')
sys.meta_path.insert(0, Forbidden())
from nexuml.core.config import ResolvedConfig
from nexuml_library.scenarios.imitation.so101 import (
    SO101DemonstrationSource, so101_behavior_cloning,
)
recipe = so101_behavior_cloning(sys.argv[1])
assert recipe.data.input_shapes == {'observation': [15]}
restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(recipe).to_yaml())
assert isinstance(restored.data.datasets[0].source, SO101DemonstrationSource)
assert restored.data.datasets[0].source.component_version == '2'
assert not attempts, attempts
""",
            str(receipt_root),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_identity_excludes_viewer_revision_and_implementation_factory(monkeypatch):
    expected = _task_identity()
    definition = IsaacLabEnvironment(task=TASK)
    selected = SimpleNamespace(
        task=definition.task,
        physics=definition.physics,
        revision="different-implementation-revision",
        viewer=object(),
        task_config=object(),
        describe=definition.describe,
    )
    monkeypatch.setattr(so101_scenario, "IsaacLabEnvironment", lambda **kwargs: selected)
    assert _task_identity() == expected
    assert expected["environment"] == {"task": TASK, "physics": "isaacsim_physx"}
    assert set(expected) == {
        "environment",
        "interaction_contract",
        "processor",
        "expert",
        "reset",
        "success",
    }


def test_capture_records_complete_projection_and_canonical_digest(implementation_sources):
    receipt = _synthetic_receipt()
    capture = receipt["capture_provenance"]
    assert receipt["schema_version"] == 2
    assert set(capture) == {"source_sha256", "packages", "python"}
    assert capture["source_sha256"] == implementation_sources
    assert set(capture["source_sha256"]) == {
        "isaaclab.py",
        "_isaaclab_so101.py",
        "_isaaclab_runtime.py",
        "_isaaclab_tasks.py",
        "isaaclab_config.py",
    }
    assert receipt["capture_sha256"] == _canonical_sha256(capture)
    assert not {"source_sha256", "packages", "python"} & set(receipt)
    info = {"total_episodes": 6, "provenance": receipt}
    assert _receipt(info) is receipt
    capture["additional_capture_context"] = {"device": "synthetic-only"}
    receipt["capture_sha256"] = _canonical_sha256(capture)
    assert _receipt(info) is receipt
    capture["additional_capture_context"]["device"] = "tampered"
    with pytest.raises(ValueError, match="capture.*digest"):
        _receipt(info)


@pytest.mark.parametrize(
    "modify",
    [
        pytest.param(lambda r: r.update(schema_version=1), id="old-schema"),
        pytest.param(lambda r: r.update(schema_version=True), id="boolean-schema"),
        pytest.param(lambda r: r.pop("capture_provenance"), id="missing-capture"),
        pytest.param(lambda r: r.pop("capture_sha256"), id="missing-capture-digest"),
        pytest.param(lambda r: r.update(capture_sha256="not-a-sha256"), id="bad-capture-digest"),
        pytest.param(
            lambda r: r["capture_provenance"]["source_sha256"].pop("isaaclab_config.py"),
            id="missing-source",
        ),
        pytest.param(
            lambda r: r["capture_provenance"]["source_sha256"].update(
                unexpected=_canonical_sha256("synthetic extra source")
            ),
            id="extra-source",
        ),
        pytest.param(
            lambda r: r["capture_provenance"]["source_sha256"].update(isaaclab="invalid"),
            id="bad-source-digest",
        ),
        pytest.param(
            lambda r: r["capture_provenance"]["packages"]["torch"].update(version="tampered"),
            id="tampered-package",
        ),
        pytest.param(
            lambda r: r["capture_provenance"].update(python="tampered"), id="tampered-python"
        ),
    ],
)
def test_receipt_rejects_missing_or_tampered_capture(modify):
    receipt = _synthetic_receipt()
    modify(receipt)
    with pytest.raises(ValueError, match="schema_version=2|capture|provenance"):
        _receipt({"total_episodes": 6, "provenance": receipt})


@pytest.mark.parametrize(
    "modify",
    [
        pytest.param(lambda c: c.pop("source_sha256"), id="missing-sources"),
        pytest.param(
            lambda c: c["source_sha256"].update({"isaaclab.py": "invalid"}), id="malformed-source"
        ),
        pytest.param(lambda c: c.pop("packages"), id="missing-packages"),
        pytest.param(lambda c: c["packages"].pop("isaacsim"), id="missing-package"),
        pytest.param(lambda c: c["packages"]["lerobot"].update(version=""), id="empty-version"),
        pytest.param(lambda c: c["packages"]["isaaclab"].pop("revision"), id="missing-revision"),
        pytest.param(
            lambda c: c["packages"]["isaaclab"].update(revision="invalid"), id="bad-revision"
        ),
        pytest.param(lambda c: c.pop("python"), id="missing-python"),
        pytest.param(lambda c: c.update(python=""), id="empty-python"),
    ],
)
def test_incomplete_capture_is_rejected_even_with_recomputed_digest(modify):
    receipt = _synthetic_receipt()
    modify(receipt["capture_provenance"])
    receipt["capture_sha256"] = _canonical_sha256(receipt["capture_provenance"])
    with pytest.raises(ValueError, match="capture|provenance"):
        _receipt({"total_episodes": 6, "provenance": receipt})


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


@pytest.fixture
def implementation_drift(synthetic_lerobot, monkeypatch, implementation_sources):
    source = _source(synthetic_lerobot)
    original = validate_so101_demonstrations(source)
    current = {
        **implementation_sources,
        "isaaclab.py": _canonical_sha256(
            [implementation_sources["isaaclab.py"], "viewer refactor"]
        ),
    }
    monkeypatch.setattr(so101_scenario, "_source_hashes", lambda: dict(current))
    return SimpleNamespace(
        root=synthetic_lerobot,
        source=source,
        original=original,
        qualification=_qualification(synthetic_lerobot, original),
    )


def test_validator_counts_hashes_and_model_inputs(synthetic_lerobot):
    source = _source(synthetic_lerobot)
    first = validate_so101_demonstrations(source)
    assert first == validate_so101_demonstrations(source)
    assert first["schema_version"] == 1  # Existing validation-result/SDK interface is unchanged.
    assert first["episodes"] == 6 and first["frames"] == 48
    assert first["train_episodes"] == 4 and first["train_windows"] == 32
    assert first["splits"]["val"] == {"episodes": 1, "windows": 8}
    assert len(first["dataset_sha256"]) == len(first["contract_sha256"]) == 64
    assert str(synthetic_lerobot) not in json.dumps(first)
    x, y = source.build()[0]
    assert set(x.keys()) == {"observation"} and set(y.keys()) == {"action"}


def test_public_validation_and_source_build_each_load_and_validate_once(
    synthetic_lerobot, monkeypatch
):
    source = so101_behavior_cloning(str(synthetic_lerobot)).data.datasets[0].source
    assert isinstance(source, LeRobotDataSource)
    original_build = LeRobotDataSource.build
    original_validate = so101_scenario._validate_demonstration_dataset
    loaded, validated = [], []

    def tracked_build(definition):
        dataset = original_build(definition)
        loaded.append(dataset)
        return dataset

    def tracked_validate(definition, dataset):
        validated.append(dataset)
        return original_validate(definition, dataset)

    monkeypatch.setattr(LeRobotDataSource, "build", tracked_build)
    monkeypatch.setattr(so101_scenario, "_validate_demonstration_dataset", tracked_validate)
    dataset = source.build()
    assert loaded == validated == [dataset]
    report = validate_so101_demonstrations(source)
    assert len(loaded) == len(validated) == 2
    assert all(first is second for first, second in zip(loaded, validated, strict=True))
    assert report["episodes"] == 6 and report["frames"] == len(dataset)


def test_restored_source_build_is_native_viewer_and_marl_import_free(
    synthetic_lerobot, implementation_sources
):
    code = """
import importlib.abc, json, sys
attempts = []
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'isaacsim', 'isaaclab', 'omni', 'pxr', 'viser', 'torchrl'}:
            attempts.append(fullname)
            raise AssertionError('unexpected optional runtime: ' + fullname)
sys.meta_path.insert(0, Forbidden())
from nexuml.core.config import ResolvedConfig
from nexuml_library.scenarios.imitation import so101
# Use the fixture's real source snapshot while other sessions edit the provider.
so101._source_hashes = lambda: json.loads(sys.argv[2])
recipe = so101.so101_behavior_cloning(sys.argv[1])
restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(recipe).to_yaml())
source = restored.data.datasets[0].source
assert isinstance(source, so101.SO101DemonstrationSource)
assert len(source.build()) == 48
assert so101.validate_so101_demonstrations(source)['frames'] == 48
assert not attempts, attempts
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(synthetic_lerobot), json.dumps(implementation_sources)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "key", ["interaction_contract", "processor", "environment", "expert", "reset", "success"]
)
def test_validator_rejects_changed_physical_identity(synthetic_lerobot, key):
    _change_info(synthetic_lerobot, lambda info: info["provenance"].update({key: {}}))
    with pytest.raises(ValueError, match="contract|hashes"):
        validate_so101_demonstrations(_source(synthetic_lerobot))


@pytest.mark.parametrize(
    "change", ["units", "joints", "asset_id", "scale", "control_period", "processor", "expert"]
)
def test_validator_rejects_same_shape_different_physical_semantics(implementation_drift, change):
    _write_qualification(implementation_drift.root, implementation_drift.qualification)

    def alter(info):
        contract = info["provenance"]["interaction_contract"]
        if change == "asset_id":
            contract["physical"][change] = "different-asset"
        elif change == "scale":
            contract["actions"]["action"]["physical"][change] = [0.25]
        elif change == "control_period":
            contract["control_period_s"] = 0.1
        elif change in ("processor", "expert"):
            info["provenance"][change]["version"] = "different-semantics"
        else:
            contract["actions"]["action"]["physical"][change].reverse()
            if change == "units":
                contract["actions"]["action"]["physical"][change] = ["degree"]

    _change_info(implementation_drift.root, alter)
    with pytest.raises(ValueError, match="exact task contract"):
        validate_so101_demonstrations(implementation_drift.source)


def test_implementation_refactor_requires_qualification_without_rewriting_data(
    implementation_drift,
):
    state = implementation_drift
    original_files = {
        path.relative_to(state.root): path.read_bytes()
        for path in state.root.rglob("*")
        if path.is_file()
    }
    info = json.loads((state.root / "meta/info.json").read_text())
    assert _receipt(info) == info["provenance"]
    with pytest.raises(ValueError, match="implementation drift.*qualification"):
        validate_so101_demonstrations(state.source)
    _write_qualification(state.root, state.qualification)
    assert validate_so101_demonstrations(state.source) == state.original
    # Sidecar bytes are not original dataset bytes and do not enter its content identity.
    (state.root / "meta/runtime_qualification.json").write_text(
        json.dumps(state.qualification, sort_keys=True, indent=4)
    )
    assert validate_so101_demonstrations(state.source) == state.original
    assert original_files == {
        path.relative_to(state.root): path.read_bytes()
        for path in state.root.rglob("*")
        if path.is_file() and path.name != "runtime_qualification.json"
    }


def test_recipe_source_build_rechecks_implementation_qualification(implementation_drift):
    state = implementation_drift
    recipe = so101_behavior_cloning(str(state.root))
    source = recipe.data.datasets[0].source
    assert isinstance(source, LeRobotDataSource)
    original_files = {
        path.relative_to(state.root): path.read_bytes()
        for path in state.root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="implementation drift.*qualification"):
        source.build()
    _write_qualification(state.root, state.qualification)
    assert len(source.build()) == 48
    assert validate_so101_demonstrations(source) == state.original
    assert original_files == {
        path.relative_to(state.root): path.read_bytes()
        for path in state.root.rglob("*")
        if path.is_file() and path.name != "runtime_qualification.json"
    }
    (state.root / "meta/runtime_qualification.json").unlink()
    with pytest.raises(ValueError, match="implementation drift.*qualification"):
        source.build()


@pytest.mark.parametrize("entry", ["session", "local_fit"])
def test_ordinary_and_sdk_fits_admit_only_qualified_recipe_source(
    implementation_drift, tmp_path, monkeypatch, entry
):
    from lightning import Trainer
    from nexuml.core.compiler import compile
    from nexuml.training import local_fit, local_fit_manifest
    from nexuml.training.lightning import NexuSession

    state = implementation_drift
    recipe = so101_behavior_cloning(str(state.root))
    recipe.training = recipe.training.model_copy(update={"max_epochs": 1, "max_steps": 1})
    initial = {name: value.clone() for name, value in compile(recipe).state_dict().items()}
    manifest = local_fit_manifest(recipe, initial)
    fit_calls = []
    original_fit = Trainer.fit

    def tracked_fit(trainer, *args, **kwargs):
        fit_calls.append(trainer)
        return original_fit(trainer, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", tracked_fit)

    def fit():
        if entry == "local_fit":
            return local_fit(
                trusted_recipe=recipe,
                initial_tensors=initial,
                model_manifest=manifest,
                local_data=recipe.data,
                budget={"steps": 1},
                seed=7,
            )
        return NexuSession(
            recipe,
            accelerator="cpu",
            devices=1,
            log_dir=tmp_path / "session",
            enable_loggers=False,
            enable_progress_bar=False,
        ).fit()

    with pytest.raises(ValueError, match="implementation drift.*qualification"):
        fit()
    assert not fit_calls
    _write_qualification(state.root, state.qualification)
    result = fit()
    assert len(fit_calls) == 1 and fit_calls[0].global_step == 1
    if entry == "local_fit":
        assert result.train_windows == 32 and result.episodes == 4
        assert result.metrics["optimizer_steps"] == 1
        assert any(
            not torch.equal(result.updated_tensors[name], value) for name, value in initial.items()
        )
    source = recipe.data.datasets[0].source
    assert isinstance(source, LeRobotDataSource)
    assert validate_so101_demonstrations(source) == state.original
    (state.root / "meta/runtime_qualification.json").unlink()
    with pytest.raises(ValueError, match="implementation drift.*qualification"):
        fit()
    assert len(fit_calls) == 1


@pytest.mark.parametrize("change", ["semantics", "capture", "frames"])
def test_recipe_source_build_rechecks_semantics_and_integrity(implementation_drift, change):
    state = implementation_drift
    source = so101_behavior_cloning(str(state.root)).data.datasets[0].source
    _write_qualification(state.root, state.qualification)
    assert len(source.build()) == 48
    if change == "semantics":
        _change_info(
            state.root,
            lambda i: i["provenance"]["interaction_contract"]["actions"]["action"][
                "physical"
            ].update(scale=[0.25]),
        )
        message = "exact task contract"
    elif change == "capture":
        _change_info(
            state.root, lambda i: i["provenance"]["capture_provenance"].update(python="tampered")
        )
        message = "capture.*digest"
    else:
        _change_frame(state.root, 0, "timestamp", 0.01)
        message = "timestamps"
    with pytest.raises(ValueError, match=message):
        source.build()


def test_selected_native_revision_drift_requires_qualification(synthetic_lerobot, monkeypatch):
    source = _source(synthetic_lerobot)
    original = validate_so101_demonstrations(source)
    definition = IsaacLabEnvironment(task=TASK)
    selected = SimpleNamespace(
        task=definition.task,
        physics=definition.physics,
        revision=sha256(b"synthetic selected native revision").hexdigest()[:40],
        describe=definition.describe,
    )
    monkeypatch.setattr(so101_scenario, "IsaacLabEnvironment", lambda **kwargs: selected)
    with pytest.raises(ValueError, match="implementation drift.*qualification"):
        validate_so101_demonstrations(source)
    _write_qualification(synthetic_lerobot, _qualification(synthetic_lerobot, original))
    assert validate_so101_demonstrations(source) == original


@pytest.mark.parametrize(
    "modify",
    [
        pytest.param(lambda q: q.update(schema_version=2), id="schema"),
        pytest.param(lambda q: q.update(schema_version=True), id="boolean-schema"),
        pytest.param(lambda q: q.pop("dataset_sha256"), id="missing-binding"),
        pytest.param(lambda q: q.update(source_allowlist=[]), id="no-allowlist"),
        pytest.param(lambda q: q.update(dataset_sha256=_canonical_sha256("other data")), id="data"),
        pytest.param(
            lambda q: q.update(capture_sha256=_canonical_sha256("other capture")), id="capture"
        ),
        pytest.param(
            lambda q: q.update(semantic_sha256=_canonical_sha256("other semantics")), id="semantics"
        ),
        pytest.param(
            lambda q: q["implementation_source_sha256"].update(
                {"isaaclab.py": _canonical_sha256("other implementation")}
            ),
            id="implementation",
        ),
        pytest.param(lambda q: q.update(isaaclab_revision="other revision"), id="native-revision"),
        pytest.param(lambda q: q["checks"].pop("native_export_replay"), id="missing-check"),
        pytest.param(lambda q: q["checks"].update(unknown={}), id="unknown-check"),
        pytest.param(lambda q: q["checks"]["sdk"].update(passed=False), id="failed-check"),
        pytest.param(lambda q: q["checks"]["sdk"].update(passed=1), id="nonboolean-check"),
        pytest.param(
            lambda q: q["checks"]["native_control"].pop("evidence_sha256"), id="missing-evidence"
        ),
        pytest.param(
            lambda q: q["checks"]["runtime"].update(evidence_sha256="short"), id="short-evidence"
        ),
        pytest.param(
            lambda q: q["checks"]["runtime"].update(evidence_sha256="x" * 64), id="nonhex-evidence"
        ),
        pytest.param(
            lambda q: q["checks"]["runtime"].update(evidence_sha256=None), id="absent-evidence"
        ),
    ],
)
def test_implementation_qualification_requires_exact_bindings_and_passed_evidence(
    implementation_drift, modify
):
    qualification = deepcopy(implementation_drift.qualification)
    modify(qualification)
    _write_qualification(implementation_drift.root, qualification)
    with pytest.raises(ValueError, match="qualification"):
        validate_so101_demonstrations(implementation_drift.source)


def test_unparseable_qualification_is_rejected(implementation_drift):
    (implementation_drift.root / "meta/runtime_qualification.json").write_text("{invalid")
    with pytest.raises(ValueError, match="qualification"):
        validate_so101_demonstrations(implementation_drift.source)


def test_qualification_does_not_waive_capture_or_split_integrity(implementation_drift):
    state = implementation_drift
    _write_qualification(state.root, state.qualification)
    path = state.root / "meta/info.json"
    original = path.read_text()
    _change_info(
        state.root, lambda i: i["provenance"]["capture_provenance"].update(python="tampered")
    )
    with pytest.raises(ValueError, match="capture.*digest"):
        validate_so101_demonstrations(state.source)
    path.write_text(original)
    _change_info(
        state.root,
        lambda i: i["provenance"]["split_episodes"].update(val=[5], test=[4]),
    )
    with pytest.raises(ValueError, match="splits"):
        validate_so101_demonstrations(state.source)


def test_qualification_binds_physically_valid_frame_bytes(implementation_drift):
    state = implementation_drift
    _write_qualification(state.root, state.qualification)
    for index, key in ((0, "next.observation"), (1, "observation")):
        _change_frame(state.root, index, key, lambda value: [*value[:5], 0.01, *value[6:]])
    with pytest.raises(ValueError, match="qualification.*identity"):
        validate_so101_demonstrations(state.source)


def test_qualification_binds_complete_original_receipt(implementation_drift):
    state = implementation_drift
    _write_qualification(state.root, state.qualification)
    _change_info(
        state.root, lambda i: i["provenance"].update(operator_note="changed after capture")
    )
    with pytest.raises(ValueError, match="qualification.*identity"):
        validate_so101_demonstrations(state.source)


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
