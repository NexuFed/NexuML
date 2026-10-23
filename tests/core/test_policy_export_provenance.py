"""Export provenance uses saved physical counters, never a training budget or optimizer step."""

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nexuml.core.compiler import compile
from nexuml.core.export import export_package, load_inference_package, load_package
from nexuml.core.policy import CompiledPolicy
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.gymnasium import pendulum_ppo


@pytest.fixture
def policy(monkeypatch):
    pytest.importorskip("gymnasium")

    def forbidden(*args, **kwargs):
        pytest.fail("provenance must not start a simulator, learner or CUDA")

    monkeypatch.setattr(GymnasiumEnvironment, "build", forbidden)
    monkeypatch.setattr(PPO, "build", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    scenario = pendulum_ppo(total_frames=1024, frames_per_batch=32)
    scenario.logging = None
    return CompiledPolicy.from_pipeline(
        compile(scenario), scenario.interaction.environment.describe()
    )


@pytest.mark.parametrize("frames", [0, 64, 128])
@pytest.mark.parametrize("owner", ["module", "trainer"])
def test_policy_export_preserves_live_and_restored_physical_frames(policy, tmp_path, frames, owner):
    module = SimpleNamespace(policy=policy, frames=torch.tensor(frames))
    kwargs: dict[str, Any] = (
        {"lightning_module": module}
        if owner == "module"
        else {"trainer": SimpleNamespace(lightning_module=module, global_step=999)}
    )
    destination = export_package(policy, tmp_path / "policy", **kwargs)
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["training_frames"] == frames
    for loader in (load_package, load_inference_package):
        _, _, reloaded = loader(destination)
        assert reloaded["training_frames"] == frames
        assert reloaded["policy_files_sha256"] == metadata["policy_files_sha256"]
    assert set(metadata["policy_files_sha256"]) == {"state_dict.pt", "action_adapter_state.pt"}
    for filename, digest in metadata["policy_files_sha256"].items():
        with (destination / filename).open("rb") as source:
            assert digest == hashlib.file_digest(source, "sha256").hexdigest()
    assert not (destination / "lightning.ckpt").exists()
    assert not (destination / "training_state.pt").exists()
    assert metadata["training_state_available"] is False


def test_explicit_checkpoint_counter_and_identity_take_precedence(policy, tmp_path):
    checkpoint = tmp_path / "source with spaces.ckpt"
    torch.save(
        {
            "state_dict": {"frames": torch.tensor(256), "batches": torch.tensor(8)},
            "global_step": 999,
            "epoch": 3,
            "optimizer_states": [{"sensitive_learner_state": "not exported as actor state"}],
        },
        checkpoint,
    )
    destination = export_package(
        policy,
        tmp_path / "policy",
        lightning_module=SimpleNamespace(policy=policy, frames=torch.tensor(0)),
        checkpoint_path=checkpoint,
    )
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["training_frames"] == 256
    assert metadata["checkpoint"]["source"] == str(checkpoint)
    with checkpoint.open("rb") as source:
        assert metadata["checkpoint"]["sha256"] == hashlib.file_digest(source, "sha256").hexdigest()
    assert not (destination / "lightning.ckpt").exists()
    assert not (destination / "training_state.pt").exists()
    state = torch.load(destination / "state_dict.pt", weights_only=True)
    assert set(state) == set(policy.pipeline.state_dict())
    _, _, loaded = load_inference_package(destination)
    assert loaded["checkpoint"] == metadata["checkpoint"]


@pytest.mark.parametrize("source", ["absent", "unrelated_module", "old_checkpoint"])
def test_unknown_physical_frames_are_not_inferred_from_other_counters(policy, tmp_path, source):
    kwargs: dict[str, Any] = {}
    if source == "unrelated_module":
        kwargs["lightning_module"] = SimpleNamespace(policy=object(), frames=torch.tensor(512))
    if source == "old_checkpoint":
        checkpoint = tmp_path / "old.ckpt"
        torch.save({"state_dict": {}, "global_step": 999, "epoch": 4}, checkpoint)
        kwargs["checkpoint_path"] = checkpoint
    destination = export_package(policy, tmp_path / source, **kwargs)
    assert json.loads((destination / "metadata.json").read_text())["training_frames"] is None


@pytest.mark.parametrize("frames", [-1, True, 3.5, torch.tensor(3.5), torch.tensor([1, 2])])
def test_invalid_physical_counter_cannot_be_exported_as_provenance(policy, tmp_path, frames):
    with pytest.raises(ValueError, match="physical.*frames"):
        export_package(
            policy,
            tmp_path / "policy",
            lightning_module=SimpleNamespace(policy=policy, frames=frames),
        )
