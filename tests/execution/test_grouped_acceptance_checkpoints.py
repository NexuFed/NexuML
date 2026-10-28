"""Hermetic acceptance assertions using synthetic tensor checkpoints, no model/runtime."""

from pathlib import Path
import runpy

import pytest
import torch


def checkpoints(tmp_path):
    def state(value, frames, step):
        return {
            "global_step": step,
            "state_dict": {
                "frames": torch.tensor(frames),
                "algorithm.updates": torch.tensor(step),
                "algorithm.actor_modules.shared.weight": torch.tensor([value]),
                "algorithm.critics.central.weight": torch.tensor([value]),
            },
            "optimizer_states": [{"state": {0: {"step": torch.tensor(step)}}} for _ in range(2)],
        }

    before, after = tmp_path / "partial.ckpt", tmp_path / "final.ckpt"
    torch.save(state(1.0, 32, 1), before)
    torch.save(state(2.0, 128, 4), after)
    return before, after


def validator():
    fixture = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_acceptance.py"
    return runpy.run_path(str(fixture))["_validate_grouped_continuation"]


def test_grouped_acceptance_requires_changed_actors_critics_and_populated_owners(tmp_path):
    before, after = checkpoints(tmp_path)
    result = validator()(before, after)
    assert result == {
        "grouped_owners": {"actor_modules": ["shared"], "critics": ["central"]},
        "optimizer_owners": 2,
        "all_actors_critics_changed_after_resume": True,
        "grouped_updates": 4,
    }


def test_grouped_acceptance_checks_every_owner_not_only_the_first(tmp_path):
    before, after = checkpoints(tmp_path)
    for path, value in ((before, 1.0), (after, 2.0)):
        raw = torch.load(path, weights_only=False)
        for kind in ("actor_modules", "critics"):
            raw["state_dict"][f"algorithm.{kind}.independent.weight"] = torch.tensor([value])
        raw["optimizer_states"] *= 2
        torch.save(raw, path)
    assert validator()(before, after)["optimizer_owners"] == 4
    raw = torch.load(after, weights_only=False)
    raw["state_dict"]["algorithm.actor_modules.independent.weight"] = torch.tensor([1.0])
    torch.save(raw, after)
    with pytest.raises(AssertionError, match="did not change finitely"):
        validator()(before, after)


@pytest.mark.parametrize(
    "defect", ["actor", "critic", "nan", "ownership", "optimizer", "updates", "frames"]
)
def test_grouped_acceptance_rejects_missing_update_or_ownership_proof(tmp_path, defect):
    before, after = checkpoints(tmp_path)
    raw = torch.load(after, weights_only=False)
    state = raw["state_dict"]
    if defect in {"actor", "critic"}:
        key = (
            "algorithm.actor_modules.shared.weight"
            if defect == "actor"
            else "algorithm.critics.central.weight"
        )
        state[key] = torch.tensor([1.0])
    elif defect == "nan":
        state["algorithm.critics.central.weight"] = torch.tensor([float("nan")])
    elif defect == "ownership":
        state["algorithm.critics.other.weight"] = state.pop("algorithm.critics.central.weight")
    elif defect == "optimizer":
        raw["optimizer_states"].pop()
    elif defect == "updates":
        state["algorithm.updates"] = torch.tensor(1)
    else:
        state["frames"] = torch.tensor(32)
    torch.save(raw, after)
    with pytest.raises(AssertionError):
        validator()(before, after)
