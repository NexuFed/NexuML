"""Worker-side frozen-resume validation, without loading a learner or simulator."""

import pytest

from nexuml.core.config import ResolvedConfig
from nexuml.core.types import ScenarioSpec
from nexuml.execution.ray_job import _validate_remote_resume


def test_matching_worker_resume_uses_original_checkpoint_path(monkeypatch, tmp_path):
    scenario = ScenarioSpec(name="frozen-resume")
    config = tmp_path / "request.yaml"
    ResolvedConfig.from_scenario(scenario).save(config)
    seen = []
    monkeypatch.setattr(
        "nexuml.training.lightning.load_scenario_from_trainer_checkpoint",
        lambda checkpoint, **kwargs: seen.append(checkpoint) or scenario,
    )
    _validate_remote_resume(config, "/worker only/saved ; checkpoint.ckpt")
    assert seen == ["/worker only/saved ; checkpoint.ckpt"]


@pytest.mark.parametrize(
    "field", ["pipeline", "training", "interaction", "policy", "policy_evaluation"]
)
def test_changed_worker_resume_semantics_fail_before_allocation(monkeypatch, tmp_path, field):
    scenario = ScenarioSpec(name="frozen-resume")
    saved = scenario.model_copy(update={field: object()})
    config = tmp_path / "request.yaml"
    ResolvedConfig.from_scenario(scenario).save(config)
    monkeypatch.setattr(
        "nexuml.training.lightning.load_scenario_from_trainer_checkpoint",
        lambda *_args, **_kwargs: saved,
    )
    monkeypatch.setattr(
        "nexuml.training.lightning.NexuSession",
        lambda *_args, **_kwargs: pytest.fail("constructed learner before resume admission"),
    )
    with pytest.raises(ValueError, match=f"unchanged checkpoint {field}"):
        _validate_remote_resume(config, "/worker/partial.ckpt")
