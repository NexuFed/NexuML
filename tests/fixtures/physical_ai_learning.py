"""Frozen L1 scalar-quality worker case; importing this fixture is stdlib-only.

The acceptance driver admits the worker and owns budgets/cleanup. Each injected
stage waits for CLI exit and raises on failure before the next native lifetime.
No viewer, recording, recovery or multiseed qualification is inferred here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import sys


RECIPE = "isaac-cartpole-learning-ppo"
FRAMES = {"untrained": 0, "early": 8_192, "intermediate": 1_048_576, "final": 2_097_152}
_METRICS = (
    "episodes",
    "length_mean",
    "return_mean",
    "terminated",
    "truncated",
    "native_truncated",
    "evaluation_horizon_limits",
)


def check_learning_metrics(metrics_by_stage):
    """Check the frozen scalar protocol, without importing any application code."""
    for label in FRAMES:
        if label not in metrics_by_stage:
            raise AssertionError(f"missing learning stage: {label}")
        metrics = metrics_by_stage[label]
        if any(f"eval/rl/{key}" not in metrics for key in _METRICS):
            raise AssertionError(f"{label}: missing required evaluation metrics")
        if not all(
            type(value) in (int, float) and math.isfinite(value) for value in metrics.values()
        ):
            raise AssertionError(f"{label}: nonfinite or nonnumeric evaluation metric")
        values = {key: metrics[f"eval/rl/{key}"] for key in _METRICS}
        if values["episodes"] != 32 or not 0 < values["length_mean"] <= 300:
            raise AssertionError(f"{label}: expected 32 episodes within native 300-step horizon")
        for key in ("terminated", "truncated", "native_truncated", "evaluation_horizon_limits"):
            value = values[key]
            if value != int(value) or not 0 <= value <= 32:
                raise AssertionError(f"{label}: invalid {key} count")
        if values["evaluation_horizon_limits"] != 0:
            raise AssertionError(f"{label}: evaluator step cap ended an episode")
        # Native termination and truncation may both be true on the same transition.
        if (
            values["truncated"] != values["native_truncated"]
            or values["terminated"] + values["native_truncated"] < 32
        ):
            raise AssertionError(f"{label}: inconsistent native ending counts")
    final = metrics_by_stage["final"]
    assertions = {
        "mean_return_at_least_4": final["eval/rl/return_mean"] >= 4.0,
        "mean_length_at_least_285": final["eval/rl/length_mean"] >= 285,
        "native_early_terminations_at_most_1": final["eval/rl/terminated"] <= 1,
        "improves_over_matched_untrained": (
            final["eval/rl/return_mean"] > metrics_by_stage["untrained"]["eval/rl/return_mean"]
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(
            "frozen L1 quality failed: "
            + ", ".join(key for key, passed in assertions.items() if not passed)
        )
    return assertions


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _assert_defaults(config):
    training = config.training
    evaluation = training.evaluation
    environment = config.interaction.environment
    if (
        config.name != RECIPE
        or type(training.seed) is not int
        or training.seed != 0
        or training.total_frames != FRAMES["final"]
        or training.frames_per_batch != FRAMES["early"]
        or training.collector.num_collectors != 1
        or training.collector.num_envs_per_collector != 512
        or environment.physics != "newton_mjwarp"
        or environment.viewer is not None
        or evaluation.episodes != 0
        or evaluation.at_start
        or evaluation.every_n_frames is not None
        or evaluation.video.enabled
        or config.policy_evaluation is not None
    ):
        raise AssertionError("canonical L1 defaults changed; explicit requalification required")


def run_learning(
    record, root, env, prepare_recipe, stage, checkpoint, validate_policy, evaluation_metrics
):
    """Run only on the guarded worker, using serial public CLI native lifetimes."""
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.scenario_registry import get_scenario_registry
    from nexuml.core.types import PolicyEvaluationSpec

    root = Path(root)
    record["gates"]["L"] = "UNVERIFIED"
    record["gate_scope"]["L"] = (
        "L1 frozen single-training-seed scalar quality only; "
        "no multiseed, camera-policy, cross-backend, viewer or recording claim"
    )
    canonical = ResolvedConfig.from_scenario(get_scenario_registry().get(RECIPE)())
    _assert_defaults(canonical)
    canonical.save(root / "learning-input.yaml")
    config = prepare_recipe(root / "learning-input.yaml", root, {"recipe": RECIPE})
    _assert_defaults(config)
    heldout = config.model_copy(deep=True)
    heldout.policy_evaluation = PolicyEvaluationSpec.model_validate(
        {
            "episodes": 32,
            "seed": 10000,
            "deterministic": True,
            "max_steps_per_episode": None,
            "video": {"enabled": False},
            "at_start": False,
            "every_n_frames": None,
        }
    )
    heldout.save(root / "heldout-config.yaml")
    protocol = {
        "recipe": RECIPE,
        "training_seed": 0,
        "training_seed_count": 1,
        "total_frames": FRAMES["final"],
        "frames_per_batch": FRAMES["early"],
        "num_envs": 512,
        "physics": "newton_mjwarp",
        "checkpoint_frames": {key: value for key, value in FRAMES.items() if value},
        "heldout_seed": 10000,
        "heldout_last_seed": 10031,
        "heldout_episodes": 32,
        "deterministic": True,
        "native_horizon_steps": 300,
        "native_horizon_seconds": 5,
        "max_steps_per_episode": None,
        "video_enabled": False,
        "thresholds": {
            "return_mean_min": 4.0,
            "length_mean_min": 285,
            "early_terminated_max": 1,
            "improvement_over_untrained": "strictly_greater",
        },
    }
    protocol_path = root / "learning-protocol.json"
    with protocol_path.open("x") as output:
        json.dump(protocol, output, indent=2, allow_nan=False)
        output.write("\n")
    protocol_path.chmod(0o400)
    learning: dict = {
        "protocol": protocol,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "input_config_sha256": _sha256(root / "learning-input.yaml"),
        "training_config_sha256": _sha256(root / "case.yaml"),
        "heldout_config_sha256": _sha256(root / "heldout-config.yaml"),
        "checkpoints": {},
        "metrics": {},
        "assertions": {},
    }
    record["learning"] = learning
    cli = [sys.executable, "-m", "nexuml.cli.main"]
    case = ["--config", str(root / "case.yaml")]
    stage(
        record,
        "learning-untrained-export",
        [
            *cli,
            "export",
            *case,
            "--output",
            str(root / "untrained"),
        ],
        env,
    )
    stage(record, "learning-train", [*cli, "train", *case], env)

    # Read saved counters, never infer policy identity from filenames or relabel a final export.
    selected = {}
    for path in sorted((root / "checkpoints").rglob("*.ckpt")):
        _, frames, step = checkpoint(path)
        if type(frames) is not int or type(step) is not int or frames < 0 or step < 0:
            raise AssertionError("checkpoint lacks valid physical-frame/global-step counters")
        selected.setdefault(frames, (path, step))
    stages = root / "stages"
    stages.mkdir()
    previous_step = 0
    for label, frames in protocol["checkpoint_frames"].items():
        if frames not in selected:
            raise AssertionError(f"missing {label} checkpoint at {frames} physical frames")
        source, step = selected[frames]
        if step <= previous_step:
            raise AssertionError("selected checkpoint global steps must be positive and advance")
        destination = stages / f"{label}.ckpt"
        with source.open("rb") as incoming, destination.open("xb") as output:
            shutil.copyfileobj(incoming, output)
        destination.chmod(0o400)
        if checkpoint(destination)[1:] != (frames, step):
            raise AssertionError(
                "immutable checkpoint copy changed physical-frame/global-step identity"
            )
        learning["checkpoints"][label] = {
            "path": str(destination),
            "source": str(source),
            "frames": frames,
            "global_step": step,
            "sha256": _sha256(destination),
        }
        previous_step = step

    policies = {"untrained": root / "untrained"}
    for label, evidence in learning["checkpoints"].items():
        policies[label] = root / ("checkpoint-policy" if label == "final" else f"{label}-policy")
        stage(
            record,
            f"learning-{label}-export",
            [
                *cli,
                "export",
                *case,
                "--checkpoint",
                evidence["path"],
                "--output",
                str(policies[label]),
            ],
            env,
        )
    final = Path(learning["checkpoints"]["final"]["path"])
    learning["assertions"].update(validate_policy(root, final, FRAMES["final"]))
    metrics_by_stage = {}
    for label, frames in FRAMES.items():
        name = f"learning-{label}-evaluate"
        stage(
            record,
            name,
            [
                *cli,
                "evaluate",
                str(policies[label]),
                "--config",
                str(root / "heldout-config.yaml"),
                "--device",
                "cuda:0",
            ],
            env,
        )
        metrics_by_stage[label] = evaluation_metrics(root / f"{name}.log", expected_frames=frames)
        learning["metrics"][label] = {
            "source_frames": frames,
            "policy": str(policies[label]),
            "metrics": metrics_by_stage[label],
        }
    learning["assertions"].update(check_learning_metrics(metrics_by_stage))
    record["gates"]["L"] = "PASS"
