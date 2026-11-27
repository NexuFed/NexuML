"""Fixed PhysX Cartpole CLI recording, retaining native task/checkpoint parity."""

import copy
from pathlib import Path
import runpy
import sys


def run_recorded_isaac(record, root, env, stage, evaluation_metrics):
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.serialization import lower_model
    from nexuml.core.types import LoggingSpec, PolicyVideoSpec, TensorBoardSpec
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment

    if record["recipe"] != "isaac-cartpole-ppo":
        raise ValueError("fixed RTX replay requires the original PhysX Cartpole recipe")
    helpers = runpy.run_path(str(Path(__file__).with_name("physical_ai_mujoco_recording.py")))
    shared = runpy.run_path(str(Path(__file__).with_name("physical_ai_logging.py")))
    config = ResolvedConfig.load(root / "eval-case.yaml")
    if (
        config.interaction is None
        or not isinstance(config.interaction.environment, IsaacLabEnvironment)
        or config.interaction.environment.physics != "isaacsim_physx"
    ):
        raise AssertionError("RTX workflow must not substitute another native backend")
    if config.policy_evaluation is None:
        raise AssertionError("native recorded replay requires the scalar evaluation protocol")
    original = config.model_dump()
    recorded = config.model_copy(deep=True)
    assert recorded.policy_evaluation is not None and recorded.interaction is not None
    recorded.policy_evaluation.video = PolicyVideoSpec(
        enabled=True, format="both", max_frames=120, max_size=320
    )
    tb = root / "logs/isaac-recorded-tb"
    recorded.logging = LoggingSpec(
        tensorboard=TensorBoardSpec(log_dir=str(tb)),
        diagram=None,
        temp_artifact_dir=str(root / "logs/isaac-recorded-staging"),
    )
    expected = copy.deepcopy(original)
    expected["policy_evaluation"]["video"] = recorded.policy_evaluation.video.model_dump()
    expected["logging"] = recorded.logging.model_dump()
    if recorded.model_dump() != expected or config.model_dump() != original:
        raise AssertionError("recording changed the native scalar task/protocol")
    config_path = root / "isaac-recorded-eval.yaml"
    recorded.save(config_path)
    frames = record["assertions"]["training_frames"]
    policy_path = root / "checkpoint-policy"
    policy = shared["_policy_identity"](policy_path, frames)
    effective = ResolvedConfig.load(policy_path / "resolved_config.yaml").model_copy(deep=True)
    assert effective.interaction is not None
    effective.logging, effective.policy_evaluation = recorded.logging, recorded.policy_evaluation
    effective.interaction.evaluation_environment = (
        recorded.interaction.evaluation_environment or recorded.interaction.environment
    )
    result = stage(
        record,
        "isaac-recorded-evaluate",
        [
            sys.executable,
            "-m",
            "nexuml.cli.main",
            "evaluate",
            str(policy_path),
            "--config",
            str(config_path),
            "--device",
            "cuda:0",
        ],
        env,
    )
    metrics = evaluation_metrics(Path(result["log_path"]), expected_frames=frames)
    if metrics != record["metrics"]:
        raise AssertionError("rendered replay differs from the scalar native outcomes")
    if shared["_policy_identity"](policy_path, frames) != policy:
        raise AssertionError("rendered replay changed the exported policy")
    directories = list(tb.glob("**/artifacts/evaluation"))
    if len(directories) != 1:
        raise AssertionError("expected one native recorded evaluation output")
    media = helpers["_media"](
        directories[0],
        frames,
        policy,
        recorded.policy_evaluation.model_dump(mode="json"),
        metrics,
        lambda path: shared["_gif_info"](path, max_size=320),
        lower_model(effective),
        capture_frames=120,
        capture_size=320,
    )
    record["isaac_recording"] = {
        "media": media,
        "physics": "isaacsim_physx",
        "full_metric_parity": True,
        "scope": "native CLI GIF/MP4 replay, not live GUI or convergence",
    }
    record["checks"]["isaac_recorded_replay"] = "PASS"
