"""Fixed remote M1 recorded replay; GUI, driver and learning gates stay unverified.

Imports stay stdlib-only until the guarded parent calls the helper after scalar replay.
The parent owns admission before training, subprocess lifecycle and final acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys


def _sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON metadata key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON metadata: {value}")

    data = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    json.dumps(data, allow_nan=False)
    return data


def _check_metrics(metrics):
    required = (
        "episodes",
        "return_mean",
        "return_std",
        "length_mean",
        "terminated",
        "truncated",
        "native_truncated",
        "evaluation_horizon_limits",
    )
    if (
        not isinstance(metrics, dict)
        or any(f"eval/rl/{name}" not in metrics for name in required)
        or not all(
            type(value) in (int, float) and math.isfinite(value) for value in metrics.values()
        )
    ):
        raise AssertionError("missing or nonfinite native evaluation metrics")
    if metrics["eval/rl/episodes"] != 2 or metrics["eval/rl/length_mean"] <= 0:
        raise AssertionError("recorded replay requires two complete native episodes")
    for name in ("terminated", "truncated", "native_truncated", "evaluation_horizon_limits"):
        value = metrics[f"eval/rl/{name}"]
        if value != int(value) or not 0 <= value <= 2:
            raise AssertionError("invalid native episode ending count")
    if (
        metrics["eval/rl/evaluation_horizon_limits"] != 0
        or metrics["eval/rl/truncated"] != metrics["eval/rl/native_truncated"]
        or metrics["eval/rl/terminated"] + metrics["eval/rl/native_truncated"] < 2
        or metrics["eval/rl/return_std"] < 0
    ):
        raise AssertionError("capture must not replace native episode endings")


def _check_metadata(
    data,
    frames,
    policy,
    evaluation,
    metrics,
    effective_config,
    *,
    capture_frames=16,
    capture_size=128,
):
    if type(data.get("frame_step")) is not int or data["frame_step"] != frames:
        raise AssertionError("media frame step differs from actual policy frames")
    if (
        json.dumps(data.get("evaluation"), sort_keys=True) != json.dumps(evaluation, sort_keys=True)
        or data.get("seeds") != [10000, 10001]
        or any(type(seed) is not int for seed in data["seeds"])
        or data.get("formats") != ["gif", "mp4"]
    ):
        raise AssertionError("media evaluation/capture protocol differs")
    _check_metrics(data["metrics"])
    if data["metrics"] != metrics:
        raise AssertionError("media metrics differ from full scalar evaluation")
    count, capped = data["captured_frames"], data["capture_capped"]
    if (
        type(count) is not int
        or not 2 <= count <= capture_frames
        or type(capped) is not bool
        or (capped and count != capture_frames)
        or count > 2 * metrics["eval/rl/length_mean"] + 1
    ):
        raise AssertionError("invalid first-episode capture count/cap")
    source, captured = data["source_size"], data["captured_size"]
    if any(
        not isinstance(size, list)
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
        for size in (source, captured)
    ):
        raise AssertionError("invalid native source/captured dimensions")
    ratio = min(1, capture_size / max(source))
    if captured != [max(1, round(value * ratio)) for value in source]:
        raise AssertionError("capture does not preserve bounded native aspect ratio")
    if type(data["fps"]) not in (int, float) or not math.isfinite(data["fps"]) or data["fps"] <= 0:
        raise AssertionError("invalid native recording FPS")
    provenance = data["provenance"]
    expected = {
        "source_package": str(Path(policy["path"]).resolve()),
        "source_training_frames": frames,
        "source_policy_files_sha256": policy["policy_files_sha256"],
        "source_checkpoint": policy["checkpoint"],
        "source_config_hash": policy["config_hash"],
    }
    if (
        not expected.keys() <= provenance.keys()
        or any(provenance[key] != value for key, value in expected.items())
        or type(provenance["source_training_frames"]) is not int
        or not isinstance(provenance.get("resolved_config"), dict)
        or not provenance["resolved_config"]
    ):
        raise AssertionError("recorded provenance differs from actual exported policy")
    if json.dumps(provenance["resolved_config"], sort_keys=True, allow_nan=False) != json.dumps(
        effective_config, sort_keys=True, allow_nan=False
    ):
        raise AssertionError(
            "reported effective configuration differs from the policy and CLI inputs"
        )


def _media(
    directory,
    frames,
    policy,
    evaluation,
    metrics,
    gif_info,
    effective_config,
    *,
    capture_frames=16,
    capture_size=128,
):
    import imageio.v2 as imageio

    base = directory / f"frame_{frames:012d}"
    if {path.name for path in directory.iterdir()} != {
        f"{base.name}.{suffix}" for suffix in ("json", "gif", "mp4")
    } or any(not path.is_file() or path.is_symlink() for path in directory.iterdir()):
        raise AssertionError("native evaluation requires exactly one actual-frame JSON/GIF/MP4")
    data = _strict_json(base.with_suffix(".json"))
    _check_metadata(
        data,
        frames,
        policy,
        evaluation,
        metrics,
        effective_config,
        capture_frames=capture_frames,
        capture_size=capture_size,
    )
    gif = gif_info(base.with_suffix(".gif"))
    if gif["size"] != data["captured_size"] or not 2 <= gif["frames"] <= data["captured_frames"]:
        raise AssertionError("decoded GIF dimensions/count differ from metadata")
    size = [value + value % 2 for value in data["captured_size"]]
    hashes, nonuniform, decoded = set(), False, 0
    with imageio.get_reader(base.with_suffix(".mp4")) as reader:
        info = getattr(reader, "get_meta_data")()
        count = getattr(reader, "count_frames")()
        if info["codec"] != "h264" or count != data["captured_frames"]:
            raise AssertionError("MP4 codec/frame count differs from native capture")
        for frame in getattr(reader, "iter_data")():
            if len(frame.shape) != 3 or frame.shape[2] != 3 or list(frame.shape[1::-1]) != size:
                raise AssertionError("decoded MP4 does not have even-padded capture dimensions")
            hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
            nonuniform |= bool((frame.min(axis=(0, 1)) < frame.max(axis=(0, 1))).any())
            decoded += 1
            if decoded > data["captured_frames"]:
                raise AssertionError("MP4 decoded more frames than captured")
    if decoded != count or len(hashes) < 2 or not nonuniform:
        raise AssertionError("MP4 decoded frames are missing, uniform or nonchanging")
    return {
        "metadata": data,
        "gif": gif,
        "mp4": {
            "codec": info["codec"],
            "frames": decoded,
            "size": size,
            "changing_nonuniform_frames": True,
        },
        "files": {
            suffix: {
                "path": str(base.with_suffix(f".{suffix}")),
                "sha256": _sha256(base.with_suffix(f".{suffix}")),
            }
            for suffix in ("json", "gif", "mp4")
        },
        "clip_scope": "first episode only; metrics aggregate two full native episodes",
    }


def run_recorded_mujoco(record, root, env, stage, evaluation_metrics):
    """Replay the identified M1 policy through the public CLI, with no retries or fallback.

    Stage must wait and raise on failure. All exceptions propagate unchanged; partial
    evidence remains in the parent receipt, without setting any C/V/L/R/D gate.
    """
    record["checks"].pop("mujoco_recorded_replay", None)
    if record["recipe"] not in {"mujoco-humanoid-ppo", "mujoco-reacher-ppo"}:
        raise ValueError("recorded MuJoCo supports only the two fixed M1 recipes")
    if env.get("MUJOCO_GL") != "egl":
        raise ValueError("recorded MuJoCo requires explicitly selected MUJOCO_GL=egl")
    root = Path(root)
    config_path = root / "recorded-eval-case.yaml"
    tb_root = root / "logs/recorded-tb"
    artifacts = root / "logs/recorded-artifacts"
    for path in (config_path, tb_root, artifacts):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    from nexuml.core.config import ResolvedConfig
    from nexuml.core.serialization import lower_model
    from nexuml.core.types import LoggingSpec, PolicyVideoSpec

    shared = runpy.run_path(str(Path(__file__).with_name("physical_ai_logging.py")))
    config = ResolvedConfig.load(root / "eval-case.yaml")
    evaluation = config.policy_evaluation
    if config.name != record["recipe"] or evaluation is None:
        raise AssertionError("recorded replay must retain the scalar task and evaluation")
    protocol = evaluation.model_dump(mode="json")
    if {key: value for key, value in protocol.items() if key != "video"} != {
        "episodes": 2,
        "seed": 10000,
        "deterministic": True,
        "max_steps_per_episode": None,
        "at_start": False,
        "every_n_frames": None,
    } or protocol["video"]["enabled"] is not False:
        raise AssertionError("scalar evaluation must use the fixed full-native protocol")
    frames = record["assertions"]["training_frames"]
    if type(frames) is not int or frames <= 0:
        raise AssertionError("missing actual policy training frames")
    scalar = copy.deepcopy(record["metrics"])
    _check_metrics(scalar)
    policy_path = root / "checkpoint-policy"
    _strict_json(policy_path / "metadata.json")
    policy = shared["_policy_identity"](policy_path, frames)
    original = config.model_dump()
    clone = config.model_copy(deep=True)
    assert clone.policy_evaluation is not None
    clone.policy_evaluation.video = PolicyVideoSpec(
        enabled=True, format="both", max_frames=16, max_size=128
    )
    clone.logging = LoggingSpec.model_validate(
        {
            "experiment_name": "physical-ai-m1-recording",
            "tensorboard": {"log_dir": str(tb_root)},
            "temp_artifact_dir": str(artifacts),
            "diagram": None,
        }
    )
    expected = copy.deepcopy(original)
    expected["policy_evaluation"]["video"] = clone.policy_evaluation.video.model_dump()
    expected["logging"] = clone.logging.model_dump()
    if clone.model_dump() != expected or config.model_dump() != original:
        raise AssertionError("recording changed task/state/action/shape or native horizon")
    clone.save(config_path)
    if ResolvedConfig.load(config_path).model_dump() != expected:
        raise AssertionError("saved recording config changed immutable scalar contracts")
    evaluation = clone.policy_evaluation.model_dump(mode="json")
    source_config = policy_path / "resolved_config.yaml"
    effective = ResolvedConfig.load(source_config).model_copy(deep=True)
    if effective.interaction is None or clone.interaction is None:
        raise AssertionError(
            "policy and supplied evaluation require the original task configuration"
        )
    # Match the CLI's explicit overrides, including its separate evaluation environment.
    effective.logging = clone.logging
    effective.policy_evaluation = clone.policy_evaluation
    effective.interaction.evaluation_environment = (
        getattr(clone.interaction, "evaluation_environment", None) or clone.interaction.environment
    )
    effective_config = lower_model(effective)
    inputs = {
        str(path): _sha256(path) for path in (root / "eval-case.yaml", config_path, source_config)
    }
    evidence = {
        "source_policy": policy,
        "config": {"path": str(config_path), "sha256": inputs[str(config_path)]},
        "scalar_metrics": scalar,
        "expected_effective_config": effective_config,
        "native_gl_selector": {
            "MUJOCO_GL": "egl",
            "scope": "explicit requested selector only",
            "driver_runtime_vendor_qualification": "UNVERIFIED",
        },
        "scope": "recorded native replay only; GUI/browser and C/V/L/R/D gates unchanged",
    }
    record["mujoco_recording"] = evidence
    argv = [
        sys.executable,
        "-m",
        "nexuml.cli.main",
        "evaluate",
        str(policy_path),
        "--config",
        str(config_path),
        "--device",
        "cpu",
    ]
    evidence["command"] = {"argv": argv}
    result = stage(record, "mujoco-recorded-evaluate", argv, env)
    evidence["command"]["stage"] = result
    replay = evaluation_metrics(Path(result["log_path"]), expected_frames=frames)
    evidence["recorded_metrics"] = replay
    _check_metrics(replay)
    if replay != scalar:
        raise AssertionError("recorded CLI metrics differ from original scalar metrics")
    if shared["_policy_identity"](policy_path, frames) != policy or any(
        _sha256(path) != digest for path, digest in inputs.items()
    ):
        raise AssertionError("recorded replay mutated source policy or evaluation configs")
    directories = [path for path in tb_root.glob("**/artifacts/evaluation") if path.is_dir()]
    if len(directories) != 1 or not directories[0].resolve().is_relative_to(tb_root.resolve()):
        raise AssertionError("expected exactly one owned native TensorBoard evaluation directory")
    evidence["media"] = _media(
        directories[0], frames, policy, evaluation, scalar, shared["_gif_info"], effective_config
    )
    evidence["full_metric_parity"] = True
    record["checks"]["mujoco_recorded_replay"] = "PASS"
