"""Fixed remote CLI logger producer; browser and full recipe gates stay unverified.

Imports are stdlib-only until the guarded worker calls run_logging. The parent
owns worker admission, deadlines, subprocess cleanup and the final receipt.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
from typing import cast


MODES = ("tb-only", "mlflow-only", "both", "disabled")
FRAMES = {"untrained": 0, "intermediate": 32, "final": 64}


def _sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def _logging_settings(mode, root):
    if mode not in MODES:
        raise ValueError("unknown fixed logging mode")
    if mode == "disabled":
        return None
    return {
        "tensorboard": {"log_dir": str(root / "logs/tb")} if mode in {"tb-only", "both"} else None,
        "mlflow": {
            "tracking_uri": f"sqlite:///{root / 'logs/mlflow.db'}",
            "artifact_location": (root / "logs/mlflow-artifacts").as_uri(),
            "auto_nested_runs": False,
            "experiment_name": f"physical-ai-logging-{mode}",
        }
        if mode in {"mlflow-only", "both"}
        else None,
        "dvclive": None,
        "diagram": None,
        "temp_artifact_dir": str(root / "staging"),
        "experiment_name": f"physical-ai-logging-{mode}",
    }


def _prepare_mode(mode, root, prepare_recipe):
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.scenario_registry import get_scenario_registry
    from nexuml.core.types import LoggingSpec, PolicyEvaluationSpec, ReinforcementLearningSpec
    from nexuml_library.reinforcement.algorithms.ppo import PPO

    scenario = get_scenario_registry().get("cartpole-ppo")()
    assert isinstance(scenario.training, ReinforcementLearningSpec)
    scenario.training.total_frames = 64
    scenario.training.frames_per_batch = 32
    scenario.training.seed = 0
    scenario.training.accelerator = "cpu"
    scenario.training.devices = 1
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=16)
    scenario.training.evaluation = PolicyEvaluationSpec.model_validate(
        {
            "episodes": 2,
            "seed": 10000,
            "deterministic": True,
            "max_steps_per_episode": None,
            "at_start": True,
            "every_n_frames": 32,
            "video": {
                "enabled": mode != "disabled",
                "format": "both",
                "max_frames": 16,
                "max_size": 128,
            },
        }
    )
    scenario.policy_evaluation = None
    canonical = ResolvedConfig.from_scenario(scenario)
    canonical.save(root / "logging-input.yaml")
    config = prepare_recipe(root / "logging-input.yaml", root, {"recipe": "cartpole-ppo"})
    # prepare_recipe rewrites tracking URIs: override only after it sets owned paths.
    settings = _logging_settings(mode, root)
    config.logging = None if settings is None else LoggingSpec.model_validate(settings)
    config.save(root / "case.yaml")
    replay = config.model_copy(deep=True)
    replay.policy_evaluation = PolicyEvaluationSpec.model_validate(
        {**config.training.evaluation.model_dump(), "at_start": False, "every_n_frames": None}
    )
    replay.save(root / "eval-case.yaml")
    replay.policy_evaluation.video.enabled = False
    replay.save(root / "scalar-case.yaml")
    return {
        "logging": None if config.logging is None else config.logging.model_dump(mode="json"),
        "configs": {
            name: {"path": str(root / name), "sha256": _sha256(root / name)}
            for name in ("logging-input.yaml", "case.yaml", "eval-case.yaml", "scalar-case.yaml")
        },
    }


def _copy_checkpoints(root, checkpoint):
    selected = {}
    for path in sorted((root / "checkpoints").glob("*.ckpt")):
        _, frames, step = checkpoint(path)
        if type(frames) is not int or type(step) is not int or frames < 0 or step < 0:
            raise AssertionError("invalid checkpoint physical-frame/global-step counters")
        selected.setdefault(frames, (path, step))
    saved = root / "selected-checkpoints"
    saved.mkdir()
    result, previous = {}, 0
    for label, frames in (("intermediate", 32), ("final", 64)):
        if frames not in selected:
            raise AssertionError(f"missing checkpoint at {frames} physical frames")
        source, step = selected[frames]
        if step <= previous:
            raise AssertionError("checkpoint global steps must be positive and advance")
        destination = saved / f"{label}.ckpt"
        with source.open("rb") as incoming, destination.open("xb") as output:
            shutil.copyfileobj(incoming, output)
        destination.chmod(0o400)
        if checkpoint(destination)[1:] != (frames, step) or _sha256(source) != _sha256(destination):
            raise AssertionError("immutable checkpoint copy differs from selected source")
        result[label] = {
            "path": str(destination),
            "source": str(source),
            "frames": frames,
            "global_step": step,
            "sha256": _sha256(destination),
        }
        previous = step
    return result


def _policy_identity(policy, frames, saved=None):
    metadata = json.loads((policy / "metadata.json").read_text())
    json.dumps(metadata, allow_nan=False)  # Invalid provenance must not enter the case receipt.
    hashes = {name: _sha256(policy / name) for name in ("state_dict.pt", "action_adapter_state.pt")}
    if (
        metadata.get("artifact_kind") != "policy"
        or type(metadata.get("training_frames")) is not int
        or metadata["training_frames"] != frames
        or metadata.get("policy_files_sha256") != hashes
    ):
        raise AssertionError("policy metadata lacks exact source frames/actor file hashes")
    if saved is not None and metadata.get("checkpoint", {}).get("sha256") != saved["sha256"]:
        raise AssertionError("policy metadata does not identify the selected checkpoint")
    return {
        "path": str(policy),
        "source_frames": frames,
        "metadata_path": str(policy / "metadata.json"),
        "metadata_sha256": _sha256(policy / "metadata.json"),
        "policy_files_sha256": hashes,
        "checkpoint": metadata.get("checkpoint"),
        "config_hash": metadata.get("config_hash"),
        "config_hash_availability": "recorded" if metadata.get("config_hash") else "UNVERIFIED",
    }


def _check_metrics(metrics):
    required = (
        "episodes",
        "length_mean",
        "return_mean",
        "terminated",
        "truncated",
        "native_truncated",
        "evaluation_horizon_limits",
    )
    if any(f"eval/rl/{name}" not in metrics for name in required) or not all(
        type(value) in (int, float) and math.isfinite(value) for value in metrics.values()
    ):
        raise AssertionError("missing or nonfinite evaluation metrics")
    if metrics["eval/rl/episodes"] != 2 or not 0 < metrics["eval/rl/length_mean"] <= 500:
        raise AssertionError("evaluation must complete two native CartPole episodes")
    for name in ("terminated", "truncated", "native_truncated", "evaluation_horizon_limits"):
        value = metrics[f"eval/rl/{name}"]
        if value != int(value) or not 0 <= value <= 2:
            raise AssertionError("invalid episode ending count")
    if (
        metrics["eval/rl/evaluation_horizon_limits"] != 0
        or metrics["eval/rl/truncated"] != metrics["eval/rl/native_truncated"]
        or metrics["eval/rl/terminated"] + metrics["eval/rl/native_truncated"] < 2
    ):
        raise AssertionError("capture must not truncate native metric episodes")


def _check_metadata(path, data, frames, policy=None):
    if (
        Path(path).name != f"frame_{frames:012d}.json"
        or type(data.get("frame_step")) is not int
        or data["frame_step"] != frames
    ):
        raise AssertionError("media filename/frame step differs from source frames")
    evaluation = data["evaluation"]
    video = evaluation["video"]
    if not {"max_steps_per_episode", "every_n_frames"} <= evaluation.keys():
        raise AssertionError("evaluation horizon/schedule must be explicitly recorded")
    if (
        evaluation.get("episodes") != 2
        or evaluation.get("seed") != 10000
        or evaluation.get("deterministic") is not True
        or evaluation.get("max_steps_per_episode") is not None
        or evaluation.get("at_start") is not (policy is None)
        or evaluation.get("every_n_frames") != (32 if policy is None else None)
        or data.get("seeds") != [10000, 10001]
        or video.get("enabled") is not True
        or video.get("format") != "both"
        or video.get("max_frames") != 16
        or video.get("max_size") != 128
        or data.get("formats") != ["gif", "mp4"]
    ):
        raise AssertionError("media evaluation/capture protocol differs")
    _check_metrics(data["metrics"])
    count, capped = data["captured_frames"], data["capture_capped"]
    if (
        type(count) is not int
        or not 2 <= count <= 16
        or type(capped) is not bool
        or (capped and count != 16)
        or count > 2 * data["metrics"]["eval/rl/length_mean"] + 1
    ):
        raise AssertionError("invalid first-episode capture count/cap")
    source, captured = data["source_size"], data["captured_size"]
    if any(
        not isinstance(size, list)
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
        for size in (source, captured)
    ) or any(c > min(128, s) for c, s in zip(captured, source, strict=True)):
        raise AssertionError("invalid source/captured media bounds")
    ratio = min(1, 128 / max(source))
    if captured != [max(1, round(value * ratio)) for value in source]:
        raise AssertionError("capture dimensions do not preserve bounded source aspect ratio")
    if type(data["fps"]) not in (int, float) or not math.isfinite(data["fps"]) or data["fps"] <= 0:
        raise AssertionError("missing native recording timing")
    if policy is not None:
        provenance = data["provenance"]
        if (
            not {"source_training_frames", "source_checkpoint", "source_config_hash"}
            <= provenance.keys()
        ):
            raise AssertionError("fresh replay source provenance fields missing")
        if (
            provenance.get("source_package") != str(Path(policy["path"]).resolve())
            or type(provenance.get("source_training_frames")) is not int
            or provenance["source_training_frames"] != frames
            or provenance.get("source_policy_files_sha256") != policy["policy_files_sha256"]
            or provenance.get("source_checkpoint") != policy["checkpoint"]
            or provenance.get("source_config_hash") != policy["config_hash"]
            or not isinstance(provenance.get("resolved_config"), dict)
            or not provenance["resolved_config"]
        ):
            raise AssertionError("fresh replay provenance differs from actual exported policy")
    return {
        "frame_step": frames,
        "metrics": data["metrics"],
        "seeds": data["seeds"],
        "evaluation": evaluation,
        "captured_frames": count,
        "capture_capped": capped,
        "source_size": source,
        "captured_size": captured,
        "fps": data["fps"],
        "clip_scope": "first episode only; metrics aggregate both episodes",
        "capture_semantics": "reset plus transitions; terminal included only within capture limit",
        "native_reset_terminal_scene_identity": "UNVERIFIED (requires scene oracle)",
        "provenance": data.get("provenance", {}),
        "in_fit_file_hash_provenance": (
            "not supplied" if not data.get("provenance") else "UNVERIFIED; raw provenance retained"
        )
        if policy is None
        else "not applicable",
    }


def _gif_info(source, *, max_size=128):
    from PIL import Image

    with Image.open(source) as image:
        count, hashes, nonuniform = getattr(image, "n_frames", 1), set(), False
        if image.format != "GIF" or count <= 1 or max(image.size) > max_size:
            raise AssertionError("native preview/artifact must be a bounded animated GIF")
        for index in range(count):
            image.seek(index)
            rgb = image.convert("RGB")
            hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())
            extrema = cast(tuple[tuple[int, int], ...], rgb.getextrema())
            nonuniform |= any(low < high for low, high in extrema)
        if len(hashes) < 2 or not nonuniform:
            raise AssertionError("GIF has uniform or nonchanging decoded frames")
        return {"frames": count, "size": list(image.size), "changing_nonuniform_frames": True}


def _check_media(directory, frames, policy=None):
    import imageio.v2 as imageio

    expected = {
        f"frame_{frame:012d}.{suffix}" for frame in frames for suffix in ("json", "gif", "mp4")
    }
    if {path.name for path in directory.iterdir()} != expected:
        raise AssertionError("native run must contain exactly one JSON/GIF/MP4 per expected frame")
    result = {}
    for frame in frames:
        base = directory / f"frame_{frame:012d}"
        data = json.loads(base.with_suffix(".json").read_text())
        entry = _check_metadata(base.with_suffix(".json"), data, frame, policy)
        entry["gif"] = _gif_info(base.with_suffix(".gif"))
        if (
            entry["gif"]["size"] != data["captured_size"]
            or entry["gif"]["frames"] > data["captured_frames"]
        ):
            raise AssertionError("GIF dimensions/frames differ from capture metadata")
        with imageio.get_reader(base.with_suffix(".mp4")) as reader:
            count = getattr(reader, "count_frames")()
            info = getattr(reader, "get_meta_data")()
            shape = getattr(reader, "get_data")(0).shape
        expected_size = [value + value % 2 for value in data["captured_size"]]
        if (
            count != data["captured_frames"]
            or info["codec"] != "h264"
            or list(shape[1::-1]) != expected_size
        ):
            raise AssertionError("MP4 codec/frames/padded dimensions differ from capture")
        entry["mp4"] = {"frames": count, "codec": info["codec"], "size": expected_size}
        entry["files"] = {
            suffix: {
                "path": str(base.with_suffix(f".{suffix}")),
                "sha256": _sha256(base.with_suffix(f".{suffix}")),
            }
            for suffix in ("json", "gif", "mp4")
        }
        result[str(frame)] = entry
    return result


def _tensorboard_runs(root):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runs = []
    directories = sorted({path.parent for path in (root / "logs/tb").rglob("*tfevents*")})
    for directory in directories:
        events = EventAccumulator(
            str(directory), size_guidance={"images": 0, "scalars": 0}
        ).Reload()
        tags = events.Tags()
        images = events.Images("eval/rl/episode") if "eval/rl/episode" in tags["images"] else []
        previews = []
        for event in images:
            previews.append(
                {
                    "step": event.step,
                    "sha256": hashlib.sha256(event.encoded_image_string).hexdigest(),
                    "decoded": _gif_info(io.BytesIO(event.encoded_image_string)),
                }
            )
        artifact = directory / "artifacts/evaluation"
        runs.append(
            {
                "log_dir": str(directory),
                "event_files": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in sorted(directory.glob("*tfevents*"))
                ],
                "scalar_history": {
                    tag: [{"step": e.step, "value": e.value} for e in events.Scalars(tag)]
                    for tag in tags["scalars"]
                },
                "image_tag": "eval/rl/episode",
                "previews": previews,
                "evaluation_dir": str(artifact) if artifact.is_dir() else None,
                "status": "UNVERIFIED (TensorBoard has no native run terminal status)",
            }
        )
    if not runs:
        raise AssertionError("configured TensorBoard produced no native event files")
    return runs


def _mlflow_runs(root, settings):
    from mlflow.tracking import MlflowClient

    if not (root / "logs/mlflow.db").is_file():
        raise AssertionError("configured MLflow produced no local SQLite store")
    client = MlflowClient(tracking_uri=settings["tracking_uri"])
    experiment = client.get_experiment_by_name(settings["experiment_name"])
    if experiment is None:
        raise AssertionError("configured native MLflow experiment missing")
    runs, token = [], None
    while True:
        page = client.search_runs([experiment.experiment_id], max_results=100, page_token=token)
        for run in page:
            mirror = root / "mlflow-downloads" / run.info.run_id
            mirror.mkdir(parents=True)
            artifacts = client.list_artifacts(run.info.run_id)
            evaluation = None
            for artifact in artifacts:
                if artifact.path == "evaluation" and artifact.is_dir:
                    evaluation = client.download_artifacts(
                        run.info.run_id, "evaluation", str(mirror)
                    )
            runs.append(
                {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "artifact_uri": run.info.artifact_uri,
                    "artifact_paths": [item.path for item in artifacts],
                    "evaluation_dir": evaluation,
                    "scalar_history": {
                        key: [
                            {"step": e.step, "value": e.value}
                            for e in client.get_metric_history(run.info.run_id, key)
                        ]
                        for key in run.data.metrics
                    },
                }
            )
        token = page.token
        if not token:
            break
    if not runs:
        raise AssertionError("configured MLflow produced no native runs")
    return runs


def _check_run_scalars(run, *, tensorboard):
    for name in next(iter(run["media"].values()))["metrics"]:
        events = run["scalar_history"].get(name, [])
        expected = {int(frame): payload["metrics"][name] for frame, payload in run["media"].items()}
        if sorted(event["step"] for event in events) != sorted(expected) or any(
            not math.isclose(
                event["value"],
                expected[event["step"]],
                rel_tol=1e-6 if tensorboard else 0,
                abs_tol=1e-6 if tensorboard else 0,
            )
            for event in events
        ):
            raise AssertionError("native evaluation scalar values/physical-frame steps differ")


def _classify_runs(runs, policies, metrics, *, tensorboard=False):
    main, fresh = [], {label: [] for label in FRAMES}
    for run in runs:
        directory = run["evaluation_dir"]
        if directory is None:
            run["role"] = "additional configured run (export or scalar-only; no media)"
            continue
        directory = Path(directory)
        names = {path.name for path in directory.glob("*.json")}
        if names == {f"frame_{frame:012d}.json" for frame in FRAMES.values()}:
            run["role"] = "main training; in-fit evaluations"
            run["media"] = _check_media(directory, list(FRAMES.values()))
            run["associated_policy_exports"] = policies
            main.append(run)
        else:
            matched = False
            for label, frame in FRAMES.items():
                if names == {f"frame_{frame:012d}.json"}:
                    run["role"] = f"fresh replay: {label}"
                    run["media"] = _check_media(directory, [frame], policies[label])
                    if run["media"][str(frame)]["metrics"] != metrics[label]:
                        raise AssertionError(
                            "fresh media metrics differ from the actual CLI report"
                        )
                    fresh[label].append(run)
                    matched = True
                    break
            if not matched:
                raise AssertionError("unexpected native evaluation artifact frame set")
        _check_run_scalars(run, tensorboard=tensorboard)
        if not tensorboard and run["status"] != "FINISHED":
            raise AssertionError("native MLflow recorded run is not FINISHED")
        if tensorboard:
            if [event["step"] for event in run["previews"]] != sorted(
                int(frame) for frame in run["media"]
            ):
                raise AssertionError(
                    "native image-summary steps must match each source frame exactly once"
                )
            for preview in run["previews"]:
                if (
                    preview["sha256"]
                    != run["media"][str(preview["step"])]["files"]["gif"]["sha256"]
                ):
                    raise AssertionError("native image-summary differs from encoded rollout GIF")
    if len(main) != 1 or any(len(items) != 1 for items in fresh.values()):
        raise AssertionError("expected one main train run and one fresh replay per source stage")
    return {
        "main": main[0],
        "fresh": {label: items[0] for label, items in fresh.items()},
        "all_runs": runs,
    }


def _collect_mode(mode, root, settings, policies, metrics):
    if mode not in {"tb-only", "both"} and list(root.rglob("*tfevents*")):
        raise AssertionError("unconfigured TensorBoard writer created events")
    if mode not in {"mlflow-only", "both"} and list(root.rglob("mlflow.db")):
        raise AssertionError("unconfigured MLflow created a SQLite store")
    if mode == "disabled":
        if (
            any(root.rglob("*.gif"))
            or any(root.rglob("*.mp4"))
            or any(root.glob("**/evaluation/*.json"))
        ):
            raise AssertionError("disabled logging created media")
        return {"writers": "none", "media": "none", "scalar_cli_metrics": metrics}
    result: dict = {}
    if mode in {"tb-only", "both"}:
        result["tensorboard"] = _classify_runs(
            _tensorboard_runs(root), policies, metrics, tensorboard=True
        )
    if mode in {"mlflow-only", "both"}:
        result["mlflow"] = _classify_runs(_mlflow_runs(root, settings["mlflow"]), policies, metrics)
    if mode == "both":
        for role in ("main", *FRAMES):
            tb = (
                result["tensorboard"]["main"]
                if role == "main"
                else result["tensorboard"]["fresh"][role]
            )
            ml = result["mlflow"]["main"] if role == "main" else result["mlflow"]["fresh"][role]
            for frame, payload in tb["media"].items():
                if {key: value["sha256"] for key, value in payload["files"].items()} != {
                    key: value["sha256"] for key, value in ml["media"][frame]["files"].items()
                }:
                    raise AssertionError("same-rollout native TB/MLflow GIF/MP4/JSON bytes differ")
        result["same_rollout_payload_bytes"] = "PASS"
    return result


def run_logging(
    record, root, env, prepare_recipe, stage, checkpoint, validate_policy, evaluation_metrics
):
    """Produce fixed four-mode native CLI evidence, never browser qualification.

    Injected stages must wait and raise on failure; exceptions propagate unchanged.
    """
    root = Path(root)
    logging = {"scope": "logger payloads only; browser and C/D/L/R/V gates UNVERIFIED", "modes": {}}
    record["logging"] = logging
    record["checks"].pop("logger_payloads", None)
    cli = [sys.executable, "-m", "nexuml.cli.main"]
    for mode in MODES:
        owned = root / mode
        owned.mkdir(mode=0o700)
        evidence = {"root": str(owned), "commands": []}
        logging["modes"][mode] = evidence
        evidence.update(_prepare_mode(mode, owned, prepare_recipe))
        mode_env = {
            **env,
            "CUDA_VISIBLE_DEVICES": "",
            "SDL_VIDEODRIVER": "dummy",
            "NEXUML_LOGS_ROOT": str(owned / "logs"),
        }

        def invoke(name, argv):
            name = f"logging-{mode}-{name}"
            command = {"name": name, "argv": argv}
            evidence["commands"].append(command)
            result = stage(record, name, argv, mode_env)
            command["stage"] = result
            return result

        case = ["--config", str(owned / "case.yaml")]
        invoke("untrained-export", [*cli, "export", *case, "--output", str(owned / "untrained")])
        invoke("train", [*cli, "train", *case])
        saved = _copy_checkpoints(owned, checkpoint)
        evidence["checkpoints"] = saved
        policies = {"untrained": owned / "untrained"}
        for label in ("intermediate", "final"):
            policies[label] = owned / (
                "checkpoint-policy" if label == "final" else "intermediate-policy"
            )
            invoke(
                f"{label}-export",
                [
                    *cli,
                    "export",
                    *case,
                    "--checkpoint",
                    saved[label]["path"],
                    "--output",
                    str(policies[label]),
                ],
            )
        evidence["actor_assertions"] = validate_policy(owned, Path(saved["final"]["path"]), 64)
        identities = {
            label: _policy_identity(policy, FRAMES[label], saved.get(label))
            for label, policy in policies.items()
        }
        evidence["policies"] = identities
        metrics = {}
        for label, policy in policies.items():
            replay = invoke(
                f"{label}-evaluate",
                [
                    *cli,
                    "evaluate",
                    str(policy),
                    "--config",
                    str(owned / "eval-case.yaml"),
                    "--device",
                    "cpu",
                ],
            )
            metrics[label] = evaluation_metrics(
                Path(replay["log_path"]), expected_frames=FRAMES[label]
            )
            _check_metrics(metrics[label])
        scalar = invoke(
            "final-scalar-evaluate",
            [
                *cli,
                "evaluate",
                str(policies["final"]),
                "--config",
                str(owned / "scalar-case.yaml"),
                "--device",
                "cpu",
            ],
        )
        scalar_metrics = evaluation_metrics(Path(scalar["log_path"]), expected_frames=64)
        _check_metrics(scalar_metrics)
        if metrics["final"] != scalar_metrics:
            raise AssertionError("recording changed final same-policy full-episode metrics")
        evidence.update(
            metrics=metrics, final_scalar_metrics=scalar_metrics, capture_metric_parity=True
        )
        for item in saved.values():
            if _sha256(item["path"]) != item["sha256"]:
                raise AssertionError("export/replay mutated an immutable selected checkpoint")
        evidence["native"] = _collect_mode(mode, owned, evidence["logging"], identities, metrics)
        evidence["ui_inputs"] = {
            "tensorboard_log_root": str(owned / "logs/tb") if mode in {"tb-only", "both"} else None,
            "mlflow_tracking_uri": evidence["logging"]["mlflow"]["tracking_uri"]
            if mode in {"mlflow-only", "both"}
            else None,
            "source_frames": FRAMES,
            "browser_proof": "UNVERIFIED; no server/browser started",
        }
    manifest = root / "logging-manifest.json"
    _write_json(manifest, logging)
    record["logging"] = {
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "modes": list(MODES),
        "scope": logging["scope"],
    }
    record["checks"]["logger_payloads"] = "PASS"
