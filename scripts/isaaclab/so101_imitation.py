"""Native SO-101 demonstrations and clean-package replay, never a training loop.

Run through scripts/isaaclab/run.sh. Each invocation owns exactly one Kit app.
Evaluation keeps per-trial native outcomes: the shared evaluate_policy API only
returns aggregates, so all references use the same bounded episode loop here.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import platform

import numpy as np
import torch
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext
from nexuml_library.environments._isaaclab_so101 import JOINT_NAMES, TARGET_SCALE
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.scenarios.imitation.so101 import (
    FPS,
    MAX_STEPS,
    REPO_ID,
    TASK,
    _observation,
    _source_hashes,
    _split_episodes,
    _task_identity,
    so101_behavior_cloning,
    validate_so101_demonstrations,
)


def _packages() -> dict:
    result = {}
    for name in ("nexuml", "nexuml-library", "torch", "lerobot", "isaaclab", "isaacsim"):
        installed = distribution(name)
        origin = json.loads(installed.read_text("direct_url.json") or "{}")
        result[name] = {"version": installed.version}
        revision = origin.get("vcs_info", {}).get("commit_id")
        if revision is not None:
            result[name]["revision"] = revision
    return result


def _features() -> dict:
    observation_names = [
        f"{kind}.{joint}" for kind in ("position", "velocity", "goal") for joint in JOINT_NAMES
    ]
    return {
        "observation": {"dtype": "float32", "shape": (15,), "names": observation_names},
        "action": {"dtype": "float32", "shape": (5,), "names": list(JOINT_NAMES)},
        "next.observation": {"dtype": "float32", "shape": (15,), "names": observation_names},
        "next.joint_error_rad": {"dtype": "float32", "shape": (1,), "names": None},
        **{
            f"next.{key}": {"dtype": "int64", "shape": (1,), "names": None}
            for key in ("success", "terminated", "truncated")
        },
    }


def _episode(runtime, *, seed: int, policy="expert", capture=False):
    observation = runtime.reset(seed=seed)
    generator = torch.Generator(device=observation.device).manual_seed(seed)
    frames = []
    total, success, error = 0.0, False, None
    outcome = {
        "seed": seed,
        "steps": 0,
        "success": False,
        "terminated": False,
        "truncated": False,
        "action_violations": 0,
        "joint_limit_violations": 0,
    }
    with torch.no_grad():
        for step in range(MAX_STEPS):
            before = observation["observation"].detach().cpu().numpy().copy()
            try:
                _observation(before)
            except ValueError:
                outcome.update(failure="invalid_physical_observation", joint_limit_violations=1)
                return frames, outcome
            if isinstance(policy, str):
                values = (
                    observation["observation"][10:15] / TARGET_SCALE
                    if policy == "expert"
                    else torch.rand(5, device=observation.device, generator=generator) * 2 - 1
                )
                action = TensorDict({"action": values}, batch_size=(), device=observation.device)
            else:
                try:
                    action = policy(observation, deterministic=True)
                except ValueError:
                    outcome.update(failure="policy_contract_violation", action_violations=1)
                    return frames, outcome
            values = action["action"]
            if values.shape != (5,) or not torch.isfinite(values).all() or (values.abs() > 1).any():
                outcome.update(failure="invalid_action", action_violations=1)
                return frames, outcome
            executed = values.detach().cpu().numpy().copy()
            transition = runtime.step(action)
            after = transition.observation["observation"].detach().cpu().numpy().copy()
            try:
                _observation(after)
            except ValueError:
                outcome.update(failure="invalid_physical_observation", joint_limit_violations=1)
                return frames, outcome
            if transition.reward is None or not torch.isfinite(transition.reward).all():
                raise ValueError("Native SO-101 reward is absent or nonfinite")
            info = transition.info or {}
            success = bool(info["task_success"].item())
            error = float(info["joint_error_rad"].item())
            if not np.isfinite(error) or error < 0:
                raise ValueError("Native SO-101 joint error is invalid")
            terminated = bool(transition.terminated.item())
            truncated = bool(transition.truncated.item())
            total += float(transition.reward.item())
            outcome.update(
                steps=step + 1,
                success=success,
                terminated=terminated,
                truncated=truncated,
                final_joint_error_rad=error,
                reward=total,
                goal_rad=before[10:15].tolist(),
            )
            if capture:
                frames.append(
                    {
                        "observation": before,
                        "action": executed,
                        "next.observation": after,
                        "next.joint_error_rad": np.array([error], dtype=np.float32),
                        **{
                            f"next.{key}": np.array([value], dtype=np.int64)
                            for key, value in (
                                ("success", success),
                                ("terminated", terminated),
                                ("truncated", truncated),
                            )
                        },
                        "task": TASK,
                    }
                )
            if terminated or truncated:
                break
            observation = transition.observation
    if not outcome["terminated"] and not outcome["truncated"]:
        outcome.update(truncated=True, failure="bounded_horizon_without_native_terminal")
    return frames, outcome


def _summary(trials: list[dict]) -> dict:
    return {
        "episodes": len(trials),
        "success_rate": sum(trial["success"] for trial in trials) / len(trials),
        "terminated": sum(trial["terminated"] for trial in trials),
        "truncated": sum(trial["truncated"] for trial in trials),
        "action_violations": sum(trial["action_violations"] for trial in trials),
        "joint_limit_violations": sum(trial["joint_limit_violations"] for trial in trials),
        "failures": sum(not trial["success"] for trial in trials),
    }


def collect(root: Path, *, episodes: int, seed: int) -> dict:
    """Write only complete successful native episodes through the maintained v3 writer.

    Returns:
        Native trial outcomes and private-dataset validation counts/hashes.

    Raises:
        FileExistsError: If collection would overwrite an existing local root.
        ValueError: If an expert episode is invalid, unsuccessful or incomplete.
    """
    splits = _split_episodes(episodes)
    if root.exists():
        raise FileExistsError("Collection root already exists; refusing to overwrite local data")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    receipt = {
        "schema_version": 1,
        **_task_identity(),
        "source_sha256": _source_hashes(),
        "packages": _packages(),
        "python": platform.python_version(),
        "seed": seed,
        "episode_seeds": list(range(seed, seed + episodes)),
        "split_episodes": splits,
    }
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=root,
        fps=FPS,
        features=_features(),
        robot_type="simulation:so101-fixed-gripper",
        use_videos=False,
        video_backend="pyav",
    )
    runtime, trials = None, []
    try:
        runtime = IsaacLabEnvironment(task=TASK).build(
            EnvironmentBuildContext(purpose="eval", seed=seed, device="cuda:0", num_envs=1)
        )
        for episode in range(episodes):
            frames, outcome = _episode(runtime, seed=seed + episode, capture=True)
            if (
                not outcome["success"]
                or not outcome["terminated"]
                or outcome["truncated"]
                or outcome["action_violations"]
                or outcome["joint_limit_violations"]
            ):
                raise ValueError(f"Native expert episode {episode} rejected: {outcome}")
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            trials.append(outcome)
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            dataset.finalize()
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["provenance"] = receipt
    path.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    scenario = so101_behavior_cloning(str(root))
    counts = validate_so101_demonstrations(scenario.data.datasets[0].source)
    result = {
        "schema_version": 1,
        "kind": "native-expert-collection",
        "task": TASK,
        "summary": _summary(trials),
        "validation": counts,
        "trials": trials,
    }
    (root / "meta/collection.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def evaluate(package: Path, *, episodes: int, seed: int, reference: str) -> dict:
    """Clean-load a trusted local policy package and retain each native trial outcome.

    Returns:
        Reproducible per-trial native outcomes and their aggregate counts.

    Raises:
        ValueError: If the package lacks the exact SO-101 deployment contract.
    """
    from nexuml.core.export import PACKAGE_FILENAME, load_inference_package
    from nexuml.core.types import InteractionContract

    policy, config, metadata = load_inference_package(package)
    if metadata.get("artifact_kind") != "policy" or any(
        getattr(policy, name, None) is None for name in ("pipeline", "action_adapter", "contract")
    ):
        raise ValueError("Evaluation requires a complete policy package, not a pipeline")
    # Match the ordinary evaluation CLI: torch.package can isolate Pydantic/policy classes.
    policy.contract = InteractionContract.model_validate(metadata["interaction_contract"])
    scenario = config.to_scenario()
    definition = IsaacLabEnvironment(task=TASK)
    if (
        policy.contract.model_dump(mode="json") != definition.describe().model_dump(mode="json")
        or scenario.interaction is None
        or scenario.interaction.environment != definition
    ):
        raise ValueError(
            "Package does not match the exact maintained SO-101 task/physical contract"
        )
    policy.to("cuda:0").eval()
    runtime = definition.build(
        EnvironmentBuildContext(purpose="eval", seed=seed, device="cuda:0", num_envs=1)
    )
    try:
        trials = [
            _episode(
                runtime, seed=seed + episode, policy=policy if reference == "package" else reference
            )[1]
            for episode in range(episodes)
        ]
    finally:
        runtime.close()
    return {
        "schema_version": 1,
        "kind": "native-closed-loop",
        "reference": reference,
        "task_identity": _task_identity(),
        "source_sha256": _source_hashes(),
        "packages": _packages(),
        "package_sha256": sha256(
            (package / PACKAGE_FILENAME if package.is_dir() else package).read_bytes()
        ).hexdigest(),
        "seed": seed,
        "summary": _summary(trials),
        "trials": trials,
        "generalization": "fresh-seed episodes on the native fixed THREE GOALS, not unseen goals",
    }


def main():
    """Collect native data or replay one package/reference in a fresh owned process."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("root", type=Path)
    collect_parser.add_argument("--episodes", type=int, default=72)
    collect_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument(
        "package", type=Path, help="Trusted local executable policy package"
    )
    evaluate_parser.add_argument("output", type=Path)
    evaluate_parser.add_argument("--episodes", type=int, default=60)
    evaluate_parser.add_argument("--seed", type=int, default=10000)
    evaluate_parser.add_argument(
        "--reference", choices=("package", "expert", "random"), default="package"
    )
    args = parser.parse_args()
    if args.seed < 0 or args.episodes < 1:
        parser.error("Seed must be nonnegative and episode count positive")
    if args.command == "collect":
        result = collect(args.root, episodes=args.episodes, seed=args.seed)
    else:
        result = evaluate(
            args.package, episodes=args.episodes, seed=args.seed, reference=args.reference
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
