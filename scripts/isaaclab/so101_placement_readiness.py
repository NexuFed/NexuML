"""Verify archived SO-101 evidence or replay one condition under the native GPU lock."""

from argparse import ArgumentParser
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
import tarfile

import torch

from so101_learning_comparison import DEVICE, HORIZON, SEEDS, WIDTH, _dump, _native_env, _sha256
from so101_learning_followup import _actor

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (
    ROOT
    / "openspec/changes/decouple-simulation-learning-and-evaluation"
    / "so101-learning-followup-receipts.tar.gz"
)
ARCHIVE_SHA = "c37f4ff4348fb0c22d483f9115634e5b9460647e0866fe3c4bcf47fad51850ce"
LEARNERS = ("rsl-rl", "nexuml")
EVAL_SEEDS = range(30101, 30133)
STAGES = ("initial", "frame-524288")


def _receipts(artifacts):
    assert _sha256(ARCHIVE) == ARCHIVE_SHA, f"Archive hash mismatch: {ARCHIVE}"
    verified = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        run = json.load(archive.extractfile("run.json"))
        for path, expected in run["sha256"].items():
            assert _sha256(ROOT / path) == expected, path
        for package, expected in run["versions"].items():
            assert version(package) == expected, package
        for seed in SEEDS:
            for learner in LEARNERS:
                name = f"{learner}-{seed}-low"
                assert [seed, learner, "low"] in run["layout"], name
                design = json.load(archive.extractfile(f"{name}/design.json"))
                assert design == {
                    "learner": learner,
                    "seed": seed,
                    "low": True,
                    "budgets": run["budgets"][1:],
                }, name
                assert design["budgets"] == [131072, 524288, 1048576], name
                fit = json.load(archive.extractfile(f"{name}/train.json"))
                assert fit["seed"] == seed and fit["frames"] == 1048576, name
                rows = fit["per_rollout_collect_update_losses"]
                assert len(rows) == fit["rollouts"] == 1024, name
                assert fit["minibatch_updates"] == 8192, name
                assert all(losses and all(map(isfinite, losses.values())) for _, _, losses in rows)
                hashes = {
                    path: digest
                    for path, digest in fit["artifact_sha256"].items()
                    if any(
                        path == f"{stage}.pt" or path.startswith(f"{stage}/") for stage in STAGES
                    )
                }
                expected_count = 2 if learner == "rsl-rl" else 6
                assert len(hashes) == expected_count, name
                for path, digest in hashes.items():
                    artifact = artifacts / name / path
                    assert _sha256(artifact) == digest, f"Artifact hash mismatch: {artifact}"
                windows = []
                for start in (4, 36, 68):
                    window = rows[start : start + 32]
                    assert len(window) == 32
                    assert all(isfinite(t) and t > 0 for row in window for t in row[:2]), name
                    windows.append([sum(row[i] for row in window) for i in (0, 1)])
                verified[name] = {"artifact_sha256": hashes, "warm_collect_update_seconds": windows}
    assert WIDTH * HORIZON == 1024
    return verified


def _load_actor(learner, source, stage, hashes):
    if learner == "rsl-rl":
        return _actor(learner, source, stage, hashes)

    from tensordict import TensorDict
    from nexuml.core.export import load_inference_package

    path = source / stage / "pipeline.package"
    assert _sha256(path) == hashes[str(path.relative_to(source))], path
    model, _, _ = load_inference_package(path)
    model = model.to(DEVICE).eval()
    return lambda obs: model(TensorDict({"observation": obs}, [1]), deterministic=True)["action"]


def _episodes(env, act):
    from nexuml_library.environments._isaaclab_so101 import GOALS

    native = env.unwrapped
    assert native.max_episode_length == 120
    goals = torch.as_tensor(GOALS, device=DEVICE)
    rows = []
    for seed in EVAL_SEEDS:
        obs, _ = env.reset(seed=seed)
        target = obs["policy"][:, 10:15].clone()
        (goal,) = (target == goals).all(dim=-1).nonzero().flatten().tolist()
        total = 0.0
        for step in range(1, 121):
            policy_obs = obs["policy"]
            assert policy_obs.shape == (1, 15) and torch.isfinite(policy_obs).all()
            action = act(policy_obs)
            assert action.shape == (1, 5) and (action.abs() <= 1).all()
            obs, reward, term, trunc, extras = env.step(action)
            assert isfinite(reward.item())
            total += reward.item()
            if term.item() or trunc.item():
                break
        else:
            raise AssertionError(f"Reset seed {seed}: no native end after 120 steps")
        assert term.item() == native.joint_reach_success.item()
        assert not trunc.item() or step == 120
        ending = extras["final_obs"]["policy"]
        assert ending.shape == (1, 15) and torch.isfinite(ending).all()
        torch.testing.assert_close(ending[:, 10:15], target, rtol=0, atol=0)
        torch.testing.assert_close(
            (ending[:, :5] - target).abs().mean(dim=-1), native.joint_reach_error, rtol=0, atol=0
        )
        rows.append(
            {
                "seed": seed,
                "goal": goal,
                "steps": step,
                "return": total,
                "success": term.item(),
                "timeout": trunc.item(),
                "joint_error_rad": native.joint_reach_error.item(),
            }
        )
    assert {row["goal"] for row in rows} == set(range(len(GOALS)))
    return rows


@torch.inference_mode()
def main():
    """Run a single fresh native replay stage or verify the fixed receipts offline."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("receipts", "control", *LEARNERS))
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if (args.stage in LEARNERS) != (args.seed is not None):
        parser.error("--seed is required only for learner replay")
    if args.stage == "receipts":
        _dump(args.output, _receipts(args.artifacts))
        return

    verified = _receipts(args.artifacts)
    torch.set_float32_matmul_precision("medium")
    with _native_env(1, EVAL_SEEDS.start) as env:
        if args.stage == "control":
            from nexuml_library.environments._isaaclab_so101 import TARGET_SCALE

            rows = _episodes(env, lambda obs: obs[:, 10:15] / TARGET_SCALE)
            assert all(row["success"] and not row["timeout"] for row in rows)
            results = {"control": rows}
        else:
            name = f"{args.stage}-{args.seed}-low"
            hashes = verified[name]["artifact_sha256"]
            results = {
                stage: _episodes(env, _load_actor(args.stage, args.artifacts / name, stage, hashes))
                for stage in STAGES
            }
        _dump(args.output, results)


if __name__ == "__main__":
    main()
