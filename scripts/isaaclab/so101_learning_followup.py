"""Run the predeclared SO-101 budget and exploration comparison."""

from argparse import ArgumentParser
from pathlib import Path
import json
from time import perf_counter

import torch

from so101_learning_comparison import (
    DEVICE,
    HORIZON,
    SEEDS,
    WIDTH,
    _dump,
    _native_env,
    _nexuml,
    _rsl_rl,
    _sha256,
)

FRAMES_PER_ROLLOUT = WIDTH * HORIZON
BUDGETS = (131_072, 524_288, 1_048_576)
EVAL_SEED = 10_000


def _actor(learner, output, stage, hashes):
    if learner == "rsl-rl":
        path = output / f"{stage}.pt"
        assert _sha256(path) == hashes[path.name]
        model = torch.jit.load(str(path), map_location=DEVICE).eval()
        return lambda obs: model(obs).clamp(-1, 1)

    from tensordict import TensorDict
    from nexuml.core.export import load_inference_package

    path = output / stage / "pipeline.package"
    assert _sha256(path) == hashes[str(path.relative_to(output))]
    model, _, _ = load_inference_package(path.parent)
    model = model.to(DEVICE).eval()
    return lambda obs: model(TensorDict({"observation": obs}, [WIDTH]), deterministic=True)[
        "action"
    ]


def _episodes(wrapped, act):
    from nexuml_library.environments._isaaclab_so101 import GOALS, TOLERANCE

    obs, _ = wrapped.reset(seed=EVAL_SEED)
    obs = obs["policy"]
    goals = torch.as_tensor(GOALS, device=DEVICE)
    target = obs[:, 10:15].clone()
    matches = (target[:, None] == goals).all(dim=-1)
    assert matches.any(dim=0).all() and (matches.sum(dim=1) == 1).all()
    goal_ids = matches.long().argmax(dim=1)
    steps = torch.zeros_like(goal_ids)
    passed = torch.zeros_like(goal_ids, dtype=torch.bool)
    timeouts = passed.clone()
    active = ~passed
    streak, longest, saturated = (steps.clone() for _ in range(3))
    best = torch.full_like(target[:, 0], float("inf"))
    first_action = None
    for _ in range(int(wrapped.unwrapped.max_episode_length)):
        action = act(obs)
        assert action.shape == (WIDTH, 5) and (action.abs() <= 1).all()
        if first_action is None:
            first_action = action.cpu().tolist()
        next_obs, _, term, trunc, extras = wrapped.step(action)
        obs = next_obs["policy"]
        done = term | trunc
        end = extras["final_obs"]["policy"] if done.any() else obs
        observed = torch.where(done[:, None], end, obs)
        error = (observed[:, :5] - target).abs().amax(dim=1)
        best = torch.where(active, torch.minimum(best, error), best)
        streak = torch.where(active & (error <= TOLERANCE), streak + 1, 0)
        longest = torch.maximum(longest, streak)
        saturated += active * (action.abs() >= 0.99).sum(dim=1)
        steps += active
        passed |= active & term
        timeouts |= active & trunc
        active &= ~done
        if not active.any():
            break
    assert not active.any() and torch.isfinite(best).all() and first_action is not None
    return [
        {
            "goal": goal,
            "steps": ticks,
            "success": success,
            "timeout": timeout,
            "best_max_error_rad": error,
            "longest_hold_ticks": hold,
            "action_saturation_fraction": hits / (ticks * 5),
            "initial_action": initial_action,
        }
        for goal, ticks, success, timeout, error, hold, hits, initial_action in zip(
            goal_ids.tolist(),
            steps.tolist(),
            passed.tolist(),
            timeouts.tolist(),
            best.tolist(),
            longest.tolist(),
            saturated.tolist(),
            first_action,
            strict=True,
        )
    ]


@torch.inference_mode()
def _evaluate(output):
    from nexuml_library.environments._isaaclab_so101 import TARGET_SCALE

    config = json.loads((output / "design.json").read_text())
    control = config["learner"] == "control"
    with _native_env(WIDTH, EVAL_SEED) as wrapped:
        if control:
            rows = _episodes(wrapped, lambda obs: obs[:, 10:15] / TARGET_SCALE)
            assert all(row["success"] and not row["timeout"] for row in rows)
            results = {"control": rows}
        else:
            budgets = config["budgets"]
            fit = json.loads((output / "train.json").read_text())
            assert fit["frames"] == budgets[-1] and fit["seed"] == config["seed"]
            stages = ["initial", *(f"frame-{frames}" for frames in budgets[:-1]), "final"]
            results = {}
            for frames, stage in zip([0, *budgets], stages, strict=True):
                act = _actor(config["learner"], output, stage, fit["artifact_sha256"])
                results[str(frames)] = _episodes(wrapped, act)
        _dump(output / "evaluation.json", config | {"evaluation_seed": EVAL_SEED, "rows": results})


def main():
    """Run one fresh-process fit, replay, or positive control."""
    entered = perf_counter()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "control"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--learner", choices=("rsl-rl", "nexuml"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--low", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("medium")
    if args.stage == "train":
        if args.learner is None or args.seed is None:
            parser.error("train needs --learner and --seed")
        budgets = (FRAMES_PER_ROLLOUT, 2 * FRAMES_PER_ROLLOUT) if args.smoke else BUDGETS
        args.output.mkdir(parents=True, exist_ok=False)
        _dump(
            args.output / "design.json",
            {"learner": args.learner, "seed": args.seed, "low": args.low, "budgets": budgets},
        )
        fit = _rsl_rl if args.learner == "rsl-rl" else _nexuml
        fit(
            args.output,
            args.seed,
            budgets[-1] // FRAMES_PER_ROLLOUT,
            entered,
            low_exploration=args.low,
            snapshots=budgets[:-1],
        )
    else:
        if args.stage == "control":
            args.output.mkdir(parents=True, exist_ok=False)
            _dump(args.output / "design.json", {"learner": "control"})
        _evaluate(args.output)


if __name__ == "__main__":
    main()
