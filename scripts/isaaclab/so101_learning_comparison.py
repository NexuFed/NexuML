"""Run one process of the predeclared native/NexuML SO-101 PPO comparison."""

from argparse import ArgumentParser
from contextlib import contextmanager
import hashlib
import json
from math import expm1, isfinite, log
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
from typing import Any
from unittest.mock import patch

import torch

DEVICE = "cuda:0"
TASK = "NexuML-SO101-Joint-Reach"
WIDTH, HORIZON, ROLLOUTS = 32, 32, 128
EPOCHS, MINIBATCHES = 2, 4
LR, GAMMA, LAMBDA, CLIP, ENTROPY, GRAD_NORM = 0.0003, 0.99, 0.95, 0.2, 0.01, 0.5
SEEDS = (17, 43, 97)
STAGES = ("initial", "final")


def _tick():
    torch.cuda.synchronize(DEVICE)
    return perf_counter()


def _dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _state(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().clone()


def _verify_update(model, before):
    after = _state(model)
    assert torch.isfinite(after).all() and not torch.equal(after, before)


@contextmanager
def _native_env(width, seed):
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    from nexuml_library.environments._isaaclab_tasks import register_reference_tasks

    register_reference_tasks()
    cfg: Any = parse_env_cfg(TASK, device=DEVICE, num_envs=width)
    cfg.seed = seed
    args = {
        "headless": True,
        "fast_shutdown": True,
        "device": DEVICE,
        "visualizer": ["none"],
        "visualizer_explicit": True,
        "kit_args": (
            "--/crashreporter/enabled=false --/crashreporter/skipOldDumpUpload=true "
            "--/app/uploadDumpsOnStartup=false --/telemetry/enableAnonymousData=false "
            "--/telemetry/enableAnonymousAppName=false --/telemetry/enableSentry=false "
            "--/structuredLog/enable=false"
        ),
    }
    with launch_simulation(cfg, args):
        env: Any = gym.make(TASK, cfg=cfg)
        try:
            yield env
        finally:
            env.close()


def _rsl_rl(output, seed, batches, entered, *, low_exploration=False, snapshots=()):
    with _native_env(WIDTH, seed) as env:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        model = {"class_name": "MLPModel", "hidden_dims": [64, 64], "activation": "tanh"}
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = OnPolicyRunner(
            wrapper,
            {
                "num_steps_per_env": HORIZON,
                "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
                "actor": model
                | {
                    "distribution_cfg": {
                        "class_name": "GaussianDistribution",
                        "init_std": 0.2 if low_exploration else 0.7,
                    }
                },
                "critic": model,
                "algorithm": {
                    "class_name": "PPO",
                    "num_learning_epochs": EPOCHS,
                    "num_mini_batches": MINIBATCHES,
                    "learning_rate": LR,
                    "schedule": "fixed",
                    "gamma": GAMMA,
                    "lam": LAMBDA,
                    "clip_param": CLIP,
                    "entropy_coef": ENTROPY,
                    "max_grad_norm": GRAD_NORM,
                    "use_clipped_value_loss": False,
                    "normalize_advantage_per_mini_batch": True,
                },
            },
            device=DEVICE,
        )
        actor, critic = runner.alg.get_policy(), runner.alg.critic
        before_actor, before_critic = _state(actor), _state(critic)
        start_export = _tick()
        runner.export_policy_to_jit(str(output), "initial.pt")
        initial_export = _tick() - start_export

        rows, optimizer_steps = [], 0

        def record_step(*_):
            nonlocal optimizer_steps
            optimizer_steps += 1

        def record_batch(**metrics):
            rows.append((metrics["collect_time"], metrics["learn_time"], metrics["loss_dict"]))
            frames = len(rows) * WIDTH * HORIZON
            if frames in snapshots:
                with torch.random.fork_rng(devices=[DEVICE]):
                    runner.export_policy_to_jit(str(output), f"frame-{frames}.pt")

        steps_before = env.unwrapped.common_step_counter
        start = _tick()
        with (
            patch.object(runner.logger, "log", record_batch),
            runner.alg.optimizer.register_step_post_hook(record_step),
        ):
            runner.learn(batches)
        training = _tick() - start
        frames = (env.unwrapped.common_step_counter - steps_before) * wrapper.num_envs
        _verify_update(actor, before_actor)
        _verify_update(critic, before_critic)
        start_export = _tick()
        runner.export_policy_to_jit(str(output), "final.pt")
        runner.save(str(output / "learner.pt"))
        _training_receipt(
            output,
            seed,
            batches,
            frames,
            optimizer_steps,
            rows,
            start - entered - initial_export,
            training,
            initial_export,
            _tick() - start_export,
        )


def _nexuml(output, seed, batches, entered, *, low_exploration=False, snapshots=()):
    from lightning.pytorch.callbacks import Callback

    from nexuml.core.export import export_package
    from nexuml.training.lightning import NexuSession
    from nexuml_library.scenarios.reinforcement.isaaclab import isaac_so101_joint_reach_ppo

    scenario = isaac_so101_joint_reach_ppo(batches * WIDTH * HORIZON, WIDTH * HORIZON, WIDTH)
    training: Any = scenario.training
    training.seed = seed
    training.algorithm = training.algorithm.model_copy(
        update={
            "actor_lr": LR,
            "critic_lr": LR,
            "gamma": GAMMA,
            "gae_lambda": LAMBDA,
            "clip_epsilon": CLIP,
            "entropy_coef": ENTROPY,
            "max_grad_norm": GRAD_NORM,
            "update_epochs": EPOCHS,
            "minibatch_size": WIDTH * HORIZON // MINIBATCHES,
        }
    )
    assert scenario.interaction is not None
    session = NexuSession(scenario, log_dir=output, enable_loggers=False, enable_progress_bar=False)
    module: Any = session.lightning_module
    if low_exploration:
        (head,) = [
            layer
            for layer in module.policy.pipeline.modules()
            if isinstance(layer, torch.nn.Linear) and layer.out_features == 10
        ]
        with torch.no_grad():
            # The first five outputs are action locations; only shift the scale logits.
            head.bias[5:].add_(log(expm1(0.2)) - log(expm1(0.7)))
    before_actor, before_critic = _state(module.policy), _state(module.algorithm.critic)
    start_export = perf_counter()
    export_package(module.policy, output / "initial", lightning_module=module)
    initial_export = perf_counter() - start_export

    class Timing(Callback):
        rows = []
        previous = collect_start = 0.0

        def on_train_start(self, trainer, pl_module):
            self.previous = self.collect_start = _tick()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.update_start = _tick()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            end = _tick()
            losses = {
                key: float(value)
                for key, value in trainer.callback_metrics.items()
                if key.startswith("train/rl/loss")
            }
            self.rows.append((self.update_start - self.previous, end - self.update_start, losses))
            frames = len(self.rows) * WIDTH * HORIZON
            if frames in snapshots:
                assert frames == int(pl_module.frames)
                with torch.random.fork_rng(devices=[DEVICE]):
                    export_package(
                        pl_module.policy, output / f"frame-{frames}", lightning_module=pl_module
                    )
            self.previous = end

    timing = Timing()
    trainer = session.trainer
    getattr(trainer, "callbacks").append(timing)
    with scenario.interaction.environment.job_scope():
        start = perf_counter()
        session.run()
        training_seconds = _tick() - start
        assert int(module.batches) == batches
        _verify_update(module.policy, before_actor)
        _verify_update(module.algorithm.critic, before_critic)
        start_export = _tick()
        export_package(module.policy, output / "final", lightning_module=module)
        trainer.save_checkpoint(output / "learner.ckpt")
        _training_receipt(
            output,
            seed,
            batches,
            int(module.frames),
            int(module.algorithm.updates),
            timing.rows,
            timing.collect_start - entered - initial_export,
            training_seconds,
            initial_export,
            _tick() - start_export,
        )


def _training_receipt(
    output, seed, batches, frames, updates, rows, setup, training, initial, final
):
    assert frames == batches * WIDTH * HORIZON
    assert updates == batches * EPOCHS * MINIBATCHES
    assert len(rows) == batches
    assert all(losses and all(isfinite(x) for x in losses.values()) for _, _, losses in rows)
    _dump(
        output / "train.json",
        {
            "seed": seed,
            "frames": frames,
            "rollouts": batches,
            "minibatch_updates": updates,
            "timing_seconds": {
                "startup": setup,
                "training_including_setup_for_nexuml": training,
                "initial_export": initial,
                "final_export_and_checkpoint": final,
            },
            "per_rollout_collect_update_losses": rows,
            "peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(DEVICE),
            "artifact_sha256": {
                str(path.relative_to(output)): _sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.suffix in {".pt", ".package", ".ckpt"}
            },
        },
    )


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@torch.inference_mode()
def _evaluate(backend, output):
    from tensordict import TensorDict

    with _native_env(1, 10000) as env:
        result = {}
        for stage in STAGES:
            if backend == "rsl-rl":
                model = torch.jit.load(str(output / f"{stage}.pt"), map_location=DEVICE).eval()

                def act(obs):
                    return model(obs).clamp(-1, 1)
            else:
                from nexuml.core.export import load_inference_package

                model, _, _ = load_inference_package(output / stage)
                model = model.to(DEVICE).eval()

                def act(obs):
                    return model(TensorDict({"observation": obs}, [1]), deterministic=True)[
                        "action"
                    ]

            episodes = []
            for seed in range(10000, 10032):
                observations, _ = env.reset(seed=seed)
                total = 0.0
                for step in range(1, env.unwrapped.max_episode_length + 1):
                    action = act(observations["policy"])
                    assert action.shape == (1, 5) and (action.abs() <= 1).all()
                    observations, reward, terminated, truncated, _ = env.step(action)
                    total += reward.item()
                    if terminated.item() or truncated.item():
                        break
                else:
                    raise AssertionError("SO-101 episode exceeded its native horizon")
                assert bool(terminated.item()) == bool(env.unwrapped.joint_reach_success.item())
                episodes.append(
                    {
                        "seed": seed,
                        "steps": step,
                        "return": total,
                        "success": bool(terminated.item()),
                        "timeout": bool(truncated.item()),
                        "joint_error_rad": env.unwrapped.joint_reach_error.item(),
                    }
                )
            result[stage] = episodes
        _dump(output / "evaluation.json", result)


def main():
    """Run one trusted reference training or fresh native evaluation process."""
    entered = perf_counter()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("rsl-rl", "nexuml"))
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("seed", type=int, choices=SEEDS)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--smoke", action="store_true", help="Run two rollouts, not the quality gate."
    )
    args = parser.parse_args()
    torch.set_float32_matmul_precision("medium")
    if args.stage == "train":
        args.output.mkdir(parents=True)
        run = _rsl_rl if args.backend == "rsl-rl" else _nexuml
        run(args.output, args.seed, 2 if args.smoke else ROLLOUTS, entered)
    else:
        assert json.loads((args.output / "train.json").read_text())["seed"] == args.seed
        _evaluate(args.backend, args.output)


if __name__ == "__main__":
    main()
