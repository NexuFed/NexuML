"""Bounded native NexuML grouped learning/current continuation/export/replay proof."""

import json
from pathlib import Path
import shutil
import sys


def infer(output):
    import importlib.abc

    class BlockNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {
                "isaaclab",
                "isaaclab_tasks",
                "isaaclab_assets",
                "isaacsim",
                "omni",
                "pxr",
                "torchrl",
                "viser",
                "newton",
                "nexuml",
                "nexuml_library",
                "isaaclab_marl_config",
            }:
                raise ImportError("Clean MARL inference blocks " + fullname)

    sys.meta_path.insert(0, BlockNative())
    import torch
    from torch.package import PackageImporter

    source = Path(__file__).resolve().parents[2]
    roots = {source, source / "src", source / "library/src", Path(__file__).resolve().parent}
    sys.path[:] = [entry for entry in sys.path if entry and Path(entry).resolve() not in roots]
    observation = torch.load(output / "observation.pt", weights_only=False)
    expected = torch.load(output / "expected.pt", weights_only=False)
    payload = PackageImporter(str(output / "policy/pipeline.package")).load_pickle(
        "nexuml_export", "artifact.pkl"
    )
    policy = payload["policy"]
    if (output / "expected-state.pt").is_file():
        expected_state = torch.load(output / "expected-state.pt", weights_only=True)
        torch.testing.assert_close(policy.state_dict(), expected_state, rtol=0, atol=0)
        parent_context = json.loads((output / "parent-context.json").read_text())
        print("PARENT_CONTEXT", json.dumps(parent_context))
        print("CHILD_INITIAL_PRECISION", torch.get_float32_matmul_precision())
        # Exact parity requires the parent's numerical execution policy.
        torch.set_float32_matmul_precision(parent_context["precision"])
        print(
            "CHILD_CONTEXT",
            json.dumps(
                {
                    "torch": str(torch.__version__),
                    "precision": torch.get_float32_matmul_precision(),
                    "threads": torch.get_num_threads(),
                    "training": policy.training,
                    "device": str(next(policy.parameters()).device),
                }
            ),
        )
    with torch.no_grad():
        actual = policy(observation)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not payload["training_state"]
    assert all("critic" not in name for name, _ in policy.named_parameters())
    for group, ids in policy.actor_ids.items():
        assert len(ids) == len(policy.contract.groups[group].agents)
        if group == "hands":
            assert ids == ("hand_actor", "hand_actor") and len(policy.actors) == 1
    assert not any(
        module.split(".")[0]
        in {"torchrl", "isaaclab", "isaacsim", "viser", "newton", "isaaclab_marl_config"}
        for module in sys.modules
    )
    print("GROUPED_EXACT_ACTOR_ONLY_CLEAN_INFERENCE_AND_SHARING_VERIFIED", flush=True)


def train(family, algorithm_name, stage, root):
    from typing import Any, cast
    from lightning.pytorch.callbacks import Callback
    import torch

    from nexuml.training.lightning import NexuSession
    from nexuml.training.reinforcement import NexuRLLightningModule
    from nexuml.core.export import export_package
    from nexuml_library.environments import _isaaclab_runtime as native_runtime
    from nexuml_library.environments.isaaclab import isaac_task_config
    from nexuml_library.scenarios.reinforcement import isaaclab_marl
    from isaaclab_marl_config import short_horizon

    assert shutil.disk_usage(".").free > 4 * 2**30
    output = root / stage
    output.mkdir(parents=True, exist_ok=False)
    if stage in {"resume", "exhausted"}:
        checkpoint = root / (
            "interrupt/intermediate.ckpt" if stage == "resume" else "resume/complete.ckpt"
        )
        session = NexuSession.from_trainer_checkpoint(
            checkpoint, log_dir=output, enable_loggers=False, enable_progress_bar=False
        )
    else:
        scenario = getattr(isaaclab_marl, f"isaac_{family}_{algorithm_name}")(
            total_frames=64, frames_per_batch=16, num_envs=2
        )
        scenario.interaction.environment = scenario.interaction.environment.model_copy(
            update={"task_config": isaac_task_config(short_horizon)}
        )
        scenario.training.algorithm = scenario.training.algorithm.model_copy(
            update={"update_epochs": 1, "minibatch_size": 16}
        )
        session = NexuSession(
            scenario, log_dir=output, enable_loggers=False, enable_progress_bar=False
        )
    module = session.lightning_module
    assert isinstance(module, NexuRLLightningModule)

    def snapshot(network):
        return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}

    before = {name: snapshot(actor) for name, actor in module.policy.actors.items()}
    before_critic = {name: snapshot(critic) for name, critic in module.algorithm.critics.items()}
    initial_frames = int(module.frames)
    initial_updates = int(module.algorithm.updates)
    steps = 0
    final_rows = 0
    extra_native_resets = 0
    native_identity = {}

    class VerifyNative(Callback):
        def on_train_start(self, trainer, pl_module):
            nonlocal steps, extra_native_resets
            runtime = native_runtime._active_runtime
            if stage == "exhausted":
                assert runtime is None
                return
            assert runtime is not None
            native = runtime.native
            assert list(native.possible_agents) == [
                agent for spec in module.policy.contract.groups.values() for agent in spec.agents
            ]
            assert native.num_envs == 2 and native.cfg.compute_final_obs
            assert native.cfg.episode_length_s == 0.1
            native_identity.update(
                {
                    "task": runtime.resolved_task,
                    "environment_type": type(native).__module__ + "." + type(native).__name__,
                    "physics_type": type(native.cfg.sim.physics).__module__
                    + "."
                    + type(native.cfg.sim.physics).__name__,
                    "device": str(native.device),
                    "num_envs": native.num_envs,
                    "physics_dt": native.physics_dt,
                    "control_dt": native.step_dt,
                    "agents": list(native.possible_agents),
                    "observation_widths": {
                        key: list(space.shape) for key, space in native.observation_spaces.items()
                    },
                    "action_widths": {
                        key: list(space.shape) for key, space in native.action_spaces.items()
                    },
                }
            )
            from nexuml_library.reinforcement.algorithms.multiagent import agent_view

            inputs = runtime._observation.clone()
            changed = inputs.clone()
            group = "cart" if "cart" in module.policy.contract.groups else "hands"
            if group == "cart":
                changed["pendulum", "observation"].add_(1.0)
            else:
                changed["hands", "observation"][:, 1].add_(1.0)
            with torch.no_grad():
                torch.testing.assert_close(
                    module.policy(inputs)[group, "action"][:, 0],
                    module.policy(changed)[group, "action"][:, 0],
                    rtol=0,
                    atol=0,
                )
                module.algorithm._critic_inputs(inputs)
                module.algorithm._critic_inputs(changed)
                critic = module.algorithm.losses[0].critic_network
                left = critic(agent_view(inputs, group, 0))[group, "state_value"]
                right = critic(agent_view(changed, group, 0))[group, "state_value"]
                assert torch.equal(left, right) is (algorithm_name == "ippo")
            print("NATIVE_LOCAL_ACTOR_AND_DECLARED_CRITIC_DEPENDENCIES_VERIFIED", flush=True)
            original_reset = native.reset

            def reset(*args, **kwargs):
                nonlocal extra_native_resets
                extra_native_resets += 1
                return original_reset(*args, **kwargs)

            native.reset = reset
            original = runtime.step

            def step(action):
                nonlocal steps, final_rows
                transition = original(action)
                steps += 1
                assert action.batch_size == (2,) and transition.observation.batch_size == (2,)
                assert transition.groups is not None
                ended = (transition.terminated | transition.truncated).flatten()
                assert torch.equal(ended, native.reset_buf)
                final_rows += int(ended.sum())
                for group, spec in module.policy.contract.groups.items():
                    child = transition.groups[group]
                    assert child.batch_size == (2, len(spec.agents))
                    for i, agent in enumerate(spec.agents):
                        torch.testing.assert_close(
                            child["reward"][:, i, 0], native.reward_dict[agent]
                        )
                        if ended.any():
                            torch.testing.assert_close(
                                transition.observation[group, "observation"][ended, i],
                                native.extras[agent]["final_obs"][ended],
                                rtol=0,
                                atol=0,
                            )
                return transition

            runtime.step = step

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            assert batch.batch_size == (2, 8) and batch.numel() == 16
            if stage == "interrupt" and int(module.frames) == 32:
                trainer.save_checkpoint(output / "intermediate.ckpt")
                raise RuntimeError("intentional grouped checkpoint boundary")

    cast(Any, session.trainer).callbacks.append(VerifyNative())
    assert session.scenario.interaction is not None
    with session.scenario.interaction.environment.job_scope():
        if stage == "interrupt":
            try:
                session.run()
            except RuntimeError as error:
                assert "intentional grouped checkpoint boundary" in str(error)
            else:
                raise AssertionError("Expected checkpoint interruption")
            assert int(module.frames) == 32 and steps == 16
            assert final_rows > 0 and extra_native_resets == 0
            print(
                "NATIVE_GROUP_FINAL_OBSERVATIONS_AND_CACHED_RESET_VERIFIED", final_rows, flush=True
            )
            print("GROUPED_CURRENT_CHECKPOINT_INTERRUPTED_AT_PHYSICAL_FRAMES", flush=True)
            return
        result = session.run()
        assert int(module.frames) == 64
        assert steps * 2 == 64 - initial_frames
        if stage != "exhausted":
            assert final_rows > 0 and extra_native_resets == 0
            print(
                "NATIVE_GROUP_FINAL_OBSERVATIONS_AND_CACHED_RESET_VERIFIED", final_rows, flush=True
            )
            for name, actor in module.policy.actors.items():
                assert any(
                    not torch.equal(before[name][key], value)
                    for key, value in snapshot(actor).items()
                ), name
            for name, critic in module.algorithm.critics.items():
                assert any(
                    not torch.equal(before_critic[name][key], value)
                    for key, value in snapshot(critic).items()
                ), name
        else:
            assert int(module.algorithm.updates) == initial_updates
        assert all(torch.isfinite(parameter).all() for parameter in module.parameters())
        ids = [
            id(p)
            for opt in result.trainer.optimizers
            for group in opt.param_groups
            for p in group["params"]
        ]
        assert len(ids) == len(set(ids))
        if family == "handover":
            assert len(module.policy.actors) == 1
        observation = module.algorithm.rollout_policy.in_keys
        from tensordict import TensorDict

        data = TensorDict({}, batch_size=(2,))
        for group, spec in module.policy.contract.groups.items():
            data[group] = TensorDict(
                {
                    key: torch.zeros(2, len(spec.agents), *field.shape)
                    for key, field in spec.observations.items()
                },
                batch_size=(2, len(spec.agents)),
            )
        assert observation
        result.trainer.save_checkpoint(output / "complete.ckpt")
        module.policy.cpu().eval()
        with torch.no_grad():
            expected = module.policy(data)
        torch.save(data, output / "observation.pt")
        torch.save(expected, output / "expected.pt")
        export_package(module.policy, output / "policy")
        (output / "proof.json").write_text(
            json.dumps(
                {
                    "frames": int(module.frames),
                    "batches": int(module.batches),
                    "updates": int(module.algorithm.updates),
                    "native_steps": steps,
                    "actor_ids": module.policy.actor_ids,
                    "native_identity": native_identity,
                    "final_rows": final_rows,
                    "extra_native_resets": extra_native_resets,
                }
            )
        )
    assert native_runtime._active_runtime is None
    assert shutil.disk_usage(".").free > 4 * 2**30
    print(
        "GROUPED_NATIVE_ACTOR_CRITIC_UPDATES_COUNTERS_EXPORT_AND_CLEANUP_VERIFIED",
        family,
        algorithm_name,
        stage,
        flush=True,
    )


def replay(output):
    import torch
    from nexuml.core.export import load_package
    from nexuml.core.policy import CompiledPolicy
    from nexuml.core.types import PolicyEvaluationSpec
    from nexuml.evaluation.policy import evaluate_policy

    assert shutil.disk_usage(".").free > 4 * 2**30
    policy, config, _ = load_package(output / "policy")
    assert isinstance(policy, CompiledPolicy)
    scenario = config.to_scenario()
    assert scenario.interaction is not None
    scenario.policy_evaluation = PolicyEvaluationSpec(episodes=2, max_steps_per_episode=12, seed=11)
    with scenario.interaction.environment.job_scope():
        metrics = evaluate_policy(policy.to("cuda:0"), scenario)
    assert any("/return_mean" in key for key in metrics)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    print("GROUPED_FRESH_NATIVE_DECENTRALIZED_REPLAY_VERIFIED", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    family, algorithm, stage, directory = sys.argv[1:]
    root = Path(directory)
    if stage == "infer":
        infer(root / "resume")
    elif stage == "replay":
        replay(root / "resume")
    else:
        train(family, algorithm, stage, root)
