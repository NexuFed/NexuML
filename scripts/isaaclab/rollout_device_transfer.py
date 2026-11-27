"""Opt-in Pendulum placement and transfer probe. Run under the native GPU flock."""

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from contextlib import nullcontext
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from time import perf_counter, time
from unittest.mock import patch

import lightning as L
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import nexuml.training.reinforcement as rl
from nexuml.core.types import CollectorSpec, RLEvaluationSpec
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.gymnasium import pendulum_ppo

WIDTH, STEPS, LIMIT, SEED = 8, 32, 7, 17
FRAMES = WIDTH * STEPS
CPU = torch.device("cpu")


def _assert_owned(tensors, device, shape):
    assert tensors.batch_size == shape, (tensors.batch_size, shape)
    assert all(
        isinstance(t, torch.Tensor) and t.device == device for t in tensors.values(True, True)
    )


def _state(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().clone()


class CopyTrace(TorchDispatchMode):
    """Measure actual cross-device ATen copies in a separate diagnostic run."""

    def __init__(self):
        self.phase = None
        self.rows = defaultdict(lambda: {"calls": 0, "bytes": 0, "seconds": 0.0})

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.phase:
            return func(*args, **kwargs)
        scalar = func is torch.ops.aten._local_scalar_dense.default
        foreach = func is torch.ops.aten._foreach_copy_.default
        if func is torch.ops.aten._to_copy.default:
            (src,) = args
            dst, device = None, kwargs.get("device") or src.device
        elif func is torch.ops.aten.copy_.default:
            dst, src, *_ = args
            device = dst.device
        elif scalar:
            (src,) = args
            dst, device = None, CPU
        elif foreach and args[0]:
            dsts, srcs = args[:2]
            dst, src = dsts[0], srcs[0]
            device = dst.device
            assert all(d.device == device and s.device == src.device for d, s in zip(dsts, srcs))
        else:
            return func(*args, **kwargs)
        if not isinstance(src, torch.Tensor) or {src.device.type, device.type} != {"cpu", "cuda"}:
            return func(*args, **kwargs)
        gpu = device if device.type == "cuda" else src.device
        torch.cuda.synchronize(gpu)
        start = perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize(gpu)
        elapsed = perf_counter() - start
        if scalar:
            kind, transferred = "internal-scalar", src.element_size()
        else:
            if not foreach:
                dsts, srcs = [result if dst is None else dst], [src]
            assert all(s.dtype == d.dtype for d, s in zip(dsts, srcs))
            host = srcs if src.device.type == "cpu" else dsts
            pins = {t.is_pinned() for t in host}
            kind = "mixed-host" if len(pins) > 1 else "pinned" if True in pins else "pageable"
            transferred = sum(d.numel() * d.element_size() for d in dsts)
        key = f"{self.phase}/{src.device.type}->{device.type}/{kind}"
        row = self.rows[key]
        row["calls"] += 1
        row["bytes"] += transferred
        row["seconds"] += elapsed
        return result


def _probe(placement, profile_copies):
    device = torch.device(placement)
    cuda = device.type == "cuda"
    batches, warmup = (4, 1) if profile_copies else (24, 4)
    scenario = pendulum_ppo(total_frames=batches * FRAMES, frames_per_batch=FRAMES)
    assert scenario.interaction is not None
    scenario.interaction.environment = GymnasiumEnvironment(
        env_name="Pendulum-v1", kwargs={"max_episode_steps": LIMIT}
    )
    training = scenario.training
    training.seed = SEED
    training.algorithm = PPO(update_epochs=1, minibatch_size=64)
    training.evaluation = RLEvaluationSpec(episodes=0)
    training.collector = CollectorSpec(
        env_device="cpu",
        policy_device=placement,
        storing_device=placement,
        num_envs_per_collector=WIDTH,
    )
    trace = CopyTrace()
    module = rl.NexuRLLightningModule(scenario)
    actor, critic = module.algorithm.rollout_policy, module.algorithm.critic
    actor_before, critic_before = _state(actor), _state(critic)
    rows, seams = [], {}
    copy_sync = rl.synchronize_policy
    load, transfer, update = (
        module.train_dataloader,
        module.transfer_batch_to_device,
        module.algorithm.update,
    )
    last_live = None
    sync_calls = native_calls = gae_checks = 0
    window_start = None

    def start(phase):
        if cuda:
            torch.cuda.synchronize(device)
        if profile_copies and len(rows) > warmup:
            trace.phase = phase
        return perf_counter()

    def stop(phase, started):
        if cuda:
            torch.cuda.synchronize(device)
        trace.phase = None
        rows[-1][phase + "_s"] = perf_counter() - started

    def dataloader():
        nonlocal collector, environment, runtime
        loader = load()
        collector = module._collector
        environment = collector.env
        runtime = environment.runtime
        assert environment.device == CPU and runtime.context.device == "cpu"
        assert collector.policy_device == collector.storing_device == device
        assert (collector.policy is actor) == (not cuda)
        assert next(collector.policy.parameters()).device == device
        assert torch.equal(_state(collector.policy), actor_before)
        seams["environment"], seams["actor_weights"] = str(environment.device), placement
        for lane, env in enumerate(runtime.envs):
            step, reset = env.step, env.reset

            def gym_step(action, *, step=step, lane=lane):
                nonlocal native_calls
                tick = perf_counter()
                result = step(action)
                rows[-1]["gym_step_s"] += perf_counter() - tick
                native_calls += 1
                if len(rows) <= 2:
                    observations[lane].append(torch.from_numpy(result[0].copy()))
                return result

            def gym_reset(*args, reset=reset, lane=lane, **kwargs):
                result = reset(*args, **kwargs)
                if len(rows) <= 2:
                    resets[lane][len(observations[lane])] = torch.from_numpy(result[0].copy())
                return result

            env.step, env.reset = gym_step, gym_reset
        runtime_step = runtime.step

        def traced_runtime_step(actions):
            result = runtime_step(actions)
            if len(rows) <= 2:
                _assert_owned(actions, CPU, (WIDTH,))
                _assert_owned(result.observation, CPU, (WIDTH,))
                assert (
                    result.reward.device
                    == result.terminated.device
                    == result.truncated.device
                    == CPU
                )
                seams["runtime_action"], seams["runtime_transition"] = "cpu", "cpu"
            return result

        runtime.step = traced_runtime_step

        def actor_inference(_, args, output):
            if len(rows) <= 2:
                observation = args[0]["observation"]
                assert observation.shape == (WIDTH, 3) and observation.device == device
                assert output["action"].device == output["action_log_prob"].device == device
                seams["actor_observation"] = str(observation.device)
                seams["actor_action_and_log_prob"] = str(output["action"].device)
                behavior_log_probs.append(output["action_log_prob"].clone())

        collector.policy.register_forward_hook(actor_inference)
        collector.pre_collect_hook, collector.post_collect_hook = before_collect, after_collect
        return loader

    def before_collect():
        nonlocal observations, resets, collect_start, window_start
        assert sync_calls == (len(rows) if cuda else 0)
        if len(rows) == warmup:
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            window_start = time()
        rows.append({"gym_step_s": 0.0})
        behavior_log_probs.clear()
        observations = [[] for _ in range(WIDTH)]
        resets = [{} for _ in range(WIDTH)]
        collect_start = start("collect")

    def after_collect(batch):
        stop("collect", collect_start)
        _assert_owned(batch, device, (WIDTH, STEPS))
        assert batch.names == [None, "time"]
        assert batch.numel() == FRAMES
        seams["storage"] = str(batch["next", "observation"].device)

    def validate_transitions(batch):
        nonlocal last_live
        next_obs = batch["next", "observation"]
        root = batch["observation"]
        assert next_obs.dtype == root.dtype == torch.float32
        terminal = batch["next", "truncated"].squeeze(-1)
        assert terminal.dtype == torch.bool
        assert not batch["next", "terminated"].any()
        assert torch.equal(terminal, batch["next", "done"].squeeze(-1))
        end = len(rows) * STEPS
        assert int(terminal.sum()) == WIDTH * (end // LIMIT - (end - STEPS) // LIMIT)
        assert torch.equal(next_obs, torch.stack([torch.stack(obs) for obs in observations]))
        if last_live is not None:
            assert torch.equal(root[:, 0], last_live)
        live = next_obs.clone()
        for lane, lane_resets in enumerate(resets):
            assert lane_resets.keys() == {
                step + 1 for step in terminal[lane].nonzero().flatten().tolist()
            }
            for step, observation in lane_resets.items():
                assert not torch.equal(next_obs[lane, step - 1], observation)
                live[lane, step - 1] = observation
        assert torch.equal(root[:, 1:], live[:, :-1])
        last_live = live[:, -1]
        action, log_prob = batch["action"], batch["action_log_prob"]
        assert action.isfinite().all() and (action.abs() <= 2).all()
        assert log_prob.isfinite().all() and batch["next", "reward"].isfinite().all()
        assert torch.equal(log_prob, torch.stack(behavior_log_probs, dim=1).cpu())
        # CPU re-evaluation may differ from CUDA inference with float32 Tensor Cores.
        with torch.no_grad():
            expected = actor.get_dist(batch.clone()).log_prob(action)
        rows[-1]["cpu_reevaluation_log_prob_max_abs_delta"] = float(
            (log_prob - expected).abs().max()
        )
        rows[-1]["terminal_checks"] = int(terminal.sum())

    def learner_transfer(batch, target, dataloader_idx):
        tick = start("learner_transfer")
        result = transfer(batch, target, dataloader_idx)
        stop("learner_transfer", tick)
        _assert_owned(result, CPU, (WIDTH, STEPS))
        assert result.names == batch.names and (result is batch) == (not cuda)
        seams["learner_batch"] = str(result["next", "observation"].device)
        if len(rows) <= 2:
            validate_transitions(result)
        return result

    def check_gae(_, args, output):
        nonlocal gae_checks
        if len(rows) <= 2:
            _assert_owned(args[0], CPU, (WIDTH, STEPS))
            assert args[0].names == [None, "time"]
            truncated = output["next", "truncated"]
            expected = (
                output["next", "reward"]
                + training.algorithm.gamma * output["next", "state_value"]
                - output["state_value"]
            )
            torch.testing.assert_close(output["advantage"][truncated], expected[truncated])
            gae_checks += int(truncated.sum())
            seams["gae"] = str(output["advantage"].device)

    def optimize(batch, **kwargs):
        tick = start("gae_and_optimize")
        metrics = update(batch, **kwargs)
        stop("gae_and_optimize", tick)
        assert metrics and all(isfinite(x) for x in metrics.values())
        rows[-1]["losses"] = {key: float(value) for key, value in metrics.items()}
        return metrics

    def synchronize(col, policy):
        nonlocal sync_calls
        if sync_calls == 0 and cuda:
            assert torch.equal(_state(col.policy), actor_before)
            assert not torch.equal(_state(policy), actor_before)
        tick = start("weight_sync")
        copy_sync(col, policy)
        stop("weight_sync", tick)
        sync_calls += 1
        if sync_calls <= 2 or sync_calls == batches:
            assert torch.equal(_state(col.policy), _state(policy))

    collector = environment = runtime = None
    observations, resets, collect_start = [], [], 0.0
    behavior_log_probs = []
    module.algorithm.gae.register_forward_hook(check_gae)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    try:
        with (
            patch.object(module, "train_dataloader", dataloader),
            patch.object(module, "transfer_batch_to_device", learner_transfer),
            patch.object(module.algorithm, "update", optimize),
            patch.object(rl, "synchronize_policy", synchronize),
            trace if profile_copies else nullcontext(),
        ):
            trainer.fit(module)
        window_end = time()
        peak = (
            {
                "allocated": torch.cuda.max_memory_allocated(device),
                "reserved": torch.cuda.max_memory_reserved(device),
            }
            if cuda
            else None
        )
    finally:
        module.close_collection()
        if collector is not None:
            assert collector.closed and environment.is_closed and not runtime.envs
    assert native_calls == int(module.frames) == batches * FRAMES
    assert len(rows) == int(module.batches) == batches
    assert sync_calls == (batches if cuda else 0)
    updates = batches * (FRAMES // 64)
    assert int(module.algorithm.updates) == updates and trainer.global_step == 2 * updates
    assert all(
        {int(value["step"]) for value in optimizer.state.values()} == {updates}
        and all(t.device == CPU for value in optimizer.state.values() for t in value.values())
        for optimizer in trainer.optimizers
    )
    delta = {}
    for name, model, before in (("actor", actor, actor_before), ("critic", critic, critic_before)):
        assert all(p.device == CPU for p in model.parameters())
        after = _state(model)
        assert after.isfinite().all() and not torch.equal(after, before)
        delta[name] = float((after - before).norm())
        seams[name + "_learner"] = "cpu"
    assert gae_checks == sum(row.get("terminal_checks", 0) for row in rows) > 0
    return {
        "collector": training.collector.model_dump(),
        "seed": SEED,
        "geometry": [WIDTH, STEPS],
        "episode_limit": LIMIT,
        "warmup": warmup,
        "warm_window_epoch_s": [window_start, window_end],
        "seams": seams,
        "initial_actor_sha256": hashlib.sha256(actor_before.numpy().tobytes()).hexdigest(),
        "initial_critic_sha256": hashlib.sha256(critic_before.numpy().tobytes()).hexdigest(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "frames": native_calls,
        "minibatch_updates": updates,
        "adam_steps_each": updates,
        "parameter_delta_l2": delta,
        "weight_syncs": sync_calls,
        "gae_terminal_checks": gae_checks,
        "collection_closed": True,
        "rows": rows,
        "warm_torch_cuda_peak_bytes": peak,
        "copy_trace": dict(trace.rows),
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("placement", choices=["cpu", "cuda:0"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--copies", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cuda = args.placement.startswith("cuda:")
    if cuda and not torch.cuda.is_available():
        parser.error("CUDA placement requires a visible GPU")
    receipt = _probe(args.placement, args.copies)
    receipt["versions"] = {
        name: version(name) for name in ("gymnasium", "lightning", "torch", "torchrl", "tensordict")
    }
    receipt["cuda_context_initialized"] = torch.cuda.is_initialized()
    if receipt["cuda_context_initialized"]:
        gc.collect()
        torch.cuda.empty_cache()
        receipt["post_cleanup_cuda_bytes"] = [
            torch.cuda.memory_allocated(0),
            torch.cuda.memory_reserved(0),
        ]
        if cuda:
            # Diagnose process-owned cuBLAS workspace separately from live rollout tensors.
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        receipt["post_workspace_clear_cuda_bytes"] = [
            torch.cuda.memory_allocated(0),
            torch.cuda.memory_reserved(0),
        ]
        assert receipt["post_workspace_clear_cuda_bytes"] == [0, 0]
    if args.copies:
        paths = {tuple(key.split("/")[:2]) for key in receipt["copy_trace"]}
        assert paths == (
            {
                ("collect", "cpu->cuda"),
                ("collect", "cuda->cpu"),
                ("learner_transfer", "cuda->cpu"),
                ("weight_sync", "cpu->cuda"),
            }
            if cuda
            else set()
        )
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    _main()
