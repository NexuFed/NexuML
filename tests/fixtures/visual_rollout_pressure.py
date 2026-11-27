"""Measure synthetic RGB placement with one FP32 CNN and Adam; not native PPO or VLA evidence."""

import argparse
import json
from collections import defaultdict
from time import perf_counter

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate


def _step(devices, phase):
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    baseline = {str(d): torch.cuda.memory_allocated(d) for d in devices}
    start = perf_counter()
    result = phase()
    for device in devices:
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start
    peak = {str(d): torch.cuda.max_memory_allocated(d) for d in devices}
    return result, {
        "seconds": seconds,
        "peak_allocated_bytes": peak,
        "peak_extra_bytes": {d: peak[d] - baseline[d] for d in peak},
        "peak_reserved_bytes": {str(d): torch.cuda.max_memory_reserved(d) for d in devices},
    }


def _model(device, channels):
    return nn.Sequential(
        nn.Conv2d(3, channels, 3, stride=2, device=device, dtype=torch.float32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(channels, 2, device=device, dtype=torch.float32),
    )


def _bytes(tensors):
    return sum(v.nbytes for v in tensors)


def _sizes(frames, side, batch, channels):
    return {
        "rollout_rgb": frames * 3 * side**2,
        "inference_rgb": batch * 3 * side**2,
        "normalized_input": batch * 3 * side**2 * 4,
        "conv_output": batch * channels * ((side - 1) // 2) ** 2 * 4,
        "model": (30 * channels + 2) * 4,
    }


def _transfer_receipt(stats, source, destination, payload_bytes, peer_access):
    split = source != destination
    return {
        **stats,
        "source": str(source),
        "destination": str(destination),
        "payload_bytes": payload_bytes,
        "cross_device_payload_bytes": payload_bytes if split else 0,
        "effective_bytes_per_second": payload_bytes / stats["seconds"] if split else None,
        "peer_access_capable": peer_access.get((source, destination)),
    }


def _check_gae(sim, learner):
    # Two environments: middle tick terminates one and times out the other.
    # An ended scene (pixel 8) and the next reset scene (pixel 32) must not alias.
    states = torch.tensor([0, 4, 32], dtype=torch.uint8, device=sim)
    next_states = torch.tensor([4, 8, 64], dtype=torch.uint8, device=sim)

    def pixels(values):
        return values.view(1, 3, 1, 1, 1).expand(2, 3, 3, 4, 4).clone()

    following = TensorDict(
        {
            "rgb": pixels(next_states),
            "reward": torch.tensor([1.0, 1.0, 2.0], device=sim).view(1, 3, 1).expand(2, -1, -1),
            "done": torch.tensor([[False, True, False]] * 2, device=sim).unsqueeze(-1),
            "terminated": torch.tensor(
                [[False, True, False], [False, False, False]], device=sim
            ).unsqueeze(-1),
        },
        [2, 3],
        device=sim,
    )
    rollout = TensorDict({"rgb": pixels(states), "next": following}, [2, 3], device=sim).to(learner)
    following = rollout["next"]
    assert rollout["rgb"].dtype == following["rgb"].dtype == torch.uint8
    advantages, targets = vec_generalized_advantage_estimate(
        0.5,
        0.5,
        rollout["rgb"][..., 0, 0, :1].float(),
        following["rgb"][..., 0, 0, :1].float(),
        following["reward"],
        following["done"],
        following["terminated"],
    )
    assert advantages.device == targets.device == learner
    advantages, targets = advantages.squeeze(-1), targets.squeeze(-1)
    torch.testing.assert_close(
        advantages, advantages.new_tensor([[2.25, -3.0, 2.0], [3.25, 1.0, 2.0]])
    )
    torch.testing.assert_close(targets, targets.new_tensor([[2.25, 1.0, 34.0], [3.25, 5.0, 34.0]]))
    return {"advantages": advantages.tolist(), "targets": targets.tolist()}


def _run(frames, side, batch, channels, sim, inference, learner):
    devices = tuple(dict.fromkeys((sim, inference, learner)))
    peer_access = {
        (src, dst): torch.cuda.can_device_access_peer(src, dst)
        for src in devices
        for dst in devices
        if src != dst
    }
    torch.manual_seed(17)
    fit, actor = _model(learner, channels), _model(inference, channels)
    actor.load_state_dict(fit.state_dict())
    optimizer = torch.optim.Adam(fit.parameters(), fused=True)
    shape = (frames, 3, side, side)
    sizes = _sizes(frames, side, batch, channels)

    def make_rollout():
        # Vary frames and spatial values so zero-filled copies cannot pass.
        ids = torch.arange(frames, device=sim, dtype=torch.int64).to(torch.uint8)
        rgb = torch.arange(3 * side**2, device=sim, dtype=torch.int64).to(torch.uint8)
        rgb = rgb.reshape(1, 3, side, side) + ids[:, None, None, None]
        return TensorDict({"rgb": rgb}, [frames], device=sim)

    td, storage = _step(devices, make_rollout)
    weight_bytes = _bytes(fit.state_dict().values())
    assert td["rgb"].nbytes == sizes["rollout_rgb"] and weight_bytes == sizes["model"]

    def iteration(version):
        actor_inputs, to_actor = _step(devices, lambda: td[:batch].to(inference))

        def predict(net, inputs):
            return net(inputs["rgb"].float() / 255)

        with torch.no_grad():
            action, forward = _step(devices, lambda: predict(actor, actor_inputs))
        returned_action, to_sim = _step(devices, lambda: action.to(sim))
        transferred, to_learner = _step(devices, lambda: td.to(learner))
        learner_inputs = transferred[:batch]
        assert transferred.batch_size == td.batch_size and transferred.device == learner
        for moved, original, device in (
            (actor_inputs["rgb"], td["rgb"][:batch], inference),
            (returned_action, action, sim),
            (transferred["rgb"], td["rgb"], learner),
        ):
            assert moved.device == device and moved.dtype == original.dtype
            assert moved.shape == original.shape
            if device == original.device:
                assert moved.data_ptr() == original.data_ptr()
            # Check every byte outside timing without allocating a second full rollout.
            for start in range(0, len(moved), batch):
                window = slice(start, start + batch)
                assert torch.equal(moved[window], original[window].to(device))

        def update():
            optimizer.zero_grad(set_to_none=True)
            behavior = predict(fit, learner_inputs)
            loss = behavior.square().mean()
            loss.backward()
            optimizer.step()
            return loss.detach(), behavior.detach()

        (loss, behavior), fit_peak = _step(devices, update)
        assert loss.device == learner and torch.isfinite(loss)
        torch.testing.assert_close(behavior, action.to(learner), rtol=1e-4, atol=1e-5)
        with torch.no_grad():
            assert torch.equal(action, predict(actor, actor_inputs))
            updated = predict(fit, learner_inputs)
            assert not torch.allclose(behavior, updated)
        _, publication = _step(devices, lambda: actor.load_state_dict(fit.state_dict()))
        for name, value in actor.state_dict().items():
            assert value.device == inference and torch.equal(
                value, fit.state_dict()[name].to(inference)
            )
        with torch.no_grad():
            torch.testing.assert_close(
                predict(actor, actor_inputs), updated.to(inference), rtol=1e-4, atol=1e-5
            )
        return {
            "behavior_version": version,
            "published_version": version + 1,
            "loss": loss.item(),
            "updated_action_max_delta": (updated - behavior).abs().max().item(),
            "to_actor": _transfer_receipt(
                to_actor, sim, inference, actor_inputs["rgb"].nbytes, peer_access
            ),
            "forward": forward,
            "forward_frames_per_second": batch / forward["seconds"],
            "to_sim": _transfer_receipt(to_sim, inference, sim, action.nbytes, peer_access),
            "to_learner": _transfer_receipt(
                to_learner, sim, learner, transferred["rgb"].nbytes, peer_access
            ),
            "update": fit_peak,
            "update_frames_per_second": batch / fit_peak["seconds"],
            "publication": _transfer_receipt(
                publication, learner, inference, weight_bytes, peer_access
            ),
        }

    iterations = {"first": iteration(0), "warm": iteration(1)}
    optimizer_tensors = [tensor for state in optimizer.state.values() for tensor in state.values()]
    assert all(t.device == learner for t in (*fit.parameters(), *optimizer_tensors))
    print(
        json.dumps(
            {
                "scope": "synthetic RGB, FP32 tensors, CNN and Adam; no simulator or PPO",
                "transfer_scope": "logical payloads; peer capability does not prove transfer route",
                "rgb_shape": shape,
                "rgb_dtype": str(td["rgb"].dtype),
                "minibatch_frames": batch,
                "channels": channels,
                "simulation_device": str(sim),
                "inference_device": str(inference),
                "learner_device": str(learner),
                "cuda_devices": {
                    str(d): {
                        "name": torch.cuda.get_device_properties(d).name,
                        "uuid": str(torch.cuda.get_device_properties(d).uuid),
                    }
                    for d in devices
                },
                "rollout_bytes": sizes["rollout_rgb"],
                "activation_tensor_bytes": {
                    name: sizes[name] for name in ("normalized_input", "conv_output")
                },
                "replica_weight_bytes": weight_bytes,
                "gradient_bytes": _bytes(p.grad for p in fit.parameters() if p.grad is not None),
                "optimizer_bytes": _bytes(optimizer_tensors),
                "storage": storage,
                "iterations": iterations,
                "terminal_gae": _check_gae(sim, learner),
            },
            indent=2,
            allow_nan=False,
        )
    )


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--channels", type=int, default=16, help="CNN width for model/activation pressure"
    )
    for role in ("simulation", "inference", "learner"):
        parser.add_argument(f"--{role}", default="cuda:0")
    args = parser.parse_args(argv)
    if not 1 <= args.batch <= args.frames or args.side < 4 or args.channels < 1:
        parser.error("provide positive frame counts and channels, with image side >= 4")
    try:
        sim, inference, learner = map(torch.device, (args.simulation, args.inference, args.learner))
    except RuntimeError as error:
        parser.error(str(error))
    if any(
        d.type != "cuda" or d.index is None or d.index >= torch.cuda.device_count()
        for d in (sim, inference, learner)
    ):
        parser.error("every role needs an explicit visible CUDA index")
    sizes = _sizes(args.frames, args.side, args.batch, args.channels)
    budget = defaultdict(int)
    budget[sim] += sizes["rollout_rgb"]
    budget[inference] += sizes["model"] + (sizes["inference_rgb"] if sim != inference else 0)
    budget[learner] += 4 * sizes["model"] + (sizes["rollout_rgb"] if sim != learner else 0)
    # Working-set estimate only; keep most free VRAM for CUDA/library workspaces and headroom.
    workspace = sizes["normalized_input"] + sizes["conv_output"]
    budget[inference] += 2 * workspace
    budget[learner] += 4 * workspace
    for device, required in budget.items():
        free, _ = torch.cuda.mem_get_info(device)
        if required > free // 4:
            parser.error(
                f"{device} estimated working set ({required} bytes) exceeds 25% of free VRAM "
                f"({free} bytes); reduce frames, batch, side, or channels; no CPU offload"
            )
    _run(args.frames, args.side, args.batch, args.channels, sim, inference, learner)


if __name__ == "__main__":
    _main()
