"""Measure synthetic RGB rollout, inference and update storage; not a learning benchmark."""

import argparse
import json
from time import perf_counter

import torch
from tensordict import TensorDict
from torch import nn


def _step(devices, phase):
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    baseline = {str(d): torch.cuda.memory_allocated(d) for d in devices}
    start = perf_counter()
    result = phase()
    for device in devices:
        torch.cuda.synchronize(device)
    return result, {
        "seconds": perf_counter() - start,
        "peak_extra_bytes": {
            str(d): torch.cuda.max_memory_allocated(d) - baseline[str(d)] for d in devices
        },
    }


def _model(device):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 2),
    ).to(device)


def _bytes(tensors):
    return sum(v.numel() * v.element_size() for v in tensors)


def _run(frames, side, batch, sim, inference, learner):
    devices = set((sim, inference, learner))
    torch.manual_seed(17)
    fit, actor = _model(learner), _model(inference)
    actor.load_state_dict(fit.state_dict())
    optimizer = torch.optim.Adam(fit.parameters(), fused=True)
    shape = (frames, 3, side, side)
    td, storage = _step(
        devices,
        lambda: TensorDict({"rgb": torch.zeros(shape, device=sim, dtype=torch.uint8)}, [frames]),
    )
    with torch.no_grad():
        actor_inputs, to_actor = _step(devices, lambda: td[:batch].to(inference))
        action, forward = _step(
            devices, lambda inputs=actor_inputs: actor(inputs["rgb"].float() / 255)
        )
        _, to_sim = _step(devices, lambda: action.to(sim))
    del actor_inputs
    learner_inputs, to_learner = _step(devices, lambda: td[:batch].to(learner))

    def update():
        loss = fit(learner_inputs["rgb"].float() / 255).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    loss, fit_peak = _step(devices, update)
    assert torch.isfinite(loss)
    assert action.device == inference and loss.device == learner and td.device in (None, sim)
    assert any(
        not torch.equal(fitted, old.to(learner))
        for fitted, old in zip(fit.parameters(), actor.parameters(), strict=True)
    )
    _, publication = _step(devices, lambda: actor.load_state_dict(fit.state_dict()))
    torch.testing.assert_close(
        actor.state_dict(), {k: v.to(inference) for k, v in fit.state_dict().items()}
    )
    print(
        json.dumps(
            {
                "rgb_shape": shape,
                "rgb_dtype": str(td["rgb"].dtype),
                "minibatch_frames": batch,
                "simulation_device": str(sim),
                "inference_device": str(inference),
                "learner_device": str(learner),
                "rollout_bytes": _bytes([td["rgb"]]),
                "replica_weight_bytes": _bytes(actor.parameters()),
                "gradient_bytes": _bytes(p.grad for p in fit.parameters() if p.grad is not None),
                "optimizer_bytes": _bytes(
                    v for state in optimizer.state.values() for v in state.values()
                ),
                "storage": storage,
                "to_actor": to_actor,
                "forward": forward,
                "to_sim": to_sim,
                "to_learner": to_learner,
                "update": fit_peak,
                "publication": publication,
                "version_barrier": "copied weights equal at update boundary",
            },
            indent=2,
        )
    )


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    for role in ("simulation", "inference", "learner"):
        parser.add_argument(f"--{role}", type=torch.device, default=torch.device("cuda:0"))
    args = parser.parse_args()
    devices = (args.simulation, args.inference, args.learner)
    if not 1 <= args.batch <= args.frames or args.side < 4:
        parser.error("provide positive frame counts and image side >= 4")
    if any(
        d.type != "cuda" or d.index is None or d.index >= torch.cuda.device_count() for d in devices
    ):
        parser.error("every role needs an explicit visible CUDA index")
    if args.frames * 3 * args.side**2 > torch.cuda.mem_get_info(args.simulation)[0] // 4:
        parser.error("RGB rollout alone exceeds one quarter of current free simulation VRAM")
    for device in set((args.inference, args.learner)):
        if args.batch * 64 * args.side**2 > torch.cuda.mem_get_info(device)[0] // 4:
            parser.error("reduce the RGB minibatch to fit inference and update workspace")
    _run(args.frames, args.side, args.batch, *devices)


if __name__ == "__main__":
    _main()
