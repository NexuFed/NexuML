"""Check NCCL rank assignment and real collective results before launching Isaac."""

import argparse
import json
import os
from datetime import timedelta

import torch
from torch import distributed as dist


def probe(device: torch.device, *, minimum_ranks: int = 2) -> dict:
    """Test scalar and gradient-sized CUDA all-reduce on the initialized group.

    Returns:
        Verified local device and collective rank.

    Raises:
        RuntimeError: If the group or collective results are invalid.
    """
    rank, size = dist.get_rank(), dist.get_world_size()
    if dist.get_backend() != "nccl" or size < minimum_ranks or device.type != "cuda":
        raise RuntimeError(f"NCCL needs CUDA and at least {minimum_ranks} rank(s)")
    for count in (1, 262144):
        tensor = torch.full((count,), rank + 1, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = size * (size + 1) / 2
        if not torch.all(tensor == expected):
            raise RuntimeError(f"Incorrect NCCL all-reduce result for {count} floats")
    return {"backend": "nccl", "rank": rank, "world_size": size, "device": str(device)}


def main():
    """Run the collective probe in the current torchrun rank.

    Raises:
        SystemExit: If there are too few ranks or visible GPUs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-ranks", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) < args.minimum_ranks:
        raise SystemExit("Insufficient NCCL ranks; launch torchrun with two or more GPUs")
    if int(os.environ["LOCAL_WORLD_SIZE"]) > torch.cuda.device_count():
        raise SystemExit("Insufficient visible CUDA devices to assign one to each local rank")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=60))
    try:
        print(json.dumps(probe(device, minimum_ranks=args.minimum_ranks)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
