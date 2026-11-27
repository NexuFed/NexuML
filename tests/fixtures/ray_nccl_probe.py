"""Bounded peer NCCL check on existing stock Ray GPU workers; not a learner."""

import argparse
import json
from datetime import timedelta
from typing import Any

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


class Worker:
    def ready(self):
        import torch

        assert torch.cuda.device_count() == 1
        return {
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "node": ray.get_runtime_context().get_node_id(),
        }

    def reserve(self):
        from torch import distributed as dist

        host = ray.util.get_node_ip_address()
        self.store = dist.TCPStore(host, 0, 2, True, timedelta(seconds=60), wait_for_workers=False)
        return host, self.store.port

    def run(self, rank, host, port):
        import torch
        from torch import distributed as dist

        from scripts.isaaclab.nccl_preflight import probe

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        timeout = timedelta(seconds=60)
        store = self.store if rank == 0 else dist.TCPStore(host, port, 2, False, timeout)
        dist.init_process_group(
            "nccl", store=store, rank=rank, world_size=2, device_id=device, timeout=timeout
        )
        try:
            return probe(device)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-resource", required=True, help="Approved whole-GPU worker selector")
    args = parser.parse_args()
    ray.init(address="auto")
    worker = ray.remote(num_cpus=1, num_gpus=1, resources={args.pool_resource: 1})(Worker)
    bundles = [{"CPU": 1, "GPU": 1, args.pool_resource: 1} for _ in range(2)]
    group = placement_group(bundles, strategy="STRICT_SPREAD")
    try:
        ray.get(group.ready(), timeout=230)
        runtime = {
            "uv": {
                "packages": ["torch==2.11.0+cu130"],
                "uv_pip_install_options": [
                    "--only-binary",
                    ":all:",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu130",
                ],
            },
            "config": {"setup_timeout_seconds": 420},
            "env_vars": {"NCCL_DEBUG": "INFO"},
        }
        workers: list[Any] = [
            worker.options(
                runtime_env=runtime,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=group, placement_group_bundle_index=rank
                ),
            ).remote()
            for rank in range(2)
        ]
        devices = ray.get([actor.ready.remote() for actor in workers], timeout=470)
        assert all(len({row[key] for row in devices}) == 2 for key in ("uuid", "node"))
        host, port = ray.get(workers[0].reserve.remote(), timeout=20)
        collectives = ray.get(
            [actor.run.remote(rank, host, port) for rank, actor in enumerate(workers)],
            timeout=120,
        )
        print("NCCL_PEER_VERIFIED", json.dumps({"devices": devices, "collectives": collectives}))
    finally:
        remove_placement_group(group)
