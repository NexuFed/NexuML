"""Execution-local collector factories and the Lightning iterable bridge."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch.utils.data import IterableDataset
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.types import RayCollectorSpec, ReinforcementLearningSpec


@dataclass(frozen=True, slots=True)
class EnvironmentFactory:
    """Serialize definitions, never a prebuilt simulator or transport."""

    definition: EnvironmentDefinition
    context: EnvironmentBuildContext
    owns_job_scope: bool = False

    def __call__(self):
        from nexuml.reinforcement.torchrl_env import _TorchRLEnvironmentAdapter

        with ExitStack() as resources:
            if self.owns_job_scope:
                resources.enter_context(self.definition.rollout_scope())
            runtime = self.definition.build(self.context)
            resources.callback(runtime.close)
            adapter = _TorchRLEnvironmentAdapter(runtime, self.context.device, self.context.seed)
            adapter._owned_resources = resources.pop_all()
            return adapter


def _ray_placement_init(placement: RayCollectorSpec) -> dict[str, Any]:
    import ray
    from ray._private.services import (
        canonicalize_bootstrap_address,
        get_ray_address_from_environment,
    )

    existing = ray.get_runtime_context().gcs_address if ray.is_initialized() else None
    address = placement.address
    if address is None:
        if existing is None:
            raise ValueError(
                "collector.ray placement requires an initialized Ray runtime or address"
            )
        return {"address": existing}
    if address == "auto":
        # Ray's auto discovery can honor RAY_ADDRESS=local; never pass that on to ray.init.
        address = get_ray_address_from_environment(address, temp_dir=None)
        if address is None or address == "auto":
            raise ConnectionError("Ray collector auto discovery did not return a concrete address")
        placement.validate_address(address)
    target = canonicalize_bootstrap_address(address)
    if target is None:
        raise ConnectionError("Ray collector placement requires an existing cluster address")
    if existing is not None and target != existing:
        raise ValueError("collector.ray address conflicts with the initialized Ray runtime")
    return {"address": target}


def build_collector(definition, training: ReinforcementLearningSpec, policy, device, rank=0) -> Any:
    """Build TorchRL collection at the execution site, with the Ray teardown repair.

    Returns:
        A finite collector owning worker-local environment resources.

    Raises:
        ImportError: If reinforcement collector prerequisites are missing.
        ValueError: If an unvalidated Ray collector topology is requested.
    """
    definition.validate_training(training, device=str(device))
    spec = training.collector
    worker_policy = policy
    policy_device = str(device) if spec.policy_device == "auto" else spec.policy_device
    for name, value in (
        ("env_device", spec.env_device),
        ("policy_device", policy_device),
        ("storing_device", spec.storing_device),
    ):
        try:
            selected = torch.device(value)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"collector.{name} is not a Torch device: {value!r}") from error
        if (
            spec.backend != "ray"
            and selected.type == "cuda"
            and (selected.index or 0) >= torch.cuda.device_count()
        ):
            raise ValueError(f"collector.{name} CUDA device {value!r} is unavailable")
    if spec.backend == "ray":
        gpu_worker = spec.ray is not None and spec.ray.num_gpus == 1
        if gpu_worker:
            spec.validate_ray_gpu_devices()
        learner_device = torch.device(device).type
        if not spec.sync or learner_device not in {"cpu", "cuda"}:
            raise ValueError(
                "Ray collection requires synchronous collectors and a CPU/CUDA learner"
            )
        if learner_device == "cuda" and (gpu_worker or spec.ray is None):
            raise ValueError(
                "A Ray CUDA learner requires explicit Ray placement and CPU rollout workers"
            )
        if not gpu_worker and any(
            torch.device(value).type != "cpu"
            for value in (spec.env_device, spec.storing_device, policy_device)
        ):
            raise ValueError("Ray collection without a GPU reservation requires CPU collectors")
        if spec.policy_sync_interval_batches != 1:
            raise ValueError("Ray collection requires policy_sync_interval_batches=1")
        width = spec.num_collectors * spec.num_envs_per_collector
        if training.frames_per_batch % width:
            raise ValueError(
                "Ray frames_per_batch must be divisible by collectors times environments"
            )
        if learner_device == "cuda":
            # Ray constructs workers before moving the policy to the worker's device.
            worker_policy = deepcopy(policy).cpu()
    try:
        from torchrl.collectors import Collector
    except ImportError as error:
        raise ImportError("Reinforcement collection requires nexuml[reinforcement]") from error
    context = EnvironmentBuildContext(
        purpose="train",
        seed=training.seed,
        device=spec.env_device,
        worker_rank=rank,
        num_envs=spec.num_envs_per_collector,
    )
    factories = [
        EnvironmentFactory(
            definition.model_copy(deep=True),
            replace(
                context,
                worker_rank=rank + index,
                seed=None if context.seed is None else context.seed + index * context.num_envs,
            ),
            owns_job_scope=spec.backend != "direct",
        )
        for index in range(spec.num_collectors)
    ]
    # Direct construction is inside Lightning setup; remote backends receive only factories.
    direct_environment = factories[0]() if spec.backend == "direct" else None
    topology: dict[str, Any] = (
        {}
        if spec.backend == "direct"
        else {"num_collectors": spec.num_collectors, "sync": spec.sync}
    )
    constructor: Any = Collector
    if spec.backend == "ray":
        try:
            from nexuml.reinforcement.ray_collector import _GracefulRayCollector
        except ImportError as error:
            raise ImportError("Ray collection requires nexuml[reinforcement,ray]") from error
        constructor = cast(Any, _GracefulRayCollector)
        remote_configs: dict[str, Any] = {"num_cpus": 1, "num_gpus": 0}
        ray_init_config: dict[str, Any] = {
            "address": "local",
            "num_cpus": spec.num_collectors,
            "num_gpus": 0,
            "include_dashboard": False,
        }
        if spec.ray is not None:
            from ray._common.ray_option_utils import validate_actor_options

            ray_init_config = _ray_placement_init(spec.ray)
            remote_configs["num_gpus"] = spec.ray.num_gpus
            if spec.ray.resources:
                remote_configs["resources"] = spec.ray.resources
            if spec.ray.label_selector:
                remote_configs["label_selector"] = spec.ray.label_selector
            validate_actor_options(remote_configs, in_options=False)
        # Exception to the front door: override its actor-kill hook, including automatic shutdown.
        topology.update(
            collector_kwargs={"auto_register_policy_transforms": False},
            remote_configs=remote_configs,
            ray_init_config=ray_init_config,
        )
    else:
        topology["backend"] = spec.backend
        # Keep NexuML's environment specs unchanged instead of opting into v0.15 transforms.
        topology["auto_register_policy_transforms"] = False
    collector = None
    try:
        collector = constructor(
            direct_environment if direct_environment is not None else factories,
            policy=worker_policy,
            **topology,
            frames_per_batch=training.frames_per_batch,
            total_frames=training.total_frames,
            env_device=spec.env_device,
            storing_device=spec.storing_device,
            policy_device=policy_device,
        )
        if spec.backend == "ray":
            synchronize_policy(collector, policy)
        return collector
    except BaseException:
        if collector is not None:
            collector.shutdown()
        elif direct_environment is not None:
            direct_environment.close()
        raise


class RolloutDataset(IterableDataset):
    """Each iteration requests one rollout; no hidden epoch-hook collection."""

    def __init__(self, collector, batches):
        super().__init__()
        self.collector = collector
        self.batches = batches

    def __len__(self):
        # Disables Lightning lookahead before policy synchronization.
        # ponytail: batch upper bound; use backend-rounded lengths if exact counts are needed.
        return self.batches

    def __iter__(self):
        yield from self.collector


def synchronize_policy(collector, policy) -> None:
    """Refresh copied policies using the installed TorchRL weight-update API."""
    collector.update_policy_weights_(TensorDict.from_module(policy).data)
