"""Execution-local collector factories and the Lightning iterable bridge."""

from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch.utils.data import IterableDataset
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.types import ReinforcementLearningSpec


@dataclass(frozen=True, slots=True)
class EnvironmentFactory:
    """Serialize definitions, never a prebuilt simulator or transport."""

    definition: EnvironmentDefinition
    context: EnvironmentBuildContext

    def __call__(self):
        from nexuml.reinforcement.torchrl_env import _TorchRLEnvironmentAdapter

        runtime = self.definition.build(self.context)
        try:
            return _TorchRLEnvironmentAdapter(runtime, self.context.device, self.context.seed)
        except BaseException:
            runtime.close()
            raise


def build_collector(definition, training: ReinforcementLearningSpec, policy, device, rank=0) -> Any:
    """Build TorchRL collection at the execution site, with the Ray teardown repair.

    Returns:
        A finite collector owning worker-local environment resources.

    Raises:
        ImportError: If reinforcement collector prerequisites are missing.
        ValueError: If an unvalidated Ray collector topology is requested.
    """
    try:
        from torchrl.collectors import Collector
    except ImportError as error:
        raise ImportError("Reinforcement collection requires nexuml[reinforcement]") from error
    spec = training.collector
    policy_device = str(device) if spec.policy_device == "auto" else spec.policy_device
    if spec.backend == "ray":
        if not spec.sync or any(
            torch.device(value).type != "cpu"
            for value in (device, spec.env_device, spec.storing_device, policy_device)
        ):
            raise ValueError(
                "Ray collection supports a CPU learner and synchronous CPU collectors only"
            )
        width = spec.num_collectors * spec.num_envs_per_collector
        if training.frames_per_batch % width:
            raise ValueError(
                "Ray frames_per_batch must be divisible by collectors times environments"
            )
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
        # Exception to the front door: override its actor-kill hook, including automatic shutdown.
        topology.update(
            remote_configs={"num_cpus": 1, "num_gpus": 0},
            ray_init_config={
                "address": "local",
                "num_cpus": spec.num_collectors,
                "num_gpus": 0,
                "include_dashboard": False,
            },
        )
    else:
        topology["backend"] = spec.backend
    collector = None
    try:
        collector = constructor(
            direct_environment if direct_environment is not None else factories,
            policy=policy,
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
