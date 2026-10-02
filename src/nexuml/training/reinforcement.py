"""Lightning-owned manual optimization and fresh policy evaluation."""

from typing import Any

import lightning as L
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentBuildContext, RLAlgorithmBuildContext
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_model, restore_model_data
from nexuml.core.types import ReinforcementLearningSpec, ScenarioSpec
from nexuml.reinforcement.collector import (
    RolloutDataset,
    build_collector,
    synchronize_policy,
)


class NexuRLLightningModule(L.LightningModule):
    """Finite rollout batches with checkpoint-visible algorithm-owned auxiliaries."""

    automatic_optimization = False
    frames: torch.Tensor
    batches: torch.Tensor

    def __init__(self, scenario, policy=None, runtime_metadata=None):
        super().__init__()
        if not isinstance(scenario, ScenarioSpec):
            scenario = ScenarioSpec.model_validate(restore_model_data(scenario, ScenarioSpec))
        if not isinstance(scenario.training, ReinforcementLearningSpec):
            raise ValueError("NexuRLLightningModule requires reinforcement training")
        assert scenario.interaction is not None and scenario.policy is not None
        self.scenario = scenario
        self.training_spec = scenario.training
        if policy is None:
            if scenario.training.seed is not None:
                L.seed_everything(scenario.training.seed, workers=True)
            contract = scenario.interaction.environment.describe()
            policy = CompiledPolicy(
                compile(scenario), scenario.policy.action_adapter.build(contract), contract
            )
        self.policy: CompiledPolicy = policy
        self.algorithm: Any = scenario.training.algorithm.build(
            RLAlgorithmBuildContext(policy, policy.contract)
        )
        if not isinstance(self.algorithm, torch.nn.Module):
            raise TypeError("RL algorithms must build a checkpointable nn.Module")
        self._collector = None
        self.register_buffer("frames", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("batches", torch.tensor(0, dtype=torch.int64))
        self.save_hyperparameters(
            {"scenario": lower_model(scenario), "runtime_metadata": runtime_metadata or {}}
        )

    @property
    def pipeline(self):
        return self.policy.pipeline

    def forward(self, observation: TensorDict, *, deterministic=True):
        return self.policy(observation, deterministic=deterministic)

    def configure_optimizers(self):
        return self.algorithm.configure_optimizers()

    def train_dataloader(self):
        remaining = max(0, self.training_spec.total_frames - int(self.frames))
        if remaining and self._collector is None:
            assert self.scenario.interaction is not None
            # Placement and checkpoint restoration precede this hook; resume respects the budget.
            self._collector = build_collector(
                self.scenario.interaction.environment,
                self.training_spec.model_copy(update={"total_frames": remaining}),
                self.algorithm.rollout_policy,
                self.device,
                self.global_rank,
            )
        return DataLoader(
            RolloutDataset(
                self._collector if self._collector is not None else (),
                batches=-(-remaining // self.training_spec.frames_per_batch),
            ),
            batch_size=None,
            num_workers=0,
            collate_fn=lambda rollout: rollout,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def training_step(self, rollout, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        metrics = self.algorithm.update(rollout, lightning_module=self, optimizers=optimizers)
        self.frames.add_(rollout.numel())
        self.batches.add_(1)
        for name, value in metrics.items():
            self.log(
                f"train/rl/{name}", value, on_step=True, on_epoch=False, batch_size=rollout.numel()
            )
        self.log("train/rl/frames", self.frames.float(), on_step=True, on_epoch=False, batch_size=1)
        self.log(
            "train/rl/reward",
            rollout["next", "reward"].float().mean(),
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )
        if int(self.batches) % self.training_spec.collector.policy_sync_interval_batches == 0:
            synchronize_policy(self._collector, self.algorithm.rollout_policy)

    def teardown(self, stage):
        self.close_collection()

    def close_collection(self):
        if self._collector is not None:
            collector, self._collector = self._collector, None
            collector.shutdown()


def evaluate_policy(policy: CompiledPolicy, scenario: ScenarioSpec) -> dict[str, float]:
    """Evaluate the complete deployable mapping using a fresh environment.

    Returns:
        Episode returns/lengths and termination/truncation statistics.

    Raises:
        ValueError: If the environment does not supply a reward.
    """
    assert isinstance(scenario.training, ReinforcementLearningSpec)
    assert scenario.interaction is not None
    spec = scenario.training.evaluation
    if spec.episodes == 0:
        return {}
    definition = scenario.interaction.evaluation_environment or scenario.interaction.environment
    device = next(policy.parameters(), torch.tensor(0)).device
    seed = spec.seed if spec.seed is not None else scenario.training.seed
    runtime = definition.build(
        EnvironmentBuildContext(purpose="eval", seed=seed, device=str(device))
    )
    was_training = policy.training
    policy.eval()
    returns, lengths = [], []
    terminated = truncated = 0
    try:
        with torch.inference_mode():
            for episode in range(spec.episodes):
                observation = runtime.reset(seed=None if seed is None else seed + episode)
                total, steps = 0.0, 0
                while True:
                    transition = runtime.step(policy(observation, deterministic=spec.deterministic))
                    if transition.reward is None:
                        raise ValueError("RL evaluation requires a task reward")
                    total += float(transition.reward.sum())
                    steps += 1
                    term = bool(transition.terminated.any())
                    trunc = bool(transition.truncated.any())
                    limit = (
                        spec.max_steps_per_episode is not None
                        and steps >= spec.max_steps_per_episode
                    )
                    if term or trunc or limit:
                        terminated += int(term)
                        truncated += int(trunc or (limit and not term))
                        break
                    observation = transition.observation
                returns.append(total)
                lengths.append(steps)
    finally:
        runtime.close()
        policy.train(was_training)
    values = torch.tensor(returns, dtype=torch.float64)
    return {
        "eval/rl/return_mean": float(values.mean()),
        "eval/rl/return_std": float(values.std(unbiased=False)),
        "eval/rl/length_mean": sum(lengths) / len(lengths),
        "eval/rl/terminated": float(terminated),
        "eval/rl/truncated": float(truncated),
    }
