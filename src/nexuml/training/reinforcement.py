"""Lightning-owned manual optimization; episode evaluation has a shared owner."""

from typing import Any

import lightning as L
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from nexuml.core.compiler import compile
from nexuml.core.components import RLAlgorithmBuildContext
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_model, restore_model_data
from nexuml.core.types import ReinforcementLearningSpec, ScenarioSpec
from nexuml.evaluation.policy import (
    _validate_evaluation_contract as _validate_evaluation_contract,
    evaluate_policy,
    resolve_policy_evaluation,
)
from nexuml.reinforcement.collector import RolloutDataset, build_collector, synchronize_policy


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
        evaluation = (
            scenario.policy_evaluation
            if scenario.policy_evaluation is not None
            else scenario.training.evaluation
        )
        scenario.interaction.environment.validate_training(
            scenario.training.model_copy(update={"evaluation": evaluation})
        )
        resolve_policy_evaluation(scenario)
        self.scenario = scenario
        self.training_spec = scenario.training
        if policy is None:
            if scenario.training.seed is not None:
                L.seed_everything(scenario.training.seed, workers=True)
            contract = scenario.interaction.environment.describe()
            policy = CompiledPolicy.from_pipeline(compile(scenario), contract)
        self.policy: CompiledPolicy = policy
        self.algorithm: Any = scenario.training.algorithm.build(
            RLAlgorithmBuildContext(policy, policy.contract)
        )
        if not isinstance(self.algorithm, torch.nn.Module):
            raise TypeError("RL algorithms must build a checkpointable nn.Module")
        self._collector = None
        self._evaluation_cache = None
        self._next_evaluation_frame = None
        self.register_buffer("frames", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("batches", torch.tensor(0, dtype=torch.int64))
        self.save_hyperparameters(
            {"scenario": lower_model(scenario), "runtime_metadata": runtime_metadata or {}}
        )

    @property
    def pipeline(self):
        return self.policy.pipeline

    @property
    def evaluation_spec(self):
        spec = self.scenario.policy_evaluation
        return self.training_spec.evaluation if spec is None else spec

    def forward(self, observation: TensorDict, *, deterministic=True):
        return self.policy(observation, deterministic=deterministic)

    def configure_optimizers(self):
        return self.algorithm.configure_optimizers()

    def on_load_checkpoint(self, checkpoint):
        if not self.policy.contract.groups:
            return
        saved = ScenarioSpec.model_validate(
            restore_model_data(checkpoint["hyper_parameters"]["scenario"], ScenarioSpec)
        )
        assert saved.interaction is not None
        assert self.scenario.interaction is not None
        if (
            saved.interaction.environment.describe() != self.policy.contract
            or saved.interaction.environment.checkpoint_identity()
            != self.scenario.interaction.environment.checkpoint_identity()
            or saved.policy != self.scenario.policy
            or not isinstance(saved.training, ReinforcementLearningSpec)
            or saved.training.algorithm != self.training_spec.algorithm
            or saved.training.collector.num_envs_per_collector
            != self.training_spec.collector.num_envs_per_collector
        ):
            raise ValueError(
                "Grouped checkpoint requires unchanged task configuration, contracts, "
                "sharing, algorithm and count"
            )

    def train_dataloader(self):
        remaining = max(0, self.training_spec.total_frames - int(self.frames))
        if remaining and self._collector is None:
            assert self.scenario.interaction is not None
            # Placement and checkpoint restoration precede this hook; resume respects the budget.
            self._collector = build_collector(
                self.scenario.interaction.environment,
                self.training_spec.model_copy(
                    update={"total_frames": remaining, "evaluation": self.evaluation_spec}
                ),
                self.algorithm.rollout_policy,
                self.device,
                self.global_rank,
            )
        return DataLoader(
            RolloutDataset(
                self._collector if self._collector is not None else (),
                # Lightning adds restored epoch progress; collection owns only remaining frames.
                batches=(int(self.batches) + -(-remaining // self.training_spec.frames_per_batch))
                if remaining
                else 0,
            ),
            batch_size=None,
            num_workers=0,
            collate_fn=lambda rollout: rollout,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def training_step(self, rollout, batch_idx):
        self._evaluation_cache = None
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
        if "reward" in rollout["next"].keys():
            self.log(
                "train/rl/reward",
                rollout["next", "reward"].float().mean(),
                on_step=True,
                on_epoch=False,
                batch_size=1,
            )
        for group in self.policy.contract.groups:
            self.log(
                f"train/rl/{group}/agent_reward_mean",
                rollout["next", group, "reward"].float().mean(),
                on_step=True,
                on_epoch=False,
                batch_size=1,
            )
        if int(self.batches) % self.training_spec.collector.policy_sync_interval_batches == 0:
            synchronize_policy(self._collector, self.algorithm.rollout_policy)
        interval = self.evaluation_spec.every_n_frames
        if interval is not None and self._next_evaluation_frame is not None:
            if int(self.frames) >= self._next_evaluation_frame:
                self.evaluate_current_policy()
                self._next_evaluation_frame = (int(self.frames) // interval + 1) * interval

    def on_fit_start(self):
        assert self.scenario.interaction is not None
        self.scenario.interaction.environment.validate_training(
            self.training_spec.model_copy(update={"evaluation": self.evaluation_spec}),
            device=str(self.device),
        )
        from nexuml.tracking.logger import preflight_evaluation_media

        spec = self.evaluation_spec
        preflight_evaluation_media(self.trainer.loggers, spec.video)
        if spec.every_n_frames is not None:
            self._next_evaluation_frame = (
                int(self.frames) // spec.every_n_frames + 1
            ) * spec.every_n_frames
        if spec.at_start and int(self.frames) == 0:
            self.evaluate_current_policy()

    def on_fit_end(self):
        spec = self.evaluation_spec
        # Fit-end also runs when a resumed checkpoint has no remaining training batches.
        if spec.episodes and (spec.at_start or spec.every_n_frames or spec.video.enabled):
            self.evaluate_current_policy()

    def evaluate_current_policy(self) -> dict[str, float]:
        """Report one evaluation per unchanged policy/frame/configuration state.

        Returns:
            The existing scalar episode metrics, reusing only matching local results.
        """
        from nexuml.tracking.logger import log_evaluation_media

        definition, spec, seed = resolve_policy_evaluation(self.scenario)
        key = (
            id(self.policy),
            int(self.frames),
            spec.model_dump_json(),
            seed,
            repr(definition),
            tuple(tensor._version for tensor in self.policy.parameters()),
            tuple(tensor._version for tensor in self.policy.buffers()),
        )
        if self._evaluation_cache is not None and self._evaluation_cache[0] == key:
            return dict(self._evaluation_cache[1])
        step = int(self.frames)

        def publish(frames, metadata):
            log_evaluation_media(self.trainer.loggers, frames, metadata, spec.video, step=step)

        metrics = evaluate_policy(self.policy, self.scenario, media_sink=publish, frame_step=step)
        if metrics:
            for logger in self.trainer.loggers:
                logger.log_metrics(metrics, step=step)
        self._evaluation_cache = (key, dict(metrics))
        return metrics

    def teardown(self, stage):
        self.close_collection()

    def close_collection(self):
        if self._collector is not None:
            collector, self._collector = self._collector, None
            collector.shutdown()
