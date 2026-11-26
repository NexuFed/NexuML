"""Lightning-owned manual optimization; episode evaluation has a shared owner."""

import logging
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from nexuml.evaluation._process import ProcessEvaluation
    from nexuml.evaluation._ray import RayEvaluation


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
        self.fit_device: str | None = None
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
        self._pending_evaluation: tuple[tuple, ProcessEvaluation | RayEvaluation] | None = None
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
        if (
            interval is not None
            and self._next_evaluation_frame is not None
            and int(self.frames) >= self._next_evaluation_frame
        ):
            self._request_fit_evaluation()
            self._next_evaluation_frame = (int(self.frames) // interval + 1) * interval
        else:
            self._finish_snapshot(wait=False)

    def on_fit_start(self):
        self.fit_device = str(self.device)
        assert self.scenario.interaction is not None
        self.scenario.interaction.environment.validate_training(
            self.training_spec.model_copy(update={"evaluation": self.evaluation_spec}),
            device=str(self.device),
        )
        from nexuml.tracking.logger import preflight_evaluation_media

        spec = self.evaluation_spec
        if spec.episodes and (spec.process is not None or spec.ray is not None):
            if self.trainer.world_size != 1:
                raise ValueError("Snapshot fit evaluation requires one learner process")
            if spec.process is not None and spec.process.device.startswith("cuda:"):
                device_index = int(spec.process.device.split(":")[1])
                if device_index >= torch.cuda.device_count():
                    raise ValueError(f"Snapshot evaluator GPU {spec.process.device} is unavailable")
                if self.device == torch.device(spec.process.device):
                    raise ValueError("Snapshot evaluator GPU must differ from the learner GPU")
            if spec.ray is not None:
                from nexuml.evaluation._ray import require_ray_runtime

                require_ray_runtime()
        preflight_evaluation_media(self.trainer.loggers, spec.video)
        if spec.every_n_frames is not None:
            self._next_evaluation_frame = (
                int(self.frames) // spec.every_n_frames + 1
            ) * spec.every_n_frames
        if spec.at_start and int(self.frames) == 0:
            self._request_fit_evaluation()

    def on_fit_end(self):
        spec = self.evaluation_spec
        # Fit-end also runs when a resumed checkpoint has no remaining training batches.
        if spec.episodes and (
            spec.at_start or spec.every_n_frames or spec.video.enabled or spec.ray is not None
        ):
            self.evaluate_current_policy()

    def _request_fit_evaluation(self) -> None:
        placement = self.evaluation_spec.process or self.evaluation_spec.ray
        if placement is not None and placement.scheduling == "skip_if_busy":
            self.evaluate_current_policy(allow_skip=True)
        else:
            self.evaluate_current_policy()

    def _finish_snapshot(self, *, wait: bool) -> None:
        if self._pending_evaluation is None:
            return
        key, pending = self._pending_evaluation
        report = pending.result(wait=wait, check_cancelled=self._check_snapshot_cancelled)
        if report is not None:
            from nexuml.evaluation._process import ProcessEvaluation

            self._check_snapshot_cancelled()
            if isinstance(pending, ProcessEvaluation):
                pending.publish_media(self.trainer.loggers, report)
            self._close_snapshot()
            self._publish_snapshot(report)
            self._evaluation_cache = (key, dict(report["metrics"]))

    def _check_snapshot_cancelled(self) -> None:
        if self.trainer.received_sigterm:
            from lightning.pytorch.utilities.exceptions import SIGTERMException

            raise SIGTERMException

    def _close_snapshot(self) -> None:
        if self._pending_evaluation is not None:
            _, pending = self._pending_evaluation
            self._pending_evaluation = None
            pending.close()

    def _publish_snapshot(self, report: dict) -> None:
        import json

        from nexuml.tracking.logger import log_text_artifact

        step = report["source_training_frames"]
        if self.trainer.loggers:
            log_text_artifact(
                self.trainer.loggers,
                json.dumps(report, allow_nan=False),
                f"evaluation-frame-{step:012d}.json",
            )
        self._publish_evaluation_metrics(report["metrics"], step)

    def _publish_evaluation_metrics(self, metrics: dict, step: int) -> None:
        if metrics:
            for logger in self.trainer.loggers:
                logger.log_metrics(metrics, step=step)

    def evaluate_current_policy(self, *, allow_skip: bool = False) -> dict[str, float]:
        """Report one evaluation per unchanged policy/frame/configuration state.

        Returns:
            Scalar episode metrics, or an empty dict when scheduled without waiting.
        """
        from nexuml.tracking.logger import log_evaluation_media

        self._finish_snapshot(wait=not allow_skip)
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

        if (spec.process is not None or spec.ray is not None) and spec.episodes:
            self._check_snapshot_cancelled()
            if self._pending_evaluation is not None:
                _, pending = self._pending_evaluation
                self._publish_evaluation_metrics({"eval/rl/snapshot_skipped": 1}, step)
                if not self.trainer.loggers:
                    logging.getLogger(__name__).info(
                        "Skipped evaluation at frame %d: snapshot %d still running",
                        step,
                        pending.frame,
                    )
                return {}
            if spec.ray is not None:
                from nexuml.evaluation._ray import RayEvaluation

                pending = RayEvaluation(self.policy, self.scenario, owner=self, spec=spec.ray)
            else:
                from nexuml.evaluation._process import ProcessEvaluation

                assert spec.process is not None
                pending = ProcessEvaluation(
                    self.policy,
                    self.scenario,
                    owner=self,
                    timeout_s=spec.process.timeout_s,
                    device=spec.process.device,
                )
            self._pending_evaluation = (key, pending)
            if allow_skip:
                return {}
            self._finish_snapshot(wait=True)
            assert self._evaluation_cache is not None and self._evaluation_cache[0] == key
            return dict(self._evaluation_cache[1])
        metrics = evaluate_policy(self.policy, self.scenario, media_sink=publish, frame_step=step)
        self._publish_evaluation_metrics(metrics, step)
        self._evaluation_cache = (key, dict(metrics))
        return metrics

    def teardown(self, stage):
        self.close_fit_resources()

    def close_fit_resources(self):
        try:
            self._close_snapshot()
        finally:
            self.close_collection()

    def close_collection(self):
        if self._collector is not None:
            collector, self._collector = self._collector, None
            collector.shutdown()
