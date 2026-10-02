"""Reference PPO wired from TorchRL objectives into Lightning manual optimization."""

import torch
from typing import Any, cast
from pydantic import Field
from torch import nn
from tensordict.nn import TensorDictModuleBase

from nexuml.core.components import RLAlgorithmBuildContext, RLAlgorithmDefinition
from nexuml.core.compiler import compile_context_from_interaction, compile_pipeline
from nexuml.core.discovery import rl_algorithm
from nexuml.core.types import LayerSpec, PipelineSpec
from nexuml_library.reinforcement.policies import PolicyHead


@rl_algorithm("PPO")
class PPO(RLAlgorithmDefinition):
    """PPO-specific value network and update settings, without a live environment."""

    critic: PipelineSpec | None = None
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    entropy_coef: float = Field(default=0.01, ge=0)
    actor_lr: float = Field(default=3e-4, gt=0)
    critic_lr: float = Field(default=1e-3, gt=0)
    update_epochs: int = Field(default=10, gt=0)
    minibatch_size: int = Field(default=256, gt=0)
    max_grad_norm: float = Field(default=0.5, gt=0)

    def build(self, context: RLAlgorithmBuildContext) -> nn.Module:
        try:
            from torchrl.modules import ProbabilisticActor
            from torchrl.objectives import ClipPPOLoss
            from torchrl.objectives.value import GAE
            from tensordict.nn import InteractionType
        except ImportError as error:
            raise ImportError("PPO requires nexuml[reinforcement]") from error

        adapter: Any = context.policy.action_adapter
        if not hasattr(adapter, "distribution_class"):
            raise ValueError("PPO requires a Gaussian or categorical action adapter")

        actor = ProbabilisticActor(
            cast(
                Any,
                _PipelineModule(
                    context.policy.pipeline,
                    list(context.contract.observations),
                    list(adapter.parameter_keys.values()),
                ),
            ),
            in_keys=adapter.parameter_keys,
            out_keys=[adapter.action_key],
            distribution_class=adapter.distribution_class,
            distribution_kwargs=adapter.distribution_kwargs,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
            log_prob_key=f"{adapter.action_key}_log_prob",
        )
        critic_spec = self.critic or PipelineSpec(
            stages={
                "value": [
                    LayerSpec(
                        component=PolicyHead(),
                        keys_in=list(context.contract.observations),
                        keys_out=["state_value"],
                    )
                ]
            }
        )
        critic_pipeline = compile_pipeline(
            critic_spec,
            context=compile_context_from_interaction(context.contract),
            resolved_config=context.policy.pipeline.resolved_config,
        )
        critic = cast(
            Any,
            _PipelineModule(critic_pipeline, list(context.contract.observations), ["state_value"]),
        )
        loss = ClipPPOLoss(
            actor,
            critic,
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            normalize_advantage=True,
            functional=False,
        )
        loss.set_keys(action=adapter.action_key, sample_log_prob=f"{adapter.action_key}_log_prob")
        gae = GAE(
            gamma=self.gamma, lmbda=self.gae_lambda, value_network=critic, deactivate_vmap=True
        )
        return _PPORuntime(self, actor, critic, loss, gae)


class _PipelineModule(TensorDictModuleBase):
    def __init__(self, pipeline, in_keys, out_keys):
        super().__init__()
        self.pipeline, self.in_keys, self.out_keys = pipeline, in_keys, out_keys

    def forward(self, tensors):
        inputs = tensors.select(*self.in_keys).reshape(-1).clone()
        output, _ = self.pipeline(inputs, None)
        output = output.reshape(tensors.batch_size)
        tensors.update(output.select(*self.out_keys))
        return tensors


class _PPORuntime(nn.Module):
    updates: torch.Tensor

    def __init__(self, definition, actor, critic, loss, gae):
        super().__init__()
        self.definition = definition
        self.rollout_policy = actor
        self.critic = critic
        self.loss = loss
        self.gae = gae
        self.register_buffer("updates", torch.tensor(0, dtype=torch.int64))

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.rollout_policy.parameters(), lr=self.definition.actor_lr),
            torch.optim.Adam(self.critic.parameters(), lr=self.definition.critic_lr),
        ]

    def update(self, rollout, *, lightning_module, optimizers):
        rollout = rollout.detach().clone()
        with torch.no_grad():
            self.gae(rollout)
        flat = rollout.reshape(-1)
        metrics = {}
        count = 0
        for _ in range(self.definition.update_epochs):
            permutation = torch.randperm(flat.numel(), device=flat.device)
            for indices in permutation.split(self.definition.minibatch_size):
                losses = self.loss(flat[indices])
                loss = sum(losses[key] for key in ("loss_objective", "loss_critic", "loss_entropy"))
                for optimizer in optimizers:
                    optimizer.zero_grad()
                lightning_module.manual_backward(loss)
                for optimizer in optimizers:
                    lightning_module.clip_gradients(
                        optimizer,
                        gradient_clip_val=self.definition.max_grad_norm,
                        gradient_clip_algorithm="norm",
                    )
                    optimizer.step()
                self.updates.add_(1)
                count += 1
                for key in ("loss_objective", "loss_critic", "loss_entropy", "clip_fraction"):
                    metrics[key] = metrics.get(key, 0.0) + float(losses[key].detach())
        return {key: value / count for key, value in metrics.items()}
