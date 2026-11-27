"""Headless CUDA reference for the frozen, physically mapped Cartpole pair."""

import json
from pathlib import Path
import sys
from typing import Any, cast

import torch
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.discovery import environment
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import PolicyEvaluationSpec, ReinforcementLearningSpec
from nexuml.interaction.runtime import EnvironmentRuntime, InteractionStep

ASSET_SHA = "d9657349921533445007a0f83eac74d2ba274ba0cc49c585618d00bb98b75704"
SUBSTEPS = 2


@environment("test.paired_cartpole.mujoco_warp")
class MujocoWarpCartpole(EnvironmentDefinition):
    """Task-owned definition; only build loads and allocates the native simulator."""

    xml_file: str

    def describe(self):
        from paired_cartpole import _contract

        return _contract("mujoco-warp:paired-cartpole", ASSET_SHA)

    def validate_training(self, training: ReinforcementLearningSpec, *, device=None):
        spec = training.collector
        if (
            spec.backend != "direct"
            or not spec.sync
            or spec.policy_device not in ("auto", spec.env_device)
            or spec.storing_device != spec.env_device
            or torch.device(spec.env_device).type != "cuda"
            or torch.device(spec.env_device).index is None
            or (device is not None and device != spec.env_device)
        ):
            raise ValueError("MuJoCo Warp reference requires synchronous direct same-GPU training")

    def build(self, context: EnvironmentBuildContext) -> EnvironmentRuntime:
        from paired_cartpole import _digest

        device = torch.device(context.device)
        if context.purpose == "play" or context.render_mode is not None:
            raise ValueError("MuJoCo Warp reference supports only headless training and evaluation")
        if (
            device.type != "cuda"
            or device.index is None
            or device.index >= torch.cuda.device_count()
        ):
            raise ValueError("MuJoCo Warp reference requires an explicit available cuda:N device")
        if _digest(self.xml_file) != ASSET_SHA:
            raise ValueError("MuJoCo Warp reference requires the frozen paired Cartpole XML")
        return _Runtime(self, context, device)


class _Runtime:
    def __init__(self, definition, context, device):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        from paired_cartpole import HORIZON

        self.wp, self.mjw, self.device = wp, mjw, device
        self.closed = False
        self.horizon = HORIZON
        self.contract = definition.describe().model_copy(update={"num_envs": context.num_envs})
        self.nworld = context.num_envs
        self.batch = torch.Size((self.nworld,)) if self.nworld > 1 else torch.Size()
        self.rng = torch.Generator(device=device)
        if context.seed is None:
            self.rng.seed()
        else:
            self.rng.manual_seed(context.seed)
        native = mujoco.MjModel.from_xml_path(definition.xml_file)  # ty: ignore[unresolved-attribute]
        wp.init()
        self.stream = torch.cuda.current_stream(device)
        with wp.ScopedStream(wp.stream_from_torch(self.stream)):
            self.model = mjw.put_model(native)
            self.data = mjw.make_data(native, nworld=self.nworld)
            self.qpos = wp.to_torch(self.data.qpos)
            self.qvel = wp.to_torch(self.data.qvel)
            self.ctrl = wp.to_torch(self.data.ctrl)
        self.steps = torch.zeros(self.nworld, device=device, dtype=torch.int32)

    def _scope(self):
        if self.closed:
            raise RuntimeError("MuJoCo Warp runtime is closed")
        current = torch.cuda.current_stream(self.device)
        if current != self.stream:
            current.wait_stream(self.stream)
            self.stream = current
        return self.wp.ScopedStream(self.wp.stream_from_torch(current), sync_enter=False)

    def _observation(self):
        # Own each returned snapshot; native qpos/qvel are mutated in place.
        state = torch.cat((self.qpos, self.qvel), dim=-1)
        return TensorDict(
            {"observation": state.reshape(*self.batch, 4)}, self.batch, device=self.device
        )

    def reset(self, *, seed=None, mask=None):
        if mask is not None and (
            mask.dtype != torch.bool
            or mask.device != self.device
            or mask.shape not in (self.batch, self.batch + (1,))
        ):
            raise ValueError("Reset mask must be a boolean tensor matching the CUDA batch")
        with self._scope():
            if seed is not None:
                self.rng.manual_seed(seed)
            selected = (
                torch.ones(self.nworld, dtype=torch.bool, device=self.device)
                if mask is None
                else mask.reshape(self.nworld).contiguous()
            )
            self.mjw.reset_data(self.model, self.data, reset=self.wp.from_torch(selected))
            # Same noise range as Gymnasium, not the same RNG sequence.
            for state in (self.qpos, self.qvel):
                noise = torch.empty_like(state).uniform_(-0.01, 0.01, generator=self.rng)
                state.copy_(torch.where(selected[:, None], state + noise, state))
            self.steps.masked_fill_(selected, 0)
            return self._observation()

    def step(self, action: TensorDict):
        if action.batch_size != self.batch:
            raise ValueError("Action batch does not match native world count")
        CompiledPolicy._validate_fields(action, self.contract.actions, "Action")
        control = action["action"]
        if control.device != self.device:
            raise ValueError("Actions must be on the native CUDA device")
        with self._scope():
            self.ctrl.copy_(control.detach().reshape(self.nworld, 1))
            for _ in range(SUBSTEPS):
                self.mjw.step(self.model, self.data)
            self.steps.add_(1)
            observation = self._observation()
            state = observation["observation"]
            failed = (~torch.isfinite(state).all(dim=-1)) | (state[..., 1].abs() > 0.2)
            return InteractionStep(
                observation=observation,
                reward=(~failed).to(torch.float32).unsqueeze(-1),
                terminated=failed.unsqueeze(-1),
                truncated=(self.steps >= self.horizon).reshape(*self.batch, 1),
            )

    def close(self):
        if not self.closed:
            try:
                self.stream.synchronize()
            finally:
                self.closed = True
                del self.ctrl, self.qpos, self.qvel, self.steps, self.data, self.model


def _probe(definition):
    import mujoco
    import numpy as np

    from paired_cartpole import HORIZON, PERIOD

    device = torch.device("cuda:0")
    cpu_model = mujoco.MjModel.from_xml_path(definition.xml_file)  # ty: ignore[unresolved-attribute]
    assert cpu_model.opt.timestep * SUBSTEPS == PERIOD
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        runtime = definition.build(EnvironmentBuildContext("eval", device=str(device), num_envs=3))
        assert isinstance(runtime, _Runtime)
        try:
            command = torch.tensor([[0.5], [-0.5], [0.0]], device=device)
            controls = TensorDict({"action": command}, (3,))
            initial = runtime.reset(seed=23)["observation"]
            torch.testing.assert_close(
                runtime.reset(seed=23)["observation"], initial, rtol=0, atol=0
            )
            runtime.qpos.zero_()
            runtime.qvel.zero_()
            known = runtime.step(controls)
            state = known.observation["observation"]
            clock = runtime.wp.to_torch(runtime.data.time)
            force = runtime.wp.to_torch(runtime.data.qfrc_actuator)[:, 0]
            for alias, native in (
                (runtime.qpos, runtime.data.qpos),
                (runtime.qvel, runtime.data.qvel),
                (runtime.ctrl, runtime.data.ctrl),
                (clock, runtime.data.time),
            ):
                assert alias.is_cuda and alias.data_ptr() == native.ptr
            cpu = []
            for control in (0.5, -0.5, 0.0):
                data = mujoco.MjData(cpu_model)  # ty: ignore[unresolved-attribute]
                data.ctrl[0] = control
                mujoco.mj_step(cpu_model, data, nstep=SUBSTEPS)  # ty: ignore[unresolved-attribute]
                cpu.append(np.concatenate((data.qpos, data.qvel)))
            cpu = np.array(cpu)
            np.testing.assert_allclose(state.cpu(), cpu, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(force, 100 * command[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(clock, torch.full_like(clock, PERIOD))
            assert known.reward is not None and (known.reward == 1).all()
            assert not (known.terminated | known.truncated).any()
            receipt = {
                "cuda_observations": state.tolist(),
                "cpu_observations": cpu.tolist(),
                "force_n": force.tolist(),
                "time_before_reset_s": clock.tolist(),
            }
            selected = torch.tensor([False, True, False], device=device)
            before_clock = clock.clone()
            reset = runtime.reset(mask=selected)["observation"]
            live = ~selected
            torch.testing.assert_close(reset[live], state[live], rtol=0, atol=0)
            assert clock[1] == 0 and runtime.ctrl[1, 0] == 0 and runtime.steps[1] == 0
            torch.testing.assert_close(clock[live], before_clock[live], rtol=0, atol=0)
            torch.testing.assert_close(runtime.ctrl[live], command[live], rtol=0, atol=0)
            assert state.tolist() == receipt["cuda_observations"]
            receipt["time_after_reset_s"] = clock.tolist()

            observation = runtime.reset(seed=31)["observation"]
            for step in range(1, HORIZON + 1):
                control = -(2 * observation[:, 1:2] + 0.2 * observation[:, 3:4]).clamp(-1, 1)
                if "terminal_step" not in receipt:
                    control[0] = 0.5
                outcome = runtime.step(TensorDict({"action": control}, (3,)))
                observation = outcome.observation["observation"]
                assert outcome.reward is not None and outcome.reward.is_cuda
                assert outcome.terminated.is_cuda and outcome.truncated.is_cuda
                if outcome.terminated.any():
                    assert "terminal_step" not in receipt and step < HORIZON
                    assert outcome.terminated.flatten().tolist() == [True, False, False]
                    assert outcome.reward.flatten().tolist() == [0.0, 1.0, 1.0]
                    assert observation[0, 1].abs() > 0.2
                    receipt.update(terminal_step=step, terminal_angle_rad=observation[0, 1].item())
                    reset = runtime.reset(mask=outcome.terminated)["observation"]
                    torch.testing.assert_close(reset[1:], observation[1:], rtol=0, atol=0)
                    assert reset[0, 1].abs() <= 0.01 and clock[0] == 0
                    observation = reset
                at_horizon = step == HORIZON
                assert outcome.truncated.flatten().tolist() == [False, at_horizon, at_horizon]
            receipt["horizon_truncated"] = outcome.truncated.flatten().tolist()
            receipt["horizon_clocks_s"] = clock.tolist()
            assert "terminal_step" in receipt
        finally:
            runtime.close()
            runtime.close()
        assert runtime.closed and not hasattr(runtime, "data") and not hasattr(runtime, "model")
    return receipt


def _train_and_reload(root, definition):
    from lightning.pytorch.callbacks import Callback

    from nexuml.core.export import (
        PACKAGE_FILENAME,
        _restore_packaged_policy,
        export_package,
        infer,
        load_inference_package,
    )
    from nexuml.evaluation.policy import evaluate_policy
    from nexuml.training.lightning import NexuSession
    from nexuml.training.reinforcement import NexuRLLightningModule

    from paired_cartpole import HORIZON, _digest, _scenario

    class Observe(Callback):
        losses = ("loss_objective", "loss_critic", "loss_entropy")

        def __init__(self):
            self.batches = []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            assert pl_module.device == torch.device("cuda:0") and batch.batch_size == (8, 16)
            for key in (
                "action",
                ("next", "observation"),
                ("next", "reward"),
                ("next", "terminated"),
                ("next", "truncated"),
            ):
                assert batch[key].device == pl_module.device
            losses = {key: trainer.callback_metrics[f"train/rl/{key}"] for key in self.losses}
            assert all(value.is_cuda and torch.isfinite(value) for value in losses.values())
            terminal = batch["next", "terminated"].squeeze(-1)
            assert (batch["next", "observation"][..., 1][terminal].abs() > 0.2).all()
            assert (batch["next", "reward"].squeeze(-1)[terminal] == 0).all()
            assert (batch["observation"][:, 1:][terminal[:, :-1]].abs() <= 0.01).all()
            self.batches.append(
                {
                    **{key: float(value) for key, value in losses.items()},
                    "terminated": int(terminal.sum()),
                    "truncated": int(batch["next", "truncated"].sum()),
                }
            )

    scenario = _scenario("isaac", Path(definition.xml_file).parent)
    scenario.name = "mujoco-warp-cartpole-reference"
    assert scenario.interaction is not None and isinstance(
        scenario.training, ReinforcementLearningSpec
    )
    scenario.interaction.environment = definition
    scenario.training.seed = 31
    session = NexuSession(
        scenario, log_dir=root / "fit", enable_loggers=False, enable_progress_bar=False
    )
    module = session.lightning_module
    assert isinstance(module, NexuRLLightningModule)
    before = torch.nn.utils.parameters_to_vector(module.policy.parameters()).detach().cpu().clone()
    observer = Observe()
    cast(Any, session.trainer).callbacks.append(observer)
    session.fit()
    assert module.fit_device == "cuda:0" and module._collector is None
    counts = int(module.frames), int(module.batches), int(module.algorithm.updates)
    assert counts == (1024, 8, 16) and session.trainer.global_step == 32
    assert len(observer.batches) == 8 and sum(b["terminated"] for b in observer.batches) > 0
    after = torch.nn.utils.parameters_to_vector(module.policy.parameters()).detach().cpu()
    delta = float(torch.linalg.vector_norm(after - before))
    assert torch.isfinite(after).all() and delta > 0

    training = {
        "frames": counts[0],
        "batches": counts[1],
        "ppo_updates": counts[2],
        "optimizer_steps": session.trainer.global_step,
        "actor_delta_l2": delta,
        "rollouts": observer.batches,
    }
    (root / "training.json").write_text(json.dumps(training, indent=2, allow_nan=False) + "\n")
    # The exported config names this task definition; inference does not need the simulator.
    package = export_package(
        module.policy, root / "policy", lightning_module=module, include_modules=[__name__]
    )
    reloaded, _, metadata = load_inference_package(package)
    assert metadata["training_frames"] == counts[0]
    device = torch.device(module.fit_device)
    example = TensorDict(
        {"observation": torch.tensor([[0, 0, 0, 0], [0.1, -0.05, 0.2, 0.1]], device=device)},
        (2,),
    )
    fixed = _restore_packaged_policy(reloaded, metadata).to(device)
    actions = infer(fixed, example)["action"]
    assert actions.device == device
    torch.testing.assert_close(
        infer(module.policy.to(device), example)["action"], actions, atol=0, rtol=0
    )
    scenario.policy_evaluation = PolicyEvaluationSpec(
        episodes=4, max_steps_per_episode=HORIZON, seed=211
    )
    reports = []
    evaluate_policy(
        fixed,
        scenario,
        frame_step=counts[0],
        episode_report_sink=reports.append,
        provenance={
            "source_policy_files_sha256": metadata["policy_files_sha256"],
            "source_training_frames": counts[0],
        },
    )
    return {
        **training,
        "reloaded_cuda_actions": actions.tolist(),
        "replay": reports[0],
        "package_sha256": _digest(package / PACKAGE_FILENAME),
    }


def main():
    import argparse
    from importlib.metadata import version

    from paired_cartpole import PERIOD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    definition = MujocoWarpCartpole(xml_file=str(args.xml.resolve()))
    assert not {"mujoco", "mujoco_warp", "warp", "isaaclab", "isaacsim"} & sys.modules.keys()
    assert definition.describe().control_period_s == PERIOD
    receipt = {
        "python": sys.version.split()[0],
        "packages": {
            name: version(name)
            for name in (
                "mujoco",
                "mujoco-warp",
                "warp-lang",
                "torch",
                "torchrl",
                "tensordict",
                "lightning",
            )
        },
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "asset_sha256": ASSET_SHA,
        "probe": _probe(definition),
    }
    (args.output / "probe.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("MUJOCO_WARP_PHYSICS_PROBE_PASSED", flush=True)
    receipt["training"] = _train_and_reload(args.output, definition)
    assert not {"isaaclab", "isaacsim"} & sys.modules.keys()
    path = args.output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(f"MUJOCO_WARP_REFERENCE_PASSED {path}", flush=True)
