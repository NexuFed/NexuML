# Isolated Isaac Lab GPU reference profile

Linux x86_64, Ubuntu 24.04, Python 3.12, Isaac Sim 6.1.0, Isaac Lab
`v3.0.0-EA` / `ae37b028ea415c91ea2bc32609efcd759ed2b974`, Torch
`2.11.0+cu128`, `isaacsim_physx`. `Dockerfile` pins the base digest; `constraints.txt`
records the resolved vendor/application/test distributions, not a list to install
into an ordinary virtualenv. No new project extra installs Isaac. Do not `uv sync`
the simulator environment or replace the running DevContainer.

This remains an EA integration with an approved shutdown exception, not a general
production/hardware certification. Scoped 42-task integration and 15 native GPU tests
pass on the rebuilt image; broader roadmap robustness gates remain unchecked.
Exact evidence and existing unrelated test/type failures:
`reports/physical-ai/phase-2-isaaclab.md`.

## Prepare once, with operator approval

Requires NVIDIA container runtime, a compatible host driver and RTX/Vulkan access
(CUDA alone is insufficient), approved NVIDIA EULA and robot asset access. Optional
telemetry remains disabled. Preserve unrelated GPU jobs. Allow up to 60 GiB new
disk usage, keep at least 20 GiB free, and stop if those limits cannot be met.

From the repository root:

```bash
docker build -f scripts/isaaclab/Dockerfile -t nexuml-isaaclab:phase2-validated .
docker run --name nexuml-isaaclab-reference --label nexuml.task=add-isaaclab-gpu-robotics \
  --gpus '"device=0"' --cpus 4 --memory 16g --shm-size 2g \
  --cap-drop ALL --security-opt no-new-privileges \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  --entrypoint /bin/bash -d nexuml-isaaclab:phase2-validated -lc 'sleep infinity'
docker exec nexuml-isaaclab-reference bash scripts/isaaclab/run.sh -m pip check
```

No privileged mode, host IPC, driver install or host source mounts are needed.
First asset use requires network access; the maintained asset resolver caches the
files. Cold RTX startup may need five minutes; cached startups are much faster.
Sim's internal build is `6.1.0-rc.26+release.49347.2d230af4.gl` in this digest.

All commands below run **inside that prepared container**, from `/workspace/nexuml`.
Use `docker exec ... bash scripts/isaaclab/run.sh ...` externally. Verify daemon
completion/exit status, not just a client that disconnected or returned early.

## Native baselines (diagnostics only)

Run Cartpole, then Franka, before claiming NexuML integration. These upstream
RSL-RL jobs are not NexuML's learner. The explicit settings match bounded controls,
no observation corruption and small frame geometry, not native convergence defaults.

```bash
bash scripts/isaaclab/run.sh /opt/isaaclab/scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Isaac-Cartpole --num_envs 8 --max_iterations 2 \
  --device cuda:0 --seed 0 --visualizer none physics=isaacsim_physx \
  agent.num_steps_per_env=16 agent.algorithm.num_learning_epochs=1 \
  agent.algorithm.num_mini_batches=2 agent.clip_actions=1.0 \
  env.observations.policy.enable_corruption=false env.compute_final_obs=true
bash scripts/isaaclab/run.sh /opt/isaaclab/scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Isaac-Reach-Franka --num_envs 8 --max_iterations 2 \
  --device cuda:0 --seed 0 --visualizer none physics=isaacsim_physx \
  agent.num_steps_per_env=16 agent.algorithm.num_learning_epochs=1 \
  agent.algorithm.num_mini_batches=2 agent.clip_actions=1.0 \
  env.observations.policy.enable_corruption=false env.compute_final_obs=true
```

The maintained local probe adds no learner. It checks GPU shapes, joint identities,
physical action scale/offset, finite rewards, partial/all-row timeouts and distinct
pre-reset observations over 64 frames. Its deliberate four-step horizon is a reset
diagnostic, not the reference training horizon. SO-101 additionally validates the
exact upstream PhysX asset/gains before acting:

```bash
bash scripts/isaaclab/run.sh scripts/isaaclab/native_probe.py Isaac-Cartpole
bash scripts/isaaclab/run.sh scripts/isaaclab/native_probe.py Isaac-Reach-Franka
bash scripts/isaaclab/run.sh scripts/isaaclab/native_probe.py NexuML-SO101-Joint-Reach
```

## Ordinary NexuML training, resume and export

Run in order: `isaac-cartpole-ppo`, `isaac-franka-reach-ppo`, then required
`isaac-so101-joint-reach-ppo`. Substitute the scenario name; no source edits needed.

```bash
bash scripts/isaaclab/run.sh -m nexuml train isaac-cartpole-ppo \
  -O training.total_frames=256 -O training.frames_per_batch=128 \
  -O training.collector.num_envs_per_collector=8 \
  -O training.algorithm.update_epochs=1 -O training.algorithm.minibatch_size=64
```

The existing `NexuSession`, Lightning, direct TorchRL collector, PPO and complete
policy export remain the only learning path. One `[8,16]` rollout is 128 real frames.
The learner, environment, rollout policy and storage use one explicit CUDA device.
Process/Ray/async/multi-GPU collection and in-fit evaluation are rejected before
simulator allocation. Frame budgets count actual batches and may round up once.

For explicit checkpoint/export paths, run an ordinary trusted session job via the
entrypoint. The saved scenario/config travels with a Lightning checkpoint:

```python
from nexuml.core.export import export_package
from nexuml.training.lightning import NexuSession
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_so101_joint_reach_ppo

scenario = isaac_so101_joint_reach_ppo(total_frames=256, frames_per_batch=128, num_envs=8)
scenario.training.algorithm.update_epochs = 1
scenario.training.algorithm.minibatch_size = 64
session = NexuSession(scenario, log_dir="/workspace/output/so101")
result = session.run()
result.trainer.save_checkpoint("/workspace/output/so101.ckpt")
export_package(result.policy, "/workspace/output/policy")
```

Save as a trusted job, then `bash scripts/isaaclab/run.sh /path/to/job.py`.
Resume using `NexuSession.from_trainer_checkpoint("/workspace/output/so101.ckpt")`
and `.run()` in a **fresh process**. A checkpoint restores actor, critic, optimizer
and actual frame/update counters; it does not restore live physics or experience.
Completed budgets allocate no simulator or collector. Policy exports are inference
artifacts, not substitutes for training checkpoints. Only load trusted executable
Python packages; portable CPU TensorDict inference needs no Isaac/assets/TorchRL.

Equivalent ordinary CLI commands (use the saved checkpoint path printed by your run):

```bash
bash scripts/isaaclab/run.sh -m nexuml train --trainer-checkpoint /workspace/output/so101.ckpt
bash scripts/isaaclab/run.sh -m nexuml export isaac-so101-joint-reach-ppo \
  --checkpoint /workspace/output/so101.ckpt --output /workspace/output/policy
```

The same commands apply to Cartpole/Franka with their scenario and checkpoint. The
export command loads weights and packages a complete policy; it is not a training
resume and does not claim to collect more frames.

## Replay and media

```bash
bash scripts/isaaclab/run.sh -m nexuml evaluate /workspace/output/policy --device cuda:0 \
  -O training.evaluation.episodes=2 -O training.evaluation.max_steps_per_episode=120
bash scripts/isaaclab/run.sh -m nexuml evaluate /workspace/output/policy --device cuda:0 \
  -O training.evaluation.episodes=2 -O training.evaluation.max_steps_per_episode=120 \
  -O training.evaluation.video.enabled=true -O training.evaluation.video.format=both \
  -O training.evaluation.video.max_frames=121 -O training.evaluation.video.max_size=320
```

Always a fresh N=1 inference-only process. Choose GIF, MP4 or both from the same
bounded first-episode frames. Existing native TensorBoard/MLflow artifacts include
media and JSON; TensorBoard previews are animated. Disable TensorBoard with
`-O 'logging={"tensorboard":null,"mlflow":{"tracking_uri":"sqlite:////workspace/output/mlflow.db","auto_nested_runs":false}}'`
for MLflow-only use (JSON null inside a mapping, not the literal string `null`).
No training camera, HDF5 recorder, second rollout, video subsystem or logger is added.
MP4's `imageio-ffmpeg` dependency is optional outside this prepared acceptance image.
Source frames use package `training_frames` when available, otherwise zero; optimizer
steps are not relabeled as environment frames. Scalar replay needs no encoder imports.

Controls: Cartpole 100×action N at 60 Hz; Franka default pose + 0.5×action rad at
30 Hz; SO-101 absolute 0.5×action rad at 30 Hz for `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`. SO-101 uses the upstream `SO101_CFG`
`so101_new_calib_SysID/so101_new_calib.usda`, PhysX variant and authored gains, with
the separate gripper fixed at 0.2555162024919699 rad. Invalid identity/limits/gains
fail before acting; no replacement robot is permitted. The 15 state values are
five positions, five velocities and five goal positions. Success requires all
joints within 0.05 rad for eight steps, with a four-second timeout and negative
normalized joint-error reward. Error/success metrics are separate from return.
No hardware, IK, imitation/VLA, manipulation success or convergence is claimed.

## Acceptance and shutdown

The manual CI `isaaclab` input uses only a pre-cached image named by repository
variable `NEXUML_ISAAC_IMAGE`; it never pulls/builds/installs in default CPU CI.
It mounts the checked-out source read-only to test that revision. For local tests,
provide the checkout/tests in the container, then:

```bash
NEXUML_RUN_SIMULATOR_TESTS=1 bash scripts/isaaclab/run.sh -m pytest \
  tests/core/test_isaaclab_integration.py -q --basetemp=/workspace/output/pytest
```

These tests deliberately fail an intermediate process, verify failed logger cleanup,
resume only remaining frames, export and prove exact CPU inference, replay and decode
all three formats. They do not assert reward convergence or run ROS/Gazebo/CARLA.
Pytest is included only in the isolated acceptance image. Remote CI execution and
human/standard-player playback are separate evidence, not implied by workflow wiring.

The pinned upstream returning shutdown segfaults during Python finalization. The
approved workaround uses **public non-graceful fast close after the whole job**.
Runtime/collector close returns first; explicitly finalize loggers and save/export
before job completion. `SimulationApp.close(exit_code=...)` preserves failures and
interruptions; Python `atexit` callbacks/destructors are not guaranteed. No upstream
patch is applied. The job scope handles SIGTERM by unwinding Python with status143
before public fast close, rather than retaining AppLauncher's immediate-exit handler.
No private-handle patch is applied. A second/caller-owned application is rejected.

Copy wanted output before stopping your own container. Do not prune shared images,
kill other jobs or restart the DevContainer. No commit, merge, archive or deployment
is part of this workflow.
