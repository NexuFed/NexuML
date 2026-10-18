# Design

## Context

See `proposal.md` for motivation and the five capability specs for observable acceptance. The source roadmap's Phase 2 and the user's explicit addition of SO-101 define scope. This document is a design, not evidence of simulator compatibility or working training.

Existing reuse seams:

- `EnvironmentDefinition.describe/build`, `EnvironmentBuildContext`, `InteractionContract`, `InteractionStep`, and the framework-neutral reset/step/close protocol.
- `src/nexuml/reinforcement/{collector,torchrl_env}.py`: execution-site factory, direct collector, GPU device settings, policy synchronization, and finite rollout iterable.
- `NexuSession` / `NexuRLLightningModule`: Lightning-owned PPO, actual frame counters, checkpoint resume, and explicit collection shutdown.
- Complete-policy export/load, ordinary registry/YAML/overrides, native logging/artifact staging, `evaluate_policy`, and the inference-only `evaluate` command.

Current limits are concrete. The library Gymnasium runtime loops over environments and sends actions through CPU/NumPy; it cannot wrap a native Isaac GPU batch unchanged. The generic TorchRL adapter assumes explicit reset semantics. Evaluation requires unbatched episodes and can open a second runtime during fit. Isaac manager/direct environments own a process-global simulation context and auto-reset within `step()`.

The installed/verified stack is Python 3.12.13, Torch 2.11.0+cu128, TorchRL 0.14.0, TensorDict 0.14.2, Lightning 2.6.5, and Gymnasium 1.3.0. NexuML's declared Python minimum is 3.12; Torch 2.11 is a verified lock choice, not its declared minimum. Isaac Lab 2.3.2's documented Python 3.11 profile is therefore not this integration's target. The user selected 3.0 Early Access.

## Goals / Non-Goals

**Goals:** One isolated supported profile; one GPU and native batch; native-first Cartpole/Franka proofs followed by required SO-101 joint reaching; correct PPO transitions; complete checkpoint/export/replay/media round trips; unchanged default CPU/supervised behavior.

**Non-Goals:** No alternative trainer/session, RL algorithm rewrite, generic simulator/plugin system, GPU process/Ray collection, asynchronous/multi-GPU execution, Newton/OV backend expansion, visual-policy training, inverse-kinematics development, learned manipulation threshold, IL/VLA, hardware, or unrelated baseline repair. Rendering is for bounded replay inspection only.

## Decisions

### D1. Freeze an isolated EA profile before integration

Candidate profile, requiring native verification before being labeled tested:

| Component | Selection |
| --- | --- |
| Isaac Lab | `v3.0.0-EA`, commit `ae37b028ea415c91ea2bc32609efcd759ed2b974` |
| Isaac Sim | `6.1.0.0` Python distribution / 6.1.0 product |
| Python | 3.12; record exact patch |
| Torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 on Linux x86_64 |
| TorchRL / TensorDict | 0.14.0 / 0.14.2, subject to a successful frozen overlay resolution |
| Physics | Explicit `isaacsim_physx`; never automatic `physx` selection |
| OS | Supported Ubuntu 24.04 userspace in a dedicated environment/container |

Pin the upstream revision and its resolved environment, then resolve the NexuML/core/library requirements as a recorded overlay without replacing the simulator's Torch/CUDA/USD stack. Keep a dedicated profile manifest/lock or constraints beside the integration documentation/tooling; do not run ordinary project synchronization over Isaac's environment. Upstream EA's NumPy/Warp/USD/Pillow requirements differ from the existing `.venv`; preserve those differences explicitly rather than copying the base lock blindly. No implicit dependency upgrades or unsafe resolver changes in the ordinary workspace.

Preparation needs explicit approval for downloads, asset access/license acceptance, and any container/system action. A proposal/apply request is not an instruction to install drivers or accept an EULA. Use an existing approved runtime if available; otherwise report the precise setup request. The current session sees a 16 GB RTX 4080 SUPER and roughly 64 GB host RAM, but is an Ubuntu 26.04 container with only about 86 GiB free workspace storage. This is inventory, not a validated allocation or simulator/graphics check. Recheck resource limits, disk/cache budget, CUDA, graphics/Vulkan, and driver passthrough before installing/running; do not use `nvidia-smi` alone as compatibility evidence.

**Alternative rejected:** Install the legacy Python 3.11 profile into the current environment, follow moving `main/latest` docs, or silently switch to kit-less physics when full Isaac Sim fails.

### D2. Keep static definitions separate from native launch

Add a small library Isaac provider and three ordinary registered scenario factories. Its immutable definition stores the selected task/profile/backend, explicit per-field contract, and declared task/action parameters. The references' static contracts are validated against the native baseline; `describe()` never imports `isaaclab`, `omni`, `pxr`, or starts an app. No retained descriptor/probe simulator is permitted during discovery, compilation, YAML restoration, or package loading.

`build(context)` uses lazy imports and the pinned public launch/config/task APIs to select `isaacsim_physx`, start the launcher, register/load the task, set native N/device/seed, and create one native environment. Launcher lifetime is held by the runtime, not an open local context manager that immediately exits. Validate native single-environment fields, joint/action mappings, control period, and runtime batch width against the declared contract before exposing steps. A mismatch fails; it does not auto-rewrite a compiled policy.

Keep per-field shapes independent of N. Training returns batched TensorDicts; evaluation at N=1 removes only the singleton environment axis and restores it when passing actions to Isaac. Disable unused camera/history/privileged-critic groups for these state-only policies; do not pretend an observation-group label provides an implemented sensor/temporal contract.

**Alternative rejected:** Feed Isaac through the current Gymnasium loop, probe an app during registry scanning, serialize native task configuration objects, or add Isaac imports to core.

### D3. Reuse direct collection, with a bounded provider bridge

Start with one learner GPU and one direct synchronous collector. The collector's environment width is the native task width; simulator, rollout policy, storage, and learner use the same CUDA device. Reject process/Ray/multiple-device/asynchronous combinations and non-divisible `frames_per_batch` before allocation. Start smokes at small N, not upstream's usual 4096; increase only after measuring real memory. Example geometry: N=32 and T=32 gives 1024 collected frames, not 32.

The framework-neutral Isaac runtime remains the canonical reset/step/close boundary. First prove installed `IsaacLabWrapper` through public composition against the transition truth table in D4. If it passes, a narrow optional runtime-to-TorchRL factory hook in the reinforcement layer can expose that maintained wrapper without wrapping it again in another EnvBase. The normal factory still uses `_TorchRLEnvironmentAdapter` for all other providers. The returned bridge's public close path must close the owned runtime/launcher as well as the native env.

If the pinned wrapper cannot preserve D4 through public configuration/info-reader/transform composition, the predetermined fallback is the existing generic TorchRL adapter over the Isaac runtime's correct cached reset/final-state mapping. Record the failing wrapper probe and fallback choice. This does not permit an alternate collector/trainer, dependency upgrade, private TorchRL monkey patch, or mandatory TorchRL methods on every core environment. Add only the factory/helper actually needed; no bridge registry or speculative option family.

Use the installed collector's supported GPU options, including `no_cuda_sync` where required, after the native probe verifies behavior. Keep existing synchronization and collect → update → synchronize → next collect ordering. Preserve rollout environment/time axes through GAE before flattening PPO minibatches. Scalar metric reporting can synchronize; eliminate CPU/NumPy action and observation round trips, not necessary correctness checks.

**Alternative rejected:** Double-vectorize using process workers, run RSL-RL as NexuML's learner, or rewrite the current PPO loop for Isaac.

### D4. Establish exact final-state/reset semantics before real PPO

Enable upstream `compute_final_obs` for supported manager-based tasks. Normalize each native step into two explicit meanings: pre-reset final observations for completed transitions, and cached post-reset observations for the next action. Clone mutable reward/done/state data before later native calls can overwrite it. A subsequent reset for an already auto-reset row returns its cached initial observation without resetting physics again. Explicit user resets still use the supported public native per-index reset API.

The compatibility probe covers no rows done, a partial done mask, all rows done, true termination, time-limit truncation, simultaneous terminal/truncation flags, and mutable reused buffers. Distinct final/reset values and a controlled value function establish the expected GAE targets. Do not assume `native_autoreset=True` solves this: the installed wrapper documents unavailable/NaN terminal observations unless additional handling supplies them. Neither zero-filling NaNs nor treating timeouts as true terminations is acceptable.

The same normalization supplies correct evaluation episode boundaries. Replay resets episodes deliberately rather than allowing returned post-reset state to be labeled terminal. Missing final observations on a supported path are a visible compatibility failure, not a green finite-loss smoke.

**Alternative rejected:** Bootstrap from the new episode, double-reset at done, or implement unrelated PPO/GAE changes to hide provider errors.

### D5. Use explicit bounded task actions

Reuse the current bounded Gaussian policy and declare normalized finite task inputs, initially within [-1, 1], with the exact per-task physical mapping recorded in the immutable provider configuration and provenance. Preserve the distribution/log-probability mapping; do not clip sampled actions differently behind the loss's back. Native comparison runs apply the identical envelope/mapping and document any difference from unmodified upstream training defaults.

Reference order and proposed registry identities:

| NexuML reference | Native task / meaning |
| --- | --- |
| `isaac-cartpole-ppo` | Manager-based `Isaac-Cartpole`; one continuous effort input, upstream force scale 100 N, policy period derived from dt × decimation |
| `isaac-franka-reach-ppo` | `Isaac-Reach-Franka`; upstream joint-position action preset, named joints, default-pose offsets/scale and actual limits validated |
| `isaac-so101-joint-reach-ppo` | Minimal local manager-based joint-reaching task using the pinned upstream SO-101 asset; D6 |

Use manager-based Cartpole rather than the direct variant so all three first references share public per-index reset and pre-reset recorder hooks. This is a task selection, not a new environment framework. Native reward/transition parity is checked with matching actions/settings; RSL-RL and NexuML PPO training curves are not promised identical.

**Alternative rejected:** Reuse categorical CartPole controls, infer finite bounds from Isaac's unbounded Box, or silently reinterpret normalized outputs as raw motor commands.

### D6. Make SO-101 a required, minimal reward-bearing reference

The pinned upstream tree includes `SO101_CFG` and `IsaacContrib-Stack-Cube-SO101-v0`, but the inherited stack configuration has `rewards=None` and targets demonstrations/teleoperation. Reuse its maintained robot asset, not its reward-free task as a fictitious PPO reference. Add only a small library-owned manager-based task configuration and necessary reward/termination terms using public Isaac APIs; do not copy its simulator, robot model, IK controller, or drivers.

Use the explicit `physx` asset variant and five ordered arm joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`. Validate the separate `gripper` joint and hold it at one documented in-limit fixed position; it is not a sixth policy control. Read and record the actual asset limits, actuator parameters, joint units, initial pose, and source/content identity. Validate signs/response with bounded scripted native motion before training.

Observations contain the five arm positions, velocities, and target positions with radians/radians-per-second semantics. Decode normalized actions into a declared in-limit joint-position target interval using the upstream position-action machinery. Pick reset/goal targets from a small seeded set of native-validated reachable joint configurations. This is joint reaching, not arbitrary 6-DOF end-effector pose control; the five-DOF arm is not assumed to reach arbitrary poses.

Use a negative mean squared joint error normalized by declared joint range and multiplied by policy step duration. Define target tolerance/required hold interval and timeout as explicit task constants/configuration, verify them in controlled native episodes, and log mean joint error plus success separately from return. No reward shaping search or learned-success threshold is required. Success in a scripted task check validates the outcome calculation; a short PPO run validates integration, not manipulation mastery.

Use `InteractionStep.info` for the small declared task outcomes (`task_success` and `joint_error_rad`) and aggregate them in the existing episode evaluator only when supplied. Preserve all established scalar keys; add per-episode success-rate/final-error aggregates without a new metric registry or public result hierarchy. Reject nonfinite/mismatched task outcomes and clear them between episodes. Other providers need not supply them.

SO-101 is implemented/tested after Cartpole and Franka, but cannot be waived by passing those two. Required evidence includes native motion/reward checks, real actor updates, finite losses, interrupted resume, clean policy inference, fresh-process replay, and media. Unavailable assets or failed compatibility leave SO-101 and change completion explicitly blocked; they do not authorize dropping this requirement or substituting Franka.

**Alternative rejected:** Make SO-101 docs-only, start by training sparse-reward cube stacking, use a mixed arm/binary-gripper policy, create an IK stack, or claim hardware calibration/safety.

### D7. Use fresh-process replay instead of competing in-fit simulators

Initial Isaac reference training config sets evaluation episodes to zero and start/periodic/video options off explicitly. Add a narrow provider-declared lifecycle restriction checked by existing fit/session preflight: unsupported automatic evaluation/media for Isaac fails before runtime allocation with a `nexuml evaluate` fresh-process example. The default behavior for non-Isaac providers and their scheduled evaluation remains unchanged. Do not silently ignore requested settings or open/close another simulator while training is alive.

After training/shutdown, the existing export and evaluation commands run in separate normal Python processes using the isolated profile. The user sets a positive replay episode count with existing overrides; no child-worker orchestration, IPC protocol, or new evaluation CLI is required. The N=1 provider runtime feeds the existing episode loop; the policy package contains no native handles/critic/optimizer. Actual callable inference in a clean process with simulator/learner/encoder imports blocked is distinct from simulation replay, which requires Isaac/assets.

Resume uses the existing training checkpoint path in a new process; restore actor/critic/optimizers/counters, build/reset a fresh simulation, and collect only remaining frames. Do not promise exact interrupted physics/randomized-trajectory continuation. A completed budget allocates no training simulator. Use only explicit source-frame metadata for replay logging, otherwise native step zero; never rename optimizer steps as environment frames. Preserve the existing source-import dependency-manifest meaning rather than redesigning it here.

**Alternative rejected:** Add a second active context, reuse/reset the collector for evaluation, silently spawn evaluation workers on the only 16 GB GPU, or construct a zero-budget learner for replay.

### D8. Reuse native media, with correct pre-reset capture

Rendering is optional and enabled only in fresh-process N=1 replay. Configure the supported headless camera before launcher/environment creation; training stays state-only without cameras. Return CPU uint8 HWC RGB frames to the existing bounded capture/media sink; preserve GIF/MP4/both selection, timing/size bounds, native TensorBoard previews, MLflow-only behavior, staging, and failure diagnostics. Do not use the upstream MoviePy recorder or a separate logging run.

For manager-based auto-reset episodes, use a minimal public pre-reset recorder term to cache the ending episode's RGB frame before physics reset. Set dataset export to `EXPORT_NONE`; this hook produces no HDF5 dataset or parallel video writer and retains at most the required cached frame. On terminal steps `render_frame()` returns that frame, not the reset scene. Prove frame/episode identity with synthetic distinct scenes and real terminal replay. A missing supported renderer/frame is a clear failure, never a silent backend switch or a reset frame mislabeled terminal.

Record capture FPS from actual policy/camera sampling semantics, with renderer identity and timing, and reuse existing encoders/native artifact dispatch. Both formats use one bounded rollout. GPU-to-CPU transfer is allowed for requested replay RGB frames, not for training actions/observations.

**Alternative rejected:** Assume SDL classic-control rendering validates Isaac, render thousands of training cameras by default, or copy the upstream video pipeline.

### D9. Make completion evidence explicit and keep the diff bounded

Fast tests use synthetic state/transition/launcher fakes and the current CPU environment. Heavy tests require both existing GPU/simulator markers plus explicit user opt-in, the isolated profile, and approved cached assets. Reuse existing manual simulator CI conventions; a dedicated Isaac profile job must be opt-in and reference a prepared runtime rather than making default CI download Isaac. No full GPU run is claimed from a skip or a workflow edit.

Add a Phase 2 report (planned `reports/physical-ai/phase-2-isaaclab.md`) with exact commands, resolved versions, sources/asset identities, resource prerequisites, timings/counts, native-to-NexuML checks, all three reference round trips, media/playback evidence, and outstanding limits. Preserve original live ROS/Gazebo/CARLA tasks and Phase 0–1 results. Existing DALI/Optuna failures, 37 baseline type diagnostics, and original strict-spec warnings are recorded separately, not hidden, installed away, or fixed outside scope.

Expected edit homes: library Isaac provider/task/scenario modules; a small optional bridge in existing reinforcement factory code if needed; targeted session/evaluator preflight/task-outcome handling; dedicated integration profile/docs/tests/report and manual CI wiring. Core policy/compiler/action adapters/logger encoders should be reused unchanged unless a failing in-scope regression proves a narrow correction necessary. Do not edit original OpenSpec requirements to claim coverage or add mandatory base Isaac dependencies.

**Alternative rejected:** Treat static compilation or a 32-frame smoke as robot-learning quality, mark all later roadmap phases complete, or repair unrelated baseline gates while adding robotics.

## Risks / Trade-offs

- **[EA drift, dependency/ABI conflict]** → immutable revision/profile and native-first checks; incompatible pins stop the integration gate instead of upgrading unrelated packages.
- **[Insufficient disk, GPU graphics access, or memory]** → resource/compatibility checks before approved installation, explicit asset caching, low native N, and rendering only during N=1 replay.
- **[Incorrect but finite PPO targets]** → numerical final/reset-state GAE oracle tests for partial/all-done cases before real updates.
- **[Wrapper support differs from documentation]** → inspect/probe the actual pinned wrapper; use only public composition or the predetermined existing generic-adapter fallback, with evidence.
- **[SO-101 source is mistaken for a ready learning task]** → reuse its asset, implement explicit minimal reward-bearing joint reaching, and require real SO-101 round trips for completion.
- **[Auto-reset corrupts recorded ending frames]** → bounded public pre-reset cache, no dataset/video side effects, and terminal scene-identity tests.
- **[Disabled in-fit Isaac evaluation surprises users]** → explicit reference defaults, early actionable rejection, and documented separate replay; Gymnasium scheduling remains supported.
- **[Simulation proof is misread as hardware readiness]** → document normalized task controls, absent calibration/safety validation, and simulation-only scope.

## Migration Plan

1. Record the current worktree baseline and operator-approved installation/resource profile; verify pinned native Cartpole/Franka before adapter claims.
2. Prove transition/launcher behavior with CPU tests and a bounded native wrapper spike; implement the minimal provider/bridge with no default-path changes.
3. Verify Cartpole then Franka full training/resume/export/replay paths; add the required SO-101 joint task and the same full acceptance afterward.
4. Integrate replay-only media, task outcomes, provenance, documentation and opt-in CI; run regression/package/import-isolation gates and actual GPU/media proofs for all three.
5. Mark implementation tasks/roadmap progress only against evidence; commit/merge/archive remain separate user-directed actions.

Rollback removes the optional provider/scenarios/profile and targeted bridge/preflight wiring, preserving normal supervised/Gymnasium code paths and previously generated policies/media. Never remove an operator's simulator installation, caches, checkpoints, or other work automatically. Any failed SO-101/GPU gate leaves corresponding tasks open and a report/backlog entry; it is not completion of a smaller unapproved phase.

## Sources checked during proposal preparation

- [EA release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA) and [pinned profile](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/pyproject.toml): release status, Python/Torch/Isaac pins, dependency ownership.
- [Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html): supported OS/graphics/driver requirements, not proof of this container's usability.
- [Pinned manager environment](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab/isaaclab/envs/manager_based_rl_env.py): same-step reset, optional final observations, pre-reset recording and simulation ownership.
- [TorchRL 0.14 Isaac guide](https://docs.pytorch.org/rl/0.14/reference/isaaclab.html) and installed `envs/libs/isaac_lab.py` / `transforms/_misc.py`: maintained wrapper, batch/device mapping and terminal/reset limitations.
- [Pinned SO-101 asset](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_assets/isaaclab_assets/robots/so101.py), [joint-position example](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/so101/stack_joint_pos_env_cfg.py), and [reward-free stack base](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/stack_env_cfg.py): five arm joints, separate gripper, physics variant and why a small reaching task is needed.

Source inspection and resource inventory were read-only. No Isaac/GPU workload, compatibility test, installation, or new application code was executed while preparing these artifacts.
