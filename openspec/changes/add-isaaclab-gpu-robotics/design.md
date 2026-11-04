# Design

## Context

See `proposal.md` for motivation and the five capability specs for observable acceptance. Sections D1–D9 describe the initial EA/PhysX integration; its 42 tasks are complete with evidence in `reports/physical-ai/phase-2-isaaclab.md`. The approved extension adds existing-DevContainer execution, a TorchRL audit and the upstream Newton-based vial-placement reference (D10–D12). Planning those additions is not execution evidence.

Existing reuse seams:

- `EnvironmentDefinition.describe/build`, `EnvironmentBuildContext`, `InteractionContract`, `InteractionStep`, and the framework-neutral reset/step/close protocol.
- `src/nexuml/reinforcement/{collector,torchrl_env}.py`: execution-site factory, direct collector, GPU device settings, policy synchronization, and finite rollout iterable.
- `NexuSession` / `NexuRLLightningModule`: Lightning-owned PPO, actual frame counters, checkpoint resume, and explicit collection shutdown.
- Complete-policy export/load, ordinary registry/YAML/overrides, native logging/artifact staging, `evaluate_policy`, and the inference-only `evaluate` command.

The initial integration uses the existing generic TorchRL adapter over canonical Isaac final/reset-state normalization, after real native wrapper probes failed terminal/reset requirements. Its launcher is lazy and process-owned; replay is fresh-process N=1 only. The current shell launcher assumes `/isaac-sim` and `/workspace/nexuml`, and the Python process-boundary module imports Torch before application startup. These are concrete DevContainer/audit concerns, not evidence that the newly selected tutorial profile works.

The installed/verified stack is Python 3.12.13, Torch 2.11.0+cu128, TorchRL 0.14.0, TensorDict 0.14.2, Lightning 2.6.5, and Gymnasium 1.3.0. NexuML's declared Python minimum is 3.12; Torch 2.11 is a verified lock choice, not its declared minimum. Isaac Lab 2.3.2's documented Python 3.11 profile is therefore not this integration's target. The user selected 3.0 Early Access.

## Goals / Non-Goals

**Goals:** Two explicit isolated profiles in the existing NVIDIA DevContainer; one GPU/native batch per job; preserve the initial three references and add upstream SO-101 vial placement; version-aware TorchRL correctness; complete checkpoint/export/replay/media round trips; unchanged CPU/supervised behavior and portable existing artifacts.

**Non-Goals:** No alternative trainer/session, RL algorithm rewrite, generic simulator/plugin system, GPU process/Ray collection, asynchronous/multi-GPU execution, general backend matrix, visual-policy training/distillation, inverse-kinematics development, learned manipulation threshold, IL/VLA, sim2real preset, hardware, or unrelated baseline repair. Newton expansion is limited to the selected upstream state task; rendering is for bounded replay inspection only.

## Decisions

### D1. Freeze the initial isolated EA profile before integration

Initial profile, now backed by the completed baseline report; DevContainer-native installation still requires its own proof:

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

Preparation requires approved downloads/asset access and operator-controlled container/system actions. Standing EULA acceptance is already recorded; optional telemetry is not approved. Recheck effective resource limits, disk/cache budget, CUDA, graphics/Vulkan and driver passthrough before installation or GPU work; do not reuse historical inventory or treat `nvidia-smi` alone as simulator compatibility evidence. Never restart the active DevContainer or disturb other experiments implicitly.

**Alternative rejected:** Install the legacy Python 3.11 profile into the current environment, follow moving `main/latest` docs, or silently switch to kit-less physics when full Isaac Sim fails.

### D2. Keep static definitions separate from native launch

Keep the small library Isaac provider and ordinary scenario factories. Its immutable definition stores the selected task/profile/backend, explicit per-field contract, and declared task/action parameters. The references' static contracts are validated against the native baseline; `describe()` never imports `isaaclab`, `omni`, `pxr`, or starts an app. No retained descriptor/probe simulator is permitted during discovery, compilation, YAML restoration, or package loading.

For the initial profile, `build(context)` uses lazy public launch/config/task APIs to select `isaacsim_physx`, hold the launcher lifetime, set native N/device/seed and create one environment. The tutorial profile uses its own pinned kit-less public launch path (D10–D11), not an unconditional Kit launch. Validate native fields, joint/action mappings, control period and runtime batch width before exposing steps. A mismatch fails; it does not auto-rewrite a compiled policy.

Keep per-field shapes independent of N. Training returns batched TensorDicts; evaluation at N=1 removes only the singleton environment axis and restores it when passing actions to Isaac. Disable unused camera/history/privileged-critic groups for these state-only policies; do not pretend an observation-group label provides an implemented sensor/temporal contract.

**Operator-approved lifecycle exception (2026-10-19):** The pinned native returning-close path crashes during Python finalization: Lab EA retains an extension hook and timeline interface past Kit shutdown. The user explicitly rejected patching upstream and authorized non-graceful shutdown. Keep upstream byte-identical and use its public fast-close API at the **whole Python job boundary**, after native environment/collector cleanup, checkpoint/export and explicit logger finalization. A small same-process Python entrypoint runs the existing CLI or user script; it is not a trainer, worker or IPC layer. App allocation stays lazy; completed-budget/discovery jobs allocate none. Refuse runtime allocation outside that process scope or against a caller-owned app. Pass actual success/failure/SystemExit/interrupt status to public `SimulationApp.close(exit_code=...)`; do not let Kit's default zero exit turn failures green. Python `atexit` callbacks are not a promised cleanup mechanism under fast exit. Returning shutdown remains a documented upstream defect, not a repaired or graceful path.

**Alternative rejected:** Feed Isaac through the current Gymnasium loop, probe an app during registry scanning, serialize native task configuration objects, or add Isaac imports to core.

### D3. Reuse direct collection, with a bounded provider bridge

Start with one learner GPU and one direct synchronous collector. The collector's environment width is the native task width; simulator, rollout policy, storage, and learner use the same CUDA device. Reject process/Ray/multiple-device/asynchronous combinations and non-divisible `frames_per_batch` before allocation. Start smokes at small N, not upstream's usual 4096; increase only after measuring real memory. Example geometry: N=32 and T=32 gives 1024 collected frames, not 32.

The framework-neutral Isaac runtime remains the canonical reset/step/close boundary. First prove installed `IsaacLabWrapper` through public composition against the transition truth table in D4. If it passes, a narrow optional runtime-to-TorchRL factory hook in the reinforcement layer can expose that maintained wrapper without wrapping it again in another EnvBase. The normal factory still uses `_TorchRLEnvironmentAdapter` for all other providers. The returned bridge's public close path must close the owned runtime/launcher as well as the native env.

The pinned wrapper did not preserve D4 through public composition; the baseline therefore uses the predetermined existing generic adapter over correct cached reset/final-state mapping. Retain its native probe evidence. Re-probe on the tutorial profile rather than assuming the old wrapper verdict transfers. Neither audit permits an alternate collector/trainer, private TorchRL patch or mandatory TorchRL methods on every core environment. Any extension overlay version choice is explicit and isolated (D10); no unrelated dependency upgrade or bridge registry.

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
| `isaac-so101-vial-place-ppo` (extension) | Upstream `IsaacTutorial-Place-Vial-SO101`, separately pinned Newton MJWarp profile; D12 |

Use manager-based Cartpole rather than the direct variant so all three first references share public per-index reset and pre-reset recorder hooks. This is a task selection, not a new environment framework. Native reward/transition parity is checked with matching actions/settings; RSL-RL and NexuML PPO training curves are not promised identical.

**Alternative rejected:** Reuse categorical CartPole controls, infer finite bounds from Isaac's unbounded Box, or silently reinterpret normalized outputs as raw motor commands.

### D6. Make SO-101 a required, minimal reward-bearing reference

The pinned upstream tree includes `SO101_CFG` and `IsaacContrib-Stack-Cube-SO101-v0`, but the inherited stack configuration has `rewards=None` and targets demonstrations/teleoperation. Reuse its maintained robot asset, not its reward-free task as a fictitious PPO reference. Add only a small library-owned manager-based task configuration and necessary reward/termination terms using public Isaac APIs; do not copy its simulator, robot model, IK controller, or drivers.

Use the explicit `physx` asset variant and five ordered arm joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`. Validate the separate `gripper` joint and hold it at one documented in-limit fixed position; it is not a sixth policy control. Read and record the actual asset limits, actuator parameters, joint units, initial pose, and source/content identity. Validate signs/response with bounded scripted native motion before training.

Observations contain the five arm positions, velocities, and target positions with radians/radians-per-second semantics. Decode normalized actions into a declared in-limit joint-position target interval using the upstream position-action machinery. Pick reset/goal targets from a small seeded set of native-validated reachable joint configurations. This is joint reaching, not arbitrary 6-DOF end-effector pose control; the five-DOF arm is not assumed to reach arbitrary poses.

Use a negative mean squared joint error normalized by declared joint range and multiplied by policy step duration. Define target tolerance/required hold interval and timeout as explicit task constants/configuration, verify them in controlled native episodes, and log mean joint error plus success separately from return. No reward shaping search or learned-success threshold is required. Success in a scripted task check validates the outcome calculation; a short PPO run validates integration, not manipulation mastery.

Use `InteractionStep.info` for the small declared task outcomes (`task_success` and `joint_error_rad`) and aggregate them in the existing episode evaluator only when supplied. Preserve all established scalar keys; add per-episode success-rate/final-error aggregates without a new metric registry or public result hierarchy. Reject nonfinite/mismatched task outcomes and clear them between episodes. Other providers need not supply them.

SO-101 is implemented/tested after Cartpole and Franka, but cannot be waived by passing those two. Required evidence includes native motion/reward checks, real actor updates, finite losses, interrupted resume, clean policy inference, fresh-process replay, and media. Unavailable assets or failed compatibility leave SO-101 and change completion explicitly blocked; they do not authorize dropping this requirement or substituting Franka.

**Alternative rejected:** Make SO-101 docs-only, rename reaching as pick-and-place, start with sparse-reward cube stacking, use a mixed arm/binary-gripper policy, create an IK stack, or claim hardware calibration/safety. D12's separate six-continuous-control task does not alter this five-control reference.

### D7. Use fresh-process replay instead of competing in-fit simulators

Initial Isaac reference training config sets evaluation episodes to zero and start/periodic/video options off explicitly. Add a narrow provider-declared lifecycle restriction checked by existing fit/session preflight: unsupported automatic evaluation/media for Isaac fails before runtime allocation with a `nexuml evaluate` fresh-process example. The default behavior for non-Isaac providers and their scheduled evaluation remains unchanged. Do not silently ignore requested settings or open/close another simulator while training is alive.

After training/shutdown, the existing export and evaluation commands run in separate normal Python processes using the isolated profile. The user sets a positive replay episode count with existing overrides; no child-worker orchestration, IPC protocol, or new evaluation CLI is required. The N=1 provider runtime feeds the existing episode loop; the policy package contains no native handles/critic/optimizer. Actual callable inference in a clean process with simulator/learner/encoder imports blocked is distinct from simulation replay, which requires Isaac/assets.

Resume uses the existing training checkpoint path in a new process; restore actor/critic/optimizers/counters, build/reset a fresh simulation, and collect only remaining frames. Do not promise exact interrupted physics/randomized-trajectory continuation. A completed budget allocates no training simulator. Use only explicit source-frame metadata for replay logging, otherwise native step zero; never rename optimizer steps as environment frames. Preserve the existing source-import dependency-manifest meaning rather than redesigning it here.

**Alternative rejected:** Add a second active context, reuse/reset the collector for evaluation, silently spawn evaluation workers on the only 16 GB GPU, or construct a zero-budget learner for replay.

### D8. Reuse native media, with correct pre-reset capture

Rendering is optional and enabled only in fresh-process N=1 replay. The initial profile uses Isaac RTX; the tutorial uses its pinned Newton renderer (D12). Configure the selected headless camera before environment creation; training stays state-only without cameras. Return CPU uint8 HWC RGB frames to the existing bounded media sink; preserve GIF/MP4/both selection, animated TensorBoard previews, MLflow-only behavior, staging and diagnostics. Do not use an upstream video writer or a separate logging run.

For manager-based auto-reset episodes, use a minimal public pre-reset recorder term to cache the ending episode's RGB frame before physics reset. Set dataset export to `EXPORT_NONE`; this hook produces no HDF5 dataset or parallel video writer and retains at most the required cached frame. On terminal steps `render_frame()` returns that frame, not the reset scene. Prove frame/episode identity with synthetic distinct scenes and real terminal replay. A missing supported renderer/frame is a clear failure, never a silent backend switch or a reset frame mislabeled terminal.

Record capture FPS from actual policy/camera sampling semantics, with renderer identity and timing, and reuse existing encoders/native artifact dispatch. Both formats use one bounded rollout. GPU-to-CPU transfer is allowed for requested replay RGB frames, not for training actions/observations.

**Alternative rejected:** Assume SDL classic-control rendering validates Isaac, render thousands of training cameras by default, or copy the upstream video pipeline.

### D9. Make completion evidence explicit and keep the diff bounded

Fast tests use synthetic state/transition/launcher fakes and the current CPU environment. Heavy tests require both existing GPU/simulator markers plus explicit user opt-in, the isolated profile, and approved cached assets. Reuse existing manual simulator CI conventions; a dedicated Isaac profile job must be opt-in and reference a prepared runtime rather than making default CI download Isaac. No full GPU run is claimed from a skip or a workflow edit.

Add a Phase 2 report (planned `reports/physical-ai/phase-2-isaaclab.md`) with exact commands, resolved versions, sources/asset identities, resource prerequisites, timings/counts, native-to-NexuML checks, all three reference round trips, media/playback evidence, and outstanding limits. Preserve original live ROS/Gazebo/CARLA tasks and Phase 0–1 results. Existing DALI/Optuna failures, 37 baseline type diagnostics, and original strict-spec warnings are recorded separately, not hidden, installed away, or fixed outside scope.

Expected edit homes: library Isaac provider/task/scenario modules; a small optional bridge in existing reinforcement factory code if needed; targeted session/evaluator preflight/task-outcome handling; dedicated integration profile/docs/tests/report and manual CI wiring. Core policy/compiler/action adapters/logger encoders should be reused unchanged unless a failing in-scope regression proves a narrow correction necessary. Do not edit original OpenSpec requirements to claim coverage or add mandatory base Isaac dependencies.

**Alternative rejected:** Treat static compilation or a 32-frame smoke as robot-learning quality, mark all later roadmap phases complete, or repair unrelated baseline gates while adding robotics.

### D10. Use the existing NVIDIA DevContainer with two isolated Python environments

Reuse `/workspaces/NexuVibe/.devcontainer/nvidia/devcontainer.json` and its shared Dockerfile. Inspect and preserve staged changes: the NVIDIA profile already selects Ubuntu 24.04 and `compute,utility,graphics`, with EGL/Vulkan preparation and standing EULA acceptance. The observed running container was Ubuntu 26.04 with compute-only capabilities and no Vulkan ICDs; prepared files do not retrofit it. Coordinate any rebuild/recreation with the operator and active experiments. Keep the normal development tools, mounts, hooks and `/env`; add neither a separate development profile nor a required worker container. NexuVibe edits must be reviewed in that repository, not assumed delivered by this NexuML change.

Prepare separate persistent Python 3.12 environments for EA/PhysX and the tutorial, using explicit interpreter paths and recorded locks/constraints. The tutorial candidate is pinned from the inspected upstream lock, not moving installation examples:

| Component | Tutorial candidate (not yet native-validated) |
| --- | --- |
| IsaacLabTutorial | `9a4286b4bc8a5720ac97f87e5e0140f7542c22d9` |
| Isaac Lab | 3.0.0, `d061bf35a58273ef0cc8aceae76392166f9b7d15` from the tutorial lock |
| Physics | Explicit `newton_mjwarp`; Newton 1.6.1, MuJoCo/MJWarp 3.12.0, Warp 1.17.0 |
| Torch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 |
| Python / OS | 3.12 (record patch), Linux x86_64 / existing Ubuntu 24.04 NVIDIA DevContainer |
| TorchRL / TensorDict / NexuML | Begin with 0.14.0 / 0.14.2 and editable core/library; freeze the successful combined overlay before runtime claims |
| Rendering | Pinned Newton renderer for replay only; no Isaac Sim requirement for the tutorial |

Clone only the pinned sources and materialize required Git LFS assets/reset data, recording their hashes/licenses. Treat downloaded reset `.pt` data as trusted executable-format input and verify the pinned source/content before loading. Preserve upstream source/lock files and original reset data. Resolve the NexuML overlay without replacing either profile's Torch/CUDA/physics/USD stack; record all resolved distributions and run dependency/import-origin checks. The ordinary project lock and `/env` remain unchanged. Unset or explicitly override inherited `UV_PROJECT_ENVIRONMENT`, `VIRTUAL_ENV`, toolchain/index selectors and library search paths when operating in the isolated environment; the existing `/env` CUDA-library path must not leak into another profile. Do not infer the actual general Python version from documentation.

Adapt existing `scripts/isaaclab/run.py` / `run.sh` to explicit environment/repository locations rather than hardcoded worker paths, preserving same-process CLI/script execution and truthful status. Add only the profile-specific launch handling needed; no new daemon, trainer or worker protocol. Document editable installation, VS Code interpreter/debug selection, exact native/NexuML commands, cache/space budgets and recovery. Metadata-only package presence checks must not initialize simulation. GPU/native probes validate runtime and graphics separately; CUDA-only Newton startup is not an RTX/Vulkan pass. Keep EULA approval separate from optional privacy/telemetry consent; never copy `PRIVACY_CONSENT=Y` from the guide.

**Alternative rejected:** Mix cu128/cu130 installations into `/env`, upgrade all existing references to the tutorial release, start a second development container, install a host driver in the container, follow moving `main` pins, suppress installation failures with `|| true`, or adopt nightlies to imitate documentation examples.

### D11. Audit the real TorchRL boundary, not just the wrapper name

Create a versioned audit section in the Phase 2 report using the user's TorchRL knowledge-base URL, the installed TorchRL source and both pinned Isaac profiles. For each applicable rule record inspected paths/versions, pass/fail/justified-not-applicable status, exact tests/commands and retained evidence. The original 15 native acceptance passes are baseline evidence, not new-profile or in-DevContainer proof.

Required audit gates:

- **Startup and ownership:** The guide requires `AppLauncher` before Torch for Isaac Sim jobs. Current `run.py` imports `_isaaclab_runtime.py`, which imports Torch at module scope; runtime task imports also precede app launch. Resolve that mismatch against the pinned EA launch contract and fresh-process evidence. Use an import-safe bootstrap if the profile requires early launch; a version-specific exception needs upstream-source justification and real startup/cancellation tests. Newton is kit-less and must not allocate a Kit app just to follow Sim-specific guidance. Discovery, export, portable inference and completed-budget resume still allocate no simulator. Unsupported ordering remains an open gate, not a documentation-only waiver.
- **Native topology/specs:** One native batch and direct collector; CUDA observation/action/storage/learner alignment; no outer vectorizer, CPU/NumPy control round trips or hidden device migration. Validate TorchRL specs on an isolated disposable native instance, never by disturbing the training environment. Camera helpers are not substitutes for a real sensor.
- **Transitions and values:** Re-probe the actual wrapper and existing-adapter fallback with no-done, partial/all-done, termination, truncation, simultaneous flags, explicit masked resets and reused buffers. Require distinct pre-reset final/post-reset initial states, exactly one native reset and numerical GAE targets. `native_autoreset=True` alone and finite losses are not correctness proofs. Never zero-fill NaNs or lose timeout bootstrapping.
- **On-policy updates:** Preserve N×T axes through GAE, real frame counters, collect → update → explicit copied-policy sync → next collect ordering, bounded action/log-probability agreement, remaining-budget resume and actual changed actor parameters. The guide's recurrent primers, shifted estimators, asynchronous eval and multi-GPU suggestions do not mandate adding those features to this feed-forward single-GPU slice.
- **Lifecycle/media:** One simulation per process, early rejection of competing in-fit evaluation, metadata-only installation checks, selected renderer prerequisites, explicit logger finalization and truthful failure/cancellation. Keep the approved whole-job fast close for EA/Kit; use owned public normal cleanup for kit-less Newton rather than unconditional process termination.

Select the maintained wrapper only if public composition passes every supported transition gate; otherwise retain the existing generic adapter over canonical provider normalization with profile-specific evidence. Fix in-scope failures at the narrowest responsible layer, add regressions and rerun both profiles. No upstream/private patches or new collection engine.

**Alternative rejected:** Declare integration correct because a wrapper imports, replace the proven fallback solely to match an example, or broaden scope to recurrent/multi-device training while auditing.

### D12. Integrate the upstream SO-101 vial-placement state task

Use the pinned tutorial package's task-discovery entry point and public `IsaacTutorial-Place-Vial-SO101` configuration. Reuse its SO-101/desk/vial/rack assets, contact model, reward/termination terms and reset curriculum rather than copying the task or upstream learner. Add the normal registered `isaac-so101-vial-place-ppo` scenario and an explicit profile/task pairing in the existing provider. Existing EA/PhysX references, serialized configurations, checkpoints and five-action reaching packages keep their identity; six-action vial packages must reject incompatible task/profile/control overrides even if shapes happen to match.

The six ordered continuous controls are `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. Under the nominal tutorial configuration, targets are measured joint positions plus normalized action × 0.033 rad for the arm and × 0.02 rad for the jaw, updated with upstream relative-action semantics at each physics step. Physics is 120 Hz with decimation four (30 Hz policy) and a 20-second timeout. Verify actual joint limits, actuator responses, units and XYZW quaternion/frame conventions in this profile; do not reuse EA's absolute-target mapping or fixed gripper. Keep the bounded Gaussian/log-probability semantics without unrecorded clipping.

Use the tutorial's concatenated `policy` state group, including its progress flags, for NexuML's existing actor and critic. Do not add a privileged-critic interface merely to copy the upstream RSL-RL architecture; disable the unused native `critic` group and record this symmetric-critic deviation explicitly. Validate the full state field ordering/dimensions statically against a native probe, with no privileged data silently added to the exported actor. Native comparisons use matching observation/action/reset settings; differing learner performance is not parity failure.

Training retains the pinned eight-phase reset curriculum and its nominal mass/friction variation. Replay uses upstream play-mode semantics: canonical home-phase starts with its nominal evaluation settings, requested N=1 and traceable seed/dataset row. A release-phase curriculum success is not a complete tabletop-to-rack trajectory. Do not enable `sim2real`, regenerate/overwrite the reset dataset or turn deterministic policy inference into Warp-wide deterministic-kernel mode; the tutorial documents a collision-buffer failure for that mode.

Validate contact-based pickup, lift, transport, insertion and release with a bounded native scripted/reference trajectory from a canonical home start. Do not attach/weld the vial, teleport it after reset, remove required contacts or substitute an upstream policy's success for NexuML learning. Test positive/negative success predicates using controlled states as a separate proof: the vial must be upright, mechanically seated in a rack hole, released and nearly still for the pinned ten-control-step hold, with required progress history. Proximity, held/airborne vials, curriculum phase selection and shaped return alone are not successful placement. Reuse the existing evaluator for `task_success` and the small available grasp/lift/insertion outcomes, cached before autoreset; no fabricated joint-reaching error and no general metric registry. Preserve established reaching aggregates unchanged.

Train through `ScenarioSpec` → `NexuSession` → Lightning/TorchRL/PPO, not the tutorial's RSL-RL runner. Require real updates, checkpoint/resume/remaining-budget proofs, complete-policy export and exact CPU inference, then fresh-process scalar and rendered replay. Routine tests need no stochastic convergence threshold; report observed full-home placement success separately and do not claim the trained NexuML policy masters the task from an upstream/scripted trajectory or a bounded update smoke.

For bounded N=1 replay, add a passive inspection camera through the pinned Newton renderer, without changing state-policy inputs or selecting the camera-training task. Prove headless RGB and pre-reset terminal scene identity using public hooks before calling it supported. Feed the existing media sink: GIF, MP4 or both from one captured rollout; animated TensorBoard previews; native MLflow-only/combined behavior; optional MP4 dependencies. Revalidate failure/cancellation cleanup and truthful logger/job status. Missing rendering support leaves the recorded-replay gate open; neither silently switching to PhysX/RTX nor dropping required media is acceptable.

**Alternative rejected:** Port vial placement back to EA, keep the gripper fixed, rename joint reaching, train the camera/distillation task, require a new algorithm/privileged-observation API, or claim tutorial sim-to-real results as NexuML evidence.

## Risks / Trade-offs

- **[EA drift, dependency/ABI conflict]** → immutable revision/profile and native-first checks; incompatible pins stop the integration gate instead of upgrading unrelated packages.
- **[Insufficient disk, GPU graphics access, or memory]** → resource/compatibility checks before approved installation, explicit asset caching, low native N, and rendering only during N=1 replay.
- **[Incorrect but finite PPO targets]** → numerical final/reset-state GAE oracle tests for partial/all-done cases before real updates.
- **[Wrapper support differs from documentation]** → inspect/probe the actual pinned wrapper; use only public composition or the predetermined existing generic-adapter fallback, with evidence.
- **[SO-101 source is mistaken for a ready learning task]** → reuse its asset, implement explicit minimal reward-bearing joint reaching, and require real SO-101 round trips for completion.
- **[Auto-reset corrupts recorded ending frames]** → bounded public pre-reset cache, no dataset/video side effects, and terminal scene-identity tests.
- **[Disabled in-fit Isaac evaluation surprises users]** → explicit reference defaults, early actionable rejection, and documented separate replay; Gymnasium scheduling remains supported.
- **[Simulation proof is misread as hardware readiness]** → document normalized task controls, absent calibration/safety validation, and simulation-only scope.
- **[Two profiles contaminate each other or the shared tools]** → separate Python environments/manifests and explicit launch paths, sanitize inherited selectors/library paths, reject wrong-profile task/package use and preserve `/env`.
- **[Moving tutorial/dependency APIs invalidate the old acceptance]** → pin the tutorial and its locked Isaac revision, audit each profile separately and rerun the old references after shared-boundary fixes.
- **[Curriculum or shaping is mistaken for learned pick-and-place]** → separate curriculum outcomes, canonical-home physical-task proof, real NexuML updates and measured learned home-start success; no borrowed success claims.
- **[Kit-less Newton media differs from RTX]** → native headless-camera and terminal-frame proof using the pinned Newton renderer, with incomplete gates for unsupported paths.

## Migration Plan

1. Record the current worktree baseline and operator-approved installation/resource profile; verify pinned native Cartpole/Franka before adapter claims.
2. Prove transition/launcher behavior with CPU tests and a bounded native wrapper spike; implement the minimal provider/bridge with no default-path changes.
3. Verify Cartpole then Franka full training/resume/export/replay paths; add the required SO-101 joint task and the same full acceptance afterward.
4. Integrate replay-only media, task outcomes, provenance, documentation and opt-in CI; run regression/package/import-isolation gates and actual GPU/media proofs for all three.
5. Mark implementation tasks/roadmap progress only against evidence; commit/merge/archive remain separate user-directed actions.
6. Preserve those 42 completed tasks; append unchecked extension work. Review existing NexuVibe changes and approve rebuild timing, then prepare the two isolated DevContainer-native environments and exact editable overlays.
7. Audit both profiles against D11; correct only proven in-scope mismatches and revalidate the initial references inside the DevContainer before claiming portability of their previous worker-image evidence.
8. Validate the pinned tutorial natively, integrate its state task through the same learning stack and prove its complete train/resume/export/scalar/media round trip; record unsupported or unrun gates explicitly.

Rollback removes the optional provider/scenarios/profile and targeted bridge/preflight wiring, preserving normal supervised/Gymnasium code paths and previously generated policies/media. Never remove an operator's simulator installation, caches, checkpoints, or other work automatically. Any failed SO-101/GPU gate leaves corresponding tasks open and a report/backlog entry; it is not completion of a smaller unapproved phase.

Extension rollback deselects/removes only the new tutorial scenario/profile wiring and restores this change's coordinated setup edits through the owning repository's review process. It does not remove the working EA integration, user-staged NexuVibe changes, existing environments, upstream checkouts or retained artifacts. Expanded change completion requires all new gates; broader roadmap follow-ups remain independent.

## Sources checked during proposal preparation

- [EA release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA) and [pinned profile](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/pyproject.toml): release status, Python/Torch/Isaac pins, dependency ownership.
- [Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html): supported OS/graphics/driver requirements, not proof of this container's usability.
- [Pinned manager environment](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab/isaaclab/envs/manager_based_rl_env.py): same-step reset, optional final observations, pre-reset recording and simulation ownership.
- [TorchRL 0.14 Isaac guide](https://docs.pytorch.org/rl/0.14/reference/isaaclab.html) and installed `envs/libs/isaac_lab.py` / `transforms/_misc.py`: maintained wrapper, batch/device mapping and terminal/reset limitations.
- [Pinned SO-101 asset](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_assets/isaaclab_assets/robots/so101.py), [joint-position example](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/so101/stack_joint_pos_env_cfg.py), and [reward-free stack base](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/stack_env_cfg.py): five arm joints, separate gripper, physics variant and why a small reaching task is needed.
- [User-linked TorchRL knowledge base](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/ISAACLAB.html), inspected 2026-10-19: startup order, native batching/reset limitations, buffer cloning, camera/graphics prerequisites and on-policy synchronization. Its example nightlies, consent variables and failure suppression are not adopted as requirements.
- [Current installation](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/installation.html) and [SO-101 tutorial](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/tutorial.html), inspected 2026-10-19: distinguish default kit-less Newton from optional Isaac Sim and the initial EA profile.
- [Pinned tutorial metadata](https://github.com/isaac-sim/IsaacLabTutorial/blob/9a4286b4bc8a5720ac97f87e5e0140f7542c22d9/pyproject.toml), [lock](https://github.com/isaac-sim/IsaacLabTutorial/blob/9a4286b4bc8a5720ac97f87e5e0140f7542c22d9/uv.lock), [task configuration](https://github.com/isaac-sim/IsaacLabTutorial/blob/9a4286b4bc8a5720ac97f87e5e0140f7542c22d9/src/isaaclab_tutorial/tasks/place_vial/config/so101/env_cfg.py) and [terminations](https://github.com/isaac-sim/IsaacLabTutorial/blob/9a4286b4bc8a5720ac97f87e5e0140f7542c22d9/src/isaaclab_tutorial/tasks/place_vial/mdp/terminations.py): exact candidate revisions/dependencies, six relative controls, privileged-critic distinction, curriculum/home reset and contact-based placement.

The initial implementation evidence is retained in the baseline report. Preparing this extension was read-only source/configuration inspection plus planning-artifact edits: no installation, rebuild/restart, GPU workload or new application code was executed, and no new profile is claimed validated. Context7 was unavailable due to its monthly quota; official URLs and pinned source/lock files were used.
