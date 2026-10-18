# Tasks

Source of truth: this proposal, design, and five capability specs, derived from roadmap Phase 2 plus the user's required SO-101 addition. All implementation tasks start unchecked. Planning, fake tests, workflow wiring, or passing earlier references do not establish GPU/SO-101 completion. Preserve Phase 0–1 work and the original live ROS/Gazebo/CARLA tasks.

## 1. Isolated compatibility and native baselines

- [ ] 1.1 Reconcile the pinned EA sources, existing collector/evaluator/export boundaries, and worktree baseline; verify with inspected-path/version records and current focused CPU/static checks in `reports/physical-ai/phase-2-isaaclab.md`, separating pre-existing failures from new regressions.
- [ ] 1.2 Inspect effective GPU/graphics/container permissions, RAM/VRAM/disk/cache budget and approved asset/license needs without stress tests; verify documented prerequisites and obtain explicit approval for any missing simulator downloads, EULA/container/system setup before performing them.
- [ ] 1.3 Create or select the dedicated Linux x86_64/Ubuntu 24.04 integration profile from D1, pin EA commit and resolved simulator/Torch/CUDA/dependency overlay; verify clean resolution/import origins and that the ordinary `.venv`, driver, base extras, and unrelated lock versions were not mutated.
- [ ] 1.4 Execute approved simulator/graphics compatibility checks and a small native manager-based Cartpole state rollout/training baseline; verify actual GPU tensors, continuous-force action mapping, frame/time geometry, finite metrics and owned shutdown with exact commands/results recorded.
- [ ] 1.5 Execute the native Franka joint-position reaching baseline in the same profile; verify action order/offsets/limits, control period, resets/timeouts, finite metrics and shutdown, recording differences from unmodified native learner settings.
- [ ] 1.6 Publish reproducible profile/setup/native-baseline instructions and retain the Phase 2 report under existing report-ignore conventions; verify the documented setup/baseline commands run as written and unavailable prerequisites remain explicit incomplete gates rather than fake passing proofs.

## 2. Portable Isaac provider and owned runtime

- [ ] 2.1 Add the minimal library definition and static contracts/profile metadata for approved task identities; verify registry/YAML/checkpoint-config round trips and description with Isaac/Omniverse imports blocked, including invalid task/backend/contract diagnostics.
- [ ] 2.2 Implement lazy public launcher/task construction at the execution site, one native N-environment GPU batch and runtime contract checks; verify fake construction-order/shape/device tests plus a small native instantiation without outer vectorization or CPU/NumPy actions.
- [ ] 2.3 Implement canonical reset/step normalization with cached post-reset states, cloned final/reward/done buffers and singleton-axis adaptation for evaluation; verify controlled no-done/partial/all-done, explicit per-index reset, stale-buffer and N=1 tests.
- [ ] 2.4 Wire idempotent ownership-aware runtime/launcher close across initialization failure, normal exhaustion, exceptions and cancellation; verify fake primary-plus-cleanup-failure cases and native sequential fresh-process startup/shutdown without leaked resources or serialized handles.
- [ ] 2.5 Document provider contract/action/profile and lifecycle boundaries as introduced; verify CPU example resolution without Isaac and native rejection of mismatched shapes, units/order/scale/control timing rather than auto-rewriting policy metadata.

## 3. Correct direct GPU collection and PPO integration

- [ ] 3.1 Probe the actual pinned `IsaacLabWrapper` via public composition using D4's final/reset-state truth table; verify numerical/mutable-buffer evidence and record whether the maintained path passes or the predetermined existing generic-adapter fallback is necessary, without private patches or a new collector engine.
- [ ] 3.2 Add only the selected provider-to-collector bridge and supported GPU collector settings; verify no nested EnvBase/vectorizer, direct single-device topology, owned close propagation, and unchanged Gymnasium/direct/process factory behavior with focused tests.
- [ ] 3.3 Add pre-allocation validation for unsupported process/Ray/async/multi-GPU/mixed-device configurations and non-divisible native frame batches; verify rejection tests allocate no launcher and document the supported geometry/device restrictions.
- [ ] 3.4 Prove truncation/termination bootstrapping against controlled distinct final/reset values and known value targets, including all rows ending together; verify exact GAE expectations and visible failure for absent terminal data, not merely finite losses.
- [ ] 3.5 Verify rollout environment/time axes, actual N×T frames, rounded finite budget behavior, action-envelope/log-probability consistency, real Cartpole PPO parameter changes and due copied-policy synchronization; use targeted CPU fixtures and opted-in GPU tests without a stochastic reward threshold.
- [ ] 3.6 Document the selected bridge/reset mapping, GPU hot-path limitations and native-versus-NexuML control comparison; verify the documented small direct Cartpole smoke and existing collect→update→sync→next-collect regressions.

## 4. Portable policy replay and lifecycle restrictions

- [ ] 4.1 Add the narrow provider-declared training-evaluation restriction and explicit zero-episode/off reference defaults; verify positive automatic episode/start/periodic/media requests fail before Isaac allocation with fresh-process guidance while Gymnasium scheduled/default behavior remains unchanged.
- [ ] 4.2 Support N=1 Isaac runtime use by the existing inference-only evaluation command and shared episode loop; verify fake contract/seed/mode/RNG/termination handling, no session/learner/optimizer/collector construction, existing override validation and owned logger/simulator finalization.
- [ ] 4.3 Route the small optional SO-101 task outcome fields through existing transition info and evaluator aggregation; verify per-episode reset, success/error aggregates, invalid-value diagnostics and unchanged scalar results for providers without task outcomes.
- [ ] 4.4 Verify complete-policy package loading/calling on controlled tensors with Isaac/assets/TorchRL/encoders blocked, plus fresh-process simulator replay and source-frame-or-zero provenance; retain the existing training-checkpoint and source-import manifest distinctions without a package-format redesign.
- [ ] 4.5 Document exact scalar replay/export/checkpoint commands, disabled in-fit evaluation, trusted executable-package scope, source-step meaning and simulator-versus-inference prerequisites; verify the commands on Cartpole and early error paths before extending the remaining references.

## 5. Cartpole and Franka reference round trips

- [ ] 5.1 Add ordinary registered `isaac-cartpole-ppo` and `isaac-franka-reach-ppo` factories with bounded continuous task controls and native logger configuration; verify Python/registry/resolve/YAML/normal CLI override tests without simulator imports and runtime contract checks in the profile.
- [ ] 5.2 Run the bounded NexuML Cartpole GPU acceptance against its native baseline; verify actual collection/actor updates, finite metrics, matching control/transition semantics and complete resource shutdown with recorded commands/counts.
- [ ] 5.3 Run the bounded NexuML Franka acceptance after Cartpole; verify named articulated joint control, real actor updates, finite metrics, compatible native comparison and shutdown without claiming matching learner curves or reaching convergence.
- [ ] 5.4 For both references, interrupt/save and resume in fresh processes, then export/reload and replay; verify actor/critic/optimizer/counter restoration, only remaining frames, no simulator for completed budgets, deterministic portable inference equality and fresh simulator evaluation.
- [ ] 5.5 Publish getting-started instructions with exact dependencies, native/NexuML train/resume/export/replay commands and expected checkpoints/events/results; verify both complete workflows require no user source edits and distinguish bounded integration proof from policy quality.

## 6. Required later SO-101 joint-reaching reference

- [ ] 6.1 Inspect the pinned upstream SO-101 asset and reward-free stack example, validate approved asset access/license/provenance, select the PhysX variant and confirm actual five-arm-joint plus separate-gripper identity/limits/actuators; verify native artifact checks and explicit missing/mismatched-asset failures.
- [ ] 6.2 Add only the minimal public-API manager task configuration for five-joint target reaching, fixed in-limit gripper, declared position/velocity/target observations, seeded reset/target set, finite normalized action mapping and control period; verify static/fake contract tests and bounded native scripted sign/scale/joint-response checks.
- [ ] 6.3 Add explicit normalized joint-error reward, tolerance/hold success, timeout rules and separate error/success info; verify numerical unit tests, deterministic native controlled success/timeout/reset episodes and no reward-free/zero-reward substitution.
- [ ] 6.4 Add the discoverable `isaac-so101-joint-reach-ppo` scenario using the shared provider/policy/PPO and evaluation restrictions; verify Python/YAML/ordinary override examples without Isaac imports and a real bounded GPU run with changed actor parameters and finite metrics.
- [ ] 6.5 Interrupt/resume SO-101, export/reload, execute clean simulator-blocked inference and fresh-process scalar replay; verify learning state/remaining frames, deterministic action equality, declared five-joint/fixed-gripper semantics, separate reaching outcomes and full shutdown in recorded evidence.
- [ ] 6.6 Document exact SO-101 asset/setup/native/train/resume/export/replay steps and simulation-only limitations; verify the commands and retain SO-101 as a required open gate until its real round trip passes, without substituting Franka or requiring learned manipulation success.

## 7. Replay-only media through existing native logging

- [ ] 7.1 Configure the supported N=1 headless replay camera before launch and add a bounded public pre-reset frame cache with dataset export disabled; verify synthetic episode/frame identity, reset/terminal ordering and no HDF5/extra writer side effects, then a native terminal-frame proof.
- [ ] 7.2 Feed RGB/FPS into the shared bounded GIF/MP4/both capture/sink and existing native loggers; verify decoded artifacts/timing/provenance and exact selected outputs for TensorBoard-only, MLflow-only and both without extra runs, rollout duplication or training-camera allocation.
- [ ] 7.3 Execute fresh-process real recorded replay for Cartpole, Franka and SO-101 across all three selected formats; verify bounded same-episode frames, media/event decoding, actual TensorBoard animation/step selection, standard-player MP4 playback and cancellation/failed-delivery cleanup without convergence claims.
- [ ] 7.4 Document/test exact recorded-replay and artifact-inspection commands with rendering/graphics/FPS/asset limitations; verify ordinary scalar replay works without encoder imports, no-target/missing-renderer diagnostics are visible and existing Gymnasium media behavior remains intact.

## 8. Integrated regression and completion evidence

- [ ] 8.1 Add manual-only Isaac test/profile CI selection following existing conventions, with prepared runtime/assets and explicit GPU+simulator opt-in; verify workflow/test selection never downloads/launches Isaac in default CPU CI and distinguish wiring from an actual remote dispatch.
- [ ] 8.2 Run focused and broader CPU/coverage regressions plus clean optional-import/package proofs; verify supervised/Gymnasium/collector/export/CLI/logger behavior, policy-only inference without Isaac/TorchRL/encoders, and explicit separation of the known DALI/Optuna/type/original-spec baseline failures.
- [ ] 8.3 Run appropriate lock, repository lint/format/type, strict docs/new-change OpenSpec, whitespace, wheel/sdist/distribution and isolated-install gates; verify exact commands/results in the report, no unrelated environment/manifest churn and no suppressed failed gates.
- [ ] 8.4 Audit the real GPU/native training/resume/replay/media evidence for all three references against all five specs; verify missing asset/GPU/SO-101 proofs leave tasks incomplete and the report distinguishes actual passes, skips, manual playback, environment limits and unrun remote/system/hardware checks.
- [ ] 8.5 Update only Phase 2 completion/progress and its linked evidence after all required proofs; verify original live ROS/Gazebo/CARLA tasks and Phases 3–8 remain unchanged, no NexuFL/hardware/IL/VLA work was added and all owned test resources were stopped.

## Workflow follow-up

- Review/commit/merge/archive require separate user direction; this proposal does not perform them.
- If an actual installation, asset, simulator, or compatibility blocker remains, record its evidence and remaining work rather than marking Phase 2 or SO-101 complete.
- Do not turn unrelated recorded baseline failures into another approval gate for independent in-scope work; do obtain approval for new destructive/system/license actions or changes to the selected compatibility profile.
