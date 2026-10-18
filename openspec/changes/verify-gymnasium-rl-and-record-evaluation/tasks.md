# Tasks

Implementation source of truth: this change's proposal, design, and four capability specs, derived from the supplied Phase 0–1 plan. All tasks are initially unchecked because this proposal performs no implementation or runtime verification. Preserve the original RL change, source roadmap, and unrelated plans.

## 1. Audit and compatibility baseline

- [x] 1.1 Reconcile the original RL proposal/design/tasks/four specs, guide, relevant source/tests, extras, and test tiers; verify completion by recording inspected paths and demonstrated gaps in `reports/physical-ai/phase-0-audit.md` without changing original completion claims.
- [x] 1.2 Verify worktree-local core/library and subprocess import origins, record commit/Python/platform/installed dependency versions and available tools, and run the focused interaction/policy/RL/logger CPU baseline; verify completion with exact commands, durations, and pass/fail/skip counts in the report.
- [x] 1.3 Run the broader CPU regression and existing lock/lint/format/type/docs/OpenSpec gates before new behavior; verify completion with recorded results that separate historical evidence, current failures, and missing optional/system prerequisites.
- [x] 1.4 Probe a tiny synthetic GIF through the active native TensorBoard writer and a tiny H.264 MP4 encode/decode using explicitly selected prerequisites; verify event bytes, playable sequences/timing, Python/platform wheels, executable/codec availability, and supported writer/hook ordering in the report before training integration.
- [x] 1.5 Publish the requested capability matrix for Gymnasium, MuJoCo, ROS/Gazebo, datasets/demonstrations, IL, VLA, Isaac, benchmarks, export, and evaluation/media; verify each classification has evidence and concrete temporal/multimodal limits remain deferred rather than implemented.
- [x] 1.6 Resolve only demonstrated in-scope baseline blockers with narrow regression tests, or report blockers without silently working around them; verify the audit gate is explicit and live ROS/Gazebo/CARLA tasks remain open before proceeding.

## 2. Verify the existing Gymnasium round trips

- [x] 2.1 Reuse/extend existing CartPole and Pendulum CPU smokes for discovery, YAML restore, ordinary CLI overrides, collection, real changed actor parameters, and finite losses/episode metrics; verify both scenarios pass without reward-convergence thresholds.
- [x] 2.2 Verify discrete/bounded-continuous actions, finite/rounded frame budgets, resets, termination versus truncation, reproducible CPU seeds, and vectorized training shapes with focused existing/missing tests; record exact smoke commands and results in the audit report.
- [x] 2.3 Verify interrupted checkpoint resume advances remaining frames and restores critic/optimizers/counters while rebuilding runtimes, plus deterministic complete-policy export/reload in a clean process without learner/simulator/recording imports; preserve the supervised-versus-RL artifact distinction in tests and checkpoint guidance.
- [x] 2.4 Reuse/extend direct/process synchronization, collect-after-update, failure/interruption cleanup, and device/topology rejection tests; verify no owned workers leak and no new Ray/asynchronous capability is enabled.
- [x] 2.5 Update the RL getting-started baseline instructions with exact scoped installation, tiny train/evaluate, checkpoint location, resume/export/reload commands, and expected artifacts; verify documented baseline commands execute without user source edits.

## 3. Add typed evaluation and recording configuration

- [x] 3.1 Add `every_n_frames`, `at_start`, and the default-instantiated recording spec with `enabled`, `format`, frame/size bounds, and FPS; verify default/invalid values, GIF/MP4/both, zero-episode incompatibility, and finite FPS using focused configuration tests.
- [x] 3.2 Preserve YAML/registry/checkpoint serialization and normal `-O` behavior for the new fields; verify round trips, override diagnostics/order, legacy configs/checkpoints with absent fields, and unchanged supervised construction.
- [x] 3.3 Document the new settings/defaults and the distinction between media file format and TensorBoard preview representation; verify the configuration examples resolve through the existing config/CLI path rather than a new parser.

## 4. Extend the shared isolated evaluator and cadence

- [x] 4.1 Extend the existing metrics-returning evaluator for a fresh single evaluation runtime, compatible I/O validation, and an optional internal media sink; verify fake-runtime tests cover override selection, incompatible/batched rejection, and no learner/collector construction.
- [x] 4.2 Guard individual module training flags and Python/NumPy/PyTorch CPU/relevant CUDA RNG state around construction/seeding/inference; verify deterministic and stochastic seeded CPU evaluations preserve parameters/gradients/random continuation and restore mixed flags after success/failure.
- [x] 4.3 Preserve full-episode return/std/length and terminal/truncated/horizon metrics over every configured episode; verify zero episodes, seed fallback/episode sequencing, policy/environment failure, cancellation, and close-failure paths without masking the primary error.
- [x] 4.4 Add initial and crossed-frame-interval evaluation through existing RL lifecycle hooks after updates/due synchronization; verify actual frame steps, non-divisible intervals, multi-threshold batch overshoot, no historical catch-up, and unchanged collector request ordering.
- [x] 4.5 Integrate final metric caching/result reuse and backward-compatible resume bookkeeping without required policy/state-dict buffers; verify final deduplication, off-interval final evaluation, partial/full-budget resume, config-change invalidation, and logging before finalization.
- [x] 4.6 Preserve default plain-fit/final-run behavior when additions are disabled and native scalar logging at evaluation frame steps; verify legacy session and supervised regressions alongside focused lifecycle tests.
- [x] 4.7 Document cadence, synchronous overhead, fixed-seed comparisons, resume, disabled behavior, and final evaluation semantics; verify the documented scalar-only examples run without rendering or encoders.

## 5. Add optional RGB capture and GIF prerequisites

- [x] 5.1 Add the narrow core `rendering` extra with a direct Pillow dependency and compose it with the existing library rendering prerequisites; verify scoped installs/lock consistency and base/reinforcement imports with recording dependencies blocked.
- [x] 5.2 Add optional render-request execution metadata and a sibling RGB/FPS runtime capability without changing mandatory reset/step/close; verify contract tests keep non-rendering/custom providers valid and reject explicitly unsupported rendering clearly.
- [x] 5.3 Extend Gymnasium construction/runtime to honor evaluation RGB-array requests and expose frames/FPS without mutating definitions or vector policy inputs; verify constructor/render-mode, original image-policy behavior, dependency diagnostics, and close tests.
- [x] 5.4 Capture copied reset/transition/terminal frames from the first episode only, resize before buffering, and enforce frame/size bounds; verify fake frames cover buffer reuse, aspect ratio, invalid/empty frames, missing FPS/override, cap metadata, and complete metrics after capture stops.
- [x] 5.5 Document rendering prerequisites, headless SDL usage/limitations, FPS selection, capture bounds, and unchanged state-policy observations; verify examples/structural tests do not require graphics in ordinary CPU CI.

## 6. Add GIF artifacts and native backend dispatch

- [x] 6.1 Add a small GIF encoder/media helper beside existing tracking helpers using one bounded buffer and native staging; verify synthetic changing-frame GIF decoding/timing, partial-file cleanup, and no encoding when disabled.
- [x] 6.2 Add the TensorBoard-only GIF image-summary presentation through its existing active writer, stable tag, frame step, and flush; verify decoded real event bytes, writer capability errors, active lifecycle ordering, and reuse of the same GIF bytes for a selected artifact.
- [x] 6.3 Dispatch finalized selected files and compact provenance JSON with existing `iter_loggers`/`log_artifact`, and scalar metrics with existing logger instances; verify frame/config/contract/seed/timing/cap provenance, native log-root behavior, and no duplicate dispatch or extra logger/run creation.
- [x] 6.4 Add logger-combination tests for TensorBoard-only, MLflow-only, both, unsupported targets, and disabled logging; verify each configured artifact-capable backend receives selected files/JSON once, unconfigured backends receive nothing, and MLflow-only never accesses a TensorBoard writer.
- [x] 6.5 Add requested-output/encoder preflight and visible encoding/delivery failure handling, plus a lightweight optional local-MLflow artifact round trip; verify existing run IDs, retrievable GIF/JSON/metrics, no remote service, no false success, and no promise of cross-backend transactional rollback.
- [x] 6.6 Document native logging selection, MLflow artifact locations, TensorBoard Images playback, per-frame artifact names, and explicit missing-target behavior; verify configuration examples use ordinary logging and no custom media logger.

## 7. Add MP4 and both-format export

- [x] 7.1 Add scoped core/library `video` extras composing rendering and compatible ImageIO/FFmpeg dependencies; verify explicit installs/lock consistency, lazy imports, H.264 executable/codec diagnostics, and GIF-only operation with the MP4 backend blocked, without implicit binary downloads.
- [x] 7.2 Encode silent H.264/yuv420p MP4 from the existing capture buffer with selected FPS, explicit even-dimension padding, and disabled implicit macroblock scaling; verify decoded frame sequence/timing/dimensions with codec-appropriate tolerances and recorded padding/codec metadata.
- [x] 7.3 Implement GIF-only, MP4-only, and both selection while sharing capture/encodings and optional TensorBoard GIF preview bytes; verify exact downloadable output sets, one environment rollout, no unnecessary GIF in MP4-only/MLflow-only, and no extra GIF artifact for MP4-only/TensorBoard.
- [x] 7.4 Finalize/close MP4 encoders before publication and preserve failure/cancellation visibility; verify missing codecs, interrupted/failed encodes, staged cleanup, playable completed artifacts, and no silent GIF fallback or partial-success claim.
- [x] 7.5 Document both-format and MP4-only commands, optional executable/platform limitations, silent-video/timing/padding behavior, and GIF-only minimal installs; verify examples and synthetic native-logger routing for all formats.

## 8. Wire reference logging and inference-only replay

- [x] 8.1 Configure the two existing references with native `LoggingSpec`/`TensorBoardSpec` defaults while keeping periodic/media options disabled and MLflow opt-in; verify default logging plus ordinary MLflow-only/both/disabled configurations and scenario round trips.
- [x] 8.2 Add `nexuml evaluate <policy-package>` using existing inference loading, embedded scenario restoration, ordinary overrides, and the shared evaluator; verify scalar/recorded CartPole/Pendulum replays construct no session/algorithm/optimizer/collector and execute no training/export side effects.
- [x] 8.3 Add replay contract/artifact/override diagnostics and reject unsupported packaged-policy replacement; verify invalid/non-policy packages, incompatible environments, override summaries, format controls, and scalar replay without encoder imports.
- [x] 8.4 Use native standalone evaluation logger ownership/finalization with source-package provenance and explicit source-frame-or-zero semantics; verify MLflow-only/both routing, no fabricated frame counts from optimizer steps, and cleanup on failure/cancellation.
- [x] 8.5 Update RL/CLI/checkpoint guidance with exact resolve/train/both-format/export/replay/resume commands, native logging selection, actual checkpoint-path discovery, and trusted executable-package limitations; verify the commands against both references without source edits.

## 9. Integrate and record completion evidence

- [x] 9.1 Run both references with short initial/16-frame periodic/final evaluations, interrupt/resume, export/reload, and replay; verify frame-zero/progress/final outputs, all metrics/artifacts, unchanged vector observations, finite budgets, and complete resource shutdown.
- [x] 9.2 Run opted-in real headless rendering for both references in GIF-only, MP4-only, and both modes, including native MLflow-only artifact availability; verify decoded real outputs and record exact commands/results separately from optional skips.
- [x] 9.3 Inspect multiple real TensorBoard evaluation steps and open a generated MP4 in a standard player; verify actual animation/video playback and record that evidence without claiming convergence or statistical robustness.
- [x] 9.4 Extend the existing opt-in `rl-simulators` CI job to install scoped video prerequisites and run recording integration tests, updating its existing documentation; verify workflow/test selection keeps default CI free of simulator downloads, external tracking services, GPU rendering requirements, and stochastic reward gates.
- [x] 9.5 Rerun appropriate full regression/coverage and existing wheel/sdist/distribution/clean-install gates; verify supervised behavior, ordinary RL without encoders, GIF-only without the video backend, and complete-policy inference with learning/simulation/recording imports blocked.
- [x] 9.6 Run `uv lock --check`, repository Ruff lint/format, `ty check`, strict MkDocs, strict validation of this and the original RL change, and `git diff --check`; verify results and remaining environmental limits are recorded in the audit report, with no unrelated change or new Phase 2+ implementation.

Stop after these Phase 0–1 requirements are verified. Keep source-plan coarse steps accurate during implementation, but do not mark them complete merely because planning artifacts exist. Review/commit/merge/archive and later roadmap phases require their own user-directed workflow; the original change's live-system proofs remain separate.
