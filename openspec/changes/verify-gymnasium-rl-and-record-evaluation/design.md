# Design

## Context

See `proposal.md` for motivation and the four adjacent `specs/*/spec.md` files for observable requirements. The supplied **NexuML Physical AI: Phases 0–1 and Recorded Evaluation** plan and `.openchamber/plans/nexuml-physical-ai-roadmap.md` define the scope; they are not implementation evidence.

The baseline is the existing branch's `add-reinforcement-learning` implementation. `NexuSession` owns a Lightning trainer; `NexuRLLightningModule` delegates updates to a checkpoint-visible algorithm, counts frames, and synchronizes TorchRL collectors. `evaluate_policy()` currently returns scalar metrics from fresh environments after fit. The environment protocol has no render capability, although the Gymnasium adapter already renders image-policy observations. The two state-policy reference factories do not configure logging.

Native `create_loggers`, `iter_loggers`, artifact staging, and `log_artifact` already support TensorBoard, MLflow, and supported local artifact backends. The current MLflow path dispatches to the existing experiment/run ID. Extend these boundaries rather than adding a media logger.

Relevant constraints:

- The original RL change remains unarchived with 84/86 tasks checked; ROS/Gazebo and live CARLA remain unrelated open proofs.
- Historical `.tasks/todo.md` results are not fresh evidence, and editable installs previously resolved to a different checkout.
- Locked PyTorch 2.11's `add_video()` imports legacy `moviepy.editor` and encodes a GIF. Its existence alone does not prove compatible playback.
- Root `reinforcement` and library `rendering` already isolate simulator dependencies. Recording must not add imports to ordinary learning or packaged policy inference.
- Canonical log-path/override/test contracts remain unchanged. Portable RL policy exports retain the baseline RL exclusion of critic/optimizer/runtime state; this change does not reinterpret supervised checkpoint exports.

## Goals / Non-Goals

**Goals:**

- Keep one episode loop for post-fit evaluation, scheduled progress checks, and package replay.
- Keep scheduling and media outputs independent of PPO implementation details and collector ownership.
- Make logging backend-neutral: native artifact dispatch is primary; TensorBoard previews are an optional backend-specific presentation.
- Bound frame memory, encoding work, and output frequency through the existing frame budget plus small typed recording settings.
- Preserve old defaults and checkpoint loadability without introducing required inference state.

**Non-Goals:**

- No new evaluator engine, training mode, public result hierarchy, plugin/encoder registry, second logger/session, or generic simulator API.
- No evaluation workers, background snapshots, extra media cadence, episode-selection strategies, grids, audio, codecs beyond GIF/H.264 MP4, or automatic dependency installation.
- No speculative temporal/multimodal policy-contract fields, dependency-manifest redesign, or unrelated cleanup.

## Decisions

### D1. Audit the actual checkout before implementation

Run the scoped baseline first and write `reports/physical-ai/phase-0-audit.md`. Check core/library import paths, including subprocesses, before accepting results. Record commands, versions, durations, failures/skips, and the requested capability matrix; historical counts remain explicitly historical. Reconcile the original RL artifacts, guide, source, tests, extras, and CI markers. Note concrete future IL/VLA contract gaps without implementing them.

Probe GIF event encoding/playback and H.264 encode/decode using tiny synthetic changing frames before training integration. Install only explicitly selected optional prerequisites; record wheel/platform/codec limitations and do not download simulators or executables implicitly. Evidence-driven fixes and missing tests are allowed at their narrowest responsible layer; unrelated failures remain documented blockers.

**Alternative rejected:** Rebuild the RL foundation or treat checked tasks as current execution proof. Both obscure actual gaps and unnecessarily enlarge scope.

### D2. Extend the existing evaluation configuration additively

Keep `RLEvaluationSpec` and existing fields. Add:

| Field | Default | Constraint / meaning |
| --- | --- | --- |
| `every_n_frames` | `None` | Positive integer when present; collected training frames |
| `at_start` | `False` | Initial evaluation only at restored frame zero |
| `video.enabled` | `False` | Explicit opt-in to capture and outputs |
| `video.format` | `"gif"` | `"gif"`, `"mp4"`, or `"both"`; downloadable formats |
| `video.max_frames` | `500` | Positive; includes the reset frame |
| `video.max_size` | `256` | Positive maximum captured-frame edge |
| `video.fps` | `None` | Provider metadata, or a finite positive override |

Use one small nested recording spec with a default factory so normal dot-path overrides work. Reject recording with zero episodes and invalid fields before runtime allocation. `episodes=0` otherwise returns empty metrics without an environment. Preserve serialized definitions and current registry identities; the new optional render request is execution metadata, not a policy field.

Validate requested encoder and artifact-target capabilities before a long fit/replay. Do not require MP4 prerequisites for GIF-only use or TensorBoard for MLflow-only use.

**Alternative rejected:** New CLI-specific configuration, per-backend media settings, format lists with duplicate-handling semantics, or enabling rendering by default. A three-value format and the native config are sufficient.

### D3. Reuse one isolated evaluator and the existing scalar result type

Extend `evaluate_policy()` in `src/nexuml/training/reinforcement.py`, preserving its metrics-dictionary return type and default invocation. A small optional internal media sink receives the single bounded recording plus evaluation metadata; do not add a public result family or move/rewrite unrelated dataset evaluation code.

Select the evaluation override or training definition, describe/validate compatible required observation/action fields and semantics, and build with `purpose="eval"`, `num_envs=1`, and the policy device. Reject unsupported vectorized evaluation instead of summing rewards and stopping on any trajectory's done flag. Frame/action bounds remain the deployable policy's responsibility; no silent action conversion is introduced.

Run the complete policy under inference mode. Preserve each module's original training flag, not only the root flag. Snapshot Python/NumPy and PyTorch CPU/relevant CUDA RNG state before evaluation construction/seeding and restore it in cleanup. Evaluation uses the resolved evaluation seed, falling back to the training seed; reset episodes with `seed + index`. Seed stochastic policy draws within the isolated scope, not the learner's global stream. Unseeded runs are reported as unseeded, not promised deterministic.

Continue all configured episodes after capture caps are reached. Preserve existing aggregate keys and terminal/truncated/horizon accounting. Use nested cleanup so runtime-close failure cannot prevent mode/RNG restoration or replace a primary policy/environment/encoding exception. Preserve parameters and gradients; evaluation never constructs or steps an algorithm/optimizer.

**Alternative rejected:** Reuse/reset collector environments, run PPO-specific inference, or repurpose dataset post-train evaluation algorithms. These violate lifecycle or inference separation.

### D4. Keep scheduling with the RL module and session, not a new scheduler

Use focused RL lifecycle hooks around the existing finite rollout iterable:

```text
restore frame progress
  -> optional frame-zero evaluation
  -> collect -> learner update -> due policy synchronization
             -> optional crossed-interval evaluation -> next collection
  -> final evaluation while native loggers are active
  -> existing TrainResult uses cached final scalar metrics
```

Compute the next interval from restored frames: `(frames // interval + 1) * interval`. After a batch crosses it, evaluate once at the actual current frames and advance to the next future interval. This handles rounded/vectorized batches and multiple crossed thresholds without modulo misses or fake historical policy states. Do not change the finite loader/lookahead or synchronization cadence.

Initial evaluation occurs only before the first update when restored frames are zero. Final evaluation for opted-in initial/periodic/media runs occurs before native logger finalization. Cache final metrics with the measured frame count and evaluation-configuration identity; invalidate on further updates/config changes. `NexuSession.run()` reuses only a matching cache, avoiding repeated episodes/artifacts. A checkpoint with no remaining training batches must still take this same finalization path while loggers are active, not skip it because no batch hook fired; confirm supported Lightning hook ordering in the baseline/integration tests.

When all additions are disabled, preserve plain `fit()` behavior and the current `run()` final evaluation. Route evaluation metrics to existing loggers at collected-frame steps; do not equate PPO optimizer `global_step` with frames.

Derive cadence from existing checkpointed frame counters. Persist only necessary namespaced evaluation bookkeeping with a default for old checkpoints; do not add required policy/state-dict buffers. Cached media/delivery handles are execution-local and are not restored as live resources. Do not replay past delivery side effects on resume.

**Alternative rejected:** Epoch scheduling, exact frame modulo, a general callback scheduler, or async evaluators. The first two are incorrect for finite PPO rollouts; the others add unnecessary ownership/concurrency.

### D5. Add an optional render capability at environment construction

Keep mandatory `EnvironmentRuntime` reset/step/close unchanged. Add a small sibling RGB-rendering capability with CPU `uint8` HWC RGB frames and renderer FPS metadata. Add optional `render_mode="rgb_array"` execution metadata to `EnvironmentBuildContext`, defaulting to no render request. Non-rendering providers remain valid; explicitly requested unsupported rendering fails clearly.

The Gymnasium adapter passes the request through its normal environment creation path and exposes frames/metadata without requiring the evaluator to access its underlying private env list. Do not mutate immutable definitions, switch render mode after creation, or set `image_shape` to record state policies. Preserve the existing image-observation path.

Capture reset and step frames of the first episode only, including a terminal frame if within the cap. Validate frames, copy reused buffers, resize with aspect ratio before retention, and stop capture at the complete buffer cap. Missing FPS needs an explicit override. Other episodes contribute metrics only. Definition description/probes must not retain graphics resources; owned runtime creation/close retains its existing failure safety.

**Alternative rejected:** Make rendering mandatory on all providers or encode provider-specific access in the evaluator. Both would unnecessarily couple ROS/CARLA/custom environments to this feature.

### D6. Encode one buffer as GIF and/or H.264 MP4

Implement small encoding functions beside the media tracking helper, without an encoder registry. Pillow encodes a GIF; ImageIO's FFmpeg plugin encodes silent H.264 MP4 with `yuv420p`. Resolve compatible bounds, supported wheels, and H.264 availability in the Phase 0 probe.

Encode each requested representation once. TensorBoard, when configured, additionally needs GIF preview bytes; reuse those bytes for the GIF artifact when selected. MP4-only plus MLflow-only must not encode an unnecessary GIF preview. Do not produce an unselected downloadable format or rerun the episode for another output.

Preserve source timing using the selected FPS; record GIF timing quantization. Pad odd MP4 dimensions by at most one pixel per axis to make them even, with no cropping/stretching; disable ImageIO's implicit macroblock-size scaling. Captured source edges remain bounded by `max_size`; metadata explicitly accounts for possible one-pixel encoded padding. Use codec-appropriate tolerances, not exact RGB equality, for MP4 round trips.

Use native staging and context-managed encoder finalization. Publish only finalized files and clean partial temporaries on failure/cancellation. Do not silently substitute GIF when MP4 fails, and do not promise transactional rollback across already-uploaded remote artifacts.

**Alternative rejected:** MoviePy solely for TensorBoard GIFs, implementing video codecs, or using deprecated video-writing APIs. The selected maintained libraries keep encoding small and explicit.

### D7. Fan out through native logging; specialize only the TensorBoard preview

`scenario.logging` stays authoritative. Training uses the session's existing logger instances; replay uses the same `create_loggers`/finalization path for its own evaluation run. The new helper is media formatting/dispatch beside `log_image`, not a logger class or independent `SummaryWriter`.

1. Send scalar metrics via existing configured loggers with the evaluation frame step.
2. Stage finalized media with a stable frame-step basename, e.g. `evaluation/frame_000000010240.gif` and `.mp4`.
3. Call existing `log_artifact` with the configured logger collection once for each selected file and once for its JSON. Native MLflow dispatch uses its existing experiment and run ID; native TensorBoard/local backends copy to their current artifact roots. Preserve `NEXUML_LOGS_ROOT` behavior.
4. Only for a configured TensorBoard logger, build an animated GIF `Summary.Image` protobuf and use its active PyTorch `FileWriter.add_summary` path at `eval/rl/episode`. Validate the supported writer capability; do not invoke private video encoders or reopen a separate writer. Flush the native writer after infrequent evaluation.

MLflow-only must work without constructing/accessing any TensorBoard writer. Both loggers receive the same selected files and JSON exactly once; only TensorBoard gets the native animated preview. MP4 remains a downloadable artifact, not an embedded TensorBoard MP4 player. An unsupported backend retains native diagnostics; explicitly enabled recording with no usable artifact target fails clearly.

The compact JSON records aggregate metrics, actual frame step, seed sequence/count/horizon/mode, environment identity, resolved-config and interaction-contract provenance, FPS, cap/captured count, selected formats, encoder/codec identity, and source/encoded dimensions/padding. Preserve run-level code/dependency provenance already logged by NexuML; do not store training trajectories or duplicate the source artifact hierarchy.

**Alternative rejected:** A custom TensorBoard logger, direct MLflow run creation in the media helper, or backend-specific config branches. These bypass native ownership and fail the user's explicit MLflow routing requirement.

### D8. Keep recording dependencies layered

- Core `rendering`: directly declare Pillow for GIF creation; do not rely on transitive torchvision or unrelated export extras.
- Library `rendering`: retain classic-control prerequisites and compose core rendering.
- Core `video`: compose rendering plus `imageio[ffmpeg]`.
- Library `video`: compose its rendering prerequisites plus core video.

Lazy-import encoders only on requested recording paths. Prefer the explicitly installed FFmpeg backend wheel's executable on supported platforms; otherwise document an operator-provided compatible executable using upstream conventions. Preflight H.264 support. Do not download a binary at runtime, install system packages automatically, add MoviePy, upgrade unrelated dependencies, or expand base reinforcement/all requirements just to cover media.

Use the existing opt-in `rl-simulators` job for both formats, installing the scoped library video extra. Default unit tests use synthetic frames/local fakes and existing optional markers. A lightweight optional local MLflow store proves native artifact round trips without a tracking server or credentials. Runtime inference-import isolation remains the acceptance test; do not redesign the source-import dependency manifest as part of this work.

**Alternative rejected:** Add every media backend to base training or require system FFmpeg unconditionally. That makes non-recording users pay for an optional inspection feature.

### D9. Replay packages without constructing training artifacts

Add one `evaluate` command in the existing CLI. Load a trusted complete package through `load_inference_package`, restore its embedded scenario configuration, apply repeatable ordinary overrides with the existing utility, validate, and call the shared evaluator directly. Construct no `NexuSession`, algorithm, optimizer, or collector. Do not execute configured training/export side effects.

Support evaluation settings, native logging, and compatible evaluation-environment selection through existing override semantics. Reject attempted replacement of the packaged executable policy/pipeline or incompatible I/O rather than pretending configuration edits changed loaded weights. Invalid paths/values retain existing diagnostics and Rich summaries; no new configuration parser or parallel flags are needed.

Create a native standalone evaluation run with source-package/config/contract provenance and finalize it in `finally`. Use explicit source collected-frame metadata only when actually available; otherwise log at standalone step zero, record source frames as unavailable, and never relabel optimizer steps or call this an untrained baseline. This does not require adding training frames to the portable export format.

Keep checkpoint resume on `train --trainer-checkpoint`; package replay is not a checkpoint-resume API. Document that packaged Python artifacts are trusted executable inputs, not sandboxed passive data.

**Alternative rejected:** Run a zero-budget learner/session to evaluate, or introduce another package/CLI format. Both unnecessarily load training dependencies and blur resume versus inference.

### D10. Trace changes to bounded files and tests

| Concern | Existing implementation/test homes |
| --- | --- |
| Typed schedule/media and render request | `src/nexuml/core/{types,components}.py`; interaction/config tests |
| Optional RGB runtime / Gymnasium | `src/nexuml/interaction/runtime.py`; library `environments/gymnasium.py`; simulator tests |
| Evaluator / hooks / final result reuse | `src/nexuml/training/{reinforcement,lightning}.py`; existing RL tests and focused `test_rl_evaluation.py` |
| Encoders / native dispatch / preview | `src/nexuml/tracking/logger.py`; `tests/tracking/test_logger.py` and optional local-store proofs |
| References / replay | existing library Gymnasium reference file; `src/nexuml/cli/main.py`; existing CLI/RL tests |
| Dependency / CI / documentation evidence | core/library pyprojects, lockfile, opt-in CI job, RL/checkpoint/CLI guides, audit report |

Reuse existing tests/fixtures before adding new files. Only add public re-exports when a newly public type requires them. No original roadmap, unrelated plan, existing task completion, or other capability spec is changed to claim progress.

## Risks / Trade-offs

- **[Wrong editable checkout produces misleading results]** → Check core/library/subprocess import origins and use worktree-local source paths for verification.
- **[Lightning hook/finalization or no-batch resume behavior differs]** → Prove ordering, old/default lifecycle, final deduplication, and full-budget checkpoint behavior before media integration; keep one final-evaluation helper.
- **[Evaluation changes training randomness or mixed module modes]** → Guard all RNG domains and individual flags; compare continuation against a controlled evaluation-disabled CPU run.
- **[Writer compatibility is narrower than high-level `add_video`]** → Keep the GIF-summary helper tiny, check active writer capability, decode real event bytes, and manually verify playback on the supported locked stack.
- **[MP4 wheel/executable/codec availability varies]** → Explicit video extra and preflight; document supported platforms and operator executable fallback, with no hidden downloads.
- **[Synchronous evaluation costs wall time/storage]** → Opt-in frame cadence, one recorded episode, bounded source dimensions/frame count, and shared encodings. No async machinery or retention service is added.
- **[GIF frame coalescing and lossy video obscure exact assertions]** → Use synthetic changing frames, GIF timing tolerances, MP4 decode tolerances, and explicit padding metadata.
- **[MLflow-only accidentally depends on TensorBoard]** → Test native routing for each backend combination, forbid writer access in MLflow-only tests, and prove a local MLflow artifact round trip.
- **[Delivery succeeds on one backend and fails on another]** → Preserve visible failure and completed-file validity; do not claim distributed atomicity or delete already delivered user artifacts.
- **[Reference media is mistaken for learning quality]** → No reward thresholds/convergence claims; GIF/MP4 illustrate behavior and do not replace statistical evaluation.

## Migration Plan

1. Record baseline and media compatibility evidence; fix only demonstrated in-scope blockers before new behavior.
2. Add typed scheduling/isolation and regression tests with existing scalar defaults intact.
3. Add optional render capture, native output dispatch, GIF/MP4 dependencies and encoder proofs, then the thin replay command/reference logging/docs.
4. Run focused, regression, optional rendering/local-tracking, package, and static/documentation/OpenSpec gates. Keep all unmet environment/system proofs explicit.

Compatibility is additive: missing new config/checkpoint metadata uses defaults; no required policy buffers or inference encoder imports are added. Removing explicit scheduling/recording settings returns users to the existing final-only mode. Rollback removes this change's optional hooks/helper/CLI/extra wiring without resetting user artifacts, changing the baseline trainer, or altering existing supervised exports. Previously generated GIF/MP4/JSON files remain usable.

Stop after both references complete the verified round trip and GIF/MP4/native logger/replay requirements. No Phase 2 implementation, simulator expansion, or convergence optimization follows automatically.
