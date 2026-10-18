# Physical AI Phase 0 audit

## Status

**Implementation complete — 2026-10-18.** All 48 change tasks and six source-plan
steps are complete, including execution and honest recording of failed baseline
gates. This does **not** mean every repository gate is green: the same four
optional-dependency test failures, 37 full-repository type diagnostics, and six
original-change strict warnings remain. No new baseline failures were introduced.
See final completion evidence below; earlier sections retain the initial audit.

Change: `verify-gymnasium-rl-and-record-evaluation`.
Baseline commit: `8da86ea802176835045d1fbb8524061bf56ee2fd`, branch
`feature/nex-212-add-reinforcement-learning`. At audit start, only the new OpenSpec
planning artifacts were untracked; tracked application files were clean.

## Approved isolated environment and current execution

The user approved an isolated dependency installation after the initial failure.
Neither `/env` nor the main checkout's environment was modified. The worktree-local
`.venv` uses Python 3.12.13; both editable packages and successful parent/child
imports point into this worktree. `uv.lock` and both pyprojects remain unchanged.

Installation command (13.041 seconds, exit 0), with `VIRTUAL_ENV` unset and
`UV_PROJECT_ENVIRONMENT="$PWD/.venv"`:

```bash
uv sync --locked --all-packages --extra dev --extra reinforcement \
  --extra rendering --extra tracking --python /env/bin/python --no-python-downloads
```

This installed 180 locked packages, including PyTorch `2.11.0+cu128`, Lightning
`2.6.5`, TensorDict `0.14.2`, TorchRL `0.14.0`, Gymnasium `1.3.0`, NumPy `2.4.6`,
Pydantic `2.13.4`, TensorBoard `2.20.0`, Pillow `12.2.0`, MLflow `3.14.0`, pytest
`9.1.1`, Ruff `0.15.19`, and ty `0.0.53`. No DALI, Ray, MuJoCo, ROS, CARLA, or
hardware runtime was installed or started.

Explicit media-probe prerequisites (0.571 seconds, exit 0):

```bash
uv pip install --python .venv/bin/python \
  'imageio[ffmpeg]>=2.37,<3' 'imageio-ffmpeg>=0.6,<0.7'
```

This added ImageIO `2.38.1`, imageio-ffmpeg `0.6.0`, and its psutil `7.2.2`
dependency. These are probe-only installations, not yet declared project extras.
The Linux x86_64 imageio-ffmpeg wheel includes FFmpeg `7.0.2`; no executable was
downloaded at runtime. Pillow and TorchRL wheels match CPython 3.12/Linux x86_64.
Other Python/platform wheel combinations have not been validated.

Use the worktree environment explicitly, including for subprocess/tool discovery:

```bash
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/library/src"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NEXUML_RUN_SIMULATOR_TESTS=0 NEXUML_RUN_SYSTEM_TESTS=0
export NEXUML_RUN_HARDWARE_TESTS=0 NEXUML_RUN_DATA_TESTS=0 TMPDIR=/tmp/opencode
```

| Command | Seconds | Result |
| --- | ---: | --- |
| `.venv/bin/python -m pytest tests/core/test_interaction.py tests/core/test_policy.py tests/core/test_reinforcement.py tests/tracking/test_logger.py -q --tb=short --disable-warnings` | 21.532 | **59 passed**, 42 warnings; no skips/failures |
| `.venv/bin/python -m pytest tests -m "not slow and not requires_gpu and not requires_data and not requires_simulator and not requires_system and not requires_hardware" -q --tb=short --disable-warnings -rs` | 24.958 | **399 passed, 4 failed, 12 skipped, 17 deselected**, 44 warnings |
| `uv lock --check` | 0.011 | Pass, lock unchanged |
| `.venv/bin/ruff check .` | 0.036 | Pass |
| `.venv/bin/ruff format --check .` | 0.015 | Pass, 218 files |
| `.venv/bin/ty check` with matching `VIRTUAL_ENV` | 0.248 | Exit 1: 37 diagnostics, detailed below |
| `DISABLE_MKDOCS_2_WARNING=true .venv/bin/mkdocs build --strict` | 7.079 | Pass; generated site only |
| `openspec validate add-reinforcement-learning --strict` | 0.388 | Exit 1: the same six long-requirement warnings |
| `openspec validate verify-gymnasium-rl-and-record-evaluation --strict` | 0.374 | Pass |

The earlier ty invocation using the `.venv` binary but inherited `VIRTUAL_ENV=/env`
searched the wrong site-packages and produced 468 diagnostics. That is an environment
selection failure, not 468 source defects. The corrected run above is authoritative:
34 unresolved optional imports (audio/data/export/DALI/Ray/Optuna), two unused-ignore
warnings, and one `Tensor | Module` argument error at
`tests/core/test_ray_collectors.py:200`. No diagnostics were suppressed or source
changed to obtain a pass.

Current CPU regression failures:

- `tests/core/test_export.py::test_export_data_module_webdataset_roundtrip` requires
  DALI's `wds2idx` executable for its default index creation.
- `tests/core/test_s3_webdataset.py::test_dali_streams_s3_shards_and_materializes_indexes`
  imports the absent `nvidia.dali` module.
- `tests/tuning/test_optuna_tuner.py::test_build_objective_rebuilds_scenario_per_trial`
  and `::test_build_objective_missing_metric_lists_available` require absent Optuna.

The 12 skips comprise the existing audio-shape conformance exclusion (1), absent
DALI (1), ONNXScript (2), Ray (5), and unrequested distribution/clean-install gates
(3). They are not successful proofs. The 17 deselections follow the explicit CPU
tier expression. No unrelated dependencies were installed to conceal these failures.

### Synthetic media and lifecycle compatibility

Temporary assert-based probes and their JSON results are under
`/tmp/opencode/nexuml-phase0-approved-20261018/` (session-local, not committed):

```bash
.venv/bin/python /tmp/opencode/nexuml-phase0-approved-20261018/probe_media.py
.venv/bin/python /tmp/opencode/nexuml-phase0-approved-20261018/probe_hooks.py
```

- Media assertions passed (0.815 seconds excluding imports): three changing
  65×33 RGB frames, 10 FPS, GIF bytes decoded exactly and written as animated
  `eval/rl/episode` summaries at steps 0 and 16 through a native configured
  TensorBoard logger's existing writer. Real event bytes contain all three frames.
  Native artifact copies match the finalized GIF/MP4 bytes.
- The same captured synthetic sequence encoded as silent H.264/yuv420p MP4:
  66×34 after explicit one-pixel edge padding, 3 decoded frames, 10 FPS, 0.3 seconds,
  no implicit macroblock scaling, and per-channel differences below four levels.
  Bundled FFmpeg exposes `libx264`. This proves this installed encoding stack,
  not end-to-end policy recording, MLflow delivery, or all-platform compatibility.
- Lifecycle assertions passed (0.424 seconds excluding imports). A real 16-frame
  CartPole fit calls `on_fit_start → on_train_start → on_train_end → on_fit_end →
  logger.finalize`. Resuming its completed-budget checkpoint calls
  `on_fit_start → on_fit_end → logger.finalize`, with no training-start/end hooks,
  no new collection, and restored frame count 16. **Final evaluation must therefore
  use the fit-end path, not rely only on train-end.**
- Browser inspection reached TensorBoard's application shell but did not expose
  the image panel in the browser snapshot. Actual animation/video playback is not
  claimed. The temporary local TensorBoard server was stopped after inspection.
  These probes do not mark application media integration or task 9.3 complete.

Raw current logs/results live under the same temporary directory. The sections
below retain the initial environment failure and source-only assessment as history.
The audit/report work is complete; feature work continues against these recorded
existing failures. The report is not a claim that all baseline gates pass.

## Initial environment and prerequisites (before installation approval)

- Working directory:
  `/home/nexuadmin/.local/share/opencode/worktree/c31e64cb7ed947a5a3a40717c489e71c6f05b6f5/feature-nex-212-add-reinforcement-learning`.
- Linux `6.1.0-51-amd64`, x86_64, glibc `2.43`.
- Active interpreter: `/env/bin/python`, Python `3.12.13`.
  Both `VIRTUAL_ENV` and `UV_PROJECT_ENVIRONMENT` point to `/env`.
- Existing main-checkout interpreter: `/workspaces/NexuML/.venv/bin/python`,
  Python `3.14.4`. It also lacks the project dependencies; switching to it does not
  unblock imports. Neither interpreter was modified.
- `importlib.metadata.version()` found none of the following distributions in
  either interpreter: `nexuml`, `nexuml-library`, `torch`, `lightning`, `tensordict`,
  `torchrl`, `gymnasium`, `numpy`, `pydantic`, `tensorboard`, `Pillow`, `imageio`,
  `imageio-ffmpeg`, `mlflow`, `pytest`, `pytest-cov`, `ruff`, `ty`, or
  `mkdocs-material`. No installed project extras or editable distribution metadata
  were found.
- Available external CLIs: uv `0.13.0`, Ruff `0.15.22`, OpenSpec `1.14.1`, Git.
  An external `pytest` launcher is on PATH, but it is not the active interpreter's
  pytest; it was deliberately not substituted. `ty` and `mkdocs` are absent from
  PATH. `/usr/bin/ffmpeg` exists, but executable presence alone proves neither
  ImageIO compatibility nor H.264 round-trip/playback support.
- Root packaging declares the ordinary training dependencies plus optional
  `reinforcement` (`torchrl>=0.14,<0.15`, `gymnasium>=1,<2`). The library has separate
  `reinforcement`, `rendering`, and `mujoco` extras. Core `rendering` and both
  `video` extras are planned, not present. Pillow is currently only a direct core
  `export` dependency. No installs, simulator launches, or binary downloads ran.

### Import-origin evidence

With `PYTHONPATH="$PWD/src:$PWD/library/src"`, `importlib.util.find_spec()` selects
`src/nexuml/__init__.py` and `library/src/nexuml_library/__init__.py` under this
worktree. Actual imports in child processes using **both** interpreters fail in
the worktree's `src/nexuml/core/post_train_layer.py:8`:

```text
from tensordict import TensorDict
ModuleNotFoundError: No module named 'tensordict'
```

Thus path resolution is established, but successful parent/child imports and
runtime isolation are not. Do not reuse the historical editable-install workaround
as a claim that the currently selected environment is usable.

## Initial commands and results (before installation approval)

Commands ran at the worktree root. For import/tests, the runner explicitly set:

```bash
export PYTHONPATH="$PWD/src:$PWD/library/src"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NEXUML_RUN_SIMULATOR_TESTS=0 NEXUML_RUN_SYSTEM_TESTS=0
export NEXUML_RUN_HARDWARE_TESTS=0
```

Wall-clock timings were measured with `time.perf_counter()` around each command.
Session-local raw logs and command arrays are retained in
`/tmp/opencode/nexuml-phase0-20261018/results.json` and adjacent `.log` files;
these temporary files are not part of the repository evidence contract.

| Command | Seconds | Result |
| --- | ---: | --- |
| `/env/bin/python -c 'import nexuml, nexuml_library; print(nexuml.__file__); print(nexuml_library.__file__)'` | 0.031 | Exit 1: missing TensorDict, before successful package import |
| `/env/bin/python -m pytest tests/core/test_interaction.py tests/core/test_policy.py tests/core/test_reinforcement.py tests/tracking/test_logger.py -q` | 0.006 | Exit 1: no pytest module; no collection |
| `/env/bin/python -m pytest tests -m "not slow and not requires_gpu and not requires_data and not requires_simulator and not requires_system and not requires_hardware" -q` | 0.006 | Exit 1: no pytest module; no collection |
| `uv lock --check` | 0.021 | Pass: 261 packages resolved; no lockfile edits |
| `ruff check .` | 0.022 | Pass |
| `ruff format --check .` | 0.014 | Pass: 218 Python files already formatted |
| `openspec validate add-reinforcement-learning --strict` | 0.380 | Exit 1: six existing long-requirement warnings become strict failures |
| `openspec validate verify-gymnasium-rl-and-record-evaluation --strict` | 0.370 | Pass |
| `git diff --check` | 0.002 | Pass for tracked changes; does not inspect untracked files |

**Initial attempt: no tests collected or executed.** Pass/fail/skip test counts were
unavailable, not zero-test success. No optional skip was counted as a proof.
`ty check` and `DISABLE_MKDOCS_2_WARNING=true mkdocs build --strict` were not run:
their tools and project imports are unavailable. Encoding/event/playback, Lightning
hook-order, package-build/clean-install, and live-system checks have not run.

The original strict OpenSpec warnings name these requirements:

- Components: “Interaction contract is portable and explicit”, “Environment
  definition separates description from runtime ownership”, and “Action adaptation
  is a typed deployable component”.
- Export: “Primary RL export is a complete deployable policy”.
- Library: “Library implements a maintained ROS 2 reference adapter”.
- Training: “Scenario supports reinforcement-learning training”.

All six exceed the current validator's 500-character requirement-text guidance.
No requirement was shortened, suppressed, or removed. This differs from historical
strict-pass notes and requires a scope decision before changing the original specs.

## Reconciled source and existing acceptance coverage

The original `openspec/changes/add-reinforcement-learning/{proposal,design,tasks}.md`
and all four `specs/reinforcement-learning-{components,training,library,export}/spec.md`
were read alongside the new change's proposal/design/tasks/four specs and the
saved Phase 0–1 plan. The original remains **84/86 checked**; live ROS/Gazebo task
9.5 and live CARLA task 10.4 remain unchecked and outside this change's completion.

| Boundary | Inspected implementation and test evidence (not freshly executed) |
| --- | --- |
| Contract, registry, YAML, data/interaction coexistence | `src/nexuml/core/{components,types}.py`; `tests/core/test_interaction.py` exercises common-registry identity, portable round trips, hybrid data/interaction, validation, and a subprocess blocking simulator/RL imports |
| Shared compilation and complete policy | `src/nexuml/core/{compiler,policy}.py`; `tests/core/test_policy.py` covers shared actor/critic contexts, supervised equivalence, multimodal inputs, field validation, arbitrary batch axes, and maintained adapters |
| Action semantics | `library/src/nexuml_library/action_adapters/adapters.py` owns direct, bounded Gaussian, and categorical inference; PPO uses the adapter's distribution configuration |
| Learner, counters, collectors | `src/nexuml/training/reinforcement.py`, `src/nexuml/reinforcement/collector.py`, library `reinforcement/algorithms/ppo.py`; existing RL tests cover actor changes, optimizer/critic state, finite/rounded collection, copied-policy synchronization, and process completion cleanup |
| Resume/export versus inference | `src/nexuml/core/export.py:905-1012`; `tests/core/test_reinforcement.py` covers partial CartPole resume, complete-policy reload for both references, metadata/state exclusions, and clean-process inference with simulator/RL imports blocked |
| Session/final evaluation | `src/nexuml/training/lightning.py:885-909`; existing RL tests cover fresh evaluation overrides and policy-failure closure. Final metrics currently log at optimizer global step after fit |
| Gymnasium and reference scenarios | `library/src/nexuml_library/environments/gymnasium.py` and `scenarios/reinforcement/gymnasium.py`; tests exercise discrete/continuous/vectorized state policies and YAML/CLI/export workflows |
| Media-adjacent logging | `src/nexuml/tracking/logger.py:422-618`; `tests/tracking/test_logger.py` has staging, existing-run MLflow-style dispatch, TensorBoard artifact copies, image summary, and log-root-related helpers |
| Optional simulator/system boundaries | `tests/core/test_simulation_scenarios.py`, `test_ros2_environment.py`, `test_carla_environment.py`; static/fake and opted-in live proofs are separate |
| Docs, packaging, CI tiers | `docs/how-to/reinforcement-learning.md`, root/library pyprojects, `.github/workflows/ci.yml`, `tests/conftest.py`; normal CPU, optional simulator/system/GPU/data, and manual hardware gates are explicit. The manual `rl-simulators` job installs rendering/MuJoCo, not media exports |

Historical `.tasks/todo.md` reports 424 passed, 19 skipped and 62.26% coverage,
plus visual/MuJoCo and Ray proofs. Those are historical records only and were not
reproduced in this audit. No current test-pass or simulator-capability claim follows.

## Capability matrix — source assessment plus current CPU evidence

“Present-and-tested” applies only to the specific current CPU paths listed, not
convergence, all optional backends, or statistical robustness. “Present-but-not-system-tested”
separates existing static/fake tests from unexecuted live integrations.

| Capability | Classification | Evidence / limitation |
| --- | --- | --- |
| Gymnasium CartPole/Pendulum | Present-and-tested | Focused 59-pass run exercises both real CPU references, updates, evaluation, checkpoints, and CLI/export paths; no convergence claim |
| Visual Gymnasium | Present-and-tested | Existing visual CartPole static and real CPU smoke: 2 passed; image-observation behavior preserved |
| MuJoCo | Present-but-not-system-tested | Existing static Humanoid/Reacher contracts pass; native MuJoCo was not installed or rerun in this scope |
| ROS/Gazebo | Present-but-not-system-tested | Maintained task, production transport, fake tests, and operator-supplied live fixture exist; original 9.5 still open |
| CARLA | Present-but-not-system-tested | Static/fake-server/export and live opt-in test exist; original 10.4 still open |
| Ordinary datasets | Present-and-tested (bounded CPU paths) | Current regression exercises synthetic/local dataset paths; DALI index/export failures remain explicit, and GPU/data tiers were deselected |
| Demonstration datasets | Missing | No maintained trajectory/LeRobot source identified in the library data inventory; ordinary samples and data/interaction coexistence do not provide episode/time/action-chunk alignment |
| Imitation learning | Deferred | Original design D22/non-goals and RL guide explicitly defer training modes; reusable policy/data seams are not an IL implementation |
| VLA | Deferred | Tensor contracts/direct adapter exist, not VLA training, tokenizer alignment, chunk execution, or temporal policy state |
| Isaac | Deferred | Original design D19/D20 and RL guide leave Isaac integration outside scope; no maintained adapter identified |
| Physical-AI benchmarks | Missing | Existing tests are structural/smoke checks without learning thresholds; no dedicated multi-seed benchmark protocol/harness identified in the inspected RL implementation |
| Policy export/resume | Present-and-tested (existing cases) | Focused package/checkpoint/clean-inference tests pass for both-reference exports and existing partial CartPole resume; new acceptance cases remain |
| Episodic scalar evaluation | Present-and-tested (existing cases) | Existing fresh-runtime deterministic override and cleanup cases pass; RNG/mixed-mode/contract gaps below remain |
| Scheduled evaluation and GIF/MP4 recording/replay | Present-and-tested | Native initial/periodic/final evaluation, resume/dedup/isolation, all three formats, MLflow-only/both routing, exported-policy CLI, and actual playback proved below |

Native GIF-summary and MP4 encoding primitives and their integrated recording/replay
paths are proved below. ROS/CARLA fake/static tests passed in the CPU regression, but
their live integrations and the original 9.5/10.4 tasks remain unverified/open.

### Concrete temporal/multimodal limits (deferred, not changes)

- `TensorFieldContract.shape` has fixed positive dimensions; batch axes are separate
  but history length, timestamps, action horizons, masks, and sensor alignment have
  no dedicated semantics. A tensor can hold chunks without defining their execution.
- The direct adapter routes one tensor action. The mandatory runtime exposes one
  `step(action)` transition, not temporal chunk scheduling or stateful policy reset.
- `compile_context_from_interaction()` accepts a fixed tensor-dtype set and rejects
  raw string observations. Language must already be encoded into supported tensors;
  a modality string does not implement tokenization, variable lengths, or alignment.
- Existing camera/joint tests validate named tensor fields, not asynchronous sensor
  synchronization, sequence datasets, demonstration episode boundaries, or VLA training.

## Prioritized gaps and baseline gate

1. **Resolved environment blocker:** the approved scoped `.venv` installation now
   imports and executes the worktree correctly. Preserve explicit environment
   selection, and do not mutate either pre-existing environment. Full regression
   and type gates still require optional dependencies outside this scope; their
   current failures/skips are listed above, not silently excluded or repaired.
2. **Known baseline failures:** original OpenSpec strict validation fails on six
   long-requirement warnings, and broader tests/types retain existing failures.
   Preserve original requirements/checkmarks and keep these failures visible in
   completion evidence. Task 1.6 permits reporting such blockers rather than
   silently fixing unrelated code; they do not require another feature approval.
3. **Planned evaluator work:** preserve all RNG streams and individual training
   flags; current cleanup can skip mode restoration if `runtime.close()` raises.
   Validate evaluation I/O/single-runtime semantics instead of `.sum()`/`.any()`
   across unvalidated batches. Add regression proofs before claiming these fixes.
4. **Planned feature work:** schedule from collected frames and deduplicate final
   results before logger finalization; add optional RGB capture, bounded encoders,
   existing-run artifact routing, reference logging, and inference-only replay.
   Existing artifact dispatch is the reuse seam, not a reason to add a logger.
5. **Missing acceptance cases:** ordinary CLI overrides, both-reference interrupted
   resume, seeded stochastic isolation, encoder-blocked inference, media formats,
   actual event bytes/playback, and real/local MLflow-only routing need execution
   evidence. Reuse existing tests rather than duplicating their structural assertions.

**Audit and scoped implementation complete.** Checkmarks represent the requested
implementation/proofs and honest blocker reporting, not all-gates-pass claims.
Original live ROS/Gazebo and CARLA tasks remain open; no unrelated fixes,
original-spec changes, commits, pushes, or archive actions are included.

## Implementation evidence (incremental)

- Focused integrated CPU run: `tests/core/test_reinforcement.py`,
  `test_rl_evaluation.py`, `tests/cli/test_overrides.py`, interaction/policy/static
  simulator contracts, and `tests/tracking/test_evaluation_media.py`, with
  `-m 'not requires_simulator' -q --tb=short --disable-warnings`: **130 passed,
  15 deselected**, 97 warnings, 28.69 seconds. Additional regressions are being
  added before final gates.
- Real recording run: `NEXUML_RUN_SIMULATOR_TESTS=1 SDL_VIDEODRIVER=dummy
  .venv/bin/python -m pytest tests/core/test_rl_evaluation.py -k real_headless
  -q --tb=short --disable-warnings`: **12 passed, 29 deselected**, 52 warnings,
  53.77 seconds. Both references exercised GIF, MP4, and both formats with native
  TensorBoard-only and MLflow-only targets; actual artifacts/event bytes decode.
- CLI stabilization regression reproduced frozen-component override failure before
  its fix (1 failed). The native override helper now validates/rebuilds immutable
  components and propagates replacements without mutating the original definition.
  Existing and new override/CLI tests pass within the integrated run above.

## Final completion evidence

### Implemented outcome

- Both existing state-policy references use native TensorBoard defaults; initial,
  periodic, and media behavior remain opt-in. Scalar default fit/run behavior stays
  intact. Evaluation uses fresh compatible single environments and preserves
  Python/NumPy/PyTorch random streams, mixed policy modes, parameters, and gradients.
- Frame-based start/progress/final evaluation runs after updates/synchronization,
  handles crossed intervals and partial/full-budget resume, and deduplicates final
  results/media. Cache identity includes the resolved fallback seed.
- Optional Gymnasium RGB capture leaves vector observations unchanged. One bounded
  first-episode buffer supplies GIF, silent H.264/yuv420p MP4, or both. Capture caps
  do not truncate scalar episodes. Native artifact dispatch serves MLflow-only,
  TensorBoard-only, and both; only TensorBoard gets an animated image summary.
- MP4 finalization verifies dimensions and the complete encoded frame count before
  publication. Runtime/encoder failures, cancellation, concurrent cleanup failures,
  upload failures, unsupported targets, and missing prerequisites remain visible.
- `nexuml evaluate <policy-package>` uses native package loading/overrides/loggers
  and the shared evaluator without a session, PPO, optimizer, collector, or export
  side effects. Standalone provenance distinguishes source frames from optimizer
  steps. Packaged Pydantic interaction contracts are normalized at this boundary.
- Direct/process lifecycle tests cover failure and cancellation. Two narrowly
  demonstrated stabilization fixes rebuild immutable components for native CLI
  overrides and lazily expose data-module convenience classes so policy loading
  does not require Pillow through eager Lightning/TorchMetrics imports.
- Scoped core/library rendering/video extras, locked dependencies, CLI/checkpoint/RL
  documentation, and the existing manual simulator CI job are wired. That job
  explicitly installs video and tracking prerequisites for local MLflow proofs;
  no default CI simulator downloads, remote service, or reward threshold was added.

### Final commands and results

Use the explicit worktree environment/CPU variables from the approved-environment
section. Raw command arrays, wall times, stdout, coverage, and gate outcomes are
under `/tmp/opencode/nexuml-evaluation-final-checks/`.

| Proof | Command / scope | Result |
| --- | --- | --- |
| Integrated focused CPU | `.venv/bin/python -m pytest tests/core/test_rl_evaluation.py tests/core/test_reinforcement.py tests/core/test_policy_evaluation_cli.py tests/tracking/test_evaluation_media.py tests/tracking/test_logger.py tests/cli -m 'not requires_simulator' -q --tb=short --disable-warnings` | **185 passed, 14 deselected**, 102 warnings, 23.83s |
| Both-reference real recording/resume/export/replay | `NEXUML_RUN_SIMULATOR_TESTS=1 SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest tests/core/test_rl_evaluation.py -k 'recorded_interrupt or real_headless' -q --tb=short --disable-warnings` | **14 passed, 42 deselected**, 62 warnings, 49.26s; 52.523s command wall time |
| Existing visual-observation regression | Same opt-in variables, `.venv/bin/python -m pytest tests/core/test_simulation_scenarios.py -k visual -q --tb=short --disable-warnings` | **2 passed, 6 deselected**, 8 warnings, 1.53s |
| Old checkpoint fields absent | `.venv/bin/python -m pytest tests/core/test_reinforcement.py -k partial_checkpoint -q --tb=short --disable-warnings` after removing all three new evaluation fields from both reference checkpoints | **2 passed**, restored defaults and remaining-frame/optimizer progress verified |
| Broader CPU/coverage | `.venv/bin/python -m pytest tests -m "not slow and not requires_gpu and not requires_data and not requires_simulator and not requires_system and not requires_hardware" --cov=nexuml --cov=nexuml_library --cov-report=xml:/tmp/opencode/nexuml-evaluation-final-checks/coverage.xml --cov-report=term -q --tb=short --disable-warnings -rs` | **530 passed, 4 failed, 12 skipped, 31 deselected**, 103 warnings, 43.90s; 48.891s command wall time; **60.67% coverage**, 50% threshold met |
| Lock/lint/format | `uv lock --check`; `.venv/bin/ruff check .`; `.venv/bin/ruff format --check .` | All pass; 263 locked packages, 221 formatted Python files |
| Types | `.venv/bin/ty check` | Exit 1: **exactly the same 37 baseline diagnostic signatures**; changed implementation/test files pass scoped ty checks |
| Strict docs | `DISABLE_MKDOCS_2_WARNING=true .venv/bin/mkdocs build --strict` | Pass, 4.897s command wall time |
| OpenSpec | Strict validation of new change and original RL change | New change passes; original has the same six long-requirement warnings and remains untouched |
| Whitespace | `git diff --check` plus explicit new-file whitespace assertions | Pass |

The four broader CPU failures are exactly the baseline DALI/wds2idx and Optuna
cases listed above; none were deselected or marked passing. The 12 skips retain
the same explanations. Additional deselections are the explicitly opted-in real
recording cases, not hidden failures. The coverage total is for this scoped CPU
selection, not the historical full/all-extras or GPU suite.

### Actual playback, not just encoded bytes

The CLI proof directory `/tmp/opencode/nexuml-rl-cli-proof/` contains both
documented resolve/train/export/scalar replay/completed-budget resume workflows,
all three recording formats, native MLflow-only/both replay, and scheduled native
events at frames 0, 16, and 32.

Against its real scheduled CartPole artifacts, headless Chromium opened TensorBoard
2.20's Images dashboard and selected steps **32 → 0 → 16** using the actual slider.
Eight screenshots of the image produced **six distinct pixel images**, proving
rendered animation rather than a static first-frame preview. Screenshots were
visually inspected. Chromium's standard video page played the MP4 to 0.0632s of
its 0.18s duration with no media error, at 256×172 padded dimensions. Native
**ffplay 8.0.1 exited 0** using SDL software/headless output.

Evidence: `/tmp/opencode/nexuml-playback-proof/result.json`, `tensorboard.png`,
`animation-*.png`, `mp4-player.png`, and `ffplay.log`. The assert-based browser probe
is `/tmp/opencode/nexuml-verify-playback.py`; it ran through the skill's
`with_server.py` helper with local TensorBoard port 6014 and artifact HTTP port
6015. Both owned servers and the browser were stopped. Playwright/Chromium were
explicit development-only installations, not application dependencies or runtime
downloads. These short videos establish usability, not convergence or policy quality.

### Distribution and optional-dependency proofs

Both current **0.2.1** wheel+sdist pairs were rebuilt from this implementation at
`/tmp/opencode/nexuml-evaluation-distributions-final/{nexuml,nexuml-library}/`:

```bash
uv build --package nexuml --out-dir /tmp/opencode/nexuml-evaluation-distributions-final/nexuml
uv build --package nexuml-library --out-dir /tmp/opencode/nexuml-evaluation-distributions-final/nexuml-library
```

- Twine 7.0.0 passed all four explicit `.whl`/`.tar.gz` files; the distribution
  artifact test passed. UV built each wheel from its source distribution.
- A fresh separate environment at
  `/tmp/opencode/nexuml-evaluation-packaging-checks/clean-env` installed locked
  dependencies and actual wheels, not editable sources. Installed core and library
  tests each passed under `python -I`, with `PYTHONPATH` removed and the packaging
  tests' normal `-c /dev/null --rootdir/--confcutdir` isolation. Latest installed
  evaluator/logger source bytes match this worktree's final implementation.
- That environment lacks ImageIO and imageio-ffmpeg. Current installed-wheel
  CartPole completed **16 frames and two PPO updates**, returning finite metrics
  without importing either backend. GIF-only delivery decoded two exact 8×8 frames
  at 100ms each with no MP4 dependency installed/imported.
- Initial pip export lost UV's per-package index binding, producing a first-index
  `certifi` resolution error. The correction used `uv sync --locked --package ...
  --no-install-workspace` into the isolated environment, preserving explicit source
  bindings, then installed the wheels with `--no-deps`. No unsafe index option,
  dependency downgrade, or repository config workaround was used. Later packaging
  harness corrections limited Twine inputs to distributions (not UV's generated
  `.gitignore`) and updated provenance assertions to the final artifact directory.
  Those failed harness attempts remain distinguishable from product failures.

Exact commands/timings are in `packaging-final-fixed.json` and
`optional-boundaries-final.json` under the final-check directory, and
`acceptance-summary.json` plus per-command logs under the packaging-check directory.
Final installed core/library test wall times were 3.209s/2.457s; no-video PPO and
GIF-only proof wall times were 2.365s/1.525s. Inference tests additionally block
Pillow, ImageIO, MoviePy, TorchRL, Gymnasium, and simulator imports in clean child
processes while loading/executing both complete policies.

### Review and remaining boundaries

Independent read-only review found no P0/P1 issues and two P2 edge cases: changing
the fallback training seed could reuse stale evaluation results, and an encoder's
close error could hide its original failure/cancellation. Both were corrected and
covered by regressions; a playable-but-truncated MP4 also cannot be published as
completed output. No unrelated baseline defect was changed to make gates appear green.

CI was wired, not remotely dispatched. GPU/other Python/platform wheels,
real-data/system/hardware tiers, live ROS/Gazebo/CARLA, and new MuJoCo runs were not
performed. Original 9.5/10.4 stay open. All changes remain uncommitted; no archive,
merge, release, Phase 2 implementation, or convergence claim is included.
