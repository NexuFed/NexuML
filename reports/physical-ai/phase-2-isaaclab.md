# Physical AI Phase 2: Isaac Lab integration

## Status — 2026-10-19

**Scoped implementation and local verification complete, using operator-approved
whole-job fast shutdown.** Change: `add-isaaclab-gpu-robotics`, spec-driven,
**42/42 tasks complete**. The expanded roadmap's separate P1 robustness/system-
identification preparation gates remain unchecked; this does not complete that
broader Phase 2 or imply hardware, remote-CI, convergence, merge or release.
The user now authorizes another local container and implementation without
restarting this DevContainer or disturbing other experiments. EULA acceptance is
standing approval. The earlier prepare-only status below is retained as history.
The official digest-pinned Isaac Sim image and isolated source/dependency overlay
built successfully. Native Cartpole/Franka baselines execute, and NexuML Cartpole
updates/exports, but returning shutdown segfaults at interpreter exit. The user
rejected an upstream patch and approved non-graceful shutdown instead. Cartpole
and Franka now each train/export/finalize/replay and exit zero through the
whole-job entrypoint; SO-101 now also trains, resumes, exports and replays through
the same path. This is not a graceful-shutdown fix. Latest evidence is below.
No current DevContainer, driver, ordinary Python environment or running experiment
was restarted or modified. All three training/resume/export/scalar replay round
trips pass; media/player, final image/profile, evidence audit and owned-resource
closeout also pass. All three task-owned containers and their children are stopped;
images/caches/artifacts remain available. Other experiments were left untouched.

Starting commit: `1e9c420892501effaeb1dedb6818f04bf58c204d`
(`Merge latest origin changes (NEX-212)`), on
`feature/nex-212-add-reinforcement-learning`. Tracked/untracked git status was clean
before these edits. Earlier Phase 0–1 implementation and the Phase 2 proposal are
already committed in this baseline; older reports' uncommitted-state notes are
historical. This apply has not made a commit. Only the roadmap's Phase 2 core
progress and linked evidence were updated; expanded and later gates remain intact.

Required reference order remains **Cartpole → Franka Reach → SO-101 Joint Reach**.
Native baselines and bounded NexuML GPU evidence are recorded below. Required
SO-101 task/native/scripted/learning round trips now pass. Approved whole-job fast
shutdown passes with truthful status; returning graceful shutdown is still broken.

### Latest verification checkpoint

- `opt-in-native-suite-v2.log`: first **four real GPU tests passed**: all three
  checkpoint/remaining-budget/completed-budget/CPU-inference/scalar-replay workflows,
  plus Cartpole's non-divisible total budget (requested193, actual256).
- Camera-native probes for all three return valid 320×240 uint8 RGB and preserve
  pre-reset terminal images despite subsequent live reset rendering. Cartpole's
  terminal scene is deliberately displaced; its SHA256 differs from reset.
  The first cold RTX start took293s; cached arm probes took~11s each.
- Media tests first exposed fixture assumptions, not a new encoder: GIF uses
  the existing recorded10ms timing quantization (60FPS→20ms frames), and CLI JSON
  mapping overrides are needed for a real null (literal `null` is a string).
  Failed attempts are retained; tests now check the encoder metadata and use the
  existing whole-logging mapping override. Full selected-format suite is running.
- Isolated worker `pip check` is now **clean**. Only `usd-validation-nvidia==1.22.0`,
  `PyJWT==2.15.1`, `aiobotocore==2.25.1` were added to repair the vendor's missing
  dependencies; all pre-existing distribution versions match the frozen inventory.
  `pytest==9.1.1` plus iniconfig/pluggy were then added for the isolated opt-in tests.
  Full320-distribution constraints are now retained, and the image recipe includes
  the additions and a clean-dependency gate. Failed default/root install attempts
  changed no existing distribution; installation succeeded in the simulator user's
  site. No host environment or shared credentials changed.
- `upstream-profile-integrity.log`: exact pinned git HEAD, clean upstream tree,
  expected Python/Torch/bridge versions, clean pip check, no app allocation.
- Broader CPU+coverage: **868 passed, same4 baseline failures,12 skipped,44
  deselected**, **61.66%** coverage (50% gate). No DALI/Optuna/Ray repair or install.
- Repository lint/format, offline unchanged lock, strict docs/new OpenSpec,
  whitespace/shell and both wheel+sdist builds pass. A fixture dynamic Trainer
  attribute added one type diagnostic; that access is now corrected to follow the
  runtime attribute explicitly. Final type rerun is pending (baseline37).
- All four distributions pass Twine; artifact checks pass. Installed-wheel proof
  executes all three actual trained policy packages with exact CPU equality and
  YAML contract round trips, importing only the installed core/library wheels and
  prepared third-party dependencies, no repository source paths, Isaac/TorchRL or
  encoders. Optional discovery PIL warnings are visible, not suppressed.
- Image rebuild is pending: the inherited DevContainer credential helper failed
  before Docker build started. Retry uses an empty task-local Docker config and
  the already cached public image, not repaired/shared credentials.

Latest logs and command/status JSON:
`/tmp/opencode/nexuml-isaaclab-integration-20261019/`. Older pending statements below
are historical; only checked authoritative tasks establish acceptance.

### Latest native media, cancellation, CLI and distribution proofs

- `final-native-acceptance.log`: **14/14 real GPU tests pass** in417.27s, including
  all three resume/export/scalar workflows, rounded budget, actual SIGTERM during
  training and all nine reference×format cases. A subsequently added fifteenth
  test, `native-rgb-replay-sigterm.log`, also passes: real SIGTERM after the second
  RTX frame exits143, closes native resources, records one MLflow **FAILED** run
  and publishes no incomplete GIF. The15 cases have not yet been run together on
  the rebuilt image; the first14 ran together in the validated source-synced worker.
- Independent read-only review found AppLauncher's SIGTERM handler could fast-close
  before job cleanup. The process scope now saves/restores the caller's handler
  and reinstalls its own after public AppLauncher construction, raising conventional
  SystemExit143 to unwind first. Before-fix native trace exits143 without cleanup;
  after-fix native and real OS subprocess tests prove ordered cleanup/143. SIGINT
  still uses normal KeyboardInterrupt. A connected CLI SystemExit path initially
  marked success/could replace143 on logger-close failure; the narrow shared CLI
  handler now finalizes failed loggers and re-raises the original status. Its test
  was red, now passes;76 CLI/process tests pass. Reviewer rechecked both fixes.
- `decoded-native-media.json`: all nine cases use nine first-episode frames,
  expected60/30Hz source/capture FPS, source-frame-unavailable/step0 provenance,
  exact selected files and paired formats from the same captured buffer. Six actual
  TensorBoard previews decode to multi-frame GIFs. MLflow-only has no new TensorBoard
  events and all successful MLflow evaluation runs finish normally.
- `standard-player-proof.json`: Chromium's standard HTML video player actually
  decodes/plays all three H.264/yuv420p files at320×240, with three frame callbacks,
  advancing decoded frames and no media errors. Durations:0.15s Cartpole,0.30s arms.
- `tensorboard-browser-proof.json`: actual native event dashboard displays six image
  tiles and four distinct animated-frame screenshots; selecting SO-101 evaluation
  runs shows two tiles at source step0. Nonzero step slider is inapplicable to these
  packages (no source training_frames); earlier Phase0–1 shared logger slider proof
  remains unchanged. Owned TensorBoard/browser processes were closed after checks.
  Screenshots and copied real GIF/MP4 files are under `playback/`.
- `documented-native-cli-results.json`: the published two upstream RSL-RL baselines
  and all three ordinary `nexuml train` commands run as written, each256 GPUframes,
  exit0. `documented-cli-roundtrip-results.json`: ordinary CLI checkpoint resume,
  complete-policy export from checkpoint and fresh scalar replay all exit0 for all
  three references. No user source edit, alternate learner or convergence threshold.
- `published-native-probes.json`: the now-maintained `scripts/isaaclab/native_probe.py`
  executes all three tasks in order,64 GPUframes each, partial/all-row timeout and
  distinct final/reset states, correct actions/limits/period. SO-101 preserves and
  checks its authored asset profile. The deliberate four-step probe horizon is not
  the training horizon or a learned success claim.
- `native-contract-rejections.json`: three actual constructed native GPU tasks reject
  observation shape, action scale and control-period mutations; shared semantic
  contract validation rejects changed joint-order/units descriptions. No control
  steps occur and restored profiles validate normally. Portable CPU/YAML examples
  still require no Isaac installation.
- Latest broad run: **869 passed, same4 DALI/Optuna failures,12 skips,45 deselections**;
  focused **512 passed,28 deselections**. The added CLI cancellation test then passes
  in the76-test focused recheck; counts overlap, not additive. Final rerun pending.
  Full types return to exactly **37 baseline diagnostics**. Repository lint/format,
  lock/docs/OpenSpec/whitespace, both wheel+sdist builds, Twine/all-four artifact
  checks and installed-wheel exact CPU inference/contract proofs pass. No baseline
  failures were repaired, hidden or treated as passing gates.
- Manual-only `isaaclab` CI selection runs a prepared cached image with explicit
  GPU/simulator opt-in, readonly checked-out source and bounded resources. Default
  CPU selection deselects native tests and installs/downloads/launches no Isaac.
  Wiring/local execution are proved; no remote workflow dispatch was requested/run.
- Rebuild attempts are explicit: inherited Docker credential helper failed before
  build; isolated empty Docker config avoids touching credentials. One owned build
  was deliberately stopped to move full constraints before upstream installation.
  The pinned retry exposed upstream's Torch-only index hiding the pinned NumPy
  distribution. Both PyPI and the CUDA128 wheel index are now available throughout,
  without loosening any version. Final rebuild is still pending, and will need the
  final CLI cancellation fix/native probe copied into its source before validation.

### Exact rebuilt profile — final acceptance running

The fully constrained build succeeded in721.51s, followed by a104.27s cached
application-layer rebuild carrying the final CLI cancellation fix and published
native probe. Final local image is
`sha256:970c503d596662687fa263c670a3c1d5214e630bffe2da3fc2b9092dfa951c0c`
(`nexuml-isaaclab:phase2-validated`),30.259GiB. All version pins apply before
upstream installation as well as the NexuML overlay. Images were not pushed.

`rebuilt-profile-integrity-v2.log` verifies the actual final image's runtime/CLI
source and constraint SHA256s, correct import origins, exact Python3.12.13 /
Torch2.11.0+cu128, working CUDA tensors, pristine pinned upstream and clean
`pip check`. All305 installed distributions match declared constraints; none are
unconstrained. Fifteen no-longer-needed distributions from the old unpinned
intermediate install are absent, rather than being installed merely to reproduce
unused CUDA13 packages. The full320-entry constraint file bounds either case.
The first integrity helper incorrectly expected build-only files at checkout paths;
its corrected image-path mapping checks constraints under `/opt` and records the
Dockerfile identity in the image build manifest. No application source was changed
inside the image to make the check pass.

The exact-image worker `nexuml-isaaclab-phase2-validated-20261019` has the existing
task label, GPU0, four CPUs,16GiB RAM,2GiB shared memory, no host mounts/privileges
and no extra telemetry opt-in. Only test fixtures and the integrity helper were
copied in; implementation/dependencies are the image's own. All15 native tests are
now running together there. Until that run finishes,1.3/1.6 and final audit/closeout
remain unchecked. Both original task workers and their children are stopped;
the new validation worker will be stopped after artifacts are retained.

Final ordinary gates after cancellation changes: **870 CPU passes**, the same
four DALI/Optuna failures,12 skips,46 explicit opt-in deselections; full types
retain37 baseline diagnostics, original RL strict validation retains its baseline
warnings. Repository lint/format, unchanged offline lock, strict docs/new change,
whitespace and final package/Twine/installed-wheel proofs pass. Exact commands/
statuses: `closeout-gates.json`, `release-gates.json`, `installed-wheel-final.log`.
No suppressed or newly repaired unrelated gate is claimed.

### Completed closeout — exact final image

`rebuilt-image-full-acceptance.log` records **15/15 passed** on the final image in
797.75s (801.78s daemon-confirmed wall time), with no native test skips. The same
image then executes all three published native64-frame probes, both upstream
256-frame RSL-RL baselines and the default CLI help command, all exit0; commands and
results are in `rebuilt-published-commands.json`. Final pre/post-test integrity
checks retain all305 constrained versions, matching implementation hashes, CUDA
operation, clean dependency resolution and byte-pristine pinned upstream.

`rebuilt-image-artifacts/` retains329 actual output files, including the new-image
policies, checkpoints, CPU oracles, events, selected media, JSON and cancellation
MLflow store. `rebuilt-image-artifact-index.json` records their hashes/sizes.
`rebuilt-image-audit.json` independently decodes nine format cases and six animated
native TensorBoard previews, with six FINISHED MLflow runs and one genuinely
cancelled FAILED run. Paired native logger metadata agrees across destinations.
The actual final-image MP4s also pass standard Chromium player callbacks without
media errors (`rebuilt-standard-player-proof.json`, `playback-rebuilt/`).

`stopped-final-resources.json`: all three labeled task containers and their children
are stopped. Owned browser/TensorBoard servers and build jobs are finished. No
driver/DevContainer/other experiment restart, image push, commit, merge, archive,
deployment, hardware or later-phase work. Images, SDK/assets/caches and evidence
are retained. Free disk186.37GiB, about27.65GiB below this continuation's entry level,
within the60GiB cap and20GiB floor. GPU management showed781MiB used /15,192MiB free
and69% activity after owned shutdown; that unrelated activity was not interrupted.

All42 scoped tasks are checked against the audit below. Only the ten core Phase2
roadmap items and its progress/evidence text are updated. Additional Phase2 P1
items/held-out gate, the coarse Phase2 checkbox, Phase0 follow-ups, original live
ROS/Gazebo/CARLA tasks and Phases3–10 remain unchanged. Final CPU/regression evidence:
**870 passed,4 known unrelated failures,12 skips,46 deselections,61.67% coverage**
(50% gate),37 existing type diagnostics and original strict warnings retained.

## Final acceptance audit map

| Capability spec | Actual evidence | Result / boundary |
| --- | --- | --- |
| `isaaclab-runtime` | Pinned clean305-distribution image; CUDA/import/source integrity; fake lifecycle/order/mismatch/ownership tests; native partial/all resets, contract rejections and SIGTERM status/cleanup | Pass; final-image15-test suite and stopped owned resources |
| `isaaclab-rollout-collection` | Real wrapper composition fails safely; canonical existing-adapter fallback; exact GAE/final-reset values; CUDA `[8,16]`, actual/rounded frames, log-probabilities, update/sync sequence; no nested vectorizer/new learner | Pass; single direct synchronous GPU only |
| `isaaclab-policy-replay` | All three scalar/recorded replays; no learner path; CPU equality under blocked optional imports; installed wheels; nine format/backend cases, animated native events, real browser/player checks; genuine RGB SIGTERM →143/MLflowFAILED | Pass; no remote CI dispatch implied |
| `isaaclab-reference-workflows` | Native Cartpole/Franka RSL baselines; all three published train/probe/resume/export/replay CLIs; actor/critic/optimizers/counters/remaining/complete budgets | Pass; final-image setup/native commands confirmed |
| `so101-simulation-reference` | Exact authored PhysX asset/gains/limits; five joints/fixed gripper; scripted reachable goals/hold/reward/timeout; real PPO/portable inference/resume/replay/media/error metrics; ordinary published CLI | Pass; no hardware/manipulation/convergence claim |

The approved non-graceful whole-job shutdown exception remains explicit. Human
reaching quality, remote CI, ROS/Gazebo/CARLA, hardware and later roadmap phases are
not established by any of these passing local tests.

The current roadmap also contains additional Phase2 P1 domain-randomization,
sim-to-sim robustness and system-identification preparation items, including a
held-out-randomization gate. These are outside this frozen42-task integration
change and have not been implemented or verified. Even after this change closes,
the broader roadmap's Phase2 checkbox must not imply those extra gates passed;
retain them unchecked and preserve all later-phase constraints.

## Reconciled versions and upstream source

The tag API still maps `v3.0.0-EA` to commit
`ae37b028ea415c91ea2bc32609efcd759ed2b974`. The release text explicitly says
**Early Access**, despite GitHub reporting `prerelease=false`. The frozen profile
remains Isaac Sim 6.1.0 (Python distribution 6.1.0.0), Python 3.12, Torch
2.11.0+cu128 / torchvision 0.26.0+cu128, Linux x86_64, explicit `isaacsim_physx`.
Ubuntu 24.04 userspace is the selected integration target, not this container's OS.

Pinned upstream source inspection confirms:

- `pyproject.toml`: Python >=3.12,<3.13, Isaac Sim extra 6.1.0.0, exact Torch
  overrides/source indexes, Warp 1.16.0, USD-exchange 2.3.0, Pillow >=12.3.0 and
  upstream resolver overrides. The ordinary NexuML lock cannot be reused as an
  Isaac environment lock; a separately resolved overlay still needs verification.
- Manager Cartpole: `slider_to_cart` continuous effort action, scale 100 N,
  dt=1/120 s, decimation=2 (60 Hz policy steps), 5 s horizon.
- Franka Reach: default joint-position action on `panda_joint.*`, scale 0.5 with
  default-pose offset; actual resolved joint order/limits still need native proof.
- `ManagerBasedRLEnv.step`: reward/termination before native same-step reset;
  `compute_final_obs` exposes `extras['final_obs']` before reset, followed by
  `record_pre_reset`, reset, and post-reset observations. Native buffers must not
  be assumed immutable. Public `ManagerBasedEnv.reset(seed, env_ids, options)`
  supports per-index reset, so private native reset patching is unnecessary.
- SO-101 asset: five arm joints plus separate `gripper`; the default asset's
  Physics variant is `physics`, so the selected PhysX task must explicitly use
  `physx`. Asset-authored limits/gains are not established by Python source alone.
  The stack base has `rewards=None`; it is not the required PPO task.

## Existing code boundaries and intended reach

| Boundary inspected | Reuse / required later change |
| --- | --- |
| `src/nexuml/core/components.py:118-147` | Static `EnvironmentDefinition.describe()` and execution-local `build(context)`; keep Isaac imports/handles out of portable definitions. |
| `src/nexuml/reinforcement/collector.py:14-128` | Factory unconditionally adds the generic TorchRL adapter; a maintained bridge is optional only after terminal/reset probes. One native GPU batch, no process vectorizer. |
| `src/nexuml/reinforcement/torchrl_env.py:14-104` | Existing batch/spec/reset/step conversion; needs correct runtime final/reset states, not a replacement collector. |
| `library/src/nexuml_library/environments/gymnasium.py:118-189` | Per-env loop and CPU/NumPy action conversion are unsuitable for Isaac's GPU hot path. Add the separate library provider, not an Isaac branch in this adapter. |
| `library/src/nexuml_library/reinforcement/algorithms/ppo.py:31-93,127-153` | Reuse probabilistic actor, critic, GAE before flattening, and Lightning optimization; no alternate learner. |
| `src/nexuml/training/reinforcement.py:82-196,230-414` | Remaining-budget collector, synchronization, evaluation lifecycle and scalar/media loop; add narrow lifecycle restrictions and optional SO-101 outcomes later. |
| `src/nexuml/training/lightning.py:787-805,885-903` | Session owns collection shutdown and final evaluation; zero episodes must remain safe for Isaac's initial training defaults. |
| `src/nexuml/core/export.py:512-661` | Existing complete-policy metadata/contract/action state and inference-only packaging; native state must stay out. |
| `src/nexuml/cli/main.py:340-474` | Existing trusted-package load, overrides, native loggers and shared evaluator; retain one fresh-process replay command. |
| Installed TorchRL `envs/libs/isaac_lab.py` | 0.14.0 maintained wrapper documents NaN/unavailable terminal observations for native autoreset; wrapper compatibility is not proved by its presence. |

Later edits must reach the library provider/task/scenario modules, the selected
collector bridge if needed, session/evaluator guards/outcomes, focused tests,
isolated profile, docs and manual-only CI. Preserve callers, typed serialization,
checkpoint compatibility, logger/media behavior and supervised/Gymnasium paths.

## Fresh CPU/static baseline

Use the worktree interpreter explicitly; no synchronization/install was run:

```bash
export VIRTUAL_ENV="$PWD/.venv"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/library/src"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NEXUML_RUN_SIMULATOR_TESTS=0 NEXUML_RUN_SYSTEM_TESTS=0
export NEXUML_RUN_HARDWARE_TESTS=0 NEXUML_RUN_DATA_TESTS=0 TMPDIR=/tmp/opencode
```

Both actual package imports and editable metadata point into this worktree.
Installed versions: Python 3.12.13, core/library 0.2.1, Torch 2.11.0+cu128,
torchvision 0.26.0+cu128, Lightning 2.6.5, TorchRL 0.14.0, TensorDict 0.14.2,
Gymnasium 1.3.0, NumPy 2.4.6, Pillow 12.2.0, ImageIO 2.38.0,
imageio-ffmpeg 0.6.0. Isaac Lab/Sim distributions are absent.

Focused test command:

```bash
.venv/bin/python -m pytest tests/core/test_interaction.py tests/core/test_policy.py \
  tests/core/test_compiler.py tests/core/test_session.py tests/core/test_reinforcement.py \
  tests/core/test_rl_evaluation.py tests/core/test_policy_evaluation_cli.py \
  tests/tracking/test_evaluation_media.py tests/tracking/test_logger.py tests/cli \
  -m 'not requires_simulator and not requires_system and not requires_gpu' \
  -q --tb=short --disable-warnings
```

Broader baseline command (no coverage remeasurement):

```bash
.venv/bin/python -m pytest tests \
  -m 'not slow and not requires_gpu and not requires_data and not requires_simulator and not requires_system and not requires_hardware' \
  -q --tb=short --disable-warnings -rs
```

| Check | Result | Command wall time |
| --- | --- | --- |
| Worktree import origins | Pass, actual imports and path assertions | 1.490 s |
| Focused CPU | **208 passed, 14 deselected**, 95 warnings | 19.866 s |
| Broader CPU | **536 passed, 4 failed, 12 skipped, 31 deselected**, 84 warnings | 33.280 s |
| `uv lock --check --offline` | Pass, 263 packages | 0.009 s |
| `.venv/bin/ruff check .` | Pass | 0.010 s |
| `.venv/bin/ruff format --check .` | Pass, 222 Python files | 0.008 s |
| `.venv/bin/ty check` | Exit 1, 37 baseline diagnostics | 0.178 s |
| `openspec validate add-isaaclab-gpu-robotics --strict` | Pass | 0.389 s |
| `openspec validate add-reinforcement-learning --strict` | Exit 1, same six long-requirement warnings | 0.361 s |

The four failures are the previously documented missing DALI/wds2idx export test,
missing DALI S3 test and two missing Optuna tests. Skips are audio-shape
conformance (1), DALI (1), ONNXScript (2), Ray (5) and unrequested distribution
proofs (3). They are not passing simulator proofs or reasons to repair unrelated
code. The type count matches the earlier baseline; this is a fresh baseline at
the merged commit, not a claim of byte-identical old diagnostic signatures.

Exact subprocess command arrays, all exit codes and raw logs are under
`/tmp/opencode/nexuml-isaaclab-preflight-20261018/` in `baseline-results.json`
and the named `.log` files. Dependency manifests/lock and application files
remain unchanged. Strict docs, builds, coverage, GPU and playback were not rerun.

## Read-only setup inspection

Resource detection used the bundled `get-available-resources` detector with the
worktree Python, stdout only. No stress test, GPU allocation or resource
fingerprint file was created. Inventory is not validated simulator usability:

- Current userspace: Ubuntu **26.04.1**, x86_64 container; not the approved 24.04
  profile. Visible/affinity CPUs: 28 logical, without a finite observed CPU quota.
- Host-visible RAM: approximately **62.6 GiB** total / **37.4 GiB** available;
  no finite observed cgroup memory cap. These are observations, not reserved RAM.
- One management-visible RTX **4080 SUPER**, approximately **16 GiB VRAM**
  (15.6 GiB free at inspection), driver **610.43.03**. Character-device read/write
  permissions exist; CUDA execution and Isaac compatibility remain untested.
- Current graphics access is compute-only: driver capabilities `compute,utility`,
  no Vulkan ICD JSON files, no `vulkaninfo`/`glxinfo`, no `/dev/dri` nodes observed.
  This does not meet the graphics prerequisite proof. Do not try to fix the active
  container/host implicitly or switch to another backend.
- Worktree filesystem: approximately **85.6 GiB** free. Docker daemon storage
  capacity, project quotas, expanded image size and asset/cache sizes are unknown.
  The documented 50 GB simulator minimum alone does not budget Lab/dependencies,
  temporary layers, caches and training.
- Docker client **29.8.2**, reachable daemon **28.0.4**, NVIDIA runtime advertised;
  no matching `*isaac*` images listed. No container was started. A supplementary
  Podman inspection failed (`cannot clone: Operation not permitted`); no rootless
  Podman workaround or permission changes were attempted.

Only registry metadata was fetched with `docker manifest inspect`, not image
layers. `nvcr.io/nvidia/isaac-sim:6.1.0` currently resolves its linux/amd64 image to
`sha256:0c16dd67d09a70ea474f2c809a13c4d09bd23c738184cf1bf28af487ecd39080`.
Its 19 compressed layers total **10,634,065,490 bytes (9.90 GiB)**. This is download
size, not expanded storage. The two manifest JSON files are in the local proof
directory. Neither OS contents nor runnable image compatibility were inspected.

## Runtime proposal before the user's prepare-only decision

Recommended next action: prepare a separate, task-owned **Ubuntu 24.04**
graphics-enabled Docker environment through the existing daemon, using the
official pinned Isaac Sim image plus the frozen Isaac Lab revision. Verify the
base image's userspace and resolved profile before use. Do not change host
drivers/toolkit or the current workspace environment to force compatibility.

Approval must cover official image/dependency downloads, narrowly required
Cartpole/Franka/SO-101 assets and caches, image/container-local setup, headless
GPU/graphics compatibility checks and bounded native rollouts/training. Isaac
Sim requires explicit license acceptance (`ACCEPT_EULA=Y`); the operator must
review the linked terms. Optional telemetry consent will **not** be enabled.

Proposed guardrails for approval: cap new setup/cache usage at **60 GiB**, keep
at least **20 GiB** available on every affected filesystem, inspect daemon storage
before downloading layers and stop if those guards or version pins cannot hold.
No full offline asset pack, unrelated image/cache deletion, host mounts beyond
task inputs/outputs, host package/driver changes, privileged container or optional
backend substitution is included. A prepared approved runtime with a sufficient
storage location is an equally valid alternative.

Until approved, task 1.2 stays unchecked, native-baseline tasks stay blocked and
application integration does not proceed past the native-first gate.

## Corrected DevContainer preparation (no rebuild or install)

The user clarified that prerequisites belong in the **existing NVIDIA
DevContainer**, not a separate Isaac profile, and that EULA acceptance is standing
approval. The unnecessary separate profile/consent gate was removed; the correction
is recorded in `.tasks/lessons.md`. The earlier separate-container proposal above
was not executed and is superseded by this configuration:

- `/workspaces/NexuVibe/.devcontainer/nvidia/devcontainer.json`: only this profile
  selects Ubuntu 24.04 and adds `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`,
  `ACCEPT_EULA=Y`, and `OMNI_KIT_ACCEPT_EULA=YES`. GPU selection, mounts, startup
  hooks, CUDA 13.3.1, general Python 3.14 / `cu130` tools and `/env` stay intact.
- `/workspaces/NexuVibe/.devcontainer/Dockerfile`: Vulkan/EGL/OpenGL loaders and
  `vulkan-tools` are added only to the NVIDIA image stage. AMD/CPU stages and all
  unrelated dependencies remain unchanged. Registry metadata confirms the existing
  CUDA 13.3.1 tag has a Linux amd64 Ubuntu 24.04 image; no layers were downloaded.
- `docs/general/isaaclab.md`: instructions for the existing profile and a future
  separate Python 3.12 Isaac environment, with no repeated consent prompts.
- `tests/test_isaac_devcontainer.py`: two focused configuration/stage-isolation
  tests replace the nine tests of the discarded separate-profile design.

No build, rebuild, package installation or simulator launch was performed. These
changes take effect only after a future user-directed NVIDIA DevContainer
recreation. Ordinary pyprojects, lock, hooks, README and existing dependency tests
remain byte-identical to their prior user-owned state. EULA approval must not be
confused with optional telemetry consent, which remains off.

Configuration tests, lint/format, JSON parsing, documented Bash-block syntax and
whitespace checks pass. No resolved Isaac/NexuML overlay or working simulator is
claimed; task 1.3 and all native/GPU gates remain incomplete.

## Resumed implementation and isolated runtime

The later authorization supersedes the historical prepare-only restriction. No
Ray/KubeRay/RPC layer is being added. The task-owned container
`nexuml-isaaclab-phase2-20261018` uses the official 6.1.0 digest above, GPU 0,
4 CPUs, 16 GiB RAM, 2 GiB shared memory, no host mounts, dropped capabilities and
`no-new-privileges`. It does not replace or restart the current DevContainer.

The normal Docker credential helper failed before any download. A task-local
`/tmp/opencode/nexuml-isaac-docker-client/config.json` containing only empty
`auths` was verified against the same daemon ID and used for the public pull.
The pinned pull succeeded in **182.38 s**, leaving **64.34 GiB** free. Initial
daemon-storage verification used only a cached-image, read-only, no-GPU helper;
the resumed setup keeps the existing 60 GiB new-usage cap / 20 GiB free floor.

The base runtime is Ubuntu **24.04.3**, Python **3.12.13**; its VERSION is
`6.1.0-rc.26+release.49347.2d230af4.gl`. NVIDIA graphics libraries and
`/etc/vulkan/icd.d/nvidia_icd.json` are present. A direct Vulkan enumeration
asserted `vkCreateInstance == 0` and discovered **NVIDIA GeForce RTX 4080 SUPER**.
This fixes the missing-graphics prerequisite, not native simulation acceptance.
The bounded compatibility application exited zero after 11.33 s, but quit before
full task validation and logged OmniHub warnings; it is not a successful task
baseline. The candidate Dockerfile applies upstream's Carb environment shim and
disables Hub detect-only mode before the full native checks.

Candidate tooling lives in `scripts/isaaclab/{Dockerfile,constraints.txt}`;
source and dependency resolution/import validation remain in progress. The
minimal source-only build context contains no environments, credentials, tests,
checkpoints or experiment data. Image/container logs and manifests remain under
`/tmp/opencode/nexuml-isaaclab-preflight-20261018/`.

Application changes reuse the existing compiler, policy, direct collector, PPO,
Lightning and inference CLI. CPU evidence so far:

- Three portable definitions/scenarios, registry/YAML/JSON/config restoration,
  inference shape/bounds and blocked-Isaac-import checks: **46 passed**; combined
  with existing interaction/policy tests: **63 passed**.
- Training/collector preflight plus existing reinforcement/evaluation regressions:
  **117 passed, 14 deselected**. Unsupported topology/devices/frame widths and
  competing in-fit evaluation fail before simulator allocation. Existing
  Gymnasium schedules/resume still execute.
- Canonical runtime synthetic reset/buffer/lifecycle checks plus installed public
  wrapper probes: **131 passed**. Exact two-step GAE targets distinguish final
  from reset states, including all-done and simultaneous termination/truncation.
  Cancellation remains primary even when both native and launcher cleanup fail.
- Replay rejects same-shape control-period/rate/mapping/joint-order/revision
  changes rather than validating dimensions alone.
- The existing replay CLI adds explicit `--device` placement for GPU replay;
  without it, CPU-portable policies remain on CPU regardless of original GPU
  training settings. Zero episodes allocate no device. Earlier CLI proof had
  **43 passed**; review expanded the placement coverage and it is being rerun.
  All three documented simulator-free `nexuml resolve` commands execute as written.
- Correctly scoped `VIRTUAL_ENV="$PWD/.venv" .venv/bin/ty check` is back to the
  **37 pre-existing diagnostics**, with no new diagnostics from this slice.

The public wrapper probe can recover final states with an info reader, but the
default all-done reset calls native reset twice. Source also shows native mode
invalidates terminal observations with NaNs. The CPU fake cannot enter the real
Isaac installation/class gate; native confirmation remains open under 3.1. The
candidate therefore exercises D3's predetermined existing-adapter/runtime-cache
fallback, without a new collector or private TorchRL patch.

NexuML learning/resume/replay/media acceptance remains open. SO-101 task and RGB
replay construction are not yet implemented and fail rather than fabricating a
runtime. Native baseline results superseding the earlier pending status follow.

### Native GPU baselines now executed

The candidate image built successfully in **977.77 s**, image
`sha256:50a1a1d34e4e26ba904331560300d9921465ce5f4fbc45d9e05bba55bb6d469d`,
leaving **52.02 GiB** free. The bounded worker is now
`nexuml-isaaclab-phase2-runtime-20261018`, with the same GPU/CPU/RAM limits and no
host mounts. Actual import origins and a successful CUDA tensor/sum assertion are
in `profile-imports.{json,log}`. Resolved profile: Torch 2.11.0+cu128,
torchvision 0.26.0+cu128, torchaudio 2.11.0+cu128, TensorDict 0.14.2,
TorchRL 0.14.0, Lightning 2.6.5, pytorch-lightning 2.6.6, Gymnasium 1.3.0,
NumPy 2.5.4, Pillow 12.3.0, Warp 1.16.0, USD-exchange 2.3.0, RSL-RL 5.4.1,
Pydantic 2.13.4 and MLflow 3.17.1. The ordinary workspace lock/environment is
unchanged. Vendor-bundle `pip check` reports four missing optional distribution
dependencies; do not hide that result or label the entire profile dependency-clean.

An environment-specific Docker CLI `exec` attach defect returns zero after about
0.6 s while the daemon process continues and drops subsequent output. This caused
misleading partial import output, **not a Torch import failure**. All later native
results wait on daemon exec state and retrieve the completed log. A two-second
shell self-check proves complete output and exact nonzero exit-code propagation
(`daemon-exec-exit-code-self-check.{json,log}`). The task-local workaround stays
under the proof directory, not in NexuML's runtime or training architecture.

| Actual native check | Geometry / result | Daemon-confirmed wall time |
| --- | --- | --- |
| RSL-RL Cartpole | N=8, T=16, 2 iterations, **256 GPU frames**, finite losses | 14.58 s |
| RSL-RL Franka Reach | N=8, T=16, 2 iterations, **256 GPU frames**, finite losses | 9.10 s |
| Cartpole state/control/reset | N=8, 8 steps, **64 GPU frames**, actual 100 N scaling, 60 Hz, final/reset separation and partial timeouts | 16.60 s |
| Franka state/control/reset | N=8, 8 steps, **64 GPU frames**, actual ordered 7-joint offset/0.5 rad mapping, authored limits, 30 Hz and final/reset separation | 7.58 s |

Commands and full output are in the respective
`native-cartpole-training-no-visualizer`, `native-franka-training`,
`native-cartpole-state-control` and `native-franka-state-control` JSON/log files.
Training uses the pinned upstream `train.py --rl_library rsl_rl`, explicit
`physics=isaacsim_physx`, `--visualizer none`, seed 0, one learning epoch/two
minibatches, action clipping 1.0 and observation corruption disabled. The EA CLI
does not accept the legacy `--headless` flag; the initial failed command remains
recorded. These are **native**, not NexuML PPO results. Bounded Gaussian sampling
in NexuML differs from the native learner's clipped Gaussian; physical mappings
are matched, not learner curves.

`scripts/isaaclab/native_probe.py` reproduces raw state/control checks and emits
actual asset/joint/limit/default-pose provenance. Its four-step timeout is a
controlled boundary probe, not the references' normal episode horizon. Both
native runs clear the owned SimulationContext. Headless display/MaterialX/cache
warnings are retained, not suppressed.

SO-101's maintained PhysX asset also instantiated and stepped. The inspected
five ordered arm joints plus gripper, actual limits/default pose/USD gains are in
`native-so101-asset-object-config.log`; this is **asset evidence only**, not a
reward-bearing reaching task or PPO acceptance. The probe found that a gripper
with no explicit target closes, so the required task must set and hold its fixed
position target explicitly.

The first NexuML attempt exposed another lifecycle issue: upstream fast shutdown
can exit the Python interpreter during collector cleanup, returning zero before
NexuML finalizes checkpoints/export or reports an error. No such zero exit is
counted as acceptance. The owned launcher now requests public
`fast_shutdown=False`; a complete train/export success sentinel is required.

### Current blocker: returning physics shutdown

With public `fast_shutdown=False`, actual NexuML Cartpole completes **256 frames**
in **two [8, 16] CUDA rollouts** through `NexuSession`, Lightning, the existing
generic adapter, TorchRL collector and existing PPO. The before/after actor vector
changes; all logged losses are finite; policy and complete checkpoint files are
written. The train/export sentinel is reached, **then Python segfaults at process
exit**. Daemon-confirmed exit is nonzero. The report therefore does not mark
tasks 2.4/5.2 or a full Cartpole round trip complete.

Evidence: `nexuml-cartpole-returning-shutdown.{json,log}` (28.63 s) and
`nexuml-cartpole-released-native-shutdown.{json,log}` (26.09 s). Releasing owned
native/cache references and collecting Python cycles did not fix the crash; the
speculative GC workaround was removed. Ordinary ownership reference release stays.

The native state/control probe with returning shutdown reproduces the same crash
without a NexuML learner/collector: `native-cartpole-returning-shutdown.{json,log}`
(15.61 s), after both the state/control and returned-close sentinels. A **bare
AppLauncher with no physics environment exits cleanly** under the same setting:
`bare-isaac-app-returning-shutdown.{json,log}` (6.08 s). This narrows the blocker to
the prepared physics/runtime teardown interaction, not a claim that every Kit
launch or Torch import fails. No private native patch, forced zero exit,
`os._exit`, alternate backend, replacement trainer or version substitution was
introduced to hide it.

The generated `/opt/nexuml-isaac-resolved.txt`, actual Cartpole policy/checkpoint
and tar SHA-256 records were copied into the proof directory before stopping the
worker. Host CPU inference on the actual trained policy succeeds with Isaac,
TorchRL and encoder imports blocked (`trained-cartpole-portable-inference.json`);
its package has no explicit `training_frames`, so replay must retain existing
step-zero/unavailable-source-frame semantics, not infer frames from optimizer steps.

The required SO-101 minimal task configuration and 10 CPU numerical/reset/hold
tests are implemented, using the inspected maintained asset/gains, five absolute
position controls and an explicit fixed gripper target. **Runtime wiring and
native reaching/reward/training/resume/replay/media proofs are still pending.**
No SO-101 task checkbox was satisfied from asset inspection or CPU tests alone.

Next native work is to resolve the exact pinned PhysX returning-shutdown issue
before calling the profile compatible or proceeding to reference acceptance.
Then confirm the real wrapper gate/fallback, run Franka through NexuML, wire and
exercise SO-101, and finish resume/replay/media/packaging/CI. This is an in-scope
compatibility blocker, not the unrelated DALI/Optuna/type baseline.

Both exact-label task-owned containers were stopped after evidence capture,
including their task-local shared Hub child. No unrelated container/cache was
removed. GPU management returned to **37 MiB used, 15,937 MiB free, 0%**; observed
filesystem free space was **49.68 GiB**, still inside the approved setup budget.
No commit, push, merge, archive, Ray deployment or roadmap completion was performed.

### Final checks for this partial implementation

Exact commands/exit codes and full logs are in `final-results.json` and
`final-*.log` in the proof directory. The CPU command uses the same exclusions as
the baseline plus coverage; no Isaac or remote workload runs in that suite.

- **814 passed, 4 failed, 12 skipped, 31 deselected**; coverage **61.37%**
  (required 50%). Failures remain the same missing DALI/wds2idx and Optuna cases.
- Ruff check and format pass (232 Python files); offline lock check passes with
  the unchanged 263-package lock; whitespace check passes.
- Strict new-change OpenSpec and strict documentation build pass.
- Correct-environment full type check remains at **37 baseline diagnostics**,
  with no newly introduced diagnostics.
- Independent targeted review confirmed both corrections: canonical `cuda`
  alias normalization and explicit-only replay placement preserving CPU replay
  from packages carrying GPU training settings.
- SO-101 numerical/reset/hold tests: **10 passed**, simulator-free; three static
  policy-package cases load and infer with Isaac/TorchRL/encoders blocked.

Wheel/sdist/reinstalled-distribution checks, remote CI, native wrapper class-gate
confirmation, Franka NexuML training, SO-101 native task, checkpoint-resume,
fresh-process simulator replay and Isaac media/playback remain unrun. Earlier
Phase 0–1 proofs do not satisfy these new Isaac gates. Original ROS/Gazebo/CARLA
and later roadmap phases remain untouched. Progress stays **9/42**, not complete.

## Shutdown diagnosis and approved non-graceful continuation

### Exact cause and Internet findings

New evidence: `/tmp/opencode/nexuml-isaaclab-shutdown-20261019/`.
GDB was staged under the worker's `/tmp` by extracting packages, without changing
its installed system/Python dependencies or native binaries. The backtrace in
`native-cartpole-gdb.log` is `omni.ext._extensions` → `carb._carb` →
`_PyModule_ClearDict` → `Py_FinalizeEx`, after physics and app close return.
Module-finalization tracing identifies `isaaclab_physx`'s retained
`_SIMULATION_MANAGER_ENABLE_HOOK`. Releasing it in a diagnostic probe removes
the first fault but exposes destruction of `PhysxManager._timeline` after Kit
shutdown. The exact EA class acquires that interface at class definition and
never releases it in `close()`.

The minimal CPU-physics probe releases all caller environment locals and collects
cycles before Kit shutdown. It still fails unchanged and with only the hook
released. Releasing **both** handles before app close produces ordinary exit zero
(`native-cpu-scoped-release-both.log`). These assignments remain diagnostic-only;
**no upstream source/binary or NexuML runtime monkey patch was applied**.

Related reports were checked, not assumed identical:

- [IsaacLab #3730](https://github.com/isaac-sim/IsaacLab/issues/3730): headless
  Sim 5.0 close crashes, not a verified fix for this EA profile.
- [Sim 6.0.1 known issues](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/known_issues.html):
  an older PhysX shutdown crash occurs earlier in extension teardown, not this
  module-finalization stack. It is not grounds to ignore this failure.
- [Sim 6.1 known issues](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/known_issues.html):
  the old example is no longer listed; no exact two-handle Lab defect/fix found.
- [Kit extension hook API](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/110.1.2/omni.ext/omni.ext.ExtensionManager.html):
  subscriptions return retained native holders. Exact local source and traces,
  not generic online segfault descriptions, establish the cause.

### User-approved boundary, upstream unchanged

The user chose **keep upstream untouched and continue with non-graceful shutdown**.
The provider now lazily constructs one public `AppLauncher` whose app belongs to
a whole-job scope. Its public `launch_simulation` context reuses that owned app,
so environment/collector close does not terminate Python. Existing session/PPO,
checkpoint/export and explicit logger cleanup finish first. The outer boundary
then invokes public `SimulationApp.close(exit_code=...)` with `fast_shutdown=True`.
No worker process, IPC or replacement learner was added.

`scripts/isaaclab/run.py` executes an ordinary script or `-m` module in that scope.
`run.sh` sources official environment setup and `exec`s the bundled Python.
Vendor `python.sh` was observed to convert `SystemExit(7)` into shell exit 1; the
owned entrypoint preserves the actual status without changing that vendor file.
The Dockerfile selects the new entrypoint. Proofs used synchronized current source
in the existing worker; the image itself has **not been rebuilt**. Runtime
allocation outside this scope or into a caller-owned app is rejected. Jobs that
never build a runtime allocate no app. Python `atexit` callbacks/interpreter
destructors are **not** a promised cleanup mechanism under fast exit.

`native-owned-fast-status-results.json` records four real CPU-physics processes:
normal **0**, explicit `SystemExit(7)` **7**, injected failure **1**, interrupt **130**.
Each environment close returns to the job and its `finally` runs before app close.
Unit tests additionally cover cleanup-plus-primary failure, nesting, lazy allocation,
arguments/import paths and rejection before allocation outside the process boundary.

### Actual GPU training and fresh-process scalar replay

Smokes ran serially in the existing 4-CPU/16-GiB-RAM/2-GiB-shm task-only worker at
lowered CPU priority. Another GPU job was active (about 736 MiB), so timings are
not exclusive-device benchmarks. No unrelated process/container or driver changed.

| Proof log | Frames / rollouts | Actual result |
| --- | --- | --- |
| `nexuml-cartpole-owned-fast.log` | 256 / two CUDA `[8,16]` batches | Actor changes, finite metrics, TensorBoard logger finalizes `success`, package/checkpoint written, daemon exit **0**, 11.60 s |
| `nexuml-franka-owned-fast.log` | 256 / two CUDA `[8,16]` batches | Named seven-joint action validation, actor changes, finite metrics, logger finalizes `success`, package/checkpoint written, daemon exit **0**, 12.08 s |
| `nexuml-cartpole-scalar-replay.log` | One bounded 8-step episode, N=1 | Existing CLI, source-frame step 0, return 0.05397, truncation 1, exit **0**, 9.59 s |
| `nexuml-franka-scalar-replay.log` | One bounded 8-step episode, N=1 | Existing CLI, source-frame step 0, return -0.06496, truncation 1, exit **0**, 10.09 s |

Exact command arrays/daemon completions are in the matching JSON files. The retained
`nexuml_train_probe.py` uses ordinary registered factories and `NexuSession` and is
copied to `/tmp/nexuml_shutdown_train_probe.py` in the worker. Training commands:

```bash
nice -n 10 bash /workspace/nexuml/scripts/isaaclab/run.sh \
  /tmp/nexuml_shutdown_train_probe.py cartpole
nice -n 10 bash /workspace/nexuml/scripts/isaaclab/run.sh \
  /tmp/nexuml_shutdown_train_probe.py franka
```

Artifacts live under `/workspace/output/shutdown-20261019/` in the owned worker.
These proofs satisfy bounded training tasks 5.2–5.3, not convergence or full
resume/media/SO-101 acceptance. Native wrapper class-gate confirmation, SO-101
runtime wiring, full resume/media, packaging and opt-in CI remain open. Subsequent
work no longer requires repairing the upstream graceful-close defect.

### Retained artifacts, checks and stopped resources

Both actual trained packages/checkpoints and replay artifacts were copied to
`artifacts/shutdown-20261019/` in the new proof directory, with hashes in
`artifacts-manifest.json`. `trained-portable-inference.json` verifies deterministic,
bounded CPU calls for both trained policies with Isaac/Omniverse/TorchRL/encoder
imports blocked. `decoded-tensorboard-events.json` decodes both finalized training
event files and both scalar replay files; replay provenance remains step zero
because the package has no explicit `training_frames` metadata.

Fresh checks (`checks.json`, `corrected-checks.json`, `final-checks.json`):

- Focused CPU: **429 passed, 14 deselected**. New process-boundary tests: **13 passed**.
- Broader CPU: **827 passed, 4 pre-existing failures, 12 skipped, 31 deselected**.
  The four failures remain missing DALI/wds2idx and Optuna; no unrelated repair.
- Repository Ruff/format, offline unchanged lock, shell syntax, strict docs,
  strict amended OpenSpec and whitespace pass. The initial new long OpenSpec
  paragraph was split into its own requirement, not suppressed.
- Correct-environment full types retain **37 baseline diagnostics**. An initial
  check mistakenly used inherited `/env` and a subsequent new mixed-list test
  diagnostic was fixed; neither is represented as a passing type gate. Final
  type checks explicitly set worktree `VIRTUAL_ENV`/`UV_PROJECT_ENVIRONMENT`.
- `upstream-integrity.log`: exact EA revision, clean `git status`, clean diff.
  Native source/binaries/version pins and the current DevContainer remain untouched.

`stopped-resources.json` confirms both exact-label owned containers are stopped,
including the task-local Hub child; images, caches and worker sources remain.
Only idle owned processes existed when stopping. The unrelated GPU allocation
remained about **781 MiB used / 15,193 MiB free**; observed disk free was **214.47 GiB**.
Do not delete unrelated data or interpret shared-device utilization as task ownership.
No commit, push, merge, archive, remote CI or later-roadmap completion occurred.

## Sources verified for this preflight

### Continued apply: real wrapper decision (2026-10-19)

`/tmp/opencode/nexuml-isaaclab-integration-20261019/wrapper-results.json` records
four successful diagnostic processes on an actual 2-environment GPU Cartpole,
with the installed `IsaacLabWrapper` native class gate active. Each tested no-done,
partial-timeout and all-timeout rows, with/without `native_autoreset` and a public
final-observation info reader. Default composition produces NaN terminal states.
The reader repairs default partial resets, but all-done still calls public native
reset again and replaces the already reset initial state. Native-autoreset mode
avoids the extra reset but retains NaN terminal states even with the reader.
CPU tests already cover the wider termination/timeout/mutable-buffer/GAE table.

Task 3.1 is verified: select D3's predetermined **existing generic adapter over the
canonical Isaac runtime**. No TorchRL internals/availability flags were patched,
and no new collector bridge abstraction is needed. Exact source/probe/commands and
all four logs are retained beside that JSON. Upstream remains unmodified.

- [Frozen EA release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
  and [tag ref](https://api.github.com/repos/isaac-sim/IsaacLab/git/ref/tags/v3.0.0-EA).
- [Pinned dependencies](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/pyproject.toml),
  [base container](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docker/Dockerfile.base)
  and [compose file](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docker/docker-compose.yaml).
- [Pinned Cartpole](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py),
  [Franka](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_tasks/isaaclab_tasks/core/reach/config/franka/franka_reach_env_cfg.py),
  [SO-101 asset](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_assets/isaaclab_assets/robots/so101.py),
  [stack base](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/stack_env_cfg.py),
  [RL step lifecycle](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)
  and [public reset](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab/isaaclab/envs/manager_based_env.py).
- [Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html):
  supported Ubuntu 22.04/24.04, 32 GB RAM / 16 GB VRAM / 50 GB SSD minimum;
  Isaac Lab training needs additional resources. Driver compatibility is still untested.
- [Container setup](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html),
  [license index](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/common/legal.html)
  and [Omniverse license linked by the container guide](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/common/NVIDIA_Omniverse_License_Agreement.html).

Context7 lookup returned its monthly quota limit; official pinned source/docs and
installed TorchRL source were used instead. No simulator execution is implied.

## Continued apply: SO-101 and checkpoint round trips (2026-10-19)

Evidence root: `/tmp/opencode/nexuml-isaaclab-integration-20261019/`.
The existing bounded owned worker was resumed; no DevContainer/driver/ordinary
environment changes. GPU was idle at entry, with 15,937 MiB free; disk 214.02 GiB.

- The real native wrapper/fallback decision above completes 3.1. Retaining the
  existing generic adapter (not adding a replacement bridge) plus actual GPU
  Cartpole/Franka/SO-101 and CPU factory regressions completes 3.2.
- `_isaaclab_runtime.py` now constructs the local SO-101 manager config through
  public APIs. Its action descriptor, exact asset URL/PhysX variant, six joint
  identities, authored limits/gains/effort/velocity limits and configured initial
  pose are checked before collection. Five absolute targets are `0.5 * action`
  radians in `[-0.5,0.5]`; the separate gripper stays at 0.2555162024919699 rad.
  CPU perturbation tests reject changed asset/control profiles.
- `so101-native-control-outcomes.log` exits zero after **240 scripted GPU frames**
  (N=8): held-home success exactly after eight steps, deliberate timeout with
  success false, and both non-home goals reached by all rows in 11/10 steps.
  Reward equals negative normalized MSE times control dt, separate mean absolute
  joint error is in radians, and the gripper stays within 0.002 rad of its target.
  Reset observations differ from terminal observations; cached success/error
  still refer to the ending goal. This completes 6.1–6.3, not learned manipulation.
- The reward/termination terms retain pre-reset error/success; runtime clones
  those into `InteractionStep.info`. The shared evaluator validates paired scalar
  tensor outcomes and emits `eval/rl/task_success_rate` and
  `eval/rl/final_joint_error_rad_mean` only when supplied, with per-episode reset.
  Invalid/nonfinite/missing/inconsistent fields close the runtime and fail.
- `so101-train-256.log`: two `[8,16]` CUDA rollouts, **256 real frames**, changed
  actor parameters, finite PPO losses, bounded actions and agreeing action log
  probabilities, native logger finalized success, checkpoint/package, exit zero.
  This completes 6.4. Scope remains state-policy simulation, not hardware/IK/IL/VLA.
- `all-resume-results.json` records **nine fresh processes** in reference order.
  Each reference saves at frame128 and intentionally raises a callback RuntimeError
  (exit1, failed logger finalized, collector closed), resumes only one remaining
  `[8,16]` rollout to frame256 (exit0), then restores the complete budget without
  any app or collector (exit0). Actor/critic/all optimizer tensors and counters
  match the intermediate checkpoint at train-start. Collection follows prior
  learner updates; rollout-policy weights and sampled log probabilities agree.
  No physics/random-trajectory continuation is claimed. Native ownership proofs
  plus the existing fake cleanup/cancellation tests complete 2.2 and 2.4.
- The actual resumed policies/checkpoints, synthetic observations and pre-export
  CPU action oracles are retained under `artifacts/{cartpole,franka,so101}/resume/`.
  `resumed-portable-inference.json` verifies **exact pre-export CPU equality** for
  all three, bounded finite controls and inference-only metadata with simulator,
  TorchRL and encoding imports blocked. Existing optional-discovery PIL warnings
  were visible, not treated as mandatory inference dependencies or repaired here.

CPU checks so far: 250 focused passes/14 deselections for SO-101/runtime/evaluator/
definition/reinforcement; 195 passes/14 deselections after RGB-specific tests were
added. These overlap, not an additive test count. Full final gates remain pending.

Replay-only camera/terminal-frame code and CPU tests are now present, but native
RGB validation and real recorded replay are still in progress; tasks 7.x remain
unchecked. Resume acceptance tasks 5.4/6.5 still await replay of the resumed exports.
The profile still reports missing `usd-validation-nvidia`, PyJWT and aiobotocore
(four dependent-distribution warnings). A constrained dry run proposes only three
pure-Python additions without altering installed distributions; nothing installed
from that proposal yet. Image rebuild, opt-in CI and final package gates remain.
