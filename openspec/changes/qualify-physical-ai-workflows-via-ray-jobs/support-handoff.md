# Physical AI qualification support handoff

## Status and authority

Verification and support handoff, not a release approval. [tasks.md](tasks.md)
remains the authority: **32/38 complete**. Required blocked rows are not waived. No publication,
PR/Linear status change, merge or archive is authorized by this handoff.

### 2026-11-24 continuation

- PR #15 published `2b482c5a…` against `main`; [required CI](https://github.com/NexuFed/NexuML/actions/runs/45181507210)
  failed on Python 3.12, 3.13 and 3.14, so published readiness is not established.
  Checkout `e2751826…` plus working-tree fixes is a different candidate.
- Reproduced CI failures with TensorDict 0.15.6, TorchRL 0.15.0 and Typer 0.29.2.
  Focused patched checks: 71 passed, one skipped. The candidate feeds Lightning
  the actual batch size, checks exact Isaac wrapper action values instead of
  tensor object identity, and reads help metadata from the current Typer tree.
  This is not a rerun of published CI.
- Campaign and guard reject overlapping roots; negative tests cannot hand real
  process cleanup a fake PID. Acceptance and transport regressions: 334 tests
  and 150 subtests passed. CLI/reporting checks: 77 passed, including the new
  explicit viewer replay command for an actual export path. These are local
  checks, not live port/display or Ray cancellation acceptance.
- Setup-enabled acceptance rejects shared venv paths before submitting. Verified
  with 198 SDK, profile and inert-campaign tests. Generic SDK submissions and
  the explicit prepared-environment opt-out are unchanged.
- Submitted one `pool_gpu_workers` preflight, job `e68d8292af604e24a1bba7f079084cb6`,
  on source `864ee52f…` (tracked checkout plus the reviewed guard and CI fixes).
  It requested one whole GPU and ended `FAILED` with
  `JOB_SUPERVISOR_ACTOR_START_TIMEOUT` after about 15 minutes: no worker
  allocation, setup, workload, or worker receipt. Kubernetes reported the
  GPU worker pod `Unschedulable` (`0/27`: 15 insufficient GPU, 8 affinity,
  3 unreachable, 1 unschedulable; preemption disabled). Remote preflight and
  parallel qualification are blocked on eligible capacity at this attempt.
  The source snapshot predates the replay message and setup-venv validation
  changes. No retry or infrastructure change was made.
- A later live inventory found eight unreserved matching GPUs. Fresh serial
  job `63cdb093d22f47b3aec103bd51022837` passed on an RTX 4070 with one visible
  GPU, a private interpreter, and a 4 GiB shared-memory object-store cap.
  Both artifacts and both receipts were retrieved and verified after execution.
  This does not prove two simultaneous native workloads or post-removal readback.
- Installed distribution job `bd7e4c6f03564220ab97479e8d3bf3ce` passed the exact
  wheel, import-origin, supervised, and actor-only checks. The 17 listed artifacts
  and both receipts were verified. The approved-data S1 and L2 gates remain open.
- Ordinary public train job `raysubmit_WAzJxtCBRrkfshwg` reached worker config
  logging and failed before fit: ruamel's round-trip `ScalarInt` in checkpoint
  callback options could not be represented by PyYAML. The shared resolved YAML
  reader now produces plain values; focused config/Ray/RL regression passed
  (421 tests). The original failed job is retained. Corrected public CLI train
  `raysubmit_LYmLXMYbSWH9VCWM`, checkpoint export `raysubmit_nXF6acw9HqR7BWrk`,
  and fresh evaluation `raysubmit_KwATciVWZ4KQ938t` all succeeded on `28f1acad…`
  with clean guard exits. Submission reported setup/learning as unknown; actual
  completion showed 64 physical frames, 80 optimizer steps, observed CPU fit
  device and the produced checkpoint/policy commands. The exported policy matched
  the retrieved checkpoint hash and evaluated two episodes at frame 64. Together
  with captured finished-viewer replay and port checks, this closes 4.1, not live V5.
- Worker guard now rejects absent/ephemeral durable mounts before setup. Explicit
  missing private data blocks preflight, rather than falling back to local data.
  Training banners avoid collisions between configured spectator and tracking UI
  ports. Current focused harness/control suite: 416 tests and 154 subtests passed.
- The two native jobs `74d2ad8c…` and `f9411fcd…` shared a roughly 4m39s
  guarded setup interval on distinct one-GPU pods (`student-worker-6` and
  `cip-gpu-8`), with separate venvs and attempt roots. Both stopped before the
  simulator started: the pinned Ubuntu archive returned HTTP 403 on the worker.
  Their original guards show child/exit 1 and no cleanup errors. The same pinned
  URLs returned HTTP 200 to an independent submitter HEAD request; this does not
  establish their availability to workers or waive the original failures.
  Task 3.5 remains open: setup overlap is not native workload overlap.
- Recovery `73ff2632…` succeeded after installed-controls producer pod `dtll8`
  disappeared. Reader `jmmjl` verified all 17 artifacts and both producer receipt
  hashes. Logging `78ff2a8d…` then passed all four tracking modes against per-run
  SQLite/S3 and matched the selected payloads. All 293 logger files including
  receipts (23,358,737 artifact bytes) were rehashed. This is not live browser
  playback, failure-state UI or the complete recovery matrix.
- Full worker regression `3c8a7877…` on `0199e3e6…` produced 3,419 passes and
  10 failed hermetic dry-run tests that assumed `.git` existed in the upload.
  The original failed stage and guard are retained. The affected tests now mock
  only submitter Git metadata; all 72 campaign tests pass against an actual
  `.git`-free bundle without weakening the submitter's revision check.
- The corrected `c2bd1647…` source passed the remote ordinary suites on Python
  3.12 (`4f78b100…`, 3,429 passed), 3.13 (`5d714f8e…`, 3,433 passed), and 3.14
  (`f7059214…`, 3,433 passed). All three guard receipts were clean and terminal
  pytest logs were hash-verified. Native and other opt-in skips are not substitute
  proof. The local Starlight check also passed; published CI is separate.
- The native setup fixture now accepts an explicitly provided cache of the same
  pinned archives, validates size and hash and fails without network fallback
  for an incomplete cache. All 13 unchanged public archives were independently
  verified before staging. Serial native job `3c6a090e…` then passed all six
  Cartpole CLI stages, including partial and exhausted resume, on an RTX 4070
  with one visible GPU. It produced 1,024 physical frames, 160 optimizer steps,
  and a changed finite actor. All 58 files were verified; this is a native
  lifecycle smoke, not a live-view or learning threshold. Two later native jobs
  `36172d03…` and `f87905e0…` both passed on distinct GPU UUIDs and private
  environments, but the second training window began 160 seconds after the
  first resumed training finished. This is not workload overlap. The fixture
  now supports a bounded, opt-in training start after both jobs finish setup.
  With that start, `199e5eb1…` and `923eb9a1…` both passed on `12d78852…`.
  Their partial-train stage processes overlapped 17.01 seconds on different
  GPU UUIDs, private interpreter paths, and attempt roots. Each logged 1,024
  physical frames and 160 optimizer steps. These cases do not start viewer
  ports or use file-based MLflow; those isolation checks are still separate.
  A separate approved browser job verified all 59 files from `199e5eb1…`
  after its producer pod disappeared. That browser job reused the second
  producer's worker, so the readback refused to use it as evidence of
  post-removal recovery for `923eb9a1…`.
  Public CLI jobs `raysubmit_nzJbSYcbk7si8TpX` and `raysubmit_1iUZGBWkPyBmPpzS`
  then showed simultaneous private Newton viewers on separate GPU workers.
  Each listener served its own loopback port (8081 or 8082); neither worker
  could connect to the other's viewer. Both jobs reached `SUCCEEDED` with
  clean guards and policies at 262,144 frames. Their databases had separate
  `FINISHED` runs and S3 artifact prefixes. The first worker retrieved the
  second job's database, guard, and policy metadata after its producer pod
  disappeared. The original controller's Ray log retrieval failure is
  retained; no worker log was reconstructed. This closes 3.5, not ROS/Gazebo
  namespace isolation or the complete recovery matrix.
- Native, browser setup and codec regressions include the exact 4 GiB floor
  stopping only the owned process group. A total of 152 tests and 122 subtests
  passed. All 31 pinned browser archives were verified from public sources.
  Submitting the full set in the upload got HTTP 413 before creating a Ray
  job; native lookup confirmed no job for that ID. The next browser attempt
  stages only pinned Ubuntu packages and leaves larger binaries on their
  existing verified HTTPS routes. The smaller upload launched successfully.
  Browser `76fe4caf…` proved actual TensorBoard preview/scalar playback and
  MLflow Overview, but MLflow artifact listing failed with HTTP 500:
  the composed profile set `AWS_CA_BUNDLE` without creating its private CA file.
  The independently prepared failure run `344fcce9…` verified the native FAILED
  UI and retained animated GIF; its intentional CLI/job failure remains failed.
  With the CA setup restored, `d9905583…` (both) and `4db08018…` (separate
  TensorBoard-only and MLflow-only) passed. Both verified selected frame-64
  GIF/MP4/JSON identities, animated GIF samples, and MP4 reaching playback end
  through the download route. All 196 browser files were verified. Together
  with the four-mode producer and failed/disabled evidence, this closes 6.2,
  not the independent native spectator or external Ray cancellation gates.
- Current Isaac spectator job `69bc7ceb…` passed all eight native selectors
  without skips, on source `adca367f…`. The 164 retained files (296,987,093
  artifact bytes) were verified. Visual spot-review showed Newton Cartpole
  geometry and markers in both training and replay after reconnect, using
  browser-side software WebGL. Top-level job `raysubmit_8jdFVPUV1YuRD9dA`
  returned HTTP 200 through its private forward before external Ray stop.
  Native status was `STOPPED`; worker guard 143/cancelled, child -9, no cleanup
  errors, and same-worker port 8081 could be rebound. This closes 6.3 without a
  claim of graceful child shutdown. The preceding WebSocket-interrupted run
  `raysubmit_iZVuqedyafQ7XGQj` stopped before viewer readiness and remains separate.
- Independent read-only review found unsafe exception rendering, potentially
  stale numeric process-group signals in test cleanup, and a nested-root
  claim race. The CLI now summarizes control-plane failures without echoing
  SDK messages; tests no longer signal recorded numeric groups; guarded
  production attempts are disjoint named leaves by construction. These
  fixes passed the independent follow-up and 131 focused tests (96 subtests).
  Worker preflight and failure-preserving guard behavior close 3.3: real
  guarded success/setup failure, admitted native/browser/codec profiles, and
  hermetic missing-prerequisite, deadline and 4 GiB floor checks. Missing
  native desktop/data deployments remain blocked. This is not the final 8.2
  recovery matrix or a claim that all profiles have been deployed.
- Section 9.1 still needs an identified project-authored scene and asset set
  loaded on a prepared Isaac worker. The deployed editor pod's Isaac container
  is not ready (exit 139); its NetBird sidecar being up is not editor readiness.
  Section 9.2 is blocked on the shared
  evaluator/Arena integration in [the owning change](../add-isaac-physical-ai-backend/tasks.md#11-common-release-evaluation-and-arena).

### PR verification and support matrix

| Surface | Verified | Blocked or out of scope |
| --- | --- | --- |
| G1/G2, I1/I2, A1 and M1 CLI | Bounded lifecycle runs on their recorded source snapshots; current Isaac sensor and two-worker isolation checks are identified below and above. | Not convergence or universal task/backend support. |
| TensorBoard, MLflow and Isaac Viser | Browser playback, download and scalar parity; independent live spectator train/replay and external stop; measured server/recording overhead. | Viser browser software graphics are not native RTX evidence. |
| M1 Reacher desktop, R1 Gazebo, C1 CARLA | Reacher EGL recording; Gazebo headless lifecycle, same-worker isolation and private live CLI viewing on the recorded Gazebo Harmonic/Mesa profile. | Reacher private display (6.4). CARLA remains external and unqualified. |
| L1, S1/L2 and B1 distribution | Cartpole L1 qualified for one seed on its recorded snapshot; current installed supervised and actor controls pass. | SO-101 original episodes and matching runtime qualification (5.6, 7.2); public cloning smoke does not substitute. |
| Authored scene and Arena | Checkpoint weight selection and package/action parity verified separately with hermetic tests. | Project scene/assets and working editor (9.1); native Arena runtime, SO-101 mapping, private live viewing and scored MLflow retrieval (9.2). |
| Release checks and recovery | Local and worker static checks, docs/package checks, installed-wheel identity and representative Ray lifecycle recovery (8.2). | Final evidence gate (8.3); published PR CI is failed and the candidate was not pushed. |

The [verification matrix](verification-matrix.md) remains the required inventory.
NexuFL/federation and later model/robot work are not closed here.

### 2026-11-25 candidate checks

- Current worktree `e2751826…` plus fixes (application and fixture source
  `e73f2951…`): Ruff, full-tree typing with the locked `ty`, TensorDict,
  TorchRL and Kubernetes versions, lock, Isaac image lock, skills and shared
  release versions passed. Initial type diagnostics came from a stale local
  environment; the check was repeated in a separate verification environment.
  Current hermetic acceptance/transport/docs regressions passed: 772 tests
  and 597 subtests with CUDA and native tiers disabled.
- Starlight build, type and link checks passed with the pinned Node version;
  docs Python checks passed (67 tests, two skips, 33 subtests). Both distributions
  built from source, passed Twine metadata checks and rebuilt from the source
  archives. All three artifact and dependency-resolver checks passed. That
  does not replace remote installed-runtime checks or original-data SO-101.
- Sensor job `448e6b1a…` succeeded on source `e73f2951…`; all 13 fixed
  native oracles passed, including terminal-scene identity and raw/processed
  sensor checks. All 288 files (286 artifacts and two receipts) were retrieved
  and hash-verified. The 32×32 Newton ending-scene diagnostic intentionally
  puts the cart offscreen; its mostly background image is not renderer or
  viewer proof. The separate PhysX terminal-RGB test exercised the native
  ending scene versus the reset scene.
- Job `f784c02c…` ran two Newton spectator-off and two spectator-on CLI fits
  at 32,768 frames per fit, on the same RTX 4070. Off/on inclusive throughput
  was 1,390–1,675 / 1,386–1,644 frames/s; CLI-to-environment-ready time
  7.45–9.11 / 7.56–11.26 s; max child RSS 2.72–2.83 / 2.77–2.82 GiB.
  Sampled device VRAM peaked at 465 MiB in each fit (165 MiB baseline).
  Per-fit retained output grew by about 4.3 kB with the viewer. Both enabled
  fits served HTTP and released port 8081; every stage passed owned cleanup.
  This measures the server without a connected rendering browser. The
  independent browser-interaction test intentionally sleeps and is not a
  throughput probe.
- The same job replayed the same Reacher policy with recording off/on (EGL):
  4.12 / 4.93 s, 1.03 / 1.06 GiB max child RSS, and 165 / 177 MiB sampled peak
  device VRAM. Retained output grew from 1,577 to 89,925 bytes. The 16-frame
  clip cap did not change either full 50-step episode or their scalar outcomes.
  All 78 files were verified. This closes 6.6 with the sensor and recovery
  evidence above. The measurement-only fixture used source `ffd6cf36…`;
  no universal overhead, renderer or native desktop claim follows.
- Installed controls job `db1b241b…` succeeded on `e73f2951…`. Both installed
  wheel hashes match the locally built wheels; 20 files were verified.
  Original SO-101 S1/L2 data remains unavailable. Static job `46d0ac18…`
  passed Ruff and format, then failed to resolve `release_versions` in three
  scripts. The failed guard, receipt and all four artifacts are retained.
  The type-check configuration now explicitly includes `scripts`. Locked
  typing and all three package checks passed again locally; wheels built
  directly and from the updated source archives are byte-identical to the
  wheels installed in `db1b241b…`. Static rerun `c261f3af…` on `9fdb29d2…`
  passed lint, format, types, docs content and lock checks. On the same
  candidate, missing-data attempt `2b383e39…` blocked with guard/child exit 2,
  no cleanup errors and no workload stages. All ten files across these
  two attempts were verified.
- Registered recovery `ce779e17…` verified both producer receipts and all
  18 installed-package artifacts from `db1b241b…` after its pod disappeared.
  The reader and producer were distinct workers.
- Missing-codec job `8889c023…` on `9fdb29d2…` completed partial training,
  resume, checkpoint export, scalar evaluation and exhausted resume. Only
  recorded Reacher evaluation failed: the explicitly unavailable FFmpeg
  executable did not provide `libx264`. Native status stayed `FAILED`,
  stage/guard exit stayed 1 and cleanup reported no errors. All 63 files
  were retrieved and hash-verified, including the checkpoint and matching
  1,024-frame policy. This is successful recovery of a failed attempt, not
  a pass for that recorded evaluation.
- Setup-negative `9aabfe02…` retained child/guard exit 141, no cleanup
  errors and no workload receipt. Public `evaluate` job
  `raysubmit_QCmZwwUEK1kTUFtR` reached worker CLI and failed on the exact
  missing package path; it did not evaluate a replacement or report
  completion. Original terminal statuses and both negative records are
  retained.

### Integrated recovery evidence (8.2)

- Partial and exhausted checkpoint continuation, checkpoint-to-policy
  identity and fresh evaluation: current Reacher job `8889c023…` before its
  intentional recording failure; earlier native Cartpole and public CLI
  lifecycle runs retain their own revisions.
- Setup, absent data/package and codec: the current negative attempts above.
  Port conflict and partial startup: native spectator `69bc7ceb…`.
  Tracking delivery failure: `344fcce9…`. Actual external stop and same-worker
  port reuse: `raysubmit_8jdFVPUV1YuRD9dA`. Deadline and 4 GiB floor
  coverage is hermetic; no live disk was filled.
- Two actual whole-GPU allocations and isolated viewers/databases: the two
  successful public CLI jobs above. Persistence: representative native
  outputs after worker removal, recovered second-worker database and
  policy metadata, and the current registered package recovery.
- The final `9fdb29d2…` candidate changes only the type-check search path
  from `e73f2951…`; application, test and script sources and the lock match.
  Installed wheel bytes match after the config fix. The browser and
  spectator snapshots also match final application sources and the lock;
  their test/runtime receipts keep their own identities. This is the
  representative Ray recovery matrix, not qualification of unavailable
  ROS/desktop, private-data, authored-task or benchmark inputs.

### 2026-11-25 Arena-first apply

- The [Arena 0.4 lock](https://github.com/isaac-sim/IsaacLab-Arena/blob/a2f421e918952ee139a9f1a7ef8f9b23047a99df/uv.lock)
  pins Isaac Sim `6.2.0.0`. Its Isaac Lab submodule is the `v3.0.0` tag
  (`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`), two commits ahead of the
  workspace pin; the upstream diff only touches documentation and its tooling.
  The [maintained install](https://github.com/isaac-sim/IsaacLab-Arena/blob/a2f421e918952ee139a9f1a7ef8f9b23047a99df/pyproject.toml)
  requires cuRobo `78fd485fa82d9b9a063fb4985e371814587e666a` from Git. No
  matching public binary or accessible prepared Arena image was verified.
  The Arena CI image is in NGC staging and anonymous manifest access is denied.
  Do not omit this dependency, bypass its pin or downgrade the stack.
- The [community embodiment](https://github.com/art-e-fact/isaaclab-so101/blob/e027ac6d47151d9d33e567e70df7a87b732de8c1/src/isaaclab_so101/embodiments/so101.py)
  uses unit-scale absolute joint targets including the jaw, or a different IK
  control interface. Neither is the NexuML five-joint, 0.5 rad target-scale,
  fixed-gripper policy interface. Compose the existing compatible task and
  controls in Arena rather than rescaling or padding that policy to fit the
  community example; the new pairing still needs native proof.
- Verified-CA Ray Jobs API returned Ray `2.57.0`. The first pass did not
  submit after observing an idle pool; that was not a capacity check.
  `pool_gpu_workers` scales from zero. Submit a bounded one-GPU job and use
  actual scheduling and worker admission evidence, not idle worker counts.
  Preflight `7150771a9d954f779944e5158a34d1c2` then succeeded from idle:
  exactly one visible NVIDIA CUDA device, 4 GiB shared-memory object-store
  reservation, zero disk-backed plasma reservation, and a private writable
  durable root. The unprepared worker had no ROS or Gazebo binary and no
  display configured. This is capacity/admission proof, not Arena or GUI proof.
- `nexuml evaluate --trainer-checkpoint PATH` now selects the same evaluation
  path as a package, after restoring trusted policy weights and exporting to
  a unique temporary worker directory. Local hermetic checks prove exclusive
  weight selection, literal remote arguments without submitter loads, decoded
  action and payload parity, source checkpoint hash/frame provenance, no
  trainer or collector, and temporary cleanup on success and failure. Source
  `6defcdd121db0e0e9e7761ee96f0ed0e7429d2ce65450627fae7e53b06e4f864`
  passed 409 scoped tests (one CUDA skip), Ruff, locked ty and strict OpenSpec.
  Independent review of the checkpoint slice found no material defects.
  Public CartPole package/checkpoint controls ran on that source and the
  approved pool. Unseeded jobs `raysubmit_QNz4SgaajizD3se4` and
  `raysubmit_QuQJr7EaFs1eimKb` both completed but had different scores;
  unmatched starts invalidate a parity claim. Separate jobs
  `raysubmit_MN5UbjU3yY6YK2Px` and `raysubmit_etswee9PTg94F51q` matched
  all scalar metrics at source frame 64 with seeds 1729 and 1730 (two
  episodes per route, mean return 9.5). No retraining was submitted.
  This control does not establish native Arena, SO-101 or final 8.3 proof.
  Verified-CA records and original unmatched scores are retained under
  `arena-qualification-20261125-fpVUgQ` in the private preservation root.
  Offline checkpoint evaluation also uses the saved compile inputs rather
  than opening its training dataset or probing training batch sizes. A
  focused offline CLI regression blocks both calls and exercises replay.
  The wider regression exposed and reproduced a local API timeout race:
  the terminal notification could arrive before the owned operation slot
  was cleared. The slot now clears under the same lock before terminal
  publication, after confirmed process exit. Four focused API regressions
  passed, followed by 199 broader regressions (one CUDA skip); the original
  failure is not an evaluation or Ray failure.
- Ordinary policy evaluation with MLflow now publishes a versioned
  `evaluation-episodes.json`. It contains seeds, measured outcomes and
  termination reasons, and the last 128 samples from the first known failed
  episode. An unknown success signal stays unknown. The affected suite
  passed 560 tests plus 45 subtests (15 explicit simulator/CUDA skips);
  current combined follow-up passed 616 tests plus 103 subtests (same
  skips), and independent scoped review passed. This is not a
  retained Arena suite artifact or post-worker recovery proof.
- Fresh native SO-101 baseline job `1d2ed31e5eba4ca7b0a7e5f346f18fd5`
  succeeded through registered `ray-jobs-sdk` on one approved GPU.
  Source `8b031e3af3611f68e6558e0b1e08930368f8268f7141f187b1c3725b66e839ca`,
  1,024 training frames, private Isaac-only setup and hash-verified Ubuntu
  client libraries. A partial training stop preserved exit 143 and its
  checkpoint; resume reached 160 optimizer steps with changed finite actor
  weights. Checkpoint export and fresh replay succeeded; exhausted resume
  made no updates. Both seeded replay episodes reached the native horizon
  without success (mean final joint error 0.14234 rad). This is stock PhysX
  lifecycle evidence, not a learning success, the authored scene, or Arena.
  The receipt and logs are retained under the private preservation root above.
- The first stock SO-101 package replay with MLflow
  (`raysubmit_Dwy8v2mWhjpT5jr4`, source `18681f34…`) reached MLflow
  initialization but segfaulted inside Kit startup before an episode ran.
  Its planned checkpoint partner was not submitted. An initial attempt to
  test with telemetry off was rejected by the submitter: the controller
  parsed the `YES` EULA string as a boolean using a different YAML dialect.
  Preserve both failures and use the application's YAML dialect for the
  separate diagnostic. With telemetry disabled, `raysubmit_FRC39xmf78gScfUg`
  still failed in the same Kit startup call. Removing the telemetry thread
  alone is not a remedy. The runtime-before-tracking diagnostic
  `raysubmit_mc9WVtUTUdPwwiZD` (`7b353aea…`) failed in the same Kit startup
  call before any MLflow initialization; the experimental ordering change
  was removed. All three failing runs and the stock passing run used
  different Ray worker instances of the same advertised RTX 4070 class.
  The experimental rollback passed 447 focused regressions. A hash-verified
  same-worker comparison (`65d486d29d4140b5a30eb8560f9a82f8`) controlled for
  placement, package and seeds: normal module CLI exited zero with and
  without MLflow; embedded Python `-c` invoking Typer directly died with
  signal 11 before rollout, even without MLflow. This establishes a launch
  path difference, not the upstream native root cause. Public SDK
  evaluation now starts the normal CLI as a child with the resolved local
  scenario and propagates its status (including `139` for signal 11).
  Focused validation passed 387 tests (14 opt-in simulator skips), Ruff,
  locked ty and strict OpenSpec on source `eae98285…`. Independent review
  found no material issue with argv, interpreter, configuration lifetime,
  exit status or Ray's descendant cancellation path; it did not exercise
  live cancellation or the final evidence gate. The diagnostic
  completed one tracked rehearsal; it was not a full public Ray parity run
  or a post-worker retrieval proof.
- The corrected public SDK package and checkpoint evaluations
  (`raysubmit_B1EzGUGPYwgtBQUS`, `raysubmit_dH6WHAep5wx61QPT`;
  `eae98285…`) both succeeded on the stock SO-101 PhysX task. Same checkpoint
  hash, 1,024 source frames, seeds 10000 and 10001, two complete native
  horizon episodes per route, 0/2 successes, and equal scalar scores
  (mean final joint error 0.14234 rad). No retraining was submitted.
  After both processes exited, readback verified two distinct `FINISHED`
  MLflow runs, identical per-episode rows and 120-sample failure traces,
  and metric steps linked to the saved source checkpoint. These reports use
  private file-backed tracking on the durable mount. After the producing pod
  autoscaled away, a different GPU worker recovered both `FINISHED` MLflow
  run IDs, report bytes and guard receipt hashes (`5f91b75a7adb4a399f483e8da4785f8e`,
  verifier `a92badfb…`). Both attempts retained two completed episode rows and
  selected failure ID 0. The retrieval used the mounted artifact store, not
  a browser or an Arena suite. The post-producer proof is hash-retained in
  `arena-qualification-20261125-fpVUgQ`.
- Read-only Kubernetes status/logs confirm the editor reached full-streaming
  startup, later logged non-finite PhysX transforms/broadphase bounds, and
  exited `139`. GLFW startup warnings also appear late in that log; neither
  observation identifies the crash root cause. NetBird stayed ready. Restore
  a private editor with a clean stage before authoring; preserve the existing
  pod evidence and PVC. No infrastructure or original scenes changed.

### Arena public evaluation admission

- Implemented the evaluation-only Arena scenario and policy bridge through
  ordinary `nexuml evaluate`. Arena owns the rollout and episode recorder.
  The SO-101 candidate retains stock five-joint absolute controls, 30 Hz,
  the fixed gripper and held-success/timeout, with three target vectors not
  used by stock training. Package and standalone checkpoint produce identical
  decoded actions on a fixed CPU test observation; this is not native parity.
  Condition denominators, native episode rows and the first recorded failure
  are routed to the existing MLflow artifact; no native run produced it yet.
- Candidate source `402779f7675b2b0297ba17ce709c7b98da63695ab8be83952eda55c8ed68d7c6`:
  638 focused CPU checks passed, 14 simulator checks skipped; scoped Ruff
  and locked `ty`, docs typecheck, locked uv, and both offline wheel/sdist
  builds passed. New distributions and hashes are retained in
  `arena-candidate-distributions-43cwXU`. The qualification change validates
  strictly; the sibling's inherited long `local-learning-handoff` requirement
  still blocks its strict validation. Published PR CI was not rerun.
- Registered one-GPU `pool_gpu_workers` admission `raysubmit_Cr5t5mcaCVkTLmML`
  reached public evaluation and failed before native startup with
  `Arena requires a prepared Isaac Lab 3.0 / Arena 0.4 runtime`. It used source
  `f8c596e8a1601c79ce8a02d0333e9f63a8c2e2b154c5cee970613d66a27cfe00`;
  newer local changes are not attributed to that run. The original logs,
  config and exact submitted source are hash-retained in `arena-admission-Ruorw5`.
  No retry, source build, dependency change or alternative pool was used.
- The upstream `release/0.4.0` branch resolves to `a2f421e9…` even though its
  package metadata says `0.3.0`. Public PyPI `nvidia-curobo==0.1` is a January
  2024 upload, not Arena's September 2026 pin `78fd485f…`; it is not a
  substitute. An approved prepared worker runtime remains missing. After it
  is available, prove native compatibility, actual private Kit viewing, then
  headless package/checkpoint score and post-worker MLflow parity in that order.
- Read-only editor status still shows exit 139 / NetBird ready. The status
  and bounded crash diagnostics are retained in `editor-diagnosis-bZrouH`.
  No editor or PVC was replaced, no source-generated USD counted as editor
  authorship, and no direct cluster-node access was used.

### Ordinary pip package follow-up

- Relaxed Arena and cuRobo commit checks and now report installed versions and
  available revisions. Lab 3.0 and Sim 6.2 remain the target series. A cuRobo
  installation without a `curobo` import package is rejected before startup.
  The runtime still needs native compatibility proof.
- A version-unrestricted `uv pip install --dry-run --only-binary :all:`
  found no `isaaclab-arena` on PyPI or NVIDIA's index. PyPI resolves
  `nvidia-curobo==0.1` and `curobo==0.2`, but both SHA-256-verified wheels
  contain distribution metadata and no importable code. Ordinary MoviePy 2
  resolves, but the current Arena release requires `>=1.0.3,<2.0.0.dev0`,
  and public MoviePy 1.0.3 is source-only. This is an artifact availability
  problem, not a failure to choose versions. No usable Arena lock was produced
  and the shared lock was not changed.
- Upstream release metadata still has no Arena release assets or cuRobo
  binary release assets. Arena's target branch explicitly says that its
  cuRobo v2 is built from a Git revision. A prebuilt conda-forge MoviePy
  1.0.3 artifact exists, but it does not supply Arena/cuRobo or qualify a
  compatible job environment.
- Continuation rechecked official PyPI/NVIDIA indexes and upstream GitHub
  releases. No importable Arena wheel or cuRobo wheel asset was available;
  the latest listed cuRobo releases through v0.9.0 have no wheel assets,
  and the NVIDIA-named PyPI wheel is still the metadata-only v0.1.
  Upstream CI artifact listings show build caches/reports, not an admitted
  binary stack. Task 9.2 requires an approved compatible artifact source.
- Candidate source `ddf363097cb20b8cfdc26f3c3777825f2099beb5f02ad9de54f23790e3757881`:
  166 focused CPU tests passed, 14 simulator checks skipped. Scoped Ruff and
  locked `ty`, strict qualification OpenSpec, and the unchanged lock check passed.
- Verified signed Jammy desktop packages for a separate Reacher admission.
  An OS-only local X/VNC check required an X cookie and VNC authentication,
  bound only to IPv4/IPv6 loopback, and removed its listener on exit. This
  did not run MuJoCo or prove the remote viewer. The first bounded Ray Jobs
  display-only preflight `2659229c53e2470fbcaccba2a5df81e9` failed before
  installing or starting the display: the probe assumed `RAY_JOB_ID` was
  always set. Explicit follow-up `8fadfc536cb8485db2d06b2362305392` passed
  the identity check, then stopped at loader preflight: `libunwind.so.8`,
  `libpixman-1.so.0`, `libGL.so.1`, and `libfreetype.so.6` were absent.
  That attempt opened no listener. Follow-up `efdb6179d74740f5989a382c351a60ea`
  supplied all four through hash-verified Jammy packages. TigerVNC passed
  dynamic linking and bound loopback briefly, then failed before display
  readiness: it calls `/usr/bin/xkbcomp` by absolute path and the non-root
  worker lacks both that program and `/tmp/.X11-unix`. Extracting `xkbcomp`
  into a private directory does not satisfy this build. The next use of this
  route needs a prepared X11 worker environment; no privilege or image change
  was made. All attempts and hashes are retained. No private forwarding or
  actual live Reacher viewing was proved; 6.4 remains open.

### Stock Ray workers and remaining non-display work

- Reacher sidecar work is the last implementation task; keep the stock
  Ray worker image and install application Python dependencies inside each
  job. The deployed Helm chart rendered a GPU-only display sidecar with the
  worker image unchanged. This was a local Helm render, not a deployment.
- Gazebo environment and acceptance fixtures: 61 tests and 58 subtests
  passed, one native test skipped. Prior worker admission has no `ros2`,
  Gazebo or prepared simulation graph. The stock workers are Jammy with
  Python 3.12. ROS 2 Humble's Jammy binaries use Python 3.10, below
  NexuML's minimum, while Jazzy's Ubuntu binaries target Noble. The public
  `ros-jazzy-*` wheels inspected for `rclpy`, message types and Fast DDS
  bindings contained only metadata; no usable Python install was found.
  Native Gazebo still needs a job-installable compatible transport and an
  isolated world/controller/reset/clock testbench. This is independent of
  the deferred Reacher display.
- SO-101 original-data admission still has no approved `data_root`. The
  Isaac workstation process remains exited (139). The editor PVC and Ray
  PVC are different volumes, so a future saved scene also needs explicit,
  hash-verified asset handoff to storage reachable by the Ray job.
- The `package` case now accepts an explicit worker `--data-root` and
  runs the existing SO-101 script/SDK fit after the ordinary installed
  controls. The fit uses the hash-checked built wheels in isolated Python,
  the locked LeRobot extra without Isaac, no data download, and the
  existing 500-step repeated-fit protocol. The native replay script no
  longer requires LeRobot merely to report runtime versions; collection
  still requires it. With approved original data mounted, use
  `scripts/physical_ai_acceptance.py --case package --data-root "$REMOTE_DATA"
  --ca "$VERIFIED_CA" --output "$NEW_RECEIPT_DIR" --submit`. This is
  an unexecuted command until the data is supplied. Neither the installed
  fit nor the separate matched native learning gate has passed.
- The current editor Pod has `restartPolicy: Never`; NetBird readiness
  does not mean the Isaac container is running. Its live Pod affinity
  selects driver 580, while the repository template requires driver major
  greater than 594. The Kubernetes API admitted a dry-run create of a
  separate Pod with both `isaac` and `netbird` containers, and the RTX
  4000 Ada inventory includes driver 595. No Pod was created or replaced.
  Private enrollment, actual WebRTC input, a saved edit and Ray asset
  readback remain unverified. Dry-run evidence is retained in
  `editor-recovery-preflight-TEI6dt`.

### Job-installed Gazebo runtime probe

- RoboStack Jazzy plus conda-forge resolved a Linux x86-64 stack for Python
  3.12 and glibc 2.35, including native `rclpy`, Fast DDS, Gazebo Harmonic,
  `ros_gz_bridge`, `gz_ros2_control`, and both joint controllers. The resolved
  plan has 648 SHA-256-pinned artifacts (~915 MiB compressed). The lock, pinned
  Pixi bootstrap and probe inputs are retained privately.
- Guarded one-GPU `ray-jobs-sdk` admission `994172bf2a8b4cea851cea9f556a5e7e`
  installed the stack in a disposable job-local prefix on the stock image.
  ROS 2 on Python 3.12.16/glibc 2.35 loaded native type support and delivered
  a loopback JointState; Gazebo Harmonic 8.10.0 ran a minimal headless world
  for ten iterations. These stages passed. The overall job stayed `FAILED`:
  the trailing fixed preflight rejected controller-only metadata included
  in its worker arguments (exit 2). The complete receipt remains in
  `gazebo-robostack-admission-Zry1os`.
- Follow-up `673ec192ba54422f933fa01d0ffe0495` repeated the ROS and
  headless world passes and constructed/closed the production ROS transport
  in the layered uv environment. The bridge did not start: the minimal
  profile omitted the `ros2 run` extension. The world-control service
  therefore timed out; no clock/reset proof. Inputs and logs retained in
  `gazebo-robostack-bridge-aOvthr`. Follow-up `0e972efb16474acab568e3daafd68c94`
  invoked the installed bridge executable directly and `SUCCEEDED`. The
  regular worker preflight passed, and the job-local NexuML environment
  observed advancing Gazebo clock messages and a successful Gazebo
  world-control reset response through ROS. This empty-world probe does not
  prove the task's `std_srvs/Empty` reset adapter or movement through its
  joint controller. The locked inputs and original logs are hash-retained
  in `gazebo-robostack-bridge-corrected-4kTWFF`. The controlled robot,
  public training/resume/export/replay and private GUI remain unqualified.
- The next run, `6d3c5075ab6b4293bb65c7efde05c7f1`, added a task-owned
  two-joint world and real `ros2_control` position controllers. It checks
  measured movement to `[0.5, -0.5]` rad and return to neutral through the
  task's `std_srvs/Empty` service, then attempts a bounded public CLI
  train/export/evaluate smoke on the same simulation. The generated URDF
  and controller declarations passed static consistency checks. Spawn and
  controller setup commands returned successfully, but the control probe
  failed before any measured movement: the Torch-first process loaded the
  worker's older `libstdc++.so.6` before `rclpy`, which needs `CXXABI_1.3.15`.
  The earlier bridge probe imported `rclpy` first. Retained original inputs
  and logs in `gazebo-two-joint-E1AxxE`. The Pixi lock remains separate from
  `uv.lock`, and the Ray image is unchanged.
- Follow-up `0fb6a8e35ede41aca5a9b1492275d106` preloaded only the locked
  job-local C++ runtime for the application subprocesses. It measured
  `[0, 0]` → `[0.47138, -0.46613]` rad under the real joint controller and
  reset to `[0.00044, -0.00044]` rad through `std_srvs/Empty`. The job stayed
  `FAILED`: CLI training reached policy export, where static source scanning
  treated installed ROS modules as archive dependencies. No replay ran.
  Original logs and hashes are retained in `gazebo-two-joint-loader-6b2n5h`.
- ROS transport now imports ROS only when constructing a live environment.
  The export regression reproduced the native `PackagingError` with ROS
  import names present on the old code; the fix exports and decodes the
  same action after blocking all ROS imports. Focused CPU checks: 153 tests
  and 58 subtests passed, one native test skipped; scoped Ruff, locked `ty`,
  diff check and strict OpenSpec passed. Job
  `bba1914ed2d84180b0a1353fb2eee193` passed measured native control, public
  CLI export at 64 physical frames and fresh headless policy replay, with
  no ROS requirements in the actor package. The overall Ray status was
  `FAILED`: the final acceptance preflight refused a receipt root that
  already contained the Gazebo logs. Original source `27570d57…` and inputs,
  logs and hashes are retained in `gazebo-two-joint-export-MBoI4D`. The
  next attempt `6a3a163056cb4f58a249aa606e1a43f2` separated receipt roots
  and `SUCCEEDED` on `9b0fb056…`. The real controller moved both joints
  and returned them to neutral. The public CLI stopped at 32 frames and
  20 optimizer steps with exit 143, resumed the unchanged partial checkpoint
  to exactly 128 frames and 80 optimizer steps, exported the matching finite
  actor and action adapter, and completed three fresh evaluation episodes.
  The worker preflight passed. The setup, receipt summaries and logs are
  hash-retained in `gazebo-two-joint-resume-MSVSW1`. Readback
  `92082aca28004e179be458cc54d3a1a0` on a different GPU worker verified the
  lifecycle log hashes, checkpoint and actor/adapter bytes, consistent ROS
  namespace, and guard/stage cleanup. All three replay episodes reached the
  native 50-step limit; this is lifecycle smoke, not learning success. This
  closes headless task 5.7. Readback evidence is retained in
  `gazebo-lifecycle-readback-dGKSxk`. Private GUI remains open.
- Concurrent isolation probe `7d7b510259bc468ca6cd6a0bd0fee2d6` succeeded
  on `9b0fb056…`. Two headless worlds overlapped on a single stock worker.
  Each saw its own clock/messages and Gazebo service but not its peer.
  A duplicate ROS domain claim was rejected and owned processes cleaned up.
  This closes 3.2 with the earlier private port, interpreter, attempt-root
  and tracking-store evidence; it does not qualify native desktop viewing.
  Inputs and logs are hash-retained in `gazebo-concurrent-isolation-BVKEEg`.
- Gazebo's job-local display probe adds prebuilt X11/Mesa packages without
  replacing any package in the working ROS/Gazebo lock. TurboVNC 3.3.2
  accepts a private `xkbcomp` directory, unlike the earlier extracted
  TigerVNC used in the Reacher probe. The first Gazebo display attempt
  failed before submission with HTTP 413 during the working-directory
  upload. It has no Ray job ID. Keep that failure in
  `gazebo-display-admission-q2vKmY`; the next attempt downloads the
  hash-pinned binary into its own worker directory instead. A GUI startup
  probe is not evidence for private viewing, camera navigation or 6.5.
- Job `4b98b0cc38e34fd4bb69d46f5fa64bb2` measured real joint movement
  and reset, then failed because the stock worker has no `curl`. A new
  attempt `9f820f05625443829404a9631246d48d` fetched and verified the same
  binary with the job Python and measured control/reset again. TurboVNC
  bound to loopback but exited with signal 11 before X11 readiness, so
  there was no private viewer or camera proof. Keep both original
  failures in `gazebo-display-admission-download-DHk55o` and
  `gazebo-display-admission-urllib-szPdMR`. The separate Gazebo-only
  fallback uses prebuilt Pixi Xvfb/Mesa and a signed Ubuntu TigerVNC
  framebuffer server. Job `ba6e69f724974b50a034e3656787c535` measured
  control/reset but failed before Xvfb: the probe looked in `share/X11`
  for keyboard rules. The hash-verified conda archive places them in
  `share/xkeyboard-config-2`. The retained `gazebo-xvfb-live-UahNoN`
  failure had no listener, camera or training proof. The next explicit
  attempt fixes the rules path and still requires authenticated forwarding,
  a camera move, actual CLI training with a passive joint/clock witness
  and owned cleanup. It does not change the Ray image or start Reacher.
- With the rules path corrected, `30515db068c94e8ca9444164b0420a18`
  measured control/reset, then Xvfb aborted inside the worker's NVIDIA
  EGL/GBM library before reporting X11 ready. The next probe explicitly
  selects the locked Mesa EGL vendor for the desktop processes only and
  corrects the TigerVNC scraper's boolean flags. The physics process and
  stock Ray image are unchanged. Original failure retained in
  `gazebo-xvfb-live-xkb-W8gQRv`; 6.5 stays open.
- The Mesa-only display in `fe370d8996964d49898ff8360d2b64a0` admitted
  authenticated X11, rejected access without the X authority, and opened
  an RFB listener only on worker loopback. It then failed waiting for
  the native Gazebo GUI render marker. No client was forwarded and no
  live CLI training ran. The original attempt is hash-retained in
  `gazebo-xvfb-live-mesa-zArb8J`. Follow-up
  `3cd4145d5375491f97d340b6046de08c` captures the actual private desktop
  and line-buffered GUI output before requiring a rendered viewport and
  camera motion. Display admission alone does not close 6.5.
- `3cd4145d5375491f97d340b6046de08c` did render the R1 robot. The
  authenticated client captured its actual Gazebo viewport via private
  `kubectl port-forward`, and the GUI requested state from
  `/world/nexuml_joint_world/state`. The job still failed: ANSI color
  codes split the scene-ready log text, so the controller never armed
  the camera/training step. Original GUI log and forwarded screenshot
  are retained in `gazebo-xvfb-gui-trace-mw5Ljp`. The follow-up strips
  those codes and also checks the world-state request. No live camera
  navigation, CLI-driven viewing or cleanup acceptance is claimed yet.
- `446c77e34a7a4f128a791e4fcefc1ca9` on `4c4f93c0…` completed public CLI
  training with the display active (256 physical frames, 160 optimizer
  steps, changed finite actor), two-joint motion under the same ROS
  namespace and worker-side display release. Private camera navigation
  changed the actual view, but the camera moved the robot out of frame;
  the training capture matched that empty view. The client timed out and
  its exact leftover loopback tunnel was stopped separately. Ray status
  remains `SUCCEEDED`; visual acceptance did not pass. Preserve screenshots
  and receipts in `gazebo-xvfb-live-qualified-IYocNY`. The next attempt
  uses a small pan, rejects out-of-frame captures and checks client-side
  tunnel closure as well as worker cleanup.
- `5061701387024d11b891b7bbeb384bd7` also succeeded on the worker at
  256 frames/160 steps. The private client captured the robot before and
  after a small pan and in two different poses during the CLI train
  window. Visual inspection and image hashes confirmed the visible
  motion. The client nevertheless exited 1 when building its receipt
  (`camera_difference` was undefined); no leftover forwarding process
  was found. Preserve this split result and images in
  `gazebo-xvfb-live-framed-eOtPCc`. A clean follow-up fixes the receipt,
  checks the camera framing, joins capture times to the worker training
  window and verifies hashes and tunnel closure.
- `637659940f5d454092ad2d6b68ccc661` failed before camera or training:
  the scene creation log preceded the first rendered robot frame. The
  authenticated viewport capture was still blank. Original evidence is
  retained in `gazebo-xvfb-live-receipt-gP5ITh`. The follow-up waits for
  actual robot geometry within a bounded window, rather than treating
  GUI initialization as sufficient.
- `71abac77de01491f9997ba728911a4ab` failed earlier, during Pixi linking:
  the locked Xvfb package places a directory at `share/X11/xkb`, while
  `xkeyboard-config` also owns that path. This failure is retained in
  `gazebo-xvfb-live-render-ready-dCsdlG`; no viewer or training ran.
  A separate rerun pins keyboard rules 2.44, which uses a real directory
  compatible with Xvfb, without changing the application pins.
- Task 6.5 closes on the observed `5061701387024d11b891b7bbeb384bd7`
  viewport and distinct-worker readback `ba221f6f23804e76beb4f4dd9191c07a`.
  Readback verified the same worker's world-state request, matching URDF,
  controller and measured ROS namespace, CLI stage hashes, policy and
  checkpoint bytes, and guard/RFB shutdown. The private client images
  show the camera pan and two in-frame robot poses during the recorded
  CLI train window, verified against the preserved file times and hashes.
  No job-owned client tunnel remained. The original client exit 1 and
  failed receipt remain unchanged; independent reconciliation is retained
  separately in `gazebo-live-evidence-review-MePNV8` and the readback in
  `gazebo-live-readback-rPY28Z`. The tested GUI used camera-only input
  through loopback VNC and `kubectl port-forward`; any physical
  intervention is debug-only and excluded from quantitative acceptance.
  This is software-rendered live viewing and lifecycle smoke, not
  learned reaching, hardware safety or a universal Gazebo profile.
- Clean repeat `b75eb69f6ddc4cacbbb6f3b364e0796d` on `4c4f93c0…`
  passed with keyboard rules 2.44: Ray succeeded, the client exited 0,
  the verified receipt joins both distinct in-frame captures to the
  CLI training window, and both the display and the client port-forward
  closed. The camera pan kept the robot in frame. Both joints moved
  in the matching namespace, and training again finished at
  256 physical frames and 160 optimizer steps with a changed finite
  actor. The lock, commands, receipts, and reviewed images are
  hash-retained in `gazebo-xvfb-live-keyboard-layout-b9a0rW`.
  For a new attempt, copy the retained inputs to a new private attempt
  directory and point its controller at a fresh client venv built from
  the retained hashed `requirements.txt`. Use the verified Ray CA and
  the registered SDK submission; the retained controller records the
  exact arguments. The client venv is disposable, not a worker dependency.
- Focused CPU regression under umask `077` exposed a test fixture that
  expected `mkdir(mode=0755)` to bypass umask. Explicitly setting the
  test directory's mode keeps the original no-mutation assertion.
  The same rerun passed 164 tests and 96 subtests, plus scoped Ruff,
  strict OpenSpec and diff checks. Receipts:
  `continuation-local-gates-uWejq1` and
  `continuation-local-gates-fixed-bSCm7T`. No native gate was counted.

### Historical October baseline

- Published head: `01d3c0a7dcde55b5d4915e9484d6f069e946fe67`, based on actual
  `origin/main`; preserve Starlight and the existing rebase backups.
- Uncommitted executable snapshot:
  `dd466d1cc6d3ecd0ed2bebeb800947a8b3b726eabe4655d6be25347f4b90b3d5`.
  This includes test-owned viewer IPC changes. The current eight-test spectator
  suite passed remotely; full spectator acceptance and release readiness remain open.
- Production `src/` and `library/src/` remain unchanged from the published head.
  Historical evidence keeps its actual source/runtime identity, never the new hash.
- Published-head CI remains failed; exact-candidate CI requires separately authorized
  publication. Local checks and historical remote passes are not that CI result.

## Historical October claim-to-evidence map

| Surface | Retained result and scope | Remaining qualification |
| --- | --- | --- |
| G1/G2 public CLI, I1 native, A1 multi-agent, M1 MuJoCo | Recorded train/resume/export/fresh-evaluation passes under their original snapshots. Bounded recipe runs are smoke, not convergence. | Final identity reconciliation and applicable recovery/resource matrix. No Humanoid GUI extrapolation. |
| I2 sensors/categorical | All nine processing-owner CLI journeys plus 13 native oracles; 810 files / 1,129,237,671 artifact bytes rehashed. Sources `43ae0b9c…` and `2e7c0da5…` remain distinct. | Does not qualify spectator viewing, universal renderer support or camera-policy convergence. |
| Frozen Cartpole L1 | Actual held-out native learning comparison passed under its recorded single-training-seed protocol. | No multiseed, other-task or hardware-safety claim. |
| Public SO-101 cloning | Job `d3e74059a63d4b0aafc58e6f7965e4a3`: 64 GPU optimizer steps, 40/5/5 public-data split and reduced held-out loss. | Not frozen S1/L2: different task, channels, rate, episode count and qualification contract. |
| Installed B1/tutorial | Job `56bbb15a360f4516ae6918a15a49023a`: exact installed-wheel controls and canonical tutorial passed; 20 files rehashed. | Approved-data S1 fits and full task5.6 remain open. |
| Four logger modes / SQLite and S3 | Job `d239a4d387474ce396f178e104351f3d`, source `43ae0b9c…`: 32 CLI stages, 16 FINISHED MLflow runs; 293 retained files verified. Actual downloads and same-rollout payload equality passed. | Not current MLflow UI/failed-run cleanup proof, private-bucket protection or SQLite metadata backup. |
| MLflow successful-run UI | Historical jobs `fa1f34d39aba40db8de1c412bad04d23` and `d63ba45d56794618a410ff3fec0ec1f1`, source `bb40b77c…`: selected native Overview rows, status, media and cleanup passed. | Current SQLite/S3 browser reconciliation, native failed-run tree and unchanged strict cancellation contract remain open. Do not discard these genuine earlier passes or relabel them current. |
| Isaac public-CLI spectator | Current job `82d65c3218a24ee0a8a7764dbd0f4021`, source `dd466d1c…`, passed all eight tests without skips. All164files /296,978,604artifact bytes verified and privately preserved. Full trajectory/PPO equality, actual WebSocket mutation denial, IPC cleanup and same-job private8081HTTP200 were reviewed. | Task6.3 still needs external cancellation of a spectator-active registered Ray job with terminal status and listener reuse. Existing cancellation self-SIGTERMs only the CLI child. Training-reconnected capture is cropped, not positive framing proof. Browser SwiftShader is not server RTX proof. |
| Failed viewer recovery | Failed `64fbe044…` and `0cfaa3d7…` retain complete recovered original files. Failed `2b85841e57cb4b13b3fa7cec9e625727` now has all49original files verified; original FAIL/guard1 is unchanged. | Recovery is not a successful rerun or qualification of the failed source. Earlier23/49records remain historical partial observations. |
| Post-worker persistence | Registered recovery `37366add3c4846279d29ea27d144335b` passed after the current viewer producer was removed. A verified owned reader retrieved all164original files. | Initial stdin-based tar transfers truncated despite kubectl0. Strict EOF/hash checks rejected them; a new stdin-free literal-argv transfer passed. This is not a universal transport-repair claim or completion of the whole recovery matrix. |
| Ordinary public CLI reporting | Verified-CA command submitted `raysubmit_hK9Mg9DDf2xnMe7N`, which FAILED before guard/setup/training: preparation omitted `runtime_env.working_dir`, so the worker lacked the relative guard script. Original config/lock/logs are retained. | At this snapshot, corrected approved-CA/upload/exclusion inputs were offline only. The later November public CLI runs above closed 4.1 without retrying this job. |
| Infrastructure | Authorized GPU-worker-only plasma fix `23c3410e…` was deployed; preflight `5424f937…` verified4GiB tmpfs plasma and zero disk-backed plasma. | Does not establish application throughput, cancellation storage health or safe two-GPU overlap. No other infrastructure changes authorized. |

Receipts and exact identities live in the existing private preservation bundles and
chronological `preparation.md` / `.tasks/todo.md`, not in Git-tracked weights or media.
The current durable bundle is `current-eight-complete-public-launch-diagnosed-s2_7wlr5`
under `/home/nexuadmin/.local/state/opencode/nexuml-preservation/scheduling-mlflow-20261029-jb0h_ud0/`:
772files /959,485,635bytes rehashed, including the complete current viewer evidence,
original failures/transports and public launch diagnosis. Credential-value scans were
clean; private execution configs were recorded by hash, not copied into that bundle.

## Historical October execution order

1. No application/controller is active. The hourly capacity trigger is disabled
   after accepted admission and remains disabled after the real public-job failure.
   Latest read-only inventory at2026-10-30T03:21:46Z found3eligible free GPUs and no
   Ray jobs/workers; this is time-bound, not parallel-memory qualification.
2. Preserve the completed current eight-test proof and all49original failed files;
   do not rerun them merely because the first byte transport failed. Do not reuse
   either public preparation's exclusive submission lock.
3. For any explicitly restarted attempt, refresh eligibility/reservations/jobs,
   physical disk and actual shared-memory admission, confirm no overlapping checkout
   writer, and record external source freeze through submission and terminal readback.
   Serial whole-GPU `pool_gpu_workers` only. Fingerprints detect changes but do not
   enforce filesystem immutability.
4. Public reporting's corrected inputs explicitly preserve the approved CA, uploaded
   checkout and data/venv/output exclusions.116hermetic checks/35subtests passed;
   no new application was submitted after diagnosis. A fresh explicit attempt is
   needed before claiming actual CLI completion. Never use the old checkpoint-export
   preparation, which would restore historical logger/callback paths. For6.3, obtain
   actual registered-Ray cancellation/cleanup/listener-reuse proof, not another
   CLI-child self-signal; keep the storage/profile gate below in force.
5. Reconcile MLflow UI evidence and remaining isolation, resource-cost, persistence
   and recovery coverage. The historical `c340934b…` browser cancellation failure
   had owned D-state threads in `jbd2_log_wait_commit` / `fdatasync`; its exact old
   FD/file/device is unknown. Require storage/profile qualification before repeating
   that known-failing journey unchanged. Do not relax limits, durability or sandboxing.
6. Demonstrate two real overlapping GPU allocations only after capacity and aggregate
   memory are qualified. Queued jobs are not overlap.

The user authorized an hourly capacity trigger: OpenChamber task
`16fbdcab-4401-4c43-9a6d-989fb73dd35a`, minute20, Europe/Berlin, first
2026-10-30 00:20CET, targeting the current session. It was disabled after the first
accepted owned preflight and must not become an application retry queue. Check
active controllers and latest checkpoint before any separately restarted launch.

## Executed procedure references and boundaries

- `docs/src/content/docs/how-to/training-backends/ray.mdx`: registered SDK schema,
  asynchronous submission/status and explicit worker paths. The November
  ordinary public-CLI reporting capture is recorded above (4.1). The executed
  command forms were `train --config`, `train --config --trainer-checkpoint`,
  `export --config --checkpoint --output`, and `evaluate POLICY --config`.
  Read the terminal Ray status and verify the produced checkpoint/policy
  hashes before using the next command. These paths live on the worker.
- `docs/src/content/docs/how-to/isaac-read-only-viser.mdx`: separate train/replay
  enabling, fixed eight-test command and KubeRay port 8081 private forwarding.
  The November spectator suite and external Ray stop above provide the 6.3
  evidence. Neither requires a public listener or claims graceful shutdown.
- `docs/src/content/docs/how-to/tracking.mdx`: per-attempt SQLite metadata, separate
  S3 artifact prefix, verified TLS, explicit failures and no implicit rollback.
- `docs/src/content/docs/how-to/isaac-sensors-and-policy-state.mdx`: processing-owner
  and sensor/backend limits; spectator pictures do not prove policy input correctness.

## External prerequisites that remain

- Approved72successful SO-101 joint-reach episodes, fixed48/12/12 split and explicit
  current-runtime qualification, or an approved replacement frozen protocol. No
  automatic recapture or private-data upload.
- Approved worker display and private access for MuJoCo Reacher live native viewing.
- Storage/profile qualification for the recorded browser journal-sync boundary;
  unrelated GPU plasma repair is not proof of that repair.
- Owner credential review/rotation after the separately reported tool-output exposure.
  Do not reproduce values or copy the raw affected output into evidence/publication.
- Separate authorization for commits/push/external statuses, then fresh required CI;
  merge and archive remain independently gated.

## Source and data hygiene

The candidate diff contains only source, test, configuration and documentation
changes. Raw episodes, checkpoints, policies, media, logs, browser traces and
private runtime profiles are excluded. Preserve the mixed uncommitted `preparation.md`,
root `.venv`, datasets, old receipts and unrelated work. No driver/image/RAM/security
changes, other-user deletion, local/DGX fallback or hardware actuation is authorized.
CARLA live integration, NexuFL/federation consumption and later model/robot roadmap
items remain separate, unqualified work—not claims closed by this change.

Task 8.4 covers this handoff, not release readiness. The required blocked rows
and the final independent evidence gate remain open.
