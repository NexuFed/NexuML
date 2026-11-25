# Physical AI qualification support handoff

## Status and authority

Partial support handoff, not a release approval. [tasks.md](tasks.md) remains the
authority: **24/38 complete**. Required blocked rows are not waived. No publication,
PR/Linear status change, merge or archive is authorized by this handoff.

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

## Gazebo qualification on stock Ray workers

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

## Claim-to-evidence map

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
| Ordinary public CLI reporting | Verified-CA command submitted `raysubmit_hK9Mg9DDf2xnMe7N`, which FAILED before guard/setup/training: preparation omitted `runtime_env.working_dir`, so the worker lacked the relative guard script. Original config/lock/logs are retained. | Task4.1 remains open. Corrected approved-CA/upload/exclusion inputs passed mocked transport checks but are OFFLINE ONLY, not submitted. No failed application auto-retry. |
| Infrastructure | Authorized GPU-worker-only plasma fix `23c3410e…` was deployed; preflight `5424f937…` verified4GiB tmpfs plasma and zero disk-backed plasma. | Does not establish application throughput, cancellation storage health or safe two-GPU overlap. No other infrastructure changes authorized. |

Receipts and exact identities live in the existing private preservation bundles and
chronological `preparation.md` / `.tasks/todo.md`, not in Git-tracked weights or media.
The current durable bundle is `current-eight-complete-public-launch-diagnosed-s2_7wlr5`
under `/home/nexuadmin/.local/state/opencode/nexuml-preservation/scheduling-mlflow-20261029-jb0h_ud0/`:
772files /959,485,635bytes rehashed, including the complete current viewer evidence,
original failures/transports and public launch diagnosis. Credential-value scans were
clean; private execution configs were recorded by hash, not copied into that bundle.

## Next execution order

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
  asynchronous submission/status and explicit worker paths; ordinary public-CLI
  current reporting capture is still pending4.1.
- `docs/src/content/docs/how-to/isaac-read-only-viser.mdx`: separate train/replay
  enabling, fixed eight-test command and KubeRay8081 private forwarding. The new
  IPC fixture has current eight-test remote proof, but external-Ray cancellation
  remains missing; do not label task6.3 complete.
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

Keep raw episodes, checkpoints, policies, media, logs, browser traces and private
runtime profiles outside commits. Preserve the mixed uncommitted `preparation.md`,
root `.venv`, datasets, old receipts and unrelated work. No driver/image/RAM/security
changes, other-user deletion, local/DGX fallback or hardware actuation is authorized.
CARLA live integration, NexuFL/federation consumption and later model/robot roadmap
items remain separate, unqualified work—not claims closed by this change.

Task8.4 remains open until final reconciliation; this draft deliberately exposes
missing evidence rather than presenting partial qualification as release readiness.
