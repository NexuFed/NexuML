# Acceptance matrix — current local candidate

See linked records for the tested source/asset hashes, interpreter and native
versions, exact invocations and raw receipts. Each result applies to its
recorded candidate. This is not a release matrix.

The first standard regression used base `fcb4a81a150b0eb272be6708315ea3fa64e00813`
and uncommitted inputs with SHA-256
`56cf595c0ce320ebba8400bb46e0a94fa6f98e73f32ee8177feb7aa16d091f9d`.
Recompute from the repository root (includes untracked candidate files):

```bash
export LC_ALL=C
git ls-files -z --cached --others --exclude-standard -- \
  src library tests scripts docs configs containers skills README.md \
  Dockerfile pyproject.toml uv.lock .github .k8s |
  sort -zu | xargs -0 sha256sum -z | sha256sum
```

| Lane | Status | Evidence or prerequisite |
| --- | --- | --- |
| Direct Isaac training and standalone replay | Pass, selected references | [Direct regressions](direct-regressions.md): Cartpole, SO-101, Pendulum IPPO; suspended restore and fresh replay. The [SO-101 learning follow-up](so101-learning-followup-results.md) confirmed 32/32 episodes per passing actor on the three built-in poses, not unseen-goal transfer. |
| Native project-owned direct task | Pass, state-only Newton | [Custom direct task](custom-direct-task.md): native RSL-RL and NexuML through the same task, 256 transitions each, control/termination/reset parity, separate exports and fresh replays. |
| Headless scripted/policy interaction, direct native bindings | Pass, selected references | [Interaction foundation](interaction-foundation.md) and [binding cleanup](native-binding-cleanup.md): known controls, fixed clock, same-step reset and read-only spectator parity. |
| Complete standalone inference and bounded local fits | Pass, scoped | [Local-fit handoff](local-fit-handoff.md): installed package and native SO-101 repeats; matched PhysX / CPU MuJoCo actor fit, test-only combination, second fit and replay. Not production federation, promotion or general learning improvement. |
| Independent local playback | Pass for 4a.2 and 4a.3 | [Playback evidence](playback-evidence.md): state and mixed-sensor headless play, N=1 and native batches, with policy-owned history and selected recurrent/chunk state resets; loopback Viser with bounded task reset options and bounded headless N=1 PhysX GIF/MP4. No viewer recording or remote playback qualified. |
| Independent evaluation snapshot | Pass for fresh CPU/CUDA process and same-host CPU Ray tasks | [Snapshot boundary](snapshot-boundary.md): periodic GPU inference overlapped a CPU learner and published bounded media with correct policy provenance. Actor/optimizer matched the no-evaluation control. Two-GPU, multi-node and native Isaac fit evaluation remain unqualified. |
| Native Kit Local/X11/TigerVNC and WebRTC playback | Unqualified | [Native Kit probes](playback-evidence.md#native-kit-display-probes-4a1-and-4a4-partial): Dev Containers X11 forwarded blank windows to the operator. The desktop probe ran actual policy ticks, but neither visible playback frames nor a mutation-restricted surface were verified. No approved end-to-end native WebRTC or TigerVNC reference. |
| Ray Jobs playback | Pass on one prepared GPU host; remote acceptance open | [Playback evidence](playback-evidence.md#real-ray-jobs-on-one-prepared-gpu-host): real CUDA playback, bounded media and API cancellation with CUDA release and the borrowed Ray runtime intact. The shared-pool producer hit admission timeout with no driver; remote GPU and viewer routing remain unqualified. |
| MuJoCo Warp native API probe | Pass, substrate only | [GPU interop probe](mujoco-warp-probe.md): the pinned packages shared CUDA state with Torch, agreed with CPU for two substeps and reset only the selected row. No NexuML GPU environment, training or throughput claim. |
| MuJoCo Warp direct reference | Pass for pinned headless Cartpole | [GPU reference](mujoco-warp-reference.md): native resets and terminal flags, 1,024 CUDA transitions and 16 PPO minibatches, fixed actor export and fresh replay. Distributed, split placement and rendering are not qualified. |
| Matched-geometry runtime profile | Pass for scoped local measurement | [Runtime profile](mujoco-warp-reference.md#task-54--matched-geometry-measured-profile): CPU MuJoCo, CUDA Warp and CUDA PhysX each ran 1,024 transitions with common actor weights. Reports raw warm timing, versions/ABI, explicit EGL success and OSMesa gap. The CPU CI test command and opt-in hardware suite passed locally; no dynamics equivalence or speedup claim. |
| Matched native Isaac and CPU MuJoCo task | Pass, paired reference (5.3) | [Paired Cartpole](local-fit-handoff.md#paired-result): signed effort oracle, shared actor/PPO, and fresh evaluation of one exported package on both engines. Distinct dynamics and reward scales; not GPU MuJoCo learning. |
| Original matched SO-101 learning/throughput gate | Fail | [Baseline](baseline.md): zero held-out successes for both learners; NexuML achieved 73% of native warm throughput. The longer-budget follow-up did not rerun this gate. |
| Placement readiness | Pass for fixed SO-101 treatment | [Independent revisit](so101-placement-readiness.md): fresh reset seeds gave 96/96 successes for each learner, with NexuML 78.52% of native warm throughput. This warrants implementing placement, not claiming it works. |
| CPU/CUDA direct rollout transfers | Pass for Gymnasium Pendulum | [Device transfer evidence](rollout-device-transfer-evidence.md): 6,144 transitions per placement, forward and storage on CUDA, CPU learning, terminal-state/GAE parity and exact replica synchronization. Transfer cost made this small reference slower than all-CPU. |
| Explicit Ray CPU collection and policy version barrier | Pass for same-host CPU PPO | [Placement evidence](ray-rollout-placement.md): borrowed runtime, native custom resource/label scheduling, complete versioned batches, per-update synchronization and worker cleanup. No GPU or multi-node claim. |
| Worker-owned rollout scope (3.2) | Pass, one host | [Worker trace](ray-rollout-placement.md#returning-native-shutdown-32): real PhysX actors ran CUDA batches with the PPO actor/critic, acknowledged native close, and released their processes and object-store references after completion, failure and cancellation. This does not enable public GPU collection. |
| Split GPU and distributed collection | Unverified | Separate collector and single-learner configuration is supported. This host exposes one GPU. The existing Ray worker groups request one GPU per pod; the only multi-GPU node reported by the Kubernetes API used V100s and driver 535, outside the pinned Isaac profile. No node access, new pool or actual split allocation was used. |
| CARLA 0.9.16 | Pass for pinned dedicated-server reference | [Live evidence](carla-live-evidence.md): exact ticks, matching camera and ego snapshots, observed collision ordering, extra-tick rejection, cleanup, and public train/evaluate/export/reload. The exported policy inferred identically without CARLA installed. |
| Modern Gazebo deterministic stepping | Pass for isolated Jazzy/Harmonic reference | [Live evidence](gazebo-exact-step-evidence.md): eight accepted native goals, exact tick/clock/sample matching, signed motion, reset epoch, two-world isolation and cancellation cleanup. The standalone proof is not a NexuML Gazebo runtime adapter. |
| Arena composed training and evaluation | Native evaluation pass; training open | [Native Arena proof](arena-native-evidence.md): pinned local runtime; existing actor scored 3/3 held-out conditions through the native runner and recorder. No Arena-composed fit or continuation yet. |
| Approved three-domain physical ROS 2 and mixed objectives | Blocked | No approved matching real episodes available; simulated pairs and hermetic evidence do not count. |
| Existing supervised/dataset regression, NEX-234 real-data run | Pass for fixtures; real-data rerun blocked | Existing CPU and CUDA dataset SDK checks passed. Original approved dataset unavailable. |
| Package and image verification | Pass for local wheels and a pinned headless Isaac image | Installed Cartpole train/export/play/evaluate passed in the binary-only workstation image. [Installed MuJoCo Warp](mujoco-warp-reference.md#installed-binary-only-profile) passed without Isaac or Ray. No Harbor upload was needed. Public release delivery is deferred until the complete change is merged. |

Integrated native regression: **10 passed in 137.19 s**. This includes the
manager/direct transition oracles across PhysX, Newton and OVPhysX, SO-101
held outcomes, scripted/actor parity, and the project-owned direct task's
native and NexuML lifecycle. Logs:
`/tmp/opencode/nexuml-decouple-merged-native-CvlNA5/`. The dependency lock
passed `uv lock --check --offline`; docs and strict OpenSpec checks passed.

Integrated native evaluation media: **4 passed in 106.62 s**. Cartpole,
SO-101 and Franka all published GIF/MP4 at the exported source frame (256);
cancelled evaluation was not marked successful in MLflow. A stale fixture
had expected the historical label zero; no product behavior changed to
make the test pass. Logs: `/tmp/opencode/nexuml-decouple-media-provenance-e9zPK5/`.

Integrated CPU regression: **734 passed, 14 skipped, 2 live cases
deselected** in 66.46 s. This covers evaluation snapshots, playback and
capture, local-fit contracts, standalone export/provenance, fake system
providers and native lifecycle/configuration without a simulator.
Logs: `/tmp/opencode/nexuml-decouple-integrated-cpu-jJCNbl/`.

Final integrated native local-fit/play regression: **3 passed in 170.80 s**.
The public SDK used two approved SO-101 actor bases, ran the paired
PhysX/CPU MuJoCo fit/reload/second round, and verified custom direct
checkpoint/export/fresh evaluation/play. Logs:
`/tmp/opencode/nexuml-decouple-integrated-native-kGbxkU/`. This is correctness
and API evidence, not a rerun of the original learning-quality gate.

The candidate wheel/source pairs built offline and passed the existing
archive checks. Installed into a disposable target against the prepared
dependency profile, the wheels passed **14 playback/CPU evaluator tests**.
Native task discovery attempted no native imports. Artifact hashes and
logs: `/tmp/opencode/nexuml-decouple-candidate-packaging-YfSteD/`.
The unrelated network-backed resolver tests were not run.

The later stateful-playback/Viser candidate was rebuilt as both wheel/source
pairs with the same build-backend version. The archive check passed and
**17 installed-wheel playback and fresh-process evaluation tests passed**.
The core and library wheel SHA-256 digests were
`35b094c28872ae3dea7fadceff7da5fa21a68b31c5efeec2975f191a0ae5b451`
and `e20b6a053d700f36c9728f44642cc722a118c5d84cab4bcce45cd4339a3ab931`.
Local artifacts: `/tmp/opencode/nexuml-decouple-final-playback-dist-BRT8ov/`.
The dependency lock, full docs check and strict change validation passed.
No hosted release or remote delivery is implied.

The same wheels passed the existing isolated distribution smokes:
**1 core-only and 2 core-plus-library checks**, with Python isolated mode,
CUDA hidden, and no editable source paths. The library check verified that
registry discovery attempted no native robotics imports. The archive check
and both opt-in external consumer resolver cases (wheel and editable) also
passed: **3 checks**. These resolver cases validate consumer backend selection,
overrides, and unsupported Python/Torch rejection; they do not run a learner.
Receipts: `/tmp/opencode/nexuml-decouple-isolated-install-r0bS9H/`.

Adding only the published `reinforcement` extra to the isolated library lane
gave **4 more passes**: the documented CPU CLI first run and the installed
Pendulum learner's blocking evaluation at frames 0/64/96, timeout cleanup,
and rejection of unsupported video. Both packages resolved from the isolated
prefix. Gymnasium supplied the classic-control task; Ray and native robotics
packages (Isaac, MuJoCo, ROS, CARLA, Newton and LeRobot) were absent.
See `learner.log` and `learner-packages.txt` in the same receipt directory.

The rebuilt `51606ce8` wheels also passed the installed CPU Ray CLI smoke.
The fresh binary-only environment used Torch 2.11.0+cpu, TorchRL 0.14.0,
Ray 2.57.0 and Gymnasium 1.3.0, without native robotics packages. A local
Ray learner run completed **32 transitions and 8 optimizer steps** on CPU
with two rollout actors, then the installed CLI exported and played the
fixed actor for eight transitions. The actor reports source frame 32,
with package SHA-256
`e14a0b94b2b3d3490dcf8e2969f4b4a9ea1572a2296b8a4294755c8e5e849260`.
Receipts and wheel hashes: `/tmp/opencode/ri-jhl2lN/`. Task 6.3 remains
open for unqualified split and distributed GPU recipes, not public release
delivery.

The existing binary-only Isaac workstation build produced image
`sha256:1f03b8cf62699c098b0e590076b972446b51102086767aaa103c475b964b2d6b`
from the pinned NVIDIA base and Lab VCS cache. Included core/library wheels
had SHA-256 `be22e9898436e9298d13fdccf75e73b46de65a0d9cade9a63cdd04a241b15382`
and `e20b6a053d700f36c9728f44642cc722a118c5d84cab4bcce45cd4339a3ab931`.
Without mounting checkout sources, the installed CLI trained Cartpole for
256 native transitions, exported the actor, played 8 ticks across 2 lanes,
and completed one fresh evaluation at source frame 256. The container
exited 0 and was removed. Evidence:
`/tmp/opencode/nexuml-decouple-isaac-image-wuM5d8/` (build) and
`/tmp/opencode/nexuml-isaac-container-proof-IvfR34/` (invocation, logs and
artifact). This qualifies those image contents, not later commits or a
display/streaming path.

The rebuilt `fc3149e9` image
(`sha256:d0e9407afdf553159074e671abc610d7a522e737884a0bbefa033011ae665aee`)
also trained to source frame **256**, exported the actor, played eight
ticks over two environments and evaluated one episode. The container
exited 0 without a checkout mount and was removed. The no-network
attempt could not fetch the native ground-plane asset; normal access to
the configured Isaac assets was required. Build and runtime receipts:
`/tmp/opencode/nexuml-isaac-current-owqwqO/` and
`/tmp/opencode/nexuml-isaac-current-runtime-BusKF0/`.

The successful image run exposed a missing CLI receipt: Kit fast
shutdown ended the process after export but before the CLI could print
fit counters, the completed package and worker identity. The CLI now
reports these after export inside the existing local job scope.
Provenance remains before export. No receipt is printed on a failed
fit, provenance capture or export; the process exit status still
determines job success. Focused ownership/CLI checks passed **125**;
bootstrap, execution dispatch and ownership checks passed **266**.

The rebuilt receipt candidate
(`sha256:1676f2c47153d1c1bc814efc2703e749d87cd766d1f3eb063f8d2015f84a0f97`)
passed the installed CLI smoke without source mounts. Training printed its
completion once, with **256 physical frames, 8 optimizer steps, `cuda:0`**,
and the exported package. Playback ran **16 transitions** and evaluation
completed one episode at source frame 256. The container exited 0 and
was removed. Receipt logs, source/wheel hashes and image ID:
`/tmp/opencode/nexuml-native-receipt-image-GyjRNv/`.

The first standard non-system regression reported **3,631 passed and 12
failed**. Failures were in execution test fixtures: an old evaluation-spec
stub, a test that expected a new local-only task in the remote inventory, reliance
on an empty Isaac import cache, and a device-less TensorDict test fixture
that synchronized CUDA after earlier tests initialized it. The fixes do not
change product behavior or add the local task to the remote allowlist.
All 14 focused cases passed with Isaac already imported and CUDA initialized.
First-run log: `/tmp/opencode/nexuml-decouple-standard-regression-A6hWF1/pytest.log`.

The full standard rerun passed **3,643 tests and 604 subtests**, with 16
skipped and 115 deselected, in 449.27 s. It used Python 3.12.16,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
`python -m pytest -q --tb=short --disable-warnings -m 'not requires_system and not requires_hardware and not requires_simulator and not requires_data'`.
Skipped and deselected tests are not qualification evidence. Log:
`/tmp/opencode/nexuml-decouple-standard-rerun-pPzzlp/pytest.log`.

After GPU/Ray evaluation and the CARLA export fix, standard regression at
`124ea1cd` passed **3,649 tests and 604 subtests** (16 skipped, 119 deselected)
in 508.84 s. Log: `/tmp/opencode/nexuml-decouple-regression-kpLIcT/pytest.log`.

The first integrated placement run passed **3,730 tests and 604 subtests**
and failed nine stale Ray Jobs receipt assertions. The receipt reports
worker allocation/device and progress as unknown until observed in job
logs; train, evaluate and export tests now assert that wording. No product
behavior was changed to make these tests pass. The focused CLI suite passed
**44**. The first run's recorded input SHA-256 is
`83e1bac24564cf20a32d769b7c90ec4996f57ff79aa85ec51825901c4710731a`.
Logs: `/tmp/opencode/nexuml-placement-standard-5YAUcL/pytest.log`
and `/tmp/opencode/nexuml-ray-receipt-qbI9Hz/pytest.log`.

The rerun at `ed0c5e4c` passed **3,739 tests and 604 subtests**
(16 skipped, 124 deselected) in 455.70 s, using the same standard marker
selection. Input SHA-256:
`acbcb0e6f62aca11b596452b4243c0e24fc739eb579209ac32bfe7fe831ed3e2`.
Log: `/tmp/opencode/nexuml-placement-standard-SfCkZE/pytest.log`.
The updated docs check passed on Node 22.23.4: 67 tests and 33 subtests,
zero Astro diagnostics, and rendered links, fragments, and assets verified.

The standard regression for the native CLI receipt fix (`d98bd710`)
passed **3,739 tests and 604 subtests** (16 skipped, 124 deselected) in
460.06 s. The recorded input hash matched the committed receipt change:
`6a491a8da1af70a525d65005832e06900da205dd8136846de1310ac692d0b635`.
Log: `/tmp/opencode/nexuml-native-receipt-regression-0sDVRT/pytest.log`.

The standard regression for acknowledged native rollout shutdown
(`808921d7`) passed **3,744 tests and 604 subtests** (16 skipped, 125
deselected) in 565.72 s. The recorded input hash matched the commit:
`04a105a86a5e9a1933da9a04487aa07faf71280b4b3cc3d34818df6a8a3a822c`.
Log: `/tmp/opencode/rollout-scope-standard-qBhSfr/pytest.log`.

## Spec precedence for this candidate

- The original Isaac [rollout](../add-isaaclab-gpu-robotics/specs/isaaclab-rollout-collection/spec.md)
  and [replay](../add-isaaclab-gpu-robotics/specs/isaaclab-policy-replay/spec.md)
  limits still apply to native Isaac training. Blocking CPU snapshot evaluation
  does not lift the native Isaac in-fit evaluation restriction.
- Isaac [read-only Viser requirements](../generalize-isaac-native-configuration-and-marl/specs/read-only-isaac-visualization/spec.md)
  still govern train and evaluation. Playback controls are explicit, play-only,
  and do not enable Kit or WebRTC.
- Ray Jobs recipe qualification does not qualify distributed collection or
  remote playback. Dataset local fit retains schema 1; schema 2 adds the
  separately qualified interactive actor update path.

This follows the [design's ownership and precedence](design.md#related-changes-ownership-and-precedence).
Sync only qualified deltas into authoritative specs at archival; do not
rewrite historical requirements or completion records.

OpenSpec task statuses are authoritative. Leave remaining tasks unchecked and
do not archive the change as complete.
