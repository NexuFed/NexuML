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
| Native Kit Local/X11/TigerVNC and WebRTC playback | Unqualified | A native X11/RDP Kit probe ran actual policy ticks but failed to qualify a bounded display surface. No approved, end-to-end native WebRTC stream or TigerVNC reference. |
| Remote Ray Jobs playback | Transport verified; GPU acceptance open | CLI/SDK worker dispatch has no local fallback. The corrected producer hit the Ray supervisor admission timeout with no driver. Local signal cleanup does not qualify remote playback, cancellation or viewer routing. |
| MuJoCo Warp native API probe | Pass, substrate only | [GPU interop probe](mujoco-warp-probe.md): the pinned packages shared CUDA state with Torch, agreed with CPU for two substeps and reset only the selected row. No NexuML GPU environment, training or throughput claim. |
| MuJoCo Warp direct reference | Pass for pinned headless Cartpole | [GPU reference](mujoco-warp-reference.md): native resets and terminal flags, 1,024 CUDA transitions and 16 PPO minibatches, fixed actor export and fresh replay. Distributed, split placement and rendering are not qualified. |
| Matched-geometry runtime profile | Pass for scoped local measurement | [Runtime profile](mujoco-warp-reference.md#task-54--matched-geometry-measured-profile): CPU MuJoCo, CUDA Warp and CUDA PhysX each ran 1,024 transitions with common actor weights. Reports raw warm timing, versions/ABI, explicit EGL success and OSMesa gap. The CPU CI test command and opt-in hardware suite passed locally; no dynamics equivalence or speedup claim. |
| Matched native Isaac and CPU MuJoCo task | Pass, paired reference (5.3) | [Paired Cartpole](local-fit-handoff.md#paired-result): signed effort oracle, shared actor/PPO, and fresh evaluation of one exported package on both engines. Distinct dynamics and reward scales; not GPU MuJoCo learning. |
| Original matched SO-101 learning/throughput gate | Fail | [Baseline](baseline.md): zero held-out successes for both learners; NexuML achieved 73% of native warm throughput. The longer-budget follow-up did not rerun this gate. |
| Placement readiness | Pass for fixed SO-101 treatment | [Independent revisit](so101-placement-readiness.md): fresh reset seeds gave 96/96 successes for each learner, with NexuML 78.52% of native warm throughput. This warrants implementing placement, not claiming it works. |
| CPU/CUDA direct rollout transfers | Pass for Gymnasium Pendulum | [Device transfer evidence](rollout-device-transfer-evidence.md): 6,144 transitions per placement, forward and storage on CUDA, CPU learning, terminal-state/GAE parity and exact replica synchronization. Transfer cost made this small reference slower than all-CPU. |
| Explicit Ray CPU collection and policy version barrier | Pass for same-host CPU PPO | [Placement evidence](ray-rollout-placement.md): borrowed runtime, native custom resource/label scheduling, complete versioned batches, per-update synchronization and worker cleanup. No GPU or multi-node claim. |
| Worker-owned rollout scope (3.2) | CPU lifecycle verified | [Worker trace](ray-rollout-placement.md#worker-owned-scope-32-cpu-slice): two workers independently closed their runtimes before leaving their scopes, including learner failure and cancellation. No native Kit or GPU resource lifecycle claim. |
| Split GPU and distributed collection | Unverified | Separate collector and single-learner configuration is supported. This host exposes one GPU. The existing Ray worker groups request one GPU per pod; the only multi-GPU node reported by the Kubernetes API used V100s and driver 535, outside the pinned Isaac profile. No node access, new pool or actual split allocation was used. |
| CARLA 0.9.16 | Pass for pinned dedicated-server reference | [Live evidence](carla-live-evidence.md): exact ticks, matching camera and ego snapshots, observed collision ordering, extra-tick rejection, cleanup, and public train/evaluate/export/reload. The exported policy inferred identically without CARLA installed. |
| Modern Gazebo deterministic stepping | Pass for isolated Jazzy/Harmonic reference | [Live evidence](gazebo-exact-step-evidence.md): eight accepted native goals, exact tick/clock/sample matching, signed motion, reset epoch, two-world isolation and cancellation cleanup. The standalone proof is not a NexuML Gazebo runtime adapter. |
| Arena composed training and evaluation | Blocked | Compatible prepared Arena runtime absent; hermetic evaluator protocol is not native composition or scoring. |
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
Task 6.3 remains open for the unqualified split and distributed recipes,
not for public release delivery.

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
