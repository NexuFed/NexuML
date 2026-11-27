# Acceptance matrix — recorded candidate evidence

**Planning amendment (not new execution evidence):** the 35/44 historical checked gates remain scoped to their recorded candidates. Primary distributed path is now rank-local GPU PPO via Lightning/Ray Train DDP/NCCL; GPU split/remote collection optional. See [execution-plan.md](execution-plan.md) for independent software/live gates. No new DDP, large-model, split-GPU, independent native fit-eval or external WebRTC claim follows from this amendment. Preserve local Kit/TigerVNC and headless/Viser evidence; missing live resources are lane-specific, not a global implementation stop.

See linked records for the tested source/asset hashes, interpreter and native
versions, exact invocations and raw receipts. Each result applies to its
recorded candidate. This is not a release matrix.

## Distributed learner follow-ups

See the [distributed progress ledger](distributed-progress.md) for raw
receipts, performance breakdowns, exact reproduction commands and scope
limits. The evidence below does not broaden the historical lanes.

| Lane | Result |
| --- | --- |
| Rank-local PPO, CPU/Gloo | Pass on two real ranks; matched single-process GAE/minibatches, actor and critic synchronization, fresh checkpoint continuation and coordinated rollout failure/cancellation. |
| Ray Train frame lifecycle | Pass on two same-host CPU workers and on two cluster CUDA workers with binary-only Kit-less Isaac Newton/Warp; exports and fresh resume. Full Kit on the Ray Train worker thread is unsupported. |
| CUDA NCCL | Scalar and 1 MiB collectives pass locally at one rank. The [cluster follow-up](distributed-progress.md#cluster-follow-up) also passed both collectives on two distinct GPU nodes. This is not native CUDA PPO. |
| Native PhysX via torchrun | Pass for two cluster RTX 4000 Ada pods on distinct nodes: CUDA rollouts, exact actor/critic agreement, rank-zero exports, fresh 256→384 frame continuation. No same-host two-GPU claim. |
| Independent native snapshot | Pass on a separate process and on a whole-GPU Ray task. On two distinct GPU nodes, a native Kit-less learner reached 4,096 frames while the evaluator ran; frame-zero and final snapshot receipts passed. |
| CUDA learner with Ray GPU collectors | Pass on three distinct nodes: simulator-free learner, two single-agent Newton/MJWarp collectors, synchronous object-store transport, export, and 128→192 frame resume. A single-shared-GPU transport proof also passed. Same-host split, separate native inference, grouped and full-Kit multi-node collectors remain unverified. |
| Matched native Cartpole placement profile | 7,750 warm frames/s direct, 5,481 two-node DDP and 3,420 with two remote Ray collectors plus a CUDA learner, all at the same global frame geometry. Hardware differs across layouts; no general speedup or held-out quality claim. |
| Large-width native SO-101 throughput | Pass for measurement, not for the 90% target. Median 28,575 / 99,015 / 283,886 frames/s at widths 256 / 1,024 / 4,096. |
| Two-node native SO-101 | At 1,024 environments per rank, 186,586 warm frames/s (88.9% of RSL-RL). Fixed-global scaling was slower than one GPU. Both fixed-global checkpoint replays achieved 0/32 held-out successes. |
| Selected-rank Viser | Pass on two native ranks: private browser attach/disconnect, bounded rank-one view, normal and cancellation cleanup. 63.8→86.8 ms median warm batch time in one matched Cartpole run. |
| Standalone private WebRTC playback | Pass for decoded Cartpole policy motion in the NVIDIA client over NetBird. TCP 49100 was established and the policy process owned UDP 47998. Separate runs without a client verified normal and graceful cancellation cleanup. The client run released both sockets after the deadline wrapper used SIGKILL, so graceful shutdown after a client session is not qualified. Forwarded Windows X11 remains unqualified. |
| Installed simulator-free CUDA learner | Pass from the generated workspace lock and installed wheels: 32 frames/eight optimizer steps, then fresh resume to 48 frames/12 steps and reload of both exports. This direct-PPO CI check does not qualify a clean native simulator install or a released image. |
| Direct native placement and visual pressure | Co-located PhysX and native RGB each passed the device, weight and behavior likelihood checks. Synthetic RGB terminal/truncation GAE matched the oracle on CPU and one GPU; split native/P2P and split visual qualification require a compatible GPU pair. |

Post-merge evidence at `70204139` and the follow-up checkpoint guard is
recorded in the [ledger](distributed-progress.md#post-merge-checks):
3,776 standard tests passed; single-GPU CUDA transport resumed with a
fresh native actor; browser Viser passed locally. The
[native RGB fit](distributed-progress.md#visual-and-memory-pressure-probes)
ran on one GPU; the two-device cases skipped. Prepared-image WebRTC
playback timed out before reporting a listener, independently of the
missing external client. That earlier evidence does not qualify peer or
live WebRTC execution; later peer runs are scoped above.

## Historical evidence

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
| Independent local playback | Pass for 4a.2 and 4a.3 | [Playback evidence](playback-evidence.md): state and mixed-sensor headless play, N=1 and native batches, with policy-owned history and selected recurrent/chunk state resets; loopback Viser with bounded task reset options and bounded headless N=1 PhysX GIF/MP4. Does not qualify viewer recording. |
| Independent evaluation snapshot | Pass for fresh CPU/CUDA process and same-host CPU Ray tasks | [Snapshot boundary](snapshot-boundary.md): periodic GPU inference overlapped a CPU learner and published bounded media with correct policy provenance. Actor/optimizer matched the no-evaluation control. Two-GPU, multi-node and native Isaac fit evaluation remain unqualified. |
| Native local Kit playback | Pass for private Xvfb and loopback TigerVNC, state-only PhysX | [Local viewport](playback-evidence.md#local-kit-viewport): public CLI and SDK ran 48 GPU transitions, with fixed actor state, changing viewport pixels, no editor extensions and exit 0. An authenticated TigerVNC client decoded live policy frames separately. Dev Containers forwarded X11 and WebRTC remain unqualified. |
| Ray Jobs playback | Pass remotely, headless (4a.5) | [Stock-pool evidence](playback-evidence.md#stock-pool-headless-playback-and-cancellation): locked job-local dependencies, hash-verified export replay on a fresh CUDA worker pod, eight ticks, clean SDK stop and GPU release. Separate one-host media proof; no remote viewer or recording claim. |
| MuJoCo Warp native API probe | Pass, substrate only | [GPU interop probe](mujoco-warp-probe.md): the pinned packages shared CUDA state with Torch, agreed with CPU for two substeps and reset only the selected row. No NexuML GPU environment, training or throughput claim. |
| MuJoCo Warp direct reference | Pass for pinned headless Cartpole | [GPU reference](mujoco-warp-reference.md): native resets and terminal flags, 1,024 CUDA transitions and 16 PPO minibatches, fixed actor export and fresh replay. Distributed, split placement and rendering are not qualified. |
| Matched-geometry runtime profile | Pass for scoped local measurement | [Runtime profile](mujoco-warp-reference.md#task-54--matched-geometry-measured-profile): CPU MuJoCo, CUDA Warp and CUDA PhysX each ran 1,024 transitions with common actor weights. Reports raw warm timing, versions/ABI, explicit EGL success and OSMesa gap. The CPU CI test command and opt-in hardware suite passed locally; no dynamics equivalence or speedup claim. |
| Matched native Isaac and CPU MuJoCo task | Pass, paired reference (5.3) | [Paired Cartpole](local-fit-handoff.md#paired-result): signed effort oracle, shared actor/PPO, and fresh evaluation of one exported package on both engines. Distinct dynamics and reward scales; not GPU MuJoCo learning. |
| Original matched SO-101 learning/throughput gate | Fail | [Baseline](baseline.md): zero held-out successes for both learners; NexuML achieved 73% of native warm throughput. The longer-budget follow-up did not rerun this gate. |
| Placement readiness | Pass for fixed SO-101 treatment | [Independent revisit](so101-placement-readiness.md): fresh reset seeds gave 96/96 successes for each learner, with NexuML 78.52% of native warm throughput. This warrants implementing placement, not claiming it works. |
| CPU/CUDA direct rollout transfers | Pass for Gymnasium Pendulum | [Device transfer evidence](rollout-device-transfer-evidence.md): 6,144 transitions per placement, forward and storage on CUDA, CPU learning, terminal-state/GAE parity and exact replica synchronization. Transfer cost made this small reference slower than all-CPU. |
| Explicit Ray CPU collection and policy version barrier | Pass for same-host CPU PPO | [Placement evidence](ray-rollout-placement.md): borrowed runtime, native custom resource/label scheduling, complete versioned batches, per-update synchronization and worker cleanup. No GPU or multi-node claim. |
| Worker-owned rollout scope (3.2) | Pass, one host | [Worker trace](ray-rollout-placement.md#returning-native-shutdown-32): real PhysX actors ran CUDA batches with the PPO actor/critic, acknowledged native close, and released their processes and object-store references after completion, failure and cancellation. |
| Public GPU rollout with a CPU learner | Pass, one host, state-only PhysX | [CLI and checkpoint evidence](ray-rollout-placement.md#public-one-host-cuda-rollout-with-a-cpu-learner-3435-slice): one whole-GPU reservation with native physics and rollouts; 64 frames and eight optimizer steps, export, fresh-worker resume, and object-store cleanup. |
| PhysX direct versus Ray collection profile | Pass, one host, 3.5 slice | [Matched Cartpole profile](ray-rollout-placement.md#matched-one-host-physx-profile-35-slice): 1,024 frames per lane, versioned CPU batches, tensor payload sizes and GPU samples. Finite Ray worker close was paid in the final fetch, not steady-state collection. No network or split-GPU throughput claim. |
| Split GPU and multiple-node collection | Unverified | This host exposes one GPU. The existing Ray worker groups request one GPU per pod; the only multi-GPU node reported by the Kubernetes API used V100s and driver 535, outside the pinned Isaac profile. No node access, new pool or actual split allocation was used. |
| CARLA 0.9.16 | Pass for pinned dedicated-server reference | [Live evidence](carla-live-evidence.md): exact ticks, matching camera and ego snapshots, observed collision ordering, extra-tick rejection, cleanup, and public train/evaluate/export/reload. The exported policy inferred identically without CARLA installed. |
| Modern Gazebo deterministic stepping | Pass for isolated Jazzy/Harmonic reference | [Live evidence](gazebo-exact-step-evidence.md): eight accepted native goals, exact tick/clock/sample matching, signed motion, reset epoch, two-world isolation and cancellation cleanup. The standalone proof is not a NexuML Gazebo runtime adapter. |
| Arena composed training and evaluation | Pass, headless SO-101 joint reach | [Native Arena proof](arena-native-evidence.md): shared native builder; scripted four-world reset/control/terminal witness; real PPO, exact actor initialization and trainer resume; fresh export replay and 3/3 held-out Arena successes. Not manipulation, from-scratch learning quality or remote Arena. |
| Approved three-domain physical ROS 2 and mixed objectives | Deferred, not passed | Real-robot learning is a follow-up. No approved matching real episodes available; simulated pairs and hermetic evidence do not count. |
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

Rebuilding both wheels at `29d60590` and using an isolated CPython 3.12
environment installed **107 packages as binaries**. PyTorch CPU wheel URLs
were explicit and hash-pinned; the remaining pinned packages used PyPI.
`uv pip check` found no incompatibilities. No Isaac, MuJoCo, CARLA, ROS,
Newton or LeRobot packages were installed. With an explicit caller-owned
Ray Core address and a 96 MiB object store, the installed CLI completed
**32 frames and eight optimizer steps**, then played eight CPU transitions
from its export. Ray retained no objects and stayed alive until the
owning caller shut it down. Ray warned the filesystem was over 95% full;
the run did not require spilling. Receipts:
`/tmp/opencode/nex212-installed-ray-r4c6RA/`. This remains a one-host CPU
proof, not binary-only installation of the stock Isaac worker profile.

With the local viewer code present, both distributions passed the
existing archive check. The 107-package binary CPU install loaded the
frame-256 PhysX actor from installed wheels, twice yielding the same
deterministic action with imports of Ray, Lightning, TorchRL, Gymnasium,
simulators and ROS explicitly blocked. No blocked import was attempted.
Candidate source and artifact hashes, dependency check and action:
`/tmp/opencode/nex212-local-viewer-distributions-zlzxZH/`.

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

The Arena composition regression (`532801d8`) passed **3,745 tests and
604 subtests** (16 skipped, 125 deselected) in 479.91 s. The recorded
input hash matched the commit:
`bb9f85c128f4eff679de2157549e055ccaf91e82a08ab10b09c08fe28e61df2c`.
Log: `/tmp/opencode/arena-composition-standard-5HyOdh/pytest.log`.

The combined rollout candidate (`b774aafa`) passed **3,758 tests and
604 subtests** (16 skipped, 127 deselected) in 515.40 s. The recorded
input hash matched that revision:
`09039ef361d6ef6c6a987cdd6f0aa9f4e206dde99a4b00fb02b2b0a848d25cd4`.
The later remote milestone changed only documentation and OpenSpec
evidence and passed its own docs check (67 tests, 33 subtests, zero Astro
diagnostics, rendered links verified).
Logs: `/tmp/opencode/rollout-final-standard-LWXuuh/pytest.log` and
`/tmp/opencode/remote-play-docs-yAehUP/docs.log`.

The local Kit candidate (`e38e4d14`) passed **3,760 tests and 604
subtests** (16 skipped, 128 deselected) with the same standard marker
selection, in 474.79 s. The before/after input hash matched the committed
candidate:
`9d2b868cb6020b7847d87e3ae86b7f50086589ad597eb91be95b12f23ae1753a`.
Ruff and type checking of the changed product modules passed. Whole-tree
`ty` in this prepared environment reported the same 126 diagnostics on
the candidate and its pre-viewer baseline, including unavailable ROS and
Kubernetes modules and pre-existing test/library typing diagnostics.
Docs and strict change validation passed. Logs:
`/tmp/opencode/nex212-local-viewport-standard-94ChtB/`,
`/tmp/opencode/nex212-ty-baseline-ivndkn/` and
`/tmp/opencode/nex212-local-viewport-docs-pjXs6q/`.

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
