# Distributed RL implementation ledger

Baseline: `fdf6ad93`. OpenSpec starts at 35/44 tasks complete. Preserve previous
native and viewer evidence; none of it qualifies co-located DDP.

- [x] Prove two real CPU/Gloo ranks against single-process PPO with matched
  minibatches. Check rank seeds, global counters, resume and failure.
- [x] Exercise the lifecycle through two CPU Ray Train ranks and run the
  available one-rank NCCL preflight. Multi-GPU proof stays separate.
- [x] Test available device placements and native snapshot evaluators.
  Keep live viewing optional and preserve the existing Kit playback path.
- [x] Measure equivalent co-located Isaac and RSL-RL at larger vector widths.
  Record throughput before deciding whether an optimization is warranted.
- [x] Run integrated regressions and record exact commands and outstanding
  resource gates; leave unqualified acceptance tasks unchecked.

Available locally: one RTX 4080 SUPER and the existing pinned Isaac runtime.
Two local CUDA devices, real robot episodes, and an operator-side WebRTC
client are not available here. Remote Ray capacity must be checked through
the approved job workflow, not node access or idle inventory.

## Learner milestone

- Two real CPU/Gloo ranks consumed 32 global Cartpole frames in two
  rollouts. Both actor and critic matched across ranks bit-for-bit and agreed
  with single-process replay of exact minibatch unions (separate GAE
  trajectories, same normalization), within `atol=2e-6, rtol=2e-5`.
  Each Adam optimizer reached eight steps. Partial resume restored both
  optimizers, built environments from fresh disjoint seeds, and completed the
  remaining 16 frames. An injected step exception on rank 1 stopped both
  ranks before any update.
- Actual Ray Train (2.57.0), two same-host CPU workers: frame-based fit,
  rank-zero package/checkpoint, and resume from a completed 32-frame
  checkpoint to 48 frames / 24 combined optimizer steps. Not multi-node.
- One actual CUDA/NCCL rank passed scalar and 1 MiB all-reduces. The
  two-local-rank probe stopped at device assignment: one visible GPU. The
  Ray Jobs HTTPS version request failed certificate validation with the
  default trust store (curl 60); `RAY_CA_BUNDLE` was not supplied. No TLS
  bypass, cluster changes, or implicit retries.
- Native Isaac snapshot process passed on CUDA after publishing its receipt
  before primary Kit teardown. The first attempt used a returning Kit close
  and segfaulted during Python finalization. The Ray GPU evaluator uses the
  same fresh child boundary, not Kit in the Ray task process.
- Regression: **169 passed**, including opt-in real Ray Train, bounded
  action likelihood parity across four dtypes, and CUDA validation. Eight
  existing native transition, SO-101 action, and terminal pixel gates passed.

```bash
NEXUML_TEST_RAY_TRAIN=1 TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/execution/test_rl_ddp.py \
  tests/core/test_policy.py tests/core/test_isaac_webrtc.py \
  tests/core/test_isaaclab_preflight.py tests/core/test_isaaclab_definition.py

NEXUML_RUN_SIMULATOR_TESTS=1 TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/core/test_isaaclab_integration.py \
  -k 'snapshot_evaluator_uses_its_own_gpu_process or native_manager_and_direct_transition_oracle or native_so101_signed or terminal_pixels'
```

## Measured native throughput

On one RTX 4080 SUPER with Isaac Sim 6.2.0.0, Isaac Lab 3.0.0
(`70ab24fec3eabf51222817db0cdb5a4e7133bd7e`), Torch 2.11.0+cu130,
TorchRL 0.14.0, TensorDict 0.14.2, and RSL-RL 5.5.1. Both learners used
the same native SO-101 joint-reach task and reward, 32 ticks per rollout,
two PPO epochs, four minibatches per epoch, seeds 17 and 43, no visualizer,
and the same number of transitions within each pair. RSL-RL uses its
maintained unbounded Gaussian with the native action clip, one optimizer
and one gradient clip. NexuML uses a bounded squashed Gaussian and
separate actor/critic optimizers and clips. This is a throughput smoke, not
a new held-out learning-quality comparison.

Median over the two per-seed warm rates (discard rollouts 1–2 of each
12-rollout fit):

| Native environments | Frames per fit | RSL-RL frames/s | NexuML frames/s | NexuML / RSL-RL | NexuML collect/update per rollout |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 98,304 | 32,694 | 28,575 | 87.4% | 250 ms / 37 ms |
| 1,024 | 393,216 | 125,859 | 99,015 | 78.7% | 288 ms / 43 ms |
| 4,096 | 1,572,864 | 457,140 | 283,886 | 62.1% | 420 ms / 43 ms |

RSL-RL collect/update times were 225/26, 242/23 and 267/21 ms.
PyTorch peak reserved CUDA memory was 52/80/248 MiB for NexuML and
28/50/118 MiB for RSL-RL; the device-wide sampler observed up to
4,496 MiB (includes background and non-Torch allocations). No 90%
claim. Baseline NexuML rates were 24,920 / 81,171 / 235,663 frames/s;
relative rates varied with run ordering and native timing. Final NexuML
absolute throughput improved at all three widths. Do not infer a generic
distributed speedup or visual-policy throughput from this state-only run.
Fresh held-out replay at width 1,024 gave **0/64** final successes per
learner (32 episodes each for seeds 17 and 43, also zero at initialization).
All episodes timed out. Mean terminal joint error fell from 0.177 to
0.106 rad for RSL-RL and 0.164 to 0.141 rad for NexuML.
The timing budget did not reach the task's learning target. This does not
replace the previously qualified longer-budget state-policy evidence.

Nsight at width 1,024 measured 11,040 baseline NexuML stream syncs vs
3,592 RSL-RL (including launch/setup). The final NexuML trace measured
6,909 synchronizations and 88 ms in that API, down from 320 ms. Py-spy
at width 4,096 found
per-distribution recomputation/synchronization of fixed unit bounds and
the dominant native physics step. The targeted fixes preserve the bounded
sample, fresh likelihood and action-validation semantics, keep minibatch
metrics on the learner GPU, and skip publishing weights to the same
in-process actor. No terminal data, validation, or cross-device policy
barrier was removed.

Raw measurements: `/tmp/opencode/nexuml-wide-ppo-baseline-9IXF1K`,
`/tmp/opencode/nexuml-wide-ppo-candidate-gP8M1U`,
`/tmp/opencode/nexuml-wide-ppo-final-l1MxPL`,
`/tmp/opencode/nexuml-rl-nsys-Aydsvv`, and
`/tmp/opencode/nexuml-rl-cpu-samples-AutsrX`. Final Nsight/CPU–CUDA
transfer probe: `/tmp/opencode/nexuml-rl-final-profile-mY2YVs`.

```bash
for width in 256 1024 4096; do
  for seed in 17 43; do
    for backend in rsl-rl nexuml; do
      flock /tmp/opencode/nexuml-decouple-native-gpu.lock \
        env TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 \
        .venv/bin/python scripts/isaaclab/so101_learning_comparison.py "$backend" train "$seed" \
        "$EVIDENCE/$backend-$width-$seed" --width "$width" --rollouts 12
    done
  done
done
```

For held-out replay, use the same `backend`, training `seed` and existing
output directory in `so101_learning_comparison.py "$backend" evaluate "$seed" "$OUTPUT"`.

### Two-node SO-101 scaling

`nex212-peer-bad1346c` used one RTX 4000 Ada per pod, the same task,
driver and pinned software, seed 17 with rank-local seed offsets, eight
rollouts of 32 ticks, and 64 minibatch updates per fit. Warm rates use the
slower rank's median collect-plus-update time after the first two
rollouts. These are aggregate physical frames, not per-rank rates.

| Layout | Global frames | NexuML frames/s | RSL-RL frames/s | NexuML / RSL-RL |
| --- | ---: | ---: | ---: | ---: |
| 1 × 1,024 | 262,144 | 105,561 | 118,993 | 88.7% |
| 2 × 512 (fixed global batch) | 262,144 | 96,971 | 112,294 | 86.4% |
| 2 × 1,024 (fixed per-rank batch) | 524,288 | 186,586 | 209,796 | 88.9% |

With 1,024 environments per rank, NexuML scaled 1.77× (88.4% weak
scaling efficiency); RSL-RL scaled 1.76× (88.2%). At fixed global width,
two ranks delivered 91.9% and 94.4% of their respective single-GPU rates.
Rank-zero median collect/update times in the wide two-rank lane were
285/65 ms for NexuML and 262/49 ms for RSL-RL, with 82 and 50 MiB of peak
Torch-reserved CUDA memory per rank. Simulator memory is not included.
Rank-zero pod `eth0` sent/received 2.91/2.90 MiB (NexuML) and
2.80/2.77 MiB (RSL-RL) across each eight-rollout wide run. These counters
include native startup/teardown traffic; they are not isolated NCCL bytes.

The throughput run did not replay the exports. A separate fixed-global
repeat (`nex212-peer-d1c34b82`) evaluated both rank-zero exports in fresh
native processes on 32 held-out seeds. Both learners scored 0/32 at
initialization and after fitting; every episode timed out. Mean terminal
joint error improved from 0.158 to 0.138 rad for NexuML and 0.163 to
0.127 rad for RSL-RL. This short budget does not establish task success.
Replay logs: `/tmp/opencode/nexuml-peer-so101-heldout-Q5Cq8B/`.
The harness writes rank-local diagnostic exports; the production rank-zero
export path is verified separately.
Both Jobs completed and were removed. Reproduce from the repository root
with `bash "$EVIDENCE/run-pods.sh" "$EVIDENCE"`, after copying the benchmark
harness and inputs from `/tmp/opencode/nexuml-peer-so101-live-4iCzNc/` to
a fresh `$EVIDENCE` directory.

A traced fixed-global repeat (`nex212-peer-76b666ce`, NCCL 2.28.9,
`NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL`) measured **2,838,732 bytes**
of logical all-reduce input per NexuML rank (230 calls) versus
**2,755,332 bytes** for RSL-RL (65 calls). Main gradient payload was
64 × 11,083 FP32 elements versus 64 × 10,763; NexuML also synchronized
global advantage moments and small status and log tensors. Counts include
collective startup but not TCP overhead. The trace contains no rollout
buffers. Trace: `/tmp/opencode/nexuml-peer-so101-nccl-smh315/`.

Batching PPO logs into one vector mean cut all-reduces from **230 to 166
per rank** (2,838,604 input bytes), with the 64 gradient buckets and 64
advantage-moment reductions unchanged. Both native pods finished the
matched repeat (`nex212-peer-fcd5402c`). Real CPU/Gloo ranks retained
metric and parameter parity with the single-process minibatch reference.
This traced run measures collective count, not a new throughput gain.
Wheel hashes and trace: `/tmp/opencode/nexuml-peer-so101-metric-vector-XhSw9L/`.

## Additional acceptance at `d8579994`

- Ray Train V2 trained and resumed native Isaac Newton/Warp Cartpole on
  one allocated RTX 4080 SUPER. Fit reached 32 frames and resumed via a
  Ray-staged checkpoint to 48 frames / 48 combined optimizer steps.
  Both jobs owned fresh simulators. This was one CUDA rank, not DDP.
  Full Kit is rejected under Ray Train V2's background training thread
  because the pinned native launcher registers process signal handlers.
  Full Kit's main-thread launch remains available via torchrun.
- The standalone native CUDA snapshot and the Ray-scheduled native CUDA
  snapshot passed using their own processes. The Ray receipt recorded a
  whole GPU assignment and matched digest, frame, batch and result.
  This does not establish simultaneous training and evaluation on
  independently reserved GPUs.
- A CUDA learner and a PhysX GPU Ray collector shared the one local
  physical GPU **only as a transport test**: 32 frames, eight optimizer
  steps, synchronous CPU-staged rollouts and the borrowed Ray runtime
  survived teardown. This is not a separate-GPU or multi-node result.
  After the Ray publication change, all three native checkpoint/resume
  cases passed again (PhysX CPU learner, PhysX CUDA learner, Newton CUDA
  learner). An earlier suite left a `ray::Collector` SIGSEGV core; the
  instrumented repeat showed PhysX context cleanup and no repeat crash in
  the retained worker logs. Do not extend this to full-Kit cluster claims.
  Receipts: `/tmp/opencode/nexuml-native-publication-logs-fv8CQO/`.
- Built and installed both application wheels into a dedicated learner
  environment without Isaac Sim/Lab, MuJoCo/Warp or ROS. TorchRL 0.15.0
  directly trained for 32 frames/eight steps on CUDA; the installed CLI
  exported and replayed eight CPU transitions. The separately tested
  TorchRL 0.14.0 remote rollout profile still needs its lifecycle shim:
  the audited 0.15 source still uses `ray.kill` for collectors and starts
  them with an unlimited budget. No broad dependency upgrade or override.

Receipts: `/tmp/opencode/nexuml-native-ray-ddp-J50Adh/`,
`/tmp/opencode/nexuml-cuda-rollout-transport-baXv4z/`, and
`/tmp/opencode/nexuml-rl-wheel-Zxby1Z/final/`.
The unchanged lock passed `uv lock --check --offline`.

### Direct versus Ray transport on one GPU

Matched Newton/MJWarp Cartpole on the same RTX 4080 SUPER, 16
environments, 512 frames per batch, 4,096 total frames and 64 optimizer
steps. Direct CUDA reached **7,337 warm frames/s**; one Ray CUDA collector
sharing the learner GPU reached **5,632 frames/s** (23% lower). Median
collect/update were 55.2/14.6 ms direct versus 67.9/22.8 ms Ray.
The Ray learner received 41,984 logical TensorDict bytes per rollout and
copied them CPU-to-GPU in a median 0.18 ms; direct CUDA held 33,792 bytes
with no host staging. Each published actor had 5,128 tensor bytes. Learner
Torch-reserved memory peaked at 22 MiB in both lanes; this excludes the
separate actor and object store. These are same-GPU transport costs, not
multi-node, dedicated-GPU or task-success conclusions.

Reproduction and raw phase timings:
`/tmp/opencode/nexuml-newton-ray-transport-profile-UfgtF4/`.

```bash
for mode in direct ray; do
  flock /tmp/opencode/nexuml-decouple-native-gpu.lock \
    env -u RAY_ADDRESS -u RAY_API_SERVER_ADDRESS TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 \
    .venv/bin/python "$EVIDENCE/rollout_timing.py" "$mode" "$OUTPUT/$mode.json"
done
```

### Matched cluster topology profile

`nex212-native-profile-f06ddadd` reused the binary Newton/MJWarp Cartpole
profile with 16 total environments, 512 global frames per batch, 4,096
frames and 64 optimizer steps. Direct and remote used the same RTX 4070
learner; the two-node DDP lane used the same RTX 4000 Ada and RTX 5080
nodes as the two remote collectors.

| Layout | Warm frames/s | Collect/update ms | Logical rollout bytes | Batch transfer ms | Learner CUDA reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct CUDA | 7,750 | 52.5 / 13.6 | 33,792 on GPU | 0.018, no host staging | 22 MiB |
| Two Ray collectors + CUDA learner | 3,420 | 85.1 / 43.3 | 41,984 through Ray | 0.323 CPU-to-GPU | 22 MiB |
| Two-node co-located DDP | 5,481 | 72.2 / 21.1 | 16,896 per GPU | 0.029, no host staging | 24 MiB per rank |

Warm rates use median batch intervals after the first two batches;
collect/update columns are separate medians. Ray published 5,128 logical
weight bytes per batch to both collectors. These are tensor sizes and
learner allocator peaks, not wire bytes or simulator/object-store peaks.
The learners used matched task and frame geometry, not identical GPU
models across layouts. This is one short systems run, not a learning
quality or speedup claim. [The earlier one-host PhysX profile](ray-rollout-placement.md#matched-one-host-physx-profile-35-slice)
contains GPU sampling, object-store cleanup and episode outcomes for that
profile only. Same-host two-GPU split remains open under 2.2 and 2a.4.

The job succeeded and all four owned collector and DDP actors stopped.
Use the Jobs replay command below with the inputs, scripts and hashes in
`/tmp/opencode/nexuml-ray-newton-topologies-BmMf6B/`.

## Visual and memory-pressure probes

Native headless sensor and terminal-pixel oracles passed (**3 tests**).
Separately, the public CLI trained the 32×32 native RGB Cartpole policy
and critic on one GPU for 128 frames and eight optimizer steps, and loaded
the resulting export. This proves the image-input fit path, not learning
quality. The split learner and split inference cases skipped for lack of a
second GPU. Log: `/tmp/opencode/nexuml-rgb-ppo-public/`.
The separate synthetic RGB probe ran a small convolutional actor, same-GPU
inference and GPU Adam with exact publication of the modified weights.
At `256×3×64×64` and `2048×3×128×128`, uint8 rollout storage was 3 and
96 MiB; each replica and the learner gradients used 1,928 bytes apiece
and Adam state used 3,872 bytes. Peak additional CUDA allocation during
the first inference/update was 9.9/12.4 MiB and 44/68.1 MiB, respectively.
These are cold phase timings and allocator peaks, not VLA throughput
or an estimate of engine/renderer memory. Split native image-policy
learning and 2-GPU transfer/latency evidence are still unqualified.

```bash
NEXUML_RUN_SIMULATOR_TESTS=1 TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/core/test_isaaclab_integration.py \
  -k native_ppo_placement
.venv/bin/python tests/fixtures/visual_rollout_pressure.py --frames 2048 --side 128 --batch 64
.venv/bin/python tests/fixtures/visual_rollout_pressure.py --frames 2048 --side 128 \
  --batch 64 --channels 128
# On a compatible two-GPU host, re-run the same workload with explicit roles:
.venv/bin/python tests/fixtures/visual_rollout_pressure.py --frames 2048 --side 128 \
  --batch 64 --simulation cuda:0 --inference cuda:0 --learner cuda:1
.venv/bin/python tests/fixtures/visual_rollout_pressure.py --frames 2048 --side 128 \
  --batch 64 --simulation cuda:0 --inference cuda:1 --learner cuda:1
```

The native split cases in this test use the state-only PhysX reference.
The native RGB reference is Newton and remains co-located by provider
validation; the separate synthetic RGB fixture exercises explicit devices
without asserting native camera qualification.

The placement test checks actual rollout leaves before and after the
learner transfer, exact field values on split transfers, collector and
critic devices, and current policy weights and behavior log likelihood
on both batches. The exported frame count and collector teardown are
checked as well. The enhanced test passed for co-located native RGB and
PhysX; its two split cases skipped on one visible GPU. Logs:
`/tmp/opencode/nexuml-native-split-oracle/`.

A separate six-transition synthetic RGB TensorDict now checks TorchRL
GAE after transfer to the configured learner, before flattening.
The oracle distinguishes true termination from timeout bootstrapping and
pre-reset pixels from the next episode. It passed on CPU and `cuda:0`;
the 96 MiB pressure run still measured 68.1 MiB peak additional update
allocation, with all phases recorded in
`/tmp/opencode/nexuml-visual-gae-6GLgAQ/visual-pressure.json`.
The separate real CUDA/Ray worker regressions passed (2 tests), including
CPU collection and a native Newton GPU worker. Neither is a same-host
split GPU receipt. Logs: `/tmp/opencode/nexuml-split-regressions-Q1V7wj/`.

Follow-up native placement oracle (§2.2): the PhysX case ends one real
episode by leaving the track and another by timeout. It checks actual
native/exported frames, terminal vs reset observations on the learner,
and both GAE batches against a backward recurrence and a fresh critic
evaluation of the ending observations. Two co-located cases passed;
both split cases skipped. Admission negatives passed (84 tests). The
local RTX 4080 SUPER and active Ray GPU worker each expose one GPU; the
only multi-GPU Kubernetes node in the inventory has V100s and driver
535.247.01, incompatible with the pinned CUDA 13 stack. No suitable
same-host allocation was available. Inventory, commands and results:
`/tmp/opencode/nex-212-task22.7cmdBm/` and
`/tmp/opencode/nexuml-task22-integration-uZDg7O/`.

Follow-up synthetic visual placement (§2a.4): checked all bytes of the
patterned RGB batch and inference parity across two weight publications.
The JSON reports logical payloads separately from payloads crossing
CUDA devices, per-phase timings and allocated/reserved peaks, and labels
peer capability separately from the actual transfer route.
On one RTX 4080 SUPER, 16- and 128-channel runs each stored 96 MiB of RGB
with zero cross-device payload; second-pass update peaks above baseline
were 60.0 and 384.1 MiB. The 32,768-channel small-image case used
3,932,168 bytes per model replica and 7,864,352 bytes of Adam state.
Oversized or unsupported placements reject before allocating the model;
native Newton RGB split, native Isaac FSDP and mixed-precision RL DDP
remain unsupported. No native split visual, PEFT or VLA claim.
Receipts: `/tmp/opencode/nex212-visual-pressure-VkjDMm/`.

Raw measurements: `/tmp/opencode/nexuml-visual-pressure-0plGFI/`.

```bash
export TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 RAY_TRAIN_V2_ENABLED=1
mkdir -p "$EVIDENCE"
# Requires one GPU and 140 MiB of free Ray object-store capacity per command.
flock /tmp/opencode/nexuml-decouple-native-gpu.lock \
  .venv/bin/python tests/fixtures/rl_ddp_worker.py "$EVIDENCE" ray-native
NEXUML_RUN_SIMULATOR_TESTS=1 NEXUML_TEST_RAY_NATIVE_SNAPSHOT=1 \
  flock /tmp/opencode/nexuml-decouple-native-gpu.lock .venv/bin/python -m pytest \
  -q tests/core/test_isaaclab_integration.py -k snapshot_evaluator
```

## Pending acceptance and reproduction

| Task | Software status | Live status |
| --- | --- | --- |
| 2.2 | Explicit GPU roles, validation, and TorchRL device transfers implemented | Separate-device and P2P receipts need a compatible GPU pair |
| 2a.2–2a.3 | NCCL preflight and matched benchmark runnable | Two-node fit/resume, strong/weak scaling, fixed-global held-out outcomes and NCCL payloads measured; same-host peer still unqualified |
| 2a.4 | Synthetic pressure, transfer, terminal/GAE and weight-parity fixture; public native RGB fit fixture | Co-located native image-input fit passes; split visual and large-model results unqualified |
| 3.4–3.5 | Actor admission, CUDA learner, published weights and native checkpoint/recovery proven with local resources | Three-node fit, export and resume passed with a simulator-free GPU learner; matched direct, remote and two-node DDP rates measured. Grouped Ray rollout unsupported |
| 4.4 | Rank-zero snapshots, separate process/Ray GPU evaluation and bounded cancellation | Native Kit-less PPO and separate-node GPU snapshot evaluation overlapped; final snapshot drained |
| 4a.1, 4a.4, 4a.6 | Typed viewer selection, restricted native Kit toolbar, endpoint admission, native headless RGB and selected-rank/collector Viser routing tested | Two-node Viser attach, disconnect and cancel passed. NVIDIA client decoded and displayed moving Cartpole policy playback via private NetBird. Socket release passed in separate runs. Forwarded Windows X11 remains unqualified. |
| 6.3 | Generated workspace lock drives the installed CUDA learner smoke in existing opt-in GPU CI; role interpreter selection implemented | Staged binary Kit-less profile passed on stock workers; clean workspace-lock native worker installs and full-Kit dependency closure still unqualified |
| 1b, 6.2a | Existing simulator-only SDK, ROS 2 adapter and safety checks retained | Real-robot learning deferred, not passed; needs authorized matching data if revisited |

**Same-host peer NCCL and co-located native DDP:** requires two compatible
GPUs in one worker, not the two-node topology already tested. On an
allocated worker with two visible GPUs, run the actual collective first.
The same preflight can be launched across nodes using an approved torchrun
rendezvous instead of `--standalone`.
The available RTX nodes each advertise one GPU. The inventory's eight-GPU
host reports V100 devices and a 535-series driver, outside the pinned
Isaac/CUDA 13 lane; separate pods on RTX nodes do not qualify same-host P2P.

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/isaaclab/nccl_preflight.py
.venv/bin/python - <<'PY'
from nexuml.core.config import ResolvedConfig
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo

spec = isaac_cartpole_ppo(total_frames=2048, frames_per_batch=1024, num_envs=64)
spec.training.devices = 2
spec.training.strategy = "ddp"
spec.training.collector.env_device = "cuda"
spec.training.collector.storing_device = "cuda"
ResolvedConfig.from_scenario(spec).save("peer-ddp.yaml")
PY
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m nexuml.execution.worker --config peer-ddp.yaml
```

**Same-host split and separate inference (2.2):** requires two compatible
logical GPUs. Compare to the same command without the device overrides,
then run the placements below. Capture actual allocator/transfer and terminal
state evidence; CLI completion alone does not establish peer-to-peer copies.

```bash
NEXUML_RUN_SIMULATOR_TESTS=1 TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/core/test_isaaclab_integration.py \
  -k native_ppo_placement
for inference in cuda:0 cuda:1; do
  nsys profile --trace=cuda,nvtx --output="$EVIDENCE/split-${inference#cuda:}" \
    .venv/bin/nexuml train isaac-cartpole-ppo \
    -O training.devices=1, -O training.collector.env_device=cuda:0 \
    -O training.collector.storing_device=cuda:0 \
    -O "training.collector.policy_device=$inference" \
    -O training.total_frames=2048 -O training.frames_per_batch=1024
  nsys stats --report cuda_gpu_mem_size_sum "$EVIDENCE/split-${inference#cuda:}.nsys-rep"
done
```

**Optional same-host native fit/evaluation overlap (4.4):** on a two-GPU worker,
run a single native fit on `cuda:0` with snapshots in a child on `cuda:1`.
Capture concurrent process/device activity, snapshot frame/digest receipts,
and cleanup on completion and cancellation; command success alone is not
an overlap measurement.

```bash
.venv/bin/python - <<'PY'
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import ProcessEvaluationSpec, RLEvaluationSpec
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo

spec = isaac_cartpole_ppo(total_frames=8192)
spec.training.evaluation = RLEvaluationSpec(
    episodes=1, at_start=True, every_n_frames=2048, max_steps_per_episode=16,
    process=ProcessEvaluationSpec(device="cuda:1", scheduling="skip_if_busy"),
)
ResolvedConfig.from_scenario(spec).save("native-overlap.yaml")
PY
.venv/bin/nexuml train --config native-overlap.yaml
```

**Distributed Viser (4a.6):** after the headless peer DDP test, generate
the same `peer-ddp.yaml` with this addition before saving, and rerun its
torchrun command with a prepared private loopback browser route. Verify
only rank 1 listens, disconnect without changing training counters or
policy, then cancel and confirm the listener closes.

```python
from nexuml_library.environments.isaaclab_config import ViserSpec
env = spec.interaction.environment
spec.interaction.environment = env.model_copy(
    update={"viewer": ViserSpec(train_rank=1, port=17391, max_visible_envs=2)}
)
```

**Multi-node independent collection (3.4/3.5) and CI profile (6.3):**
Use the separately pinned learner and native worker inputs from the
[three-node Kit-less fit and resume](#cluster-follow-up). The replayable
Jobs harness verifies actual actor nodes, CUDA learner batches, versions
and exported frames; logical device indices alone do not prove placement.
That receipt does not qualify grouped remote tasks or stock-worker full Kit.
Use the authenticated dashboard tunnel below if external ingress trust is
unavailable.

```bash
curl --fail --cacert "$RAY_CA_BUNDLE" "$RAY_API_SERVER_ADDRESS/api/version"
.venv/bin/nexuml backend preflight --config "$QUALIFIED_COLLECTOR_CONFIG"
REQUESTS_CA_BUNDLE="$RAY_CA_BUNDLE" .venv/bin/nexuml train \
  --config "$QUALIFIED_COLLECTOR_CONFIG"
```

The config uses `ray-jobs-sdk` with one learner GPU, `collector.backend:
ray`, `num_collectors: 2`, `collector.ray.num_gpus: 1`, actor role
selectors, prepared `py_executable` and CPU object-store storage. Also
reserve the evaluator GPU if testing periodic native snapshots. Do not
rerun with TLS disabled or on the idle-inventory assumption.
For full Kit on stock workers, the
[wheel audit](ray-rollout-placement.md#worker-dependency-setup-diagnostic-34-not-native-qualification)
records `idna-ssl==1.1.0`, `antlr4-python3-runtime==4.9.3` and
the exact Isaac Lab source pin. The last two have reviewed image wheels;
full Kit still needs an approved stock-worker binary closure.

**WebRTC runtime and media (4a.4):** run the ordinary `nexuml play` command in the
[WebRTC how-to](../../../docs/src/content/docs/how-to/isaac-native-configuration.mdx#native-webrtc-playback)
on an approved private graphics worker, with Vulkan/NVENC, TCP 49100 and
UDP 47998. Require client-decoded policy motion and released sockets on
exit. Prior TigerVNC playback remains qualified; a Kit editor session
or open signaling listener does not count as a policy stream.

```bash
timeout -k 10s 300s .venv/bin/nexuml play "$POLICY" --scenario-file "$PLAY_SCENARIO" \
  --viewer webrtc --device cuda:0 --num-envs 1 --steps 120 --real-time
```

The installed-wheel playback test in the prepared Isaac image reached
Vulkan/RTX initialization but timed out twice (150 s and 100 s), using
both the explicit minimal rendering experience and the maintained Lab
streaming selection. Neither showed a signaling receipt, policy ticks,
or a client connection. Kit logged OmniHub and NGX problems; the logs do
not establish the low-level graphics cause. A separate native-only
`KitLauncher` run through the image's `/usr/local/bin/nexuml-isaac` entrypoint
also timed out (85 s), with no NexuML imports, task, policy or local-toolbar
settings. Its 45 s traceback stopped in Isaac Sim's
`SimulationApp._wait_for_viewport` → `_update_without_ready`. This
reproduces the **prepared-runtime startup failure** outside the training
and playback code. First rerun `kit_only.py` in the approved graphics
profile; require `KIT_ONLY_SIGNAL_LISTENER` before repeating the playback
test. The approved external media client is still a separate gate.
No host ports were published. Script and stack:
`/tmp/opencode/nexuml-kit-only-stream-3dYSbY/`. Playback receipts:
`/tmp/opencode/nexuml-stream-configured-6hHQ5o/` and
`/tmp/opencode/nexuml-stream-native-experience-Zb1wOX/`.

The standalone NetBird pod recipe is documented in
[`.k8s/isaac/README.md`](../../../.k8s/isaac/README.md#try-standalone-policy-playback).
A fresh protected vendor-control Job (`nex212-vendor-gkkfl1`) did not
start: the scheduler reported insufficient GPUs/memory on the cached
candidate nodes. The Job was deleted without using the existing
workstation pod or changing network policy. The same vendor command
in a local Docker container with no network reached renderer startup
but did not produce the vendor ready marker in the retained log.
Docker attach did not capture the harness's socket checks, so this is
not a signaling or NetBird media receipt.
Manifests, scheduler events, launcher script and logs:
`/tmp/opencode/nexuml-vendor-stream-control-gkKfL1/` and
`/tmp/opencode/nexuml-local-vendor-stream-control-hKK9G2/`.
The repeat captured socket checks inside the isolated container. The
vendor CLI opened TCP 49100 on the RTX 4080 SUPER (driver 610.43.03).
The immediate shutdown check still found that listener after the
wrapper stopped, so the test removed its container without starting
the paired Python process. This qualifies signaling only, not the
vendor ready marker, graceful release, or decoded media.
Receipt: `/tmp/opencode/nexuml-vendor-control-receipt-y4CEgm/`.
In a separate network-isolated container on the same local GPU/image,
the current NexuML wheels also opened TCP 49100 (28 s) before app
initialization returned. At 75 s the watchdog found
`SimulationApp.__init__` still blocked in `_update_without_ready`.
The process exited without policy ticks or the CLI endpoint receipt,
and the socket was closed. A bound port is not evidence that Kit was
ready or that a client could decode media; the remaining failure is
after socket bind, before usable playback.
Receipt: `/tmp/opencode/nexuml-webrtc-signaling-parity-FJNO2S/`.

The same pinned image on an allocated RTX 4000 Ada (`580.126.20`) reproduced
the short-bound result: installed playback bound TCP at 29 s, and the vendor
editor at 31 s. Neither reached a ready receipt in the test window. The
ingress-denied task Job was deleted; it was not a NetBird client test.
Receipt: `/tmp/opencode/nexuml-stream-admitted-NbkGgC/`.

Do not attribute that timeout to the stream extension alone. NVIDIA's
Python example stalled before it enabled the extension, and moving
NexuML's extension enable after Kit launch still hit the startup watchdog.
Extending the baseline's cold bound exposed the actual Python defect:
Kit reached Lab's post-launch setup around 130 s, then failed to import
`omni.kit.widget.toolbar`. Lab initializes that toolbar for livestream as
well as local playback. `24d8a2ec` enables the existing restricted toolbar
in both modes without enabling X11 presentation or authoring in WebRTC.
The patched wheels reached native scene construction with no toolbar
error. In a network-isolated pod/container, supply the required task
assets locally; the initial patched local test lacked the ground-plane USD.
Control logs and exact scripts:
`/tmp/opencode/nexuml-vendor-python-example-fjfMVb/` and
`/tmp/opencode/nexuml-stream-toolbar-HPcKP7/`.

With five checksum-recorded NVIDIA Cartpole assets supplied through
`ISAACSIM_ASSET_ROOT`, the same installed wheels ran **120 steps / 120
transitions**, exit 0, in a network-isolated container on the RTX 4080 SUPER.
The endpoint receipt showed TCP `0.0.0.0:49100` and UDP `awaiting client:47998`.
A second process observed changing native policy observations and sent
SIGTERM after five steps; it exited 143. Neither process left a TCP listener
or UDP socket on the requested ports. This proves server-side playback
and shutdown, not UDP delivery or decoded frames. Reproduce from the
receipt directory with the preserved wheels and assets:

```bash
d=/tmp/opencode/nexuml-stream-toolbar-HPcKP7
sha256sum -c "$d/wheels.sha256"
sha256sum -c "$d/assets.sha256"
RECEIPT="$d/replay-complete" WHEELS="$d/wheels" ASSETS="$d/assets" \
  bash "$d/run-with-assets.sh"
RECEIPT="$d/replay-cancel" WHEELS="$d/wheels" ASSETS="$d/assets" PROBE="$d/cancel.py" \
  bash "$d/run-with-assets.sh"
```

The patched non-preempting cluster replay could not schedule, including
after broadening the allowed RTX models and staging the same asset set.
Both owned Jobs were deleted; no existing workstation, enrollment or
network settings changed. The exact qualification script and scheduler
events are in `cluster.sh`, `cluster-ada-admission.log`, and
`cluster-admission.log` in the receipt directory. Use the
[standalone private pod profile](../../../.k8s/isaac/README.md#try-standalone-policy-playback)
for the actual NVIDIA client/UDP test, not the isolated harness.
Local Kit regression on private Xvfb passed **24 steps / 48 transitions**
and cleared the native context:
`/tmp/opencode/nexuml-toolbar-local-regression-xBDJbZ/`.

Follow-up private-viewer readiness at `37e555ee`: no authorized NVIDIA
client or route to a task-owned playback workload was available. The
compatible Ready node set advertised 15 GPUs with 15 already requested;
this was an API snapshot, not a new admission attempt. The previously
forwarded X11 display `:11` could not be opened. The local server receipts
and wheel/asset checksums still pass; neither a running NetBird sidecar nor
an X11 proxy socket qualifies the client path. No existing workstation
was used. Checks and receipts:
`/tmp/opencode/nexuml-4a4-private-readiness-jKTxfc/`.

After the workstation was recreated, Kit reported the full streaming
app loaded and TCP signaling listening. The operator then supplied a
screenshot of the decoded Isaac editor and its rendered viewport in the
NVIDIA WebRTC Streaming Client over NetBird. This confirms the recovered
editor video path, not live policy motion or playback socket cleanup.
Server receipt: `/tmp/opencode/nexuml-workstation-recovery-WX8ETo/`.

The private client test inputs are staged at
`/tmp/opencode/nexuml-4a4-private-capsule-MUSHMk/`. Both wheels match the
passing isolated-server receipt byte for byte. All 15 checksums pass;
an isolated CPU check confirmed the export and scenario are compatible.
The scenario reads `ISAACSIM_HOST` from the prepared pod's `wt0` address.
Follow the bundle's README for the offline install and bounded
18,000-step playback command. The separate private Cartpole Job
`rglitza/nex212-cartpole-nhk9pp` passed admission dry-run with NetBird
injection and an isolated volume. It then waited 15 minutes without
preemption; the scheduler reported insufficient GPUs on 15 nodes.
The Job and its pending pod were removed. Playback never started,
and the recovered editor remained ready. Reproduction script, inputs
check and scheduler events: `/tmp/opencode/nexuml-private-cartpole-nhk9Pp/`.
The retry `rglitza/nex212-cartpole-lmroq3` was admitted on `cip-gpu-2`.
It initialized the native Cartpole and reported the private NetBird
endpoint on TCP 49100, with UDP 47998 awaiting a client. It was still
inside a policy-driven native step when the 20-minute process deadline
sent SIGTERM; it did not report completing the 36,000-step budget.
The native context cleared, both sockets were released, and the Job
was deleted. No decoded policy video was confirmed. Logs and replay
scripts: `/tmp/opencode/nexuml-private-cartpole-retry-lMroq3/`.

The client retry `rglitza/nex212-cartpole-ruwlwk` used the same verified
wheels, policy and assets on `student-worker-5` (RTX 4090, driver
580.126.20). The operator supplied a screenshot of the Cartpole viewport
in the NVIDIA WebRTC Streaming Client and confirmed the cart and pole
were moving. The server recorded an established TCP 49100 connection on
the advertised NetBird address and the policy process owning UDP 47998.
This is decoded policy playback over the prepared private route, not
editor video or a TCP-forwarding test. The editor was not interrupted.
Earlier runs verified socket release on completion and cancellation;
the client retry was still running at confirmation. The optional
forwarded Windows X11 profile was unavailable. Exact launch scripts,
logs and socket receipt: `/tmp/opencode/nexuml-cartpole-client-retry-RuWlWk/`.

**Real ROS 2 three-domain and mixed-objective proofs (1b/6.2a):** no
authorized episodes with a matching embodiment, calibrated controlled
joints, units, timing and held-out splits were supplied. The
[local-fit handoff](local-fit-handoff.md) names those prerequisites.
Do not run a synthetic/real mix or physical control as a substitute.

## Cluster follow-up

Kubernetes API access and a loopback-only port-forward to the existing Ray
dashboard allowed normal Jobs submissions without changing ingress trust or
bypassing TLS on the external endpoint. Do not use this tunnel for WebRTC.

- `nex212-gpu-roles-c11c8d68`: a `STRICT_SPREAD` reservation delivered
  three distinct Ray nodes and three whole GPUs (RTX 5080, 4070 and 4000 Ada).
  Each worker confirmed its assigned accelerator and device UUID. This
  proves scheduler admission, not training. The probe released its owned
  placement group. Receipts: `/tmp/opencode/nexuml-ray-gpu-admission-ZX4sW5/`.
- `nex212-nccl-fabric-da7c3c56`: actual **two-node NCCL**, RTX 4070 and
  RTX 5080, Torch 2.11.0+cu130, NCCL 2.28.9, network socket transport.
  Scalar and 1 MiB all-reduces matched the exact sums on both GPUs.
  The stock worker images were unchanged; Torch was installed from wheels
  into job runtime environments. This does not prove native simulation or
  replica updates. Receipts: `/tmp/opencode/nexuml-ray-nccl-plain-5sk1VB/`.
- `nex212-peer-e1608057`: **two-node native PhysX PPO via torchrun**, using
  the pinned workstation image and installed NexuML wheels on two RTX
  4000 Ada pods. Peer collectives ran before Isaac startup. Eight native
  environments per rank retained rollout/action tensors on CUDA; actor
  and critic parameters matched bit-for-bit across ranks after each
  global batch. Training reached 256 frames/eight optimizer steps and
  exported only on rank zero. A fresh public-worker invocation resumed
  to 384 frames, with seeds 49 and 57 rather than 17 and 25. Both contexts
  cleared and both jobs completed. Rank one's torchrun agent warned about
  the TCPStore exit barrier after the resume; this did not prevent either
  completion receipt. This is not same-host CUDA P2P or Ray Train evidence.
  Setup, proof scripts, wheel hashes, placement and cleanup receipts:
  `/tmp/opencode/nexuml-pod-physx-peer-cached-r8xd2J/`. From the repository
  root rerun `bash "$EVIDENCE/run-pods.sh" "$EVIDENCE"` using that receipt
  directory and substituting a currently approved cached RTX worker in
  `job_manifest.py` if needed.
- A separate two-pod PhysX run injected a native step error on rank 1
  before the first PPO update. Rank 0 reported a collective rollout
  failure naming rank 1; both simulators cleared and both torchrun
  processes exited nonzero before the deadline. The two owned Jobs and
  their config map were removed.
  Receipts: `/tmp/opencode/nexuml-pod-ddp-failure-u1KcXp/`.
- Initial prepared-image jobs were scheduled to GPU nodes but took longer
  than the wait window to pull the image from Harbor. An extended
  attempt verified the pinned image and CUDA runtime on one RTX 4000
  Ada; the other pod could not allocate the GPU on the node targeted
  for its cached image. No changes were made to the old workstation,
  worker pools, nodes, or networking. Receipts:
  `/tmp/opencode/nexuml-pod-peer-kfKet2/` and
  `/tmp/opencode/nexuml-pod-peer-long-Pz4zu8/`.
- On a subsequently allocated RTX 4000 Ada, the installed CLI fit and
  exported 64 PhysX frames (four optimizer steps). The protected WebRTC
  play process stalled before the listener, first at Kit UI setup and
  again in `SimulationApp._update_without_ready` after removing local
  X11/toolbar options and the authoring experience. No external media
  test was possible. Receipts: `/tmp/opencode/nexuml-pod-stream-cached-H7C5hk/`
  and `/tmp/opencode/nexuml-minimal-stream-pod-LdDUuJ/`.
  Separate one-variable checks of renderer idle settings and an Xvfb
  display did not get past native startup either. No unqualified
  rendering overrides were retained. Local Kit still ran 12 native
  transitions and cleared its context after the streaming-only change.
- The Kit-less Newton/Ray profile resolves with binary wheels while
  omitting the full Isaac Sim extra and its `idna-ssl` dependency. The
  reviewed Lab and ANTLR wheel hashes matched; the Lab install retained
  genuine VCS metadata via the existing offline cache workflow. See
  `/tmp/opencode/nexuml-kitless-binary-nik4LR/` for the hash-qualified
  dependency list and actual 239-package install with clean `uv pip check`.
  Ray Train's placement-group cleaner inherited the GPU-only interpreter
  and could not launch on the unmounted stock head; the first fit was
  intentionally stopped. This does not establish a binary PhysX profile.
- `nex212-kitless-ddp-37b358db`: **two CUDA Ray Train ranks** ran
  Kit-less Newton/MJWarp Cartpole, 256 global frames/eight optimizer
  steps, and resumed from the Ray-staged checkpoint to 384 frames/12
  steps with fresh simulators. The job-local command launched the
  verified Python prefix on training workers and the stock Ray Python
  on the head where Ray Train's placement cleaner runs. No base
  interpreter installs or runtime image changes. Reproduction and
  install, fit and resume receipts:
  `/tmp/opencode/nexuml-kitless-head-safe-IUn2Cj/`.
- `nex212-native-eval-491fd21a`: **native GPU learner and GPU snapshot
  evaluator on distinct Ray nodes**. The learner fit Kit-less Cartpole
  to 4,096 frames. The evaluator task was in `RUNNING` state before and
  after each of the 16 PPO batches. Two fresh-process snapshot receipts
  retained frame 0 and 4,096 and different hashes; the final evaluation
  drained even though the periodic snapshot was skipped while busy.
  The initial `working_dir` run timed out without a receipt; the passing
  run shipped input files with Ray `py_modules` and used a bounded
  16-step evaluation. This was one native learner, not an overlap test
  with multiple DDP ranks. Submission, hashes and logs:
  `/tmp/opencode/nexuml-ray-native-eval-modules-wCUBFF/`.
- `nex212-newton-split-2ebe664f`: two whole-GPU Newton/MJWarp collectors
  and the CUDA learner used three distinct Ray nodes. The fit reached
  128 frames/eight optimizer steps; independent actor identities resumed
  to 192 frames/12 steps. Batches crossed the object store on CPU and
  reached PPO on CUDA with complete versions `[1, 2, 1]`; the Ray state
  API confirmed all four fit/resume actors stopped. This used one Python
  prefix for both roles. A post-fit assertion that no simulator module
  had been imported failed: TorchRL's optional `mujoco.mjx` discovery
  imports the installed `mujoco` parent package even without building a
  simulator. The fit/resume assertions passed, but the isolation
  assertion failed and the wrapper later reported deadline exit 124.
  The logs do not distinguish Python shutdown from prefix cleanup.
  Separate learner dependency isolation remains a distinct gate.
  Receipt: `/tmp/opencode/nexuml-ray-newton-publication-tKLqO8/`.
  Earlier attempts failed the inherited working directory or TorchRL's
  default Gloo rendezvous actor import, not GPU capacity. The current
  adapter publishes weights through the object store without that actor.
- `nex212-newton-split-0f237f78` passed the same three-node fit and resume
  with **separate hash-pinned binary environments**: 240 native worker
  packages and 126 learner packages, both with clean `uv pip check`.
  The CUDA learner had no Isaac Lab, Isaac Sim, Newton or MuJoCo
  packages. CPU-staged rollouts arrived on the learner CUDA device with
  complete versions `[1, 2, 1]`. Both exports reloaded at the expected
  frame counts (128 and 192); optimizer steps progressed from 8 to 12.
  Resume created fresh actors using the configured worker seed mapping;
  it did not restore simulator trajectories. The job succeeded and Ray
  state confirmed all four owned collector actors stopped.
  Reproduction, binary locks, wheel hashes and cleanup:
  `/tmp/opencode/nexuml-ray-newton-lean-iEIzKG/`.

- `nex212-peer-d12f1247`: two native CUDA ranks ran Newton/MJWarp with
  Viser on rank 1 only. Chromium connected through an authenticated
  loopback Kubernetes tunnel; rank 1 saw the client then its disconnect,
  and the browser exposed no simulator mutation controls. The visible
  subset was bounded to two environments. Both ranks finished 256 frames
  and eight optimizer steps with matching weights; the listener closed
  before the job exited. The browser handshake deliberately pauses the
  acceptance harness, so this does not measure spectator overhead.
  Logs, screenshot and replayable harness:
  `/tmp/opencode/nexuml-pod-ddp-viser-handshake-tlzKGz/`.
  A second run injected coordinated Lightning cancellation while a
  browser was attached; both ranks unwound and the selected listener
  refused further connections before exit.
  Receipt: `/tmp/opencode/nexuml-pod-viser-cancel-EXRtDM/`.
- Matched two-node Newton Cartpole timing with and without a connected
  spectator on rank 1: 16 environments per rank, 512 global frames per
  rollout, 4,096 frames and 64 optimizer steps per fit. On the same GPU
  pair, median warm batch time was **63.8 ms headless vs 86.8 ms with
  Viser** (about 36% longer, or 8.0k vs 5.9k frames/s). This is one
  matched pair, excluding startup and browser handshake, not a universal
  overhead estimate. Receipt: `/tmp/opencode/nexuml-pod-viser-timing-MFRJs7/`.

Copy the Ray Jobs submission and its input directory to a new `$EVIDENCE`
directory to preserve prior receipts. Run from the repository root.
This dashboard-only tunnel is not a media route:

```bash
(
  kubectl -n ray-cluster port-forward --address 127.0.0.1 \
    svc/ray-cluster-kuberay-head-svc :8265 > "$EVIDENCE/tunnel.log" 2>&1 &
  tunnel=$!
  trap 'kill "$tunnel" 2>/dev/null || true; wait "$tunnel" 2>/dev/null || true' EXIT
  env -u RAY_ADDRESS -u RAY_API_SERVER_ADDRESS TMPDIR=/tmp/opencode \
    .venv/bin/python "$EVIDENCE/submit.py"
)
```

To rerun the two-node NCCL probe, use one terminal for the authenticated
Kubernetes tunnel and another for the submission:

```bash
kubectl -n ray-cluster port-forward --address 127.0.0.1 \
  svc/ray-cluster-kuberay-head-svc 18265:8265
```

```bash
input=$(mktemp -d /tmp/opencode/nexuml-peer-input-XXXXXX)
mkdir -p "$input/scripts/isaaclab"
cp scripts/isaaclab/nccl_preflight.py "$input/scripts/isaaclab/"
cp tests/fixtures/ray_nccl_probe.py "$input/"
env -u RAY_ADDRESS -u RAY_API_SERVER_ADDRESS .venv/bin/ray job submit \
  --address http://127.0.0.1:18265 --working-dir "$input" --entrypoint-num-cpus 0.1 \
  --runtime-env-json '{"env_vars":{"UV_CACHE_DIR":"/mnt/data/uv-cache"}}' \
  -- timeout -k 15s 890s python -u ray_nccl_probe.py --pool-resource pool_gpu_workers
```

## Post-merge checks

- Verified local browser Viser train/replay, read-only parity, failure and
  cancellation cleanup, plus isolated sensor/trajectory storage. **3
  passed** with existing Chromium; no distributed native rank claim.
  Receipts: `/tmp/opencode/nexuml-viser-local-browser-v2/`.
- The installed learner used the binary lock with the `reinforcement`,
  `ray` and `library` extras and both application wheels. `uv pip check`
  passed, and the installed CLI again trained 32 frames on `cuda:0` and
  replayed eight CPU ticks. The disposable environment was removed after
  preserving wheels, package versions, hashes and receipts.
- The existing opt-in Isaac CI job now exercises the same generated
  `uv.lock`: a fresh learner prefix uses `--no-default-groups
  --no-install-workspace --no-build` with the three existing extras, then
  installs the two application wheels. With the regenerated lock, CUDA
  passed the 126-package check (Torch 2.11.0+cu130, TorchRL 0.14.0,
  TensorDict 0.14.2), confirmed no simulator packages, and fit/exported
  32 frames. Resume entered training with 32 frames and eight restored
  optimizer steps, then reached 48 frames and 12 steps. Both exports
  reloaded. This CI smoke validates direct PPO. Exact step and log:
  `/tmp/opencode/nexuml-locked-cuda-resume-OkQAgL/{run.sh,ci.log}`.
- A temporary Kit-less dependency-group trial consumed the same workspace
  resolution after staging the already approved Lab and ANTLR artifacts.
  Its 240-package prefix passed `uv pip check`, ran 32 native CUDA
  Cartpole frames/eight optimizer steps and reloaded the export without
  Isaac Sim. The trial required `--inexact` and skipping both packages
  during sync; it is not the requested clean install and that group was
  not retained. `uv sync --locked --extra isaac --no-build` into an
  empty prefix fails first on the Lab Git source. Binary-only resolution
  also rejects Isaac Sim 6.2's `idna-ssl==1.1.0`, which has no usable
  wheel. A compatible upstream release or an approved binary source is
  needed; no alternate lock, source build or dependency omission added.
  Official PyPI/NVIDIA index and upstream release metadata exposed no
  matching Lab or ANTLR wheel source. The two approved local wheels
  could install as direct references, but uv recorded an archive
  origin, not a VCS commit (even with the approved SHA256 in the URL).
  A future wheel source must also preserve verifiable revision identity
  for the runtime and Viser checks; a version-only install is not enough.
  Receipts: `/tmp/opencode/nexuml-package-lane.quhTJJaf/` and
  `/tmp/opencode/nexuml-staged-kitless-lock-Jkfajr/`. Archive origin:
  `/tmp/opencode/nexuml-direct-wheel-provenance-ZxfpqH/`.
- The follow-up found no retained upstream Lab wheel artifact for the
  exact commit. Fresh `uv sync --locked --no-build` still rejected the
  Git requirement with the genuine reviewed cache available; the
  workaround in the prepared image uses `uv pip install` and does not
  establish a clean workspace sync. The ANTLR wheel on piwheels does
  not match the reviewed hash. Its unapproved `idna-ssl` wheel imports
  `ssl.match_hostname`, which Python 3.12 no longer provides. The
  checked NVIDIA index did not offer a newer 6.2 kernel wheel. The
  remaining inputs are a supported immutable Lab binary with verifiable
  revision identity, approved ANTLR wheel delivery, and a compatible
  full-Kit dependency fix. Neither project manifest nor the lock was
  changed. Probes: `/tmp/opencode/nexuml-package-6.3-followup.TTaxhz8Z/`.
- NVIDIA's 6.2 packaging source and published CPython 3.12 Linux wheel
  metadata both list `idna-ssl` unconditionally under `isaacsim-kernel`.
  A scan of the wheel's Python and configuration files found the pin,
  but no `idna_ssl` import or patch call; that does not prove external
  extensions or native binaries never need it. No checked official 6.2
  profile keeps Kit PhysX, RTX and interactive playback while avoiding
  the kernel dependency. No packages were omitted.
  Source and metadata receipts: `/tmp/opencode/nexuml-idna-upstream.Qtil4Yjc/`.
- A disposable copy of the whole workspace replaced only the Lab and
  ANTLR artifact sources with the two reviewed local wheels and added a
  Kit-less group (Lab without the `isaacsim` extra, CUDA Torch/torchvision).
  `uv lock` generated one lock for that copy: 388 packages and no version
  changes versus the checked-in lock. A fresh `uv sync --locked --no-build`
  installed 238 dependencies without source builds or native package skips.
  Both installed application wheels brought the count to 240;
  `uv pip check` passed and Isaac Sim was absent. The unmodified runtime
  rejected the wheel's ordinary archive origin. The runtime now verifies
  the exact reviewed Lab wheel's SHA-256, its recorded files, and the
  imported Lab entry point without generating fake VCS metadata.
  The installed Kit-less Newton Cartpole fit and fresh-context resume
  passed on CUDA: **32→48 frames, 8→12 optimizer steps**, both exports
  reloaded, and both simulators closed.
  Exact replay (requires the preserved generated lock, artifacts, and
  cached dependency wheels):

  ```bash
  bash /tmp/opencode/nexuml-kitless-wheel-lock-rEiKHN/replay.sh
  ```

  The generated test lock is not tracked or a second runtime lock.
  Removing the local wheel inputs made `uv lock --check --offline` and
  even the simulator-free learner's `uv sync --locked --dry-run` fail on
  metadata. As a result, the shared lock was not switched to local
  paths. Stable delivery of these wheel inputs is still required for a
  portable native worker install; full Kit additionally retains the
  upstream `idna-ssl` edge. Viser, Arena and full Kit were not qualified
  with the archive-origin path.
- The prepared-image CI smoke now uses the image entrypoint and
  `/env/bin/python`, with checkout sources first on `PYTHONPATH` and
  pulls disabled. The previous checkout `.venv` command failed in a
  clean checkout. The corrected command collected 69 tests offline.
  A local replay of that command filtered to `native_ppo_placement
  and physx-colocated` used the checkout, cached image and five verified
  assets in a network-isolated GPU container. Exit 0, 128 exported
  frames, eight optimizer steps asserted by the fixture, terminal/GAE
  parity, and native context cleanup. This was not hosted CI or a clean
  native workspace install. Workflow command and results:
  `/tmp/opencode/nexuml-package-6.3-followup.TTaxhz8Z/` and
  `/tmp/opencode/nexuml-ci-native-candidate-23Ac3a/`.
- CUDA learner plus native CUDA rollout actor resumed on one shared GPU
  from a partial checkpoint: fresh actor identity, exact restored policy
  and Adam state, policy version 1 in the fresh collector, CPU transport,
  native CUDA learner and 32 frames/eight optimizer steps. **1 passed**;
  `/tmp/opencode/nexuml-final-cuda-transport-v2/`.
- Standalone CUDA and Ray whole-GPU snapshot evaluators both passed again
  after the worker-visible device check (**2 passed**), without a
  concurrent native learner. Receipt: `/tmp/opencode/nexuml-snapshot-assignment-proof/`.
- Role-owner, Ray Train and missing-resource admission suite: **6 passed**.
  The missing actor role timed out without shutting down borrowed Ray.
  Geometry, viewer/provider, CI and DDP guard suite: **152 passed, 17 skipped**.
- Docs check with Node 22.23.4 built both targets and verified rendered
  links; locked dependencies, Ruff and strict OpenSpec passed.
- At `70204139`, both wheels and source distributions rebuilt offline
  and passed the package layout/metadata check (1 passed, 2 opt-in
  resolver checks skipped). Artifacts and hashes:
  `/tmp/opencode/nexuml-stable-artifacts-MLAL8r/`.
- The standard suite at `70204139` passed: **3,776 passed, 154 skipped,
  604 subtests passed**. The final ordered actor graph checkpoint guard
  rejected changed actor definitions before collection and passed the
  focused PPO, grouped, evaluation and DDP suite (**172 passed, 15 skipped**).
- The standard CPU suite at `5835df61` passed: **3,779 passed, 158 skipped,
  604 subtests passed**. The separate real Ray admission/publication slice,
  including partial admission with native-scope-style cleanup, passed
  **3 tests** at `df8f86cb`. CPU suite log:
  `/tmp/opencode/nexuml-decouple-final-cpu-WVutA9/suite.log`.
- Focused type checks pass. The repository-wide type checker still reports
  115 diagnostics with the prepared runtime (including absent optional
  packages and existing fixture issues); the preceding tree reports 126.
- Follow-up regression: real CPU/Gloo **5 passed** (including parity,
  checkpoint resume, failure/cancel and rank-zero export); viewer, native
  definitions/runtime and Ray collector batch **295 passed, 19 skipped**
  before the toolbar fix. The final viewer/runtime slice with the toolbar
  and interface/graphics/port tests **196 passed**. Gloo and batch receipts:
  `/tmp/opencode/nexuml-final-gloo-VUyUlz/` and
  `/tmp/opencode/nexuml-decouple-followup-YX4v5Q/`.
- Parallel follow-up across the non-ROS gates: **142 passed** for visual
  placement guards, native preflight and definitions, and WebRTC.
  Ruff, formatting and targeted typing passed for the placement changes.
  Receipt: `/tmp/opencode/nexuml-nonros-placement-buEULU/`.

```bash
UV_PROJECT_ENVIRONMENT="$(mktemp -d /tmp/opencode/nexuml-native-cold-XXXXXX)/.venv" \
  uv sync --locked --python 3.12 --extra isaac --no-build

TMPDIR=/tmp/opencode CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NEXUML_RUN_SYSTEM_TESTS=0 NEXUML_RUN_SIMULATOR_TESTS=0 NEXUML_RUN_HARDWARE_TESTS=0 \
  .venv/bin/python -m pytest --basetemp=/tmp/opencode/nexuml-decouple-cpu -q

NEXUML_RUN_SYSTEM_TESTS=1 NEXUML_TEST_RAY_TRAIN=1 \
  TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/execution/test_rl_ddp.py tests/core/test_ray_collectors.py \
  -k 'ddp or ray_train_round_trip or admission'
```
