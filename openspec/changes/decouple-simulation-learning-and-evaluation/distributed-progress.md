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
  -k native_rgb_ppo_placement
.venv/bin/python tests/fixtures/visual_rollout_pressure.py --frames 2048 --side 128 --batch 64
# On a compatible two-GPU host, re-run the same workload with explicit roles:
.venv/bin/python tests/fixtures/visual_rollout_pressure.py --frames 2048 --side 128 \
  --batch 64 --simulation cuda:0 --inference cuda:0 --learner cuda:1
.venv/bin/python tests/fixtures/visual_rollout_pressure.py --frames 2048 --side 128 \
  --batch 64 --simulation cuda:0 --inference cuda:1 --learner cuda:1
```

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
| 2a.2–2a.3 | NCCL preflight and matched benchmark runnable | Single GPU passes functional checks; peer scaling, collective bytes and interference unmeasured |
| 2a.4 | Synthetic pressure, transfer and weight-parity fixture; public native RGB fit fixture | Co-located native image-input fit passes; split visual and large-model results unqualified |
| 3.4–3.5 | Actor admission, CUDA learner, published weights and native checkpoint/recovery proven with local resources | Two distinct collector nodes plus learner GPU never allocated; remote grouped rollout remains unsupported |
| 4.4 | Rank-zero snapshots, separate process/Ray GPU evaluation and bounded cancellation | Native snapshots pass; native fit/eval overlap on distinct devices unverified |
| 4a.1, 4a.4, 4a.6 | Typed WebRTC preflight and selected-rank/collector Viser routing | Native WebRTC startup failed; external media, forwarded X11 and actual distributed spectator unverified |
| 6.3 | Binary installed learner and prepared-image CI recipe retained; role interpreter selection implemented | Stock-worker native wheels and matching deployed role profiles unqualified |
| 1b, 6.2a | Existing simulator-only SDK and recipe validation retained | Real ROS 2 proof needs authorized matching data |

**Peer NCCL and co-located native DDP:** requires two compatible GPUs.
On an allocated worker with two visible GPUs, run the actual collective
first. The same preflight can be launched across nodes using your approved
torchrun rendezvous instead of `--standalone`.

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
for inference in cuda:0 cuda:1; do
  .venv/bin/nexuml train isaac-cartpole-ppo \
    -O training.devices=1, -O training.collector.env_device=cuda:0 \
    -O training.collector.storing_device=cuda:0 \
    -O "training.collector.policy_device=$inference" \
    -O training.total_frames=2048 -O training.frames_per_batch=1024
done
```

**Native fit/evaluation overlap (4.4):** on the same two-GPU worker,
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
spec.policy_evaluation = RLEvaluationSpec(
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
require the trusted Jobs CA, approved binary native worker profile and
two simultaneously schedulable GPU actor nodes plus a learner GPU.
Use the configured pool resources and inspect the actual actor nodes:
logical indices are not proof of distinct placement. Grouped remote
PhysX collection is still rejected; do not use the state reference
as its substitute. Verify borrowed runtime ownership, missing/stale
batches, resumed native workers and network/throughput before widening
the support matrix.

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

**WebRTC (4a.1/4a.4):** run the ordinary `nexuml play` command in the
[WebRTC how-to](../../../docs/src/content/docs/how-to/isaac-native-configuration.mdx#native-webrtc-playback)
on an approved private graphics worker, with Vulkan/NVENC, TCP 49100 and
UDP 47998. Require client-decoded policy motion and released sockets on
exit. Prior TigerVNC playback remains qualified; a Kit editor session
or open signaling listener does not count as a policy stream.
The installed-wheel playback test in the prepared Isaac image reached
Vulkan/RTX initialization but timed out twice (150 s and 100 s), using
both the explicit minimal rendering experience and the maintained Lab
streaming selection. Neither showed a signaling receipt, policy ticks,
or a client connection. Kit logged OmniHub and NGX problems; the logs do
not isolate the exact native startup fault. Treat this as a **native
startup failure**, separate from the absent approved external client.
No host ports were published. Receipts:
`/tmp/opencode/nexuml-stream-configured-6hHQ5o/` and
`/tmp/opencode/nexuml-stream-native-experience-Zb1wOX/`.

**Real ROS 2 three-domain and mixed-objective proofs (1b/6.2a):** no
authorized episodes with a matching embodiment, calibrated controlled
joints, units, timing and held-out splits were supplied. The
[local-fit handoff](local-fit-handoff.md) names those prerequisites.
Do not run a synthetic/real mix or physical control as a substitute.

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
- CUDA learner plus native CUDA rollout actor resumed on one shared GPU
  from a partial checkpoint: fresh actor identity, exact restored policy
  and Adam state, policy version 1 in the fresh collector, CPU transport,
  native CUDA learner and 32 frames/eight optimizer steps. **1 passed**;
  `/tmp/opencode/nexuml-final-cuda-transport-v2/`.
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
- Focused type checks pass. The repository-wide type checker still reports
  115 diagnostics with the prepared runtime (including absent optional
  packages and existing fixture issues); the preceding tree reports 126.

```bash
TMPDIR=/tmp/opencode CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NEXUML_RUN_SYSTEM_TESTS=0 NEXUML_RUN_SIMULATOR_TESTS=0 NEXUML_RUN_HARDWARE_TESTS=0 \
  .venv/bin/python -m pytest --basetemp=/tmp/opencode/nexuml-decouple-cpu -q

NEXUML_RUN_SYSTEM_TESTS=1 NEXUML_TEST_RAY_TRAIN=1 \
  TMPDIR=/tmp/opencode OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/execution/test_rl_ddp.py tests/core/test_ray_collectors.py \
  -k 'ddp or ray_train_round_trip or admission'
```
