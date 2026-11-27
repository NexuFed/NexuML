# Distributed RL implementation ledger

Baseline: `fdf6ad93`. OpenSpec starts at 35/44 tasks complete. Preserve previous
native and viewer evidence; none of it qualifies co-located DDP.

- [x] Prove two real CPU/Gloo ranks against single-process PPO with matched
  minibatches. Check rank seeds, global counters, resume and failure.
- [ ] Exercise the same lifecycle through Ray Train and run NCCL before native
  distributed CUDA. Distinguish rank-local DDP from remote collectors.
- [ ] Test independent device placement and snapshot evaluators. Keep live
  viewing optional and preserve the existing Kit playback path.
- [ ] Measure equivalent co-located Isaac and RSL-RL at larger vector widths.
  Record throughput before deciding whether an optimization is warranted.
- [ ] Run integrated regressions and record exact commands and outstanding
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

Nsight at width 1,024 measured 11,040 baseline NexuML stream syncs vs
3,592 RSL-RL (including launch/setup). Py-spy at width 4,096 found
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
`/tmp/opencode/nexuml-rl-cpu-samples-AutsrX`.

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
