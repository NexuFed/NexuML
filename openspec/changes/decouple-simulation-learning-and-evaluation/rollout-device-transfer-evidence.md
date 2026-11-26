# 2.3 — CPU/CUDA rollout ownership and transfer evidence

**Pass for the direct Gymnasium Pendulum reference.** The CPU simulator and
learner used a CUDA rollout actor and CUDA storage, compared with all-CPU
placement. This is independent of the native Isaac split path in
[design §4](design.md#4-direct-and-distributed-collection) and is not
evidence for task 2.2, GPU physics, P2P, or Ray.

## Correctness and ownership

[Opt-in probe](../../../scripts/isaaclab/rollout_device_transfer.py), run on
2026-11-26 at `61374679d72cd4500706425e8d60faf97fe6513f` plus the probe.
Pinned `.venv`: Torch `2.11.0+cu130`, TorchRL `0.14.0`, TensorDict `0.14.2`,
Gymnasium `1.3.0`, Lightning `2.6.5`. Hardware: i7-14700K; one visible
RTX 4080 SUPER (`cuda:0`, driver `610.43.03`, 16,376 MiB).

Both layouts used ordinary `pendulum_ppo`, `CollectorSpec`, the
`build_collector` → `EnvironmentFactory` → TorchRL adapter path, and
`NexuRLLightningModule` with actual `Trainer.fit`, PPO/GAE, and two Adam
optimizers. Seed 17, eight CPU environments × 32 steps per rollout, a
seven-step time limit, one PPO epoch with 64-sample minibatches, and no
evaluation. Initial actor and critic hashes matched across all four runs.

| Observed tensor seam | All CPU | Split |
| --- | --- | --- |
| Runtime action, observation, reward and end flags | CPU | CPU |
| Rollout actor parameters, input, action and behavior log probability | CPU | CUDA:0 |
| Collector storage, all Torch leaves | CPU | CUDA:0 |
| Learner input, GAE, actor/critic and optimizer state | CPU | CPU |

- Both timed fits executed **6,144** actual Gym steps, **24** rollouts
  and **96** steps per optimizer. Actor/critic parameter L2 changes
  were **0.1982/4.3238** (CPU) and **0.3042/4.3181** (split); losses were
  finite. Both counted **24** weight syncs. Collection waited for each
  sync. The split replica still held its old weights after the first
  CPU update; spot checks after sync 1, 2, and 24 matched learner weights
  exactly. The all-CPU actor was the learner actor itself.
- Every storage and learner batch had `[8,32]` and names `[None,"time"]`;
  GAE saw those same axes before flattening. In the first two rollouts,
  all stored next observations matched the raw Gym steps. The **72**
  time-limit transitions per layout kept pre-reset observations in
  `next`, with the raw reset observations on the following live carrier.
  Live observations also matched across the first rollout boundary.
  `done` matched truncation, `terminated` was false, and each timeout
  advantage matched `reward + gamma * V(final_obs) - V(current_obs)`.
- Behavior log probabilities produced by the actor were bitwise
  preserved through storage and learner transfer in those two rollouts.
  Recomputing them on the CPU learner is not bitwise equivalent: with
  the runtime's `medium` float32 matmul setting, maximum absolute split
  differences were `0.001099` and `0.000533` (CPU baseline max `<1e-6`).
  This is a cross-device numerical difference, not altered stored
  behavior statistics or evidence of stale weights.
- `close_collection` shut down the collector and closed all eight
  environments. Split warm Torch allocator peaks were **9,844,224 B
  allocated / 23,068,672 B reserved**. After fit, dropping references,
  GC, and `empty_cache` left **9,568,256 / 23,068,672 B**; clearing the
  process-owned cuBLAS workspaces in the standalone probe then gave
  **0 / 0 B**. The normal teardown does not clear global BLAS workspaces.
  GPU compute process listings were unchanged after each timed process
  exited (one pre-existing 922 MiB entry). The CPU run initialized a CUDA
  context but left **0 / 0 B** in the Torch allocator.

## Timing

Warm medians over rollouts 5–24, one serial timed run per layout, with
CUDA phase barriers in the split case and **no per-copy tracing**. Gym
timing sums only the CPU `env.step()` calls (including Gym wrappers)
inside collect; do not add it to the total.

| Phase | CPU, ms/rollout | Split, ms/rollout |
| --- | ---: | ---: |
| Collect (inference, environment, staging and storage) | 30.016 | 48.580 |
| CPU Gym step only, within collect | 3.091 | 3.054 |
| Full batch to learner | 0.006 | 0.133 |
| GAE and optimization | 9.380 | 8.923 |
| Weight synchronization | 0.238 | 0.277 |
| Transitions/s across summed phases | 6,431 | 4,326 |

`nvidia-smi` device-wide 200 ms samples within the warm window (including
background use): all CPU **0% / 1,228 MiB** (4 samples); split **10–13% /
1,302 MiB** (6 samples; median **12.5%**). This is coarse observation of a
small rollout, not a GPU saturation test.

Separate process per copy diagnostic: four rollouts, discard the first.
Both did 1,024 Gym steps and 16 updates per optimizer. CPU baseline
observed **zero** cross-device copy calls and reused the same learner
batch object. The split diagnostic intercepted actual ATen cross-device
`_to_copy`, `copy_`, `_foreach_copy_`, and CUDA scalar extraction. It
synchronized the GPU around each measured operation. Byte counts are
logical tensor payloads, not measured bus traffic. Host pinning is from
the CPU tensor's `is_pinned()`; scalar transfers use an internal buffer.

| Split phase / direction / host buffer | ATen calls | Payload bytes | Total ms | Effective MiB/s |
| --- | ---: | ---: | ---: | ---: |
| Collect, H→D, pageable | 1,468 | 39,408 | 14.804 | 2.54 |
| Collect, H→D, pinned | 384 | 12,288 | 3.319 | 3.53 |
| Collect, D→H, pinned | 480 | 18,432 | 5.735 | 3.06 |
| Collect, D→H, internal scalars | 700 | 896 | 5.438 | 0.16 |
| Learner batch, D→H, pinned | 42 | 44,544 | 0.390 | 108.90 |
| Weight sync, H→D, pageable (foreach) | 3 | 54,552 | 0.148 | 351.06 |

This is observed copy cost for these payloads, including staging,
dispatch, and completion waits. Do not subtract these diagnostic timings
from the non-profiled run or interpret them as peak PCIe bandwidth. The
split path was slower for this fixture; one seed and different device RNG
streams do not establish equivalent learning. Pendulum exercises time
limits, not true terminations, and this geometry resets every lane on
the same tick.

## Reproduction and receipts

Raw JSON, logs, GPU samples, process listings, and source checksums:
`/tmp/opencode/nexuml-rollout-transfer-m3cqe4/`. Four fresh probe
processes exited 0 under the shared lock. The probe and eight runtime
source hashes matched before and after those runs. The runtime hashes
also match the Git objects at the measured revision; subsequent
worktree changes to the collector and types are outside this receipt.
Probe SHA-256:
`66eff41bd8e51f5b36ba48b212e881d945ed07e136e5b565b567b696eac6c99b`.
The documented public CLI placement also completed a 64-transition smoke
with eight optimizer steps and reported learner device `cpu`:
`/tmp/opencode/nexuml-cuda-rollout-cli-WpEqx8/cli.log`.
The SHA-256 of the `sha256sum cpu.json split.json cpu-copies.json
split-copies.json cpu-gpu.csv split-gpu.csv` output is
`73b470e396cdc43580317fad316e93c7e1c179ce136f4af6cde22e77fdc01236`.

From the repo root, with physical GPU index 0 matching logical `cuda:0`:

```bash
set -eu
out=$(mktemp -d /tmp/opencode/nexuml-rollout-transfer-XXXXXX)
export TMPDIR="$out" PYTHONDONTWRITEBYTECODE=1 TZ=UTC
script=scripts/isaaclab/rollout_device_transfer.py
(
  flock -w 600 9
  for name in cpu split; do
    placement=cpu
    if [ "$name" = split ]; then placement=cuda:0; fi
    nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv >"$out/$name-apps-before.csv"
    timeout 110s nvidia-smi -i 0 --query-gpu=timestamp,utilization.gpu,memory.used \
      --format=csv,noheader,nounits --loop-ms=200 >"$out/$name-gpu.csv" &
    sampler=$!
    rc=0
    timeout --kill-after=10s 90s .venv/bin/python -u "$script" "$placement" \
      "$out/$name.json" >"$out/$name.log" 2>&1 || rc=$?
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv >"$out/$name-apps-after.csv"
    test "$rc" -eq 0
    timeout --kill-after=10s 90s .venv/bin/python -u "$script" "$placement" \
      "$out/$name-copies.json" --copies >"$out/$name-copies.log" 2>&1
  done
) 9>/tmp/opencode/nexuml-decouple-native-gpu.lock
```

Focused regression checks: `tests/core/test_reinforcement.py` selected
by `-k 'cpu_fit_evaluate_checkpoint_and_worker_local_materialization or
lightning_collects_only_after_the_previous_learner_update or
environment_is_closed_when_learning_raises or
vectorized_tensor_shapes_and_reset_masks or
copied_policy_synchronization_changes_rollout_weights'`: **8 passed,
16 deselected**, log
`/tmp/opencode/nexuml-rollout-transfer-tests-zsUZBv/pytest.log`.
