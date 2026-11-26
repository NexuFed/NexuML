# Ray collection placement and policy versions

**Verified on one host with Ray 2.57.0 and TorchRL 0.14.0, CPU only.**
`collector.ray` selects an existing Ray Core target or a borrowed runtime.
Ray's native options carry custom resources and label selectors to each
one-CPU actor. Conflicting targets and explicit discovery without an existing
cluster fail without starting local Ray. Omitting the placement preserves
the default CPU runtime behavior. This does not change Ray Jobs or Ray Train.

The existing TorchRL weight sync remains the only transport. Each worker
tags its transitions with the upstream integer version tracker. The copied
action belongs to the post-step version, not the previous observation's
version retained at the first tick of the next batch. The coordinator
checks that all workers contributed the complete `[worker, ..., time]`
batch and that every transition has the published version, before the
learner consumes the batch. Each update must synchronize all workers;
asynchronous, multi-learner and GPU Ray layouts remain rejected.

In a two-worker, two-environment probe, the native next-version rows were
uniform at **1, 2, 3** across successive eight-transition batches, paired
with copied actions **0, 1, 2**. The root version was one behind at the first
tick and is not the behavior-policy evidence. The ordinary suite also ran
canonical CPU PPO, borrowed-runtime resource and label checks, worker
cleanup, and missing/stale/short batch cases. **47 passed** in 91.59 s,
with the local Ray probes serialized. An overlapping earlier attempt hit
Ray's host memory safeguard; the passing run retained that safeguard.

```bash
export NEXUML_RUN_SYSTEM_TESTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
env -u RAY_ADDRESS CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  -q tests/core/test_ray_collectors.py tests/core/test_ray_rollout_barrier.py
```

Receipts: `/tmp/opencode/rv-x0oJJ9/probe.log` and
`/tmp/opencode/rb-AJNTEi/pytest.log`. No remote native Kit worker, GPU
rollout, two-node topology or grouped distributed learner is qualified.

## Worker-owned scope (3.2, CPU slice)

Distributed factories enter the provider's `job_scope()` in the rollout
worker before building the environment, and close the runtime before
leaving that scope. Direct collection retains the caller's whole-job
scope. The CPU Ray trace observed `enter → build → close → leave` in each
worker PID on iterator exhaustion, learner failure, and interruption.
A failed build also unwound its scope. The combined Ray suite passed
**48 tests**; **202** non-system evaluation, grouped-learning, and
Isaac preflight checks passed separately. Ray receipt:
`/tmp/opencode/rw-vjuOZy/pytest.log`. Native Kit lifetime and GPU placement
remain unqualified.

Shutdown now waits for all workers to close, within the same 30-second
bound, before reading results. A failure from one worker no longer
interrupts the others' grace period or skips TorchRL's weight-sync
transport/actor cleanup. The suite passed **49 checks**, including
close failure, timeout, and repeatable shutdown:
`/tmp/opencode/rs-S3RSeI/pytest.log`.

## Checkpoint continuation (3.5, CPU slice)

The public session saved an ordinary Pendulum checkpoint after 16
transitions and four optimizer steps, then interrupted training. Resume
restored actor and optimizer tensors exactly and created two new Ray
actors before consuming the remaining batch. That generation started
at policy version 1 (versions are collector-local). Training reached
32 transitions, two rollouts, and eight optimizer steps; the exported
actor reloaded with identical weights and the borrowed Ray runtime
remained alive. Targeted test receipt: `/tmp/opencode/rr-Phtsbn/pytest.log`.
This does not qualify native GPU failure/resume or distributed performance.
