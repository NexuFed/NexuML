# Ray collection placement and policy versions

**Initial CPU qualification on one host with Ray 2.57.0 and TorchRL 0.14.0.**
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
asynchronous, multi-learner and GPU Ray layouts were rejected in this
initial CPU candidate.

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

### Native actor shutdown blocker

At `dd1bd204`, a real Ray actor received the existing `EnvironmentFactory`,
reserved one GPU, entered its own Kit 6.2 / Isaac Lab 3.0 scope and built
the unchanged two-world PhysX Cartpole. Three zero-action steps produced
**six physical transitions** with CUDA rewards and terminal/truncation
tensors. On ordinary factory-owned close, Kit cleared `SimulationContext`
and the actor process exited, but Ray returned `ActorDiedError` instead of
acknowledging shutdown. The caller-owned Ray runtime remained alive.

Passing `fast_shutdown=False` to the published Lab launcher in an isolated
diagnostic did not change the result. The pinned `KitLauncher` allowlist
does not forward `fast_shutdown` to `SimulationApp`, whose default fast
close may terminate its owning process. Do not treat a generic actor death
as successful teardown; it is indistinguishable at that boundary from a
genuine lost worker. Native collection stays disabled pending a supported,
acknowledged lifecycle, independently of multi-node availability.
Receipts: `/tmp/opencode/jr-x2o97S/` and `/tmp/opencode/jr-full-2E6fPu/`.
These probes did not run TorchRL distributed PPO or verify native
cancellation, checkpoint resume, or object-store cleanup.

### Finite actor budgets preserve worker errors

Pinned TorchRL launches its actors with an infinite frame budget even
for a finite synchronous run. Ray 2.57 tests actor truthiness when
serializing an exception. TorchRL derives a non-integer `__len__` from
the infinite budget, so Ray killed the actor while trying to report
the real error. The same failure reproduced through the ordinary CPU
collector, without Kit.

Each worker now receives its share of the finite full-batch budget.
A two-worker run requesting 20 frames at eight per batch still
returned three eight-frame batches. The worker reported its original
exception, both workers closed, and borrowed Ray stayed available.
The real and hermetic Ray suites passed **51 checks**. Failure and
passing receipts: `/tmp/opencode/ray-error-before-Owalx7/` and
`/tmp/opencode/ray-frame-bounds-pgX6EK/`. A placement fixture now
uses bounded actions so it does not fail randomly on contract validation.

### Native Warp factory lifecycle (3.2, one-host slice)

At `5989d6a5`, two successive Ray actors each reserved the same RTX 4080
SUPER and built the unchanged native MuJoCo Warp Cartpole via the same
worker-owned `EnvironmentFactory`. Each ran three ticks over two worlds
with CUDA reward and termination tensors. Both Ray close calls returned
after the runtime released its model and data; repeated close was safe.
In the second actor, an injected exception after real physics steps
crossed Ray, followed by the same successful close. Both actor processes
exited and the caller-owned Ray runtime stayed up. Source hashes, exact
probe and receipts: `/tmp/opencode/wr-3DqxFe/`.

This isolates the observed non-returning close to the Kit lane, not the
shared factory. The probe did not use the public Ray PPO collector,
exercise cancellation or move rollout batches over the object store.
GPU Ray training remains disabled.

### Returning native shutdown (3.2)

Native rollout workers now enter an explicit worker scope, separate from
the whole-job scope used by direct CLI train, evaluate and play. The pinned
Lab launcher drops the `SimulationApp` keyword `fast_shutdown`. Its
supported Kit argument sets `/app/fastShutdown=false`, so Kit tears down
its extensions and returns. The worker scope propagates the original
Python failure instead of asking Kit to terminate the host with an exit
code. Missing acknowledgements and real actor death remain errors.

On one RTX 4080 SUPER, a Ray actor using the ordinary `EnvironmentFactory`
ran six CUDA PhysX transitions, cleared the simulation context, and
acknowledged native close in 11.15 s. Then the existing TorchRL-backed
collector ran three fresh, successive native workers. Each reserved a
whole GPU and placed the project PPO actor/critic and two PhysX worlds
together; CPU-staged four-transition batches had finite log probabilities
and values and the expected policy version. On ordinary exhaustion, an
exception injected after a real GPU step, and cancellation with a batch
already in flight, respectively, Kit stopped, the exact actor PID exited,
and the borrowed runtime remained usable. Plasma returned to zero
objects after each phase. No partial failed/cancelled batch was consumed.
The probe supplied the actor and critic together explicitly; the
existing public PPO algorithm and its learner-side GAE were not changed.

Proofs: `/tmp/opencode/kit-native-setting-ZgGrX9/` (native flag),
`/tmp/opencode/kp-6X5RF2/` (production scope and simple policy),
`/tmp/opencode/kv-l27AWt/` (production scope and PPO actor/critic).
The headless focused suites passed **347** unit checks and **2** native
direct frame-budget/cancellation regressions. Task 3.2 qualifies the
worker lifecycle on one host, not public GPU PPO, multi-node placement,
GPU checkpoint continuation or distributed throughput (3.4–3.5).

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

## Public one-host CUDA rollout with a CPU learner (3.4/3.5 slice)

The ordinary training command now accepts an explicit `collector.ray.num_gpus: 1`.
Ray schedules that reservation for each actor on the named Core cluster;
the worker uses its logical `cuda:0` for PhysX and policy inference and
returns CPU-staged batches. The existing synchronous version barrier and
PPO learner-side GAE are unchanged. The pinned Isaac lane rejects grouped,
interactive, Arena, non-PhysX and non-state requests. Neither a GPU learner
nor a private local Ray fallback is enabled in this configuration.

The unmodified public CLI, with CUDA hidden from the learner process, ran
two two-world PhysX rollouts for **64 physical frames / eight optimizer
steps** and exported at frame 64. The exact worker PID exited, the owner
could allocate the GPU again, and Ray reported zero Plasma objects. The
native checkpoint regression interrupted a 32-frame fit at frame 16,
then resumed using a fresh GPU actor. Exact saved actor and optimizer
state, the first new policy version, finite behavior log probabilities,
final counters and reloaded export all passed. The prior native
failure/cancellation proofs remain scoped to the worker lifetime.

Receipts: `/tmp/opencode/native-ray-ppo-public-vVWr0v/` (CLI),
`/tmp/opencode/rgp-iWwTyg/` (native resume) and
`/tmp/opencode/gpu-ray-regression-7wbf47/` (184 real CPU Ray, barrier,
Isaac preflight and Arena checks). This is one host and a CPU learner,
not separate GPU learner, two-node worker placement, grouped rollout
or a matched-throughput claim.

At `da4c2347`, binary-only NexuML wheels installed in an independent
CPU virtual environment with matching Torch `2.11.0+cpu`, torchvision
`0.26.0+cpu`, Ray `2.57.0`, TensorDict `0.14.2`, TorchRL `0.14.0`, and
lock-matched `virtualenv==21.7.8`. The installed learner had no Isaac,
Arena, MuJoCo or ROS runtime and ran outside the source directory with
Python isolated mode. The prepared local worker still ran the pinned
CUDA PhysX runtime. The public train command produced another **64
frames / eight optimizer steps**, a checkpoint and a frame-64 actor
package. Worker PID departure, GPU availability and zero object-store
references were verified again. A fresh native playback process loaded
the export and ran **eight ticks / 24 CUDA transitions** before clearing
the simulation context. This is a dependency-boundary proof, not a
cross-node or published-image proof.
Wheels, hashes and receipts: `/tmp/opencode/gpu-ray-learner-wheels-kt05Cc/`.

## CUDA publication to CPU workers (3.4 slice)

A learner on `cuda:0` ran ordinary Pendulum PPO with two CPU Ray workers
on an explicit borrowed runtime. The driver serialized one CPU policy
copy for worker construction; later updates used the existing versioned
TorchRL transport from the original CUDA learner. Both workers'
parameters matched the learner bit-for-bit at versions **2 and 3**;
the learner reached **32 physical frames / eight optimizer steps**.
The collector closed without shutting down the caller's Ray runtime.
No CUDA worker was requested in this pairing. A CUDA learner together
with CUDA workers still fails preflight pending that topology's proof.
Receipt: `/tmp/opencode/rgcl-PxTiUd/`.

The combined regression at `f1f66e52` passed **188 checks**, including
both native device directions, worker cleanup, Ray barriers, direct Isaac
preflight and Arena. Source hashes and results:
`/tmp/opencode/final-ray-GF37RO/`. The docs check also passed: **67 tests,
33 subtests**, zero Astro diagnostics and verified rendered links
(`/tmp/opencode/ray-cuda-transfer-docs-XaEA9B/`).

## Worker dependency setup diagnostic (3.4, not native qualification)

Ray's `py_executable` can run `uv run --frozen`, but that puts dependency
setup inside worker startup. A cold-cache miss failed in the worker
process, and the raylet retried after its 60 s registration timeout.
Stock-pool Isaac cold setup took nearly seven minutes, so this probe does
not justify putting the entire install inside `py_executable`.
Receipt: `/tmp/opencode/rev-Qc2M3g/`.

Ray 2.57's native actor `runtime_env.uv` instead installs before launching
the actor. In a clean local Ray/pip/virtualenv runtime, the plugin installed
a lock-exported CPU wheel with `--require-hashes --only-binary :all:` and
passed `uv pip check`. The actor loaded the package from its own environment;
the owner could not import it before or after. An invalid hash raised
`RuntimeEnvSetupError` before the actor ran, and the caller could still
submit a separate task. Receipts: `/tmp/opencode/rh-35EdBB/`.

The plugin requires `pip` and `virtualenv` in the base runtime. If Ray is
started inside a virtualenv, the plugin clones that environment first. A
probe against the full development environment timed out while copying it
and left a clone helper, which was stopped before removing that attempt's
temporary copy (`/tmp/opencode/ru-mW40Z2/`). Launching Ray from an already
heavy learner environment is not a qualified substitute for stock workers.

No new NexuML config is exposed by these probes. The full hash-pinned
Isaac binary set, installed learner import boundary, distinct worker
nodes, GPU learner and actor admission/cleanup still require an actual
combined run.

The frozen `isaac` + `ray` lock export (CPython 3.12, Linux x86-64,
excluding workspace wheels) resolved **319** packages. **316** had a
compatible hash-pinned wheel. There was no matching locked wheel for
`antlr4-python3-runtime==4.9.3`, `idna-ssl==1.1.0`, or the Isaac Lab
3.0.0 Git pin. The earlier stock-worker `uv sync --locked` log shows
source builds for all three. That playback proves the functional path,
not a binary-only install. To test the full binary-only Ray environment,
first supply approved immutable wheels for these three inputs or a
compatible upstream lock. No source build, dependency exclusion, or new
worker config was used in this audit. Receipt:
`/tmp/opencode/nex212-isaac-ray-lock-C1UYDj/`.

The standalone Isaac image manifest already pins the local Lab and ANTLR
wheel hashes (`containers/isaac/artifacts.json`). These receipts establish
the two image inputs, not an Isaac+Ray Python dependency closure on the
stock worker. The standalone profile takes Sim from its NVIDIA base.
The pip-based stock profile still needs an approved immutable `idna-ssl`
binary or a compatible upstream lock, followed by a full binary-only
worker install. A matching hash for a locally built cached `idna-ssl`
wheel is not approval to use it.

The `idna-ssl` 1.1.0 and ANTLR 4.9.3 PyPI release APIs advertise only
sdists. PyPI now has a Lab 3.0.0 wheel, but its SHA-256 differs from the
reviewed wheel in the image manifest and the vendor `v3.0.0` release
points to `e3f7d9f`, not the pinned `70ab24f`. The version label alone is
not sufficient to substitute it. A read-only search inside the local
pinned combined image also found no `idna-ssl` wheel in `/isaac-sim` or
`/env`; this does not justify extracting and repackaging installed files.
Receipt: `/tmp/opencode/nex212-vendor-wheel-audit-VjLEya/`.

Read-only admission prerequisites were refreshed without submitting
an incompatible job. The local RTX 4080 SUPER is the only visible GPU.
The only Kubernetes allocation with more than one GPU was eight V100s
on driver 535.247.01, outside this Isaac profile. Both Ray GPU worker
groups request one GPU per pod. This inventory does not measure free
capacity or prove admission across suitable single-GPU nodes. Receipts:
`/tmp/opencode/nex212-placement-prereqs-BxJtrk/`.

## Matched one-host PhysX profile (3.5 slice)

The existing opt-in Cartpole profile ran the same native PhysX task in
two fresh processes on the RTX 4080 SUPER (driver 610.43.03): direct
CUDA collection and CUDA learner, then one whole-GPU Ray actor and a CPU
learner. Both used the frozen actor initialized at seed 17, fit seed 31,
eight worlds, 16 decisions per world per batch, and one PPO epoch with
minibatches of 64. Each completed
**1,024 physical frames, eight batches, 16 PPO minibatch updates and 32
optimizer steps** with finite losses. The Ray learner never imported Kit;
the worker accepted and returned all eight expected behavior versions.

| Lane | Fit start to first collection (s) | Batches 3–4 (frames/s) | Batches 5–6 (frames/s) | Batches 7–8 (frames/s) | Total fit (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct CUDA | 4.96 | 959 | 962 | 394 | 6.83 |
| One CUDA Ray worker, CPU learner | 9.96 | 1,065 | 1,035 | 16 | 27.10 |

The windows measure collection/handoff and optimizer/weight publication;
they are not independent replicates. In batches 3–6, the Ray lane took
0.4403 s collecting and 0.0476 s updating/publishing, compared with
0.4737 s and 0.0597 s for direct. The final Ray fetch took 16.08 s:
the pinned TorchRL finite collector calls `env.close()` before yielding
its last batch. The otherwise comparable final-window time therefore
includes Kit teardown, not just rollout latency. Retain the full fit time
and report that cost separately from steady-state throughput.

Ray batches carried **10,496 bytes of tensor data** each; one policy
publication carried **18,440 tensor bytes**. These are logical payloads,
not serialized wire bytes. The receiver verified a fresh GPU allocation
after native close and exact actor PID reaping. The 96 MiB Plasma store
had zero retained objects. Device-wide GPU samples until the final batch
peaked at 3,797 MiB (direct) and 4,040 MiB (Ray), including the 1,098 MiB
ambient baseline. Mean utilization over the three measured windows was
13% (6 samples) and 3% (83 samples, dominated by close); these are not
worker-exclusive GPU measurements. Learner process peak RSS was 4,509,436
KiB and 1,161,016 KiB respectively; the latter excludes Ray and Kit
worker RSS.

Direct/Ray runs ended 56/57 episodes (all terminations), with sampled
reward sums 13.82966/13.78006 and actor L2 changes 0.228491/0.179992.
Neither equal actor initialization nor native task identity makes the
stochastic CPU/GPU runs trajectory-equivalent. This is a local systems
profile, not a learning-quality comparison or a speedup claim.

All four opt-in Cartpole profile lanes passed in 62.62 s:
`NEXUML_RUN_SIMULATOR_TESTS=1 .venv/bin/python -m pytest -q -ra tests/fixtures/test_mujoco_warp_profile.py`.
Source, lock, phase and GPU receipts:
`/tmp/opencode/nex212-cartpole-ray-profile-dfEXwr/`. Split-GPU and
multi-node timings, network bytes, and grouped native rollouts remain
unqualified; task 3.5 stays open.
