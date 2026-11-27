# Native baseline and comparison protocol

## Starting point

Implementation baseline: `c745927b505b1ca2402186217949c7d454e8ee45`; checkout
`fcb4a81a150b0eb272be6708315ea3fa64e00813` changes planning, not source. The
worktree interpreter is Python 3.12.16 with PyTorch 2.11.0+cu130, TensorDict
0.14.2, TorchRL 0.14.0, Ray 2.57.0, Lightning 2.6.5, Isaac Sim 6.2.0.0, Isaac
Lab 3.0.0 from `70ab24fec3eabf51222817db0cdb5a4e7133bd7e`, and RSL-RL
5.5.1. The local CUDA probe saw one RTX 4080 SUPER with driver 610.43.03.
Arena is not installed in this interpreter. No dependency changes were made.

| Boundary | Baseline and requirement ownership |
| --- | --- |
| Training and native runtime | `nexuml train` / `NexuSession` use a serialized `EnvironmentDefinition` and execution-local `EnvironmentBuildContext`. Native Isaac accepts one synchronous direct collector and one learner on the same CUDA device. `add-reinforcement-learning` owns the canonical learner; `add-isaaclab-gpu-robotics` owns existing direct support. New placement here does not generalize historical qualification. |
| Native configuration, sensors, groups, viewers | Keep NC1–NC6 and V1–V5 from `generalize-isaac-native-configuration-and-marl`. The runtime supports native manager, direct and grouped tasks. The viewer is opted in per purpose and read-only. Direct tasks have no public arbitrary row reset; acknowledge actual native autoresets without stepping or resetting a second time. |
| Evaluation and collection | `nexuml evaluate` accepts complete packages and trusted trainer checkpoints. Ordinary Isaac replay is fresh-process, N=1; automatic in-fit Isaac evaluation is rejected. `ArenaBenchmarkEnvironment` is evaluation-only. Current TorchRL Ray collection is synchronous CPU-only and starts a local runtime if not already attached. Ray Train and Ray Jobs are separate; `qualify-physical-ai-workflows-via-ray-jobs` owns the existing Jobs evidence and pool policy. |
| Shared policy and provider handoff | `export_package` and NEX-234 `local_fit` / `local_fit_manifest` retain the existing dataset-only identity, scope and numeric validation. `add-isaac-physical-ai-backend` owns that behavior; NexuFL owns federation. `add-mujoco-reference-workflows` owns the CPU workflow; do not replace it with a GPU dependency lane. `integrate-robotics-stack-and-teleoperation` consumes this change's generic runtime/inference seams and owns robot integration, not simulation/learning topology. |

Historical evidence is not a current-head pass. The Ray
[support handoff](../qualify-physical-ai-workflows-via-ray-jobs/support-handoff.md)
records Cartpole lifecycle on the RTX 4070 (1,024 frames / 160 optimizer steps)
and a separately qualified one-seed learning result. Prior local native SO-101
sign/limit/held-success/timeout and PPO resume/export/replay were recorded in
`reports/physical-ai/isaac-backend.md` §§17 and 20. They prove neither a matched
native learning comparison nor learned manipulation. The grouped reference
workflow already has explicit native acceptance fixtures; those are not a
distributed qualification. NEX-248 Arena benchmark evidence does not establish
ordinary Arena training. Historical task records remain unchanged.

## Pinned interfaces and placement probe

- Lab 3.0 `launch_simulation` resolves physics and enables cameras from the
  actual task configuration; launch precedes native environment construction.
  The current single-Kit process scope closes the environment before the
  public non-returning fast shutdown. Manager `compute_final_obs` preserves
  pre-reset observations and public `reset(env_ids=...)`. Do not use the
  restore-from-reset observation as a timeout value target or duplicate resets.
- The TorchRL 0.14 `IsaacLabWrapper` default mode with a public `final` reader
  preserved final/timeout bootstrap truth and avoided double resets in the
  actual manager oracle, but rejected an explicit partial reset outside the done
  mask. Native-autoreset mode lacks the required final observation in the
  existing oracle. Retain the existing runtime adapter. Isaac Lab's
  `RslRlVecEnvWrapper` emits `time_outs`; RSL-RL 5.5.1 PPO bootstraps that reward
  adjustment with the previous value, not the explicit final observation.
- [Pinned Arena builder](https://github.com/isaac-sim/IsaacLab-Arena/blob/a2f421e918952ee139a9f1a7ef8f9b23047a99df/isaaclab_arena/environments/arena_env_builder.py)
  exposes `build_env_cfg()` (configuration and constructor kwargs) separately
  from native registration/construction. It imports `isaaclab_arena_curobo`
  even with the optional relation solver disabled; a prepared compatible Arena
  runtime is needed to verify import order, construction and teardown. No
  production Arena training path is qualified by reading the builder.
- The installed TorchRL RayCollector accepts `ray_init_config` and per-worker
  `remote_configs`, forwarding to Ray's native actor resources. Ray 2.57
  accepts whole-GPU requests, custom resources and actor `label_selector`.
  `update_policy_weights_` uses the maintained Ray weight sync path with
  blocking receiver acknowledgments when configured synchronously. Verify the
  barrier and worker teardown live before enabling distributed Isaac; the
  existing CPU-only shutdown repair and borrowed runtime behavior stay in place.
- Kubernetes API, read-only: RayCluster ready; the normal GPU worker group
  declares 0–4 replicas with one GPU per replica. The DGX group is not a
  substitute. Node inventory and worker declarations do not establish that three
  slots are simultaneously schedulable, have the right runtime, or are on
  distinct nodes. No scheduling or infrastructure changes were made.
- Offline cross-image serialization probe: Ray 2.58.1 cloudpickle serialized
  a CPU TensorDict with `[env,time]` axes, `float32` observations, and separate
  terminated/truncated masks in both directions between the local Isaac image
  `fc3bcf75436d…` (Python 3.12, TensorDict 0.14.2) and the simulator-free release
  image `e2b0c096ed21…` (Python 3.13, TensorDict 0.15.6). Both exited 0 with exact
  axes, values and flags. These are not a valid shared Ray runtime tuple:
  Python minor differs and the release image lacks TorchRL. No live actor,
  GPU/object-store transfer or production worker compatibility is implied.

## Direct correctness and performance

Run the existing headless native oracles and lifecycle tests before measuring.
Use the existing eight-environment, 16-step, 256-frame, seed-0 state reference
for Cartpole (Kit PhysX) and SO-101 (Kit PhysX), one epoch and 64-sample
minibatches. Use the existing grouped Pendulum IPPO fixture (seed 0, two
environments, eight steps/batch, 64 frames, one epoch and 16-sample minibatches).
The fixtures check actual counts, changed finite weights, continuation,
export and fresh replay; the SO-101 known-action oracle separately checks both
directions per named joint, held success, and timeout distinct from success.
Short-horizon grouped fixtures exercise final observations; they do not
establish grouped-policy learning quality.

Record process wall time, peak RSS and actual physical frame/update counts for
these probes. Time rollout/update performance separately from startup, resume,
export and replay; a whole-test wall time is not frames/s. For SO-101 learning
quality and warmed timings, use the independent protocol below.

Current direct correctness result: **5 passed in 131.37 s** in the pinned
worktree environment, with exact output under
`/tmp/opencode/nexuml-decouple-direct-rfy6CO/`. Cartpole and SO-101 each resumed
from 128 to 256 physical transitions in two batches; the exported policy
manifests report frame 256. Pendulum IPPO resumed from 32 to 64 physical
transitions, with six ending rows verified in each fit segment and no double
resets. Both fresh scalar replays and grouped decentralized replay passed.
The Cartpole and SO-101 actor state hashes are `cdfdd0e7fa0c…` and
`5995874571c6…`; grouped actor hash is `b7b4fbe60ca2…`. No performance or quality
claim is made from the whole-test wall time.

Separate fresh fits used exactly those seed-0 configurations on `cuda:0`, with
viewers and loggers off. These small fits include warm-up and are not sustained
throughput estimates. Collection and update intervals were synchronized at the
Lightning batch boundaries; process wall includes interpreter setup and native
teardown. The lifecycle tests above, not this timing probe, cover resume,
artifact production and independent replay.

| Direct reference | Frames / minibatch updates | Startup / collect / update (s) | Fit frames/s | Process wall (s) | Peak RSS / sampled GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cartpole PPO, Kit PhysX | 256 / 4 | 4.642 / 0.460 / 0.144 | 424 | 8.659 | 4694 / 3402 |
| SO-101 PPO, Kit PhysX | 256 / 4 | 5.630 / 0.490 / 0.162 | 393 | 9.693 | 4636 / 3410 |
| Pendulum IPPO, Newton | 64 / 4 | 4.126 / 1.033 / 0.196 | 52 | 10.526 | 2628 / 1376 |

PyTorch peak reserved memory was 22 MiB for each timing probe. GPU memory is
total device allocation sampled every 250 ms, including the desktop and
non-PyTorch simulator allocations. Raw timing rows and the exact probe source
are in the receipts below.

## Matched SO-101 decision gate (predeclared)

- Reuse the exact native `NexuML-SO101-Joint-Reach` task, its PhysX asset
  manifest and 4-second horizon. Use 15 ordered state inputs and five
  normalized absolute position targets; the gripper stays fixed. No viewer,
  camera, observation normalization, observation corruption, or online task
  changes. Same installed stack, GPU and serial fresh-process runs.
- Train NexuML PPO and the published Isaac Lab RSL-RL wrapper/runner on seeds
  **17, 43, 97**: **32 environments × 32 control steps per rollout =
  1,024 transitions**, **131,072 transitions**, 128 updates of the shared
  rollout budget (not 128 optimizer steps). Two epochs, four minibatches,
  64×64 tanh actor and critic, Adam 0.0003, gamma 0.99, lambda 0.95, ratio
  clip 0.2, entropy coefficient 0.01 and gradient norm 0.5. Native uses the
  maintained clipped Gaussian (initial std 0.7); NexuML uses the existing
  state-dependent tanh-squashed Gaussian. Compare against each path's own
  initialization; seeds, model width and control envelope match, not
  stochastic samples or exact initial weights. Record value-loss,
  advantage-normalization and timeout-bootstrap differences rather than
  patching one implementation to mimic the other.
- Evaluate deterministic initial and trained policies in the **same native
  environment**, with 32 full episodes at seeds 10000–10031, independent of
  training; use the task's success and timeout flags, return, final mean
  absolute joint error and episode length. Verify finite state/loss, actual
  counts, strict action bounds, a changed actor/critic, and fresh reload.
  Report means and between-seed spread; report time-to-target at observed
  evaluation checkpoints only.
- Drop the first four batches from warm timing. Report collection and update
  times over each of the next three nonoverlapping 32-batch windows for each
  seed, end-to-end time, export time, process peak RSS/VRAM, and GPU
  utilization. Use the same timing boundaries and note any logging or
  instrumentation differences. Label unsupported or unavailable metrics.
- Retain the shared path only if correctness passes, its pooled held-out
  success improves over initialization by at least 20 percentage points and
  is no more than 10 points below native, its final mean joint error is no
  more than 0.02 rad worse than native, and median warm throughput is at least
  70% of native. If both policies learn poorly, report inconclusive quality.
  If capable but overhead is isolated to redundant glue, simplify that seam
  and requalify; if native is clearly preferable and interoperability or
  maintenance costs are excessive, narrow the framework. A blocked or failed
  comparison does not authorize broad expansion.
- Try the maintained TorchRL Isaac wrapper against the existing terminal and
  partial-reset oracle. Only use it for a matched minimal learner if it
  preserves ending observations and timeout targets for the required
  rollout path. Otherwise record the exact failure and keep algorithm versus
  orchestration cost attribution unresolved. Do not implement a third
  trainer or modify vendor code just to obtain a comparison.

The executable comparison is `scripts/isaaclab/so101_learning_comparison.py`.
Run a new process for each backend, seed and stage; the evaluation stage uses
only trusted artifacts produced by the training stage and the original native
task. A two-rollout `--smoke` exercises the pipeline, not quality acceptance.

```bash
.venv/bin/python scripts/isaaclab/so101_learning_comparison.py rsl-rl train 17 "$OUTPUT"
.venv/bin/python scripts/isaaclab/so101_learning_comparison.py rsl-rl evaluate 17 "$OUTPUT"
```

Repeat with `nexuml` and distinct outputs for the three seeds. Keep `train.json`,
`evaluation.json`, the native logs and artifact hashes with the source revision.
The training receipt counts PPO minibatch updates: NexuML takes a separate
optimizer step per actor/critic for each update; RSL-RL takes a joint step.

## Matched SO-101 result and scope decision

Revision `fcb4a81a150b0eb272be6708315ea3fa64e00813`, with the exact benchmark
driver SHA-256 `fac48e643e3f94be591dcfb42727a2e563b7f38cfd75eb34e5841638af0c2f47`.
All 12 processes exited 0. Each fit collected 131,072 transitions across 128
rollouts and completed 1,024 minibatch updates. Actors and critics changed,
losses stayed finite, and every initial and final artifact reloaded in a fresh
process with finite actions inside `[-1, 1]`. The native reward term records
joint error before the same-step reset; it is not measured from reset state.
The goal is held success at 0.05 rad per joint for at least 0.25 s, not merely
a reduction in average error.

| Learner | Initial → final successes | Initial → final return, mean ± seed SD | Initial → final joint error, rad, mean ± seed SD |
| --- | --- | --- | --- |
| RSL-RL PPO | 0/96 → 0/96 | −0.149 ± 0.036 → −0.083 ± 0.024 | 0.168 ± 0.021 → 0.115 ± 0.019 |
| NexuML PPO | 0/96 → 0/96 | −0.144 ± 0.018 → −0.079 ± 0.025 | 0.164 ± 0.006 → 0.126 ± 0.027 |

Every episode reached the 120-step timeout. Neither learner reached the
success target at either evaluation checkpoint; no held-out learning curve or
time-to-target can be inferred between checkpoints. The receipts retain all
episode rows, per-rollout losses, timings and model/checkpoint hashes.

| Learner / seed | Warm window 1, collect + update s | Window 2 | Window 3 | Train / replay wall s | Train peak RSS / sampled GPU MiB | Train sampled GPU utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RSL-RL / 17 | 5.639 + 0.632 | 5.625 + 0.623 | 5.495 + 0.584 | 31.233 / 42.112 | 4338 / 3410 | 42.0% |
| RSL-RL / 43 | 6.478 + 0.648 | 5.573 + 0.619 | 5.525 + 0.610 | 32.492 / 44.132 | 4334 / 3410 | 41.0% |
| RSL-RL / 97 | 5.696 + 0.629 | 5.654 + 0.591 | 5.128 + 0.567 | 31.882 / 42.133 | 4333 / 3410 | 42.0% |
| NexuML / 17 | 7.158 + 1.301 | 7.169 + 1.306 | 7.307 + 1.327 | 45.748 / 56.897 | 4873 / 3414 | 35.1% |
| NexuML / 43 | 6.990 + 1.280 | 6.980 + 1.293 | 7.943 + 1.462 | 45.261 / 53.076 | 4866 / 3414 | 32.8% |
| NexuML / 97 | 7.649 + 1.371 | 7.209 + 1.351 | 7.980 + 1.572 | 46.250 / 56.020 | 4870 / 3414 | 34.0% |

Median warm throughput across nine windows was **5,247 vs 3,828 frames/s**
(NexuML **73.0%** of RSL-RL; ranges 4,598–5,754 vs 3,431–3,962).
Median startup was 5.652 vs 8.292 s, initial export 0.012 vs 1.055 s, and
final export plus checkpoint 0.012 vs 0.261 s. PyTorch peak reserved memory was
22 vs 24 MiB. Sampled device memory/utilization include other GPU activity and
are not per-process measurements. RSL-RL uses native runner phase timers;
NexuML uses CUDA-synchronized batch hooks. Both update intervals include GAE
and optimization; collection also includes inference and wrapper bookkeeping.
Do not attribute their gaps to physics alone, to Lightning alone, or to a
specific removable adapter call.

Both paths reused the same task registration, PhysX assets and reward; no
task-specific learner changes or new dependencies were needed. Native
RSL-RL uses the Lab vector wrapper and its installed `rsl-rl-lib` runner; the
NexuML workflow adds its installed framework/library, Lightning and TorchRL
for learning. Both depend on Torch and TensorDict. The native comparison
uses squared value loss and a joint optimizer with separate actor/critic
gradient clipping. NexuML uses TorchRL smooth L1 value loss and separate
actor/critic optimizers. Both normalize advantages per minibatch; the
distributions and timeout targets differ as described above. The public
TorchRL wrapper failed the required reset oracle, so there is no third
matched measurement or justified replacement yet. NexuML's package replay
works, but existing external-weight `local_fit` is dataset-only; neither
native actor init nor external-weight interactive fitting was qualified.

**Decision: narrow the next slice to the existing portable policy lifecycle
and a bounded local-fit handoff.** Preserve the verified direct adapter and
keep native direct training available. The point estimates pass the
throughput floor (73%), the native success-gap bound (0 points) and the
joint-error bound (+0.0107 rad). The required improvement over initialization
fails (0 points, minimum 20). Quality is therefore inconclusive, not evidence
that the systems are equivalent or that RSL-RL is better at this task. Do not
extend learner/distributed placement, widen providers, or remove the adapter
on this evidence. Revisit expansion only after a separately predeclared gate
that yields held-out success and a maintainer decision; do not retune this
completed gate after seeing the results.

Raw evidence: [baseline-receipts.tar.gz](baseline-receipts.tar.gz), SHA-256
`83cce1ac212ac8966d6fb1f7c3b2546890487096003e67a663047717d957284a`.
The archive contains the executed driver, invocation and exit logs, GPU
samples, unaggregated train/evaluation JSON and direct timings. Files can be
inspected without loading any code or checkpoints (`tar -tzf`, `tar -xOzf`).
Native process logs and the trusted generated artifacts remain local under
`/tmp/opencode/nexuml-decouple-comparison-tcfKkS/`; all artifact hashes were
verified against the receipts.

## Native actor initialization feasibility

The matched RSL-RL baseline's `MLPModel` has a five-output mean network and a
separate global five-value Gaussian scale. Isaac Lab's native wrapper clips
physical inputs to `[-1,1]`. The existing NexuML PPO actor has a ten-output
head (five means plus observation-dependent softplus scales), with a
tanh-squashed distribution. Copying just the shared hidden layers and five
means does not preserve the deterministic action (`clip(mean)` versus
`tanh(mean)`), the probability of clipped actions at the bounds, or scale
parameter ownership. Disabling normalization on both paths does not remove
these differences. Native JIT reload demonstrates inference through its
original clipping, not trainable initialization into NexuML.

An import path needs one specifically supported native actor/probability
parameterization with an exact bound and scale mapping, plus equivalent
preprocessing and a fresh local PPO fit on the supplied actor. Do not silently
convert the benchmark checkpoints or import native optimizer state. No
trainable native-weight compatibility is claimed.

## Standalone inference handoff

The current `mixed-package` native sensor acceptance passed in 41.64 s:
Newton-Warp sensor capture, interrupted/resumed PPO to frame 256, complete
policy export, and fresh native replay. In a separate Python 3.12.16
environment with CUDA hidden, `uv pip check` passed with only the six
requirements listed by the package and their transitive dependencies:
PyTorch 2.11.0+cu130, TensorDict 0.14.2, NumPy 2.3.1, Pydantic 2.13.4,
ruamel.yaml 0.19.1, and Gymnasium 1.3.0. The latter is currently listed by the
exporter; no simulator or Gymnasium environment was constructed. No
NexuML installation, Lightning, collector/TorchRL, Ray, Isaac, ROS or
NexuFL import was available. This does not qualify a different PyTorch
wheel build or convert the artifact to an untrusted input format.

Executing the existing `tests/fixtures/isaaclab_job.py mixed-package infer`
proof with the isolated interpreter reproduced exact actions and all
processor-state values for the recorded two-row RGB/depth/segmentation/state
sequence, including the selective reset of the first row. The policy
rejected a missing RGB field before inference. Package SHA-256:
`9932eb99123b3eb23356d5458e64cb51471cd8758014f5ffd42c1e8410acd4d8`.
Sequence SHA-256:
`cc2c684d3379f01d9099cce6ee458842d41a5037b33cad645dd3ab9fc1ffadc2`.
The [standalone SDK example](../../../docs/src/content/docs/how-to/export.mdx)
loads a trusted artifact and keeps reset state with the caller. This proof is
for inference on recorded observations, not simulation playback or deployment.

Separately, 131 existing local-fit, export dependency/provenance and
independent policy evaluation tests passed. This preserves the evidence for
the dataset-only SDK; it does not establish interactive external-weight fits.

## Semantic compatibility regression

Ordinary policy evaluation now separates declared embodiment and calibration
from asset source and revision. It accepts matching typed policy semantics
across declared asset provenance, records both identities in the evaluation
report, and rejects a built environment that contradicts its own declared
physical provenance before reset. Existing untyped actions continue to check
the description where it carries the only control mapping. The original
contract hashes, resume checks, and dataset/local-fit manifests do not use
this relaxed comparison.

479 focused tests passed (14 opt-in cases skipped). The native post-change
rerun passed 4/4: Cartpole PPO, SO-101 PPO, Pendulum IPPO, and the mixed-sensor
actor, each including resume, export and fresh replay. The alternate-provider
comparison is a contract regression, not a claim of matched native dynamics.
