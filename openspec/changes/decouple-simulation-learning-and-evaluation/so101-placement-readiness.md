# SO-101 placement readiness: independent gate revisit

## Predeclared protocol

This is an independent replay of one fixed treatment from the exploratory
[learning follow-up](so101-learning-followup.md), not a retuning or a revision
of either prior experiment. Use only the **lower-exploration 524,288-transition
inference snapshots** and their own initial actors for **both** RSL-RL and
NexuML PPO at training seeds **17, 43, 97**. No other budget, setting, seed,
substitute artifact, or retraining. Validate trusted artifact SHA-256 against
the archived follow-up train receipts before loading; verify treatment, fit
counts and finite recorded losses against the archived design/train receipts.

- Use reset seeds **30101–30132** inclusive, in ascending order. These are
  separate from the training seeds, original `10000–10031`, and exploratory
  vector reset `10000`; do not use them to select or change treatment or
  thresholds. Use the original evaluation geometry: **one native lane,
  one full deterministic episode per seed per policy**. These are fresh
  resets of the same built-in goals, not unseen goals. Report realized goal
  counts and treat the three training seed pairs as the comparison units,
  not the individual episodes as independent learner replicates.
- Use the same `NexuML-SO101-Joint-Reach` native task, assets, headless
  PhysX, ordered 15-input state, bounded five-joint commands, success and
  timeout semantics, and 4 s / 120-step horizon. Record native success,
  timeout, episode length, return, and the exact terminal **mean absolute
  joint error in radians** from `joint_reach_error`, cross-checked against
  the pre-reset `final_obs` and original target (not best, worst-joint or
  post-reset error). Check finite observations, rewards and bounded actions,
  correct accounting and fresh reloads. First replay the scripted
  goal-to-actuator positive control in a fresh process with the same seeds;
  it must solve every episode without timeout, and all three goals must occur.
- Keep training evidence fixed at the archived lower-exploration fits. For
  warm throughput, retain exactly baseline geometry: 32 lanes × 32 steps,
  discard rollouts 1–4, use inclusive windows 5–36, 37–68, 69–100 in each
  of the six fits, with each window's 32,768 transitions divided by the sum
  of native collect + update phase timings. Use the median over nine windows
  per learner, and report the NexuML/RSL-RL ratio. Validate serial matched
  timing provenance, phase boundaries, complete windows and no in-window
  exports. If that is not supported by retained evidence, name the missing
  condition; do not substitute baseline timing, replay speed or new fits.
- Replay under `/tmp/opencode/nexuml-decouple-native-gpu.lock` with the
  existing `.venv`, one native process per learner/training seed (initial
  then 524,288 actions from separately loaded artifacts), in seed order,
  RSL-RL then NexuML. Run no concurrent native GPU job or physical hardware.
- Apply the unchanged [baseline thresholds](baseline.md#matched-so-101-result-and-scope-decision):
  correctness must pass; NexuML pooled success must gain at least **20
  percentage points** over its own initialization and be no more than **10
  points** behind RSL-RL at 524,288; its pooled final mean terminal joint
  error may be at most **0.02 rad** worse; median warm throughput must be
  at least **70%** of RSL-RL. Pool the 32 episodes per training seed with
  equal weight and report individual seed-pair results and between-seed
  spread. If both learn poorly, quality stays inconclusive. Failures or
  unavailable conditions do not pass. If required evidence is missing
  or the positive control fails, stop short of a readiness claim; do not
  change the protocol. A pass could support a maintainer decision to
  implement and qualify placement, not a claim of split placement or Ray
  support.

## Retained training evidence

Protocol SHA-256 before replay:
`7a1094a4868cb5de31bde85392271401d6e8d04c0c1361454eb203487630855f`.
The treatment is already known from the exploratory comparison; the new
reset-seed outcomes, not treatment discovery, are independent. The reset
range was not used in either earlier comparison (checked all six original
and thirteen follow-up evaluation receipts).

Verified [archived receipts](so101-learning-followup-receipts.tar.gz)
(`c37f4ff4348fb0c22d483f9115634e5b9460647e0866fe3c4bcf47fad51850ce`),
its three source digests, installed versions (Isaac Lab 3.0.0, Isaac Sim
6.2.0.0, Torch 2.11.0+cu130, TensorDict 0.14.2, TorchRL 0.14.0 and RSL-RL
5.5.1), six matched designs and 24 artifact hashes (12 actor exports plus
12 NexuML state/adapter sidecars). The archive records 1,048,576 physical
transitions, 1,024 rollouts, 8,192 minibatch updates and finite nonempty
loss rows per complete fit. The predeclared artifact was taken after
rollout 512 (524,288 transitions), not at end of fit. The source checks
actor/critic changes and finiteness at end of fit; no critic was reloaded
or validated at the inference snapshot boundary.

The warm windows are comparable by the original method. Retained logs
corroborate serial, nonoverlapping native fit intervals in the recorded
seed-block order, on the same RTX 4080 SUPER and native stack. The first
in-fit export follows rollout 128, outside every measured window.
Timing boundaries match the original: RSL-RL's own collect/learn timers
vs synchronized NexuML batch hooks, with GAE and optimization in both
update intervals. Differences in instrumentation and PPO implementations
remain; this is not a measure of physics-only or adapter-only cost.

| Learner / training seed | Rollouts 5–36, collect + update s | Rollouts 37–68 | Rollouts 69–100 |
| --- | ---: | ---: | ---: |
| RSL-RL / 17 | 5.841 + 0.631 | 6.607 + 0.735 | 6.232 + 0.680 |
| NexuML / 17 | 8.513 + 1.620 | 7.459 + 1.344 | 7.097 + 1.302 |
| RSL-RL / 43 | 6.485 + 0.729 | 7.408 + 0.970 | 6.461 + 0.791 |
| NexuML / 43 | 7.547 + 1.411 | 7.325 + 1.357 | 8.942 + 1.703 |
| RSL-RL / 97 | 5.750 + 0.588 | 5.717 + 0.602 | 5.541 + 0.619 |
| NexuML / 97 | 8.206 + 1.495 | 7.099 + 1.298 | 7.170 + 1.323 |

**Throughput passes:** medians **4,741 vs 3,722 transitions/s** (RSL-RL
range 3,911–5,319, NexuML 3,078–3,902); NexuML is **78.52%** of RSL-RL
(floor 70%). No use of original comparison times or new training. Peak
process RSS ranges were 4,325–4,345 vs 4,579–4,679 MiB, with 22 vs 24 MiB
peak Torch CUDA reserved memory. Recorded training intervals were
230.09–242.69 vs 334.10–352.06 s (NexuML includes simulator setup); initial
exports took 0.013–0.021 vs 1.343–1.518 s, and end-of-fit export plus
checkpoint took 0.012–0.013 vs 0.268–0.339 s. These receipts do not measure
device-wide peak VRAM or GPU utilization, or exact end-to-end process wall
time.

## Native replay evidence

Replay used checkout `124ea1cdab617e21068e9c695216c1e4d659e933` plus the
experiment driver, with the pinned SO-101 authored PhysX layer manifest
`sha256:101f6f319c5a283fabb9e730264981938ca4fabd6937ce7bc9c0072e5ebe61bf`.
Compared with the follow-up base revision, the native factory now validates
the loaded asset, publishes outcome metadata and offers an opt-in playback
goal override. This replay does not set that override; the random reset,
reward, ordered policy observation, action term and hold criterion are
unchanged.

Seven fresh native processes exited 0, serially on `cuda:0` under the
shared lock (18:33:34–18:37:12 CET on 2026-11-26). The scripted control
solved **32/32** episodes in 8–11 ticks without timeouts. Every policy
replay used the same goal per reset seed as the control: **10 home, 11
positive, 11 negative**. All 416 episodes passed the finite state/reward,
bounded action, native stop and pre-reset error checks. In every pair,
initial actors timed out at 120 steps and both snapshots succeeded
without timeouts in 8–10 steps.

| Learner / training seed | Initial → snapshot successes | Initial → snapshot terminal mean joint error (rad) |
| --- | ---: | ---: |
| RSL-RL / 17 | 0/32 → 32/32 | 0.174554 → 0.009480 |
| NexuML / 17 | 0/32 → 32/32 | 0.168455 → 0.004661 |
| RSL-RL / 43 | 0/32 → 32/32 | 0.195913 → 0.003466 |
| NexuML / 43 | 0/32 → 32/32 | 0.176944 → 0.009418 |
| RSL-RL / 97 | 0/32 → 32/32 | 0.162168 → 0.007978 |
| NexuML / 97 | 0/32 → 32/32 | 0.179162 → 0.008109 |

| Learner | Initial → snapshot return (mean ± training-seed SD) | Initial → snapshot terminal error (rad, mean ± training-seed SD) | Snapshot ticks (mean ± training-seed SD) |
| --- | ---: | ---: | ---: |
| RSL-RL | −0.163300 ± 0.033511 → −0.000180 ± 0.000004 | 0.177545 ± 0.017070 → 0.006974 ± 0.003130 | 9.146 ± 0.397 |
| NexuML | −0.159022 ± 0.018788 → −0.000253 ± 0.000041 | 0.174854 ± 0.005651 → 0.007396 ± 0.002457 | 9.260 ± 0.198 |

**Independent gate passes for this treatment.** Correctness and positive
control pass. NexuML improved by **100 percentage points** (0/96 → 96/96,
minimum +20), was **0 points** behind RSL-RL (maximum 10), and was
**0.000421 rad** worse on exact pooled terminal error (maximum 0.02).
The paired terminal error differences, NexuML minus RSL-RL, were
−0.004819, +0.005952, and +0.000132 rad for seeds 17, 43 and 97. Warm
throughput was **78.52%** of native (minimum 70%). These are observed
outcomes on three training seed pairs, not estimates from 96 independent
fits or evidence of unseen-goal performance.

The original gate remains failed and the exploratory conclusion stays
unchanged. This supports a maintainer decision to **implement and
independently qualify** further placement work, not a support claim. The
run used one GPU and serial replay; it did not exercise split-GPU or Ray
collection, transport or synchronized learner updates. It does not
qualify physical SO-101, new tasks or goals, or policy continuation.

Decision: proceed with implementation and independent qualification of
placement. Keep native same-GPU defaults and leave split-GPU and distributed
support unqualified until their own resource, transition and cleanup gates
pass.

The local configuration proof selected learner index `0,` through the
public CLI. The real GPU job retained simulation, rollout inference and
storage on `cuda:0`; its TensorBoard transitions were exactly 32 then 64 and
the checkpoint reported eight optimizer steps. Invalid or unavailable
indices are rejected without constructing a collector. The focused
preflight, evaluation and grouped regression passed 223 tests.
Receipt: `/tmp/opencode/nexuml-explicit-learner-cuda-Zr5MVD/`.

## Reproduction

Run from the repository root with the retained, trusted artifacts and the
prepared `.venv`. This only reads prior experiment files and writes new
receipts and logs to disk-backed scratch.

Raw receipts and process logs: `/tmp/opencode/nexuml-so101-placement-jflpho/`.
The SHA-256 of the `LC_ALL=C sha256sum *.json` output from that directory
is `c3fe8b5e74c7dbc8f7104086ec424cf5399b5b7e1abd1b1a2b4c05451bdf91b7`.
Replay driver SHA-256:
`1a8f5b6a16273ebba21f934877c53924d09fbe6004693129e6ab0a72895d439e`.

```bash
set -eu
out=$(mktemp -d /tmp/opencode/nexuml-so101-placement-XXXXXX)
export TMPDIR="$out" PYTHONDONTWRITEBYTECODE=1
root=/tmp/opencode/nexuml-so101-budget-full-uayBXC
driver=scripts/isaaclab/so101_placement_readiness.py
.venv/bin/python "$driver" receipts "$root" "$out/receipts.json"
(
  flock -w 600 9
  timeout --kill-after=15s 180s .venv/bin/python -u "$driver" control \
    "$root" "$out/control.json" >"$out/control.log" 2>&1
  test -s "$out/control.json"
  for seed in 17 43 97; do
    for learner in rsl-rl nexuml; do
      name=$learner-$seed
      timeout --kill-after=15s 180s .venv/bin/python -u "$driver" "$learner" \
        "$root" "$out/$name.json" --seed "$seed" >"$out/$name.log" 2>&1
      test -s "$out/$name.json"
    done
  done
) 9>/tmp/opencode/nexuml-decouple-native-gpu.lock
```

The focused CPU suite, `.venv/bin/python -m pytest -q -p no:cacheprovider
--basetemp "$scratch/pytest" tests/core/test_isaaclab_so101.py`, had 30
passes and one unrelated existing failure:
`test_canonical_asset_identity_is_part_of_the_current_portable_contract`.
Its import guard rejects the local relative module `isaaclab_config` as
though it were an Isaac Lab dependency. The experiment driver is not
imported by this test; no out-of-scope fix was made.
