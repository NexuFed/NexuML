# Task 1.5 — direct regression evidence

**Task 1.5 fixture regressions passed on the final candidate:** 6 native,
177 CPU supervised/dataset, 1 CUDA dataset and 17 hermetic Arena cases.
The before/after source indexes matched. The original NEX-234 dataset fit and
native Arena remain unavailable, separate from the fixture acceptance. The
matched SO-101 learning gate is historical evidence, not a current rerun.

## Runtime and reproduction

`HEAD` was `fcb4a81a150b0eb272be6708315ea3fa64e00813`; both NexuML 0.2.1
packages were editable imports from the dirty worktree, not release wheels.
The interpreter was Python 3.12.16 with Torch 2.11.0+cu130, TensorDict 0.14.2,
TorchRL 0.14.0, Lightning 2.6.5, Isaac Sim 6.2.0.0, Isaac Lab 3.0.0 (installed
from `70ab24fec3eabf51222817db0cdb5a4e7133bd7e`), Newton 1.6.1, Warp 1.17.0,
RSL-RL 5.5.1 and LeRobot 0.6.1. One RTX 4080 SUPER, driver 610.43.03.

Final proof root `P=/tmp/opencode/nexuml-decouple-task15-verified-pTY2na`
(2026-11-26 12:15:29–12:17:51 +01:00). Executed
`bash "$P/matrix.sh" "$P"` (exit 0). The driver runs native pytest and then
CUDA dataset pytest within **one foreground**
`flock /tmp/opencode/nexuml-decouple-native-gpu.lock bash -c …` invocation.
The CPU and Arena batches run with CUDA hidden, independently of the lock.
The pytest commands inside the driver and its two fixture scripts were:

```bash
P=/tmp/opencode/nexuml-decouple-task15-verified-pTY2na
export TMPDIR="$P/tmp" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
n=tests/core/test_isaaclab_integration.py
m=tests/core/test_isaaclab_marl_integration.py
# Inside the shared lock, in this order:
NEXUML_RUN_SIMULATOR_TESTS=1 .venv/bin/python -m pytest -q -ra -p no:cacheprovider \
  --basetemp="$P/native" --junitxml="$P/native.xml" \
  "$n::test_native_checkpoint_remaining_budget_export_and_scalar_replay[cartpole]" \
  "$n::test_native_checkpoint_remaining_budget_export_and_scalar_replay[so101]" \
  "$m::test_native_grouped_updates_resume_export_and_fresh_replay[ippo-pendulum]" \
  "$n::test_native_so101_signed_controls_and_held_outcomes" \
  "$n::test_native_public_wrapper_does_not_meet_the_complete_reset_contract[default-reader-Isaac-Cartpole]" \
  "$n::test_native_manager_and_direct_transition_oracle[isaacsim_physx-Isaac-Cartpole]"
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest -q -ra -p no:cacheprovider \
  --basetemp="$P/dataset-gpu" --junitxml="$P/dataset-gpu.xml" \
  tests/core/test_local_fit.py -m requires_gpu
# Outside the lock:
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -ra -p no:cacheprovider \
  --basetemp="$P/cpu" --junitxml="$P/cpu.xml" -m 'not requires_gpu' \
  tests/core/test_offline_policy_training.py tests/core/test_local_fit.py tests/core/test_so101_imitation.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -ra -p no:cacheprovider \
  --basetemp="$P/arena" --junitxml="$P/arena.xml" tests/core/test_arena_policy.py
```

Use new output directories for reproduction. All final pytest commands
exited 0; JUnit reported zero skips, failures and errors. Native ran in
133.76 s, CPU in 42.18 s (one CUDA deselection), CUDA in 2.46 s (65 CPU
deselections), and Arena in 4.34 s. `HEAD` and the SHA-256 indexes of the
tracked and unignored code, tests, fixtures, configs and dependency manifests
matched before/after (source comparison exit 0; index SHA-256
`a06aba8cb65ccd56cfe63c56d1690204ecc140414b5b2d0868048e54926b5698`).
The driver SHA-256 is `d04e8228079ab0807529db2a8fea2d8719e7853f98f4add319709247bf09db9d`.
`receipts.sha256` indexes the commands, exit receipts, logs, JUnit reports and
artifact index; readback verified all entries (exit 0; index SHA-256
`607378a0acbce1bb6d761735676f46c89104282b12951d304c59fa60f2c4a254`).

Earlier reruns in `EARLY=/tmp/opencode/nexuml-decouple-task15-tWKKC5` passed
the six native and 177 CPU cases but were not final-candidate evidence:
foundation runtime sources changed during both batches. The earlier
`final/source.diff` retains the second source mismatch.

## Native direct and supervised outcomes

Selection: `test_isaaclab_integration.py`'s existing checkpoint/resume/export/
scalar replay cases `[cartpole]` and `[so101]`, SO-101 signed-control oracle,
manager transition oracle `[isaacsim_physx-Isaac-Cartpole]` and TorchRL wrapper
oracle `[default-reader-Isaac-Cartpole]`; `test_isaaclab_marl_integration.py`'s
native grouped lifecycle `[ippo-pendulum]`. No other physics lanes, grouped
recipes, viewers or hardware were selected. The optional upstream source-CLI
comparison was not selected (`NEXUML_ISAAC_CHECKOUT` unset; installed Lab wheel
is not a source checkout); it is not the completed SO-101 benchmark below.

| Seed-0 fixture | Physical transitions, interrupt → final | Rollouts / minibatch updates / optimizer steps at final | Exported actor `state_dict.pt` SHA-256 |
| --- | ---: | ---: | --- |
| Cartpole PPO, Kit PhysX | 128 → 256 | 2 / 4 / 8 | `cdfdd0e7fa0cba1a6b4f7420af6c1fa9133390cacaa79a85f3d98914bc6265e4` |
| SO-101 PPO, Kit PhysX | 128 → 256 | 2 / 4 / 8 | `5995874571c66283d7317c5bad3b16eaa4d812220d477b9cedbc3bf3195c0e35` |
| Pendulum IPPO, Newton | 32 → 64 | 4 / 4 / 16 | `b7b4fbe60ca2ca6c413e68b92a6b2f7d22e3d47875bcb38030c9379ed6f5df25` |

Counts come from the locally generated trusted checkpoints and native fixture
reports. The two scalar interrupts intentionally exited 1 after checkpoint
and failed-logger finalization; resume, exhausted budget without another
simulator, CPU inference parity and fresh GPU replay exited 0. The grouped
fixture exited 0 at every stage, verified six ending rows in each fit segment
with zero extra resets, and reloaded the exported actor for fresh decentralized
replay. Scalar replay completed two 8-step horizon-limited episodes each;
grouped replay completed two native 5-step truncations. These are lifecycle
checks, not learning quality. SO-101 verified signed joint commands, limits,
fixed gripper, held success and timeout distinct from success. Cartpole
verified seven terminal/GAE cases and explicit masked reset. The TorchRL
wrapper preserved terminal observations and timeout targets but raised
`Cannot partially reset` for a mask not matching done; the nonconformance
assertion passed. Full final-run package, checkpoint and metadata hashes are
in `$P/native-artifacts.sha256`; readback verified every entry (exit 0;
index SHA-256 `7e22d180eb45c7854ee3a17a28dd5f099b590195b2aca804ba4fff93c0458113`).

CPU selection: `test_offline_policy_training.py` (17 passed),
`test_local_fit.py` (65 passed) and `test_so101_imitation.py` (95 passed).
Ordinary supervised training ran three steps without an actor; the supervised
policy ran two reproducible eight-step fits and passed checkpoint resume,
package reload and action parity. The CPU filter deselected the explicit
CUDA local-fit test; it passed separately (exit 0, 2.65 s). The synthetic
dataset SDK fixture used two distinct external bases with bitwise-repeatable
fits, each with eight eligible windows, two episodes and three optimizer
steps; CUDA checked three real optimizer steps and returned CPU tensors.
The SO-101 LeRobot fixture verified admitted and rejected receipt/source drift
through the ordinary session and public SDK; the admitted SDK case ran one
step on 32 eligible windows from four episodes. Authored NEX-234 contract
fixture SHA-256: `b9f02f813bb0763660e9265167b13c345e480e988f80effe20d984dcfc769352`.

In the earlier proof, the documented CPU command
`.venv/bin/python scripts/so101_local_fit.py /tmp/opencode/so101-episodes "$EARLY/original-nex234-fit" --steps 500 --seed 42`
exited 1 **before fitting** at `library/src/nexuml_library/scenarios/imitation/so101.py:250`:
`/tmp/opencode/so101-episodes/meta/info.json` was absent. It was still absent
at final review, and no documented alternative approved path was found in
the local handoff, docs or reports. A real-data rerun needs the original
approved episode files and capture receipt, and current-runtime qualification
if their implementation hashes differ. No new capture or replacement data
was used; no real-data fit or quality result is claimed.

## Hermetic Arena and historical native admission

The final `test_arena_policy.py` run passed all 17 cases. The updated fixture
accepts changed action prose without changing decoded actions and rejects
an equal-width typed `absolute_position` → `relative_position` mismatch.
Package/checkpoint action parity, policy/frame/checkpoint provenance and
suite denominators also passed. Test file SHA-256:
`f5b0a8a62d42d5da87af95ab3db7c3b264100df4056571cf541c6a33870577c7`.
The earlier run had 16 passed / 1 failed because its `[mode]` case changed
only prose and expected rejection. The fixture now matches the
[semantic contract](specs/portable-simulator-interaction/spec.md); no Arena
product code changed for that fix.

The separate original NEX-248 native Arena admission did not score any
episodes. Job `raysubmit_Cr5t5mcaCVkTLmML` ended `FAILED` before native startup
with `Arena requires a prepared Isaac Lab 3.0 / Arena 0.4 runtime`; source
SHA-256 `f8c596e8a1601c79ce8a02d0333e9f63a8c2e2b154c5cee970613d66a27cfe00`
and worker log SHA-256
`284a4574ed9c394587c6cef19d3660d65d5fe24f3291ed788ecf2372b5a67353`.
The final interpreter still has no importable `isaaclab_arena`,
`isaaclab_arena_curobo` or `curobo`. No native Arena rerun or suite artifact
was claimed.

## Original matched SO-101 benchmark (not rerun)

The [matched SO-101 benchmark](baseline.md) used the archived driver
SHA-256 `fac48e643e3f94be591dcfb42727a2e563b7f38cfd75eb34e5841638af0c2f47`,
not the current driver (`b86e8b7e09a87de1af53fc432aec7f2e252b68413543e53a6cb3e2abd7f23aea`).
[Baseline receipts](baseline-receipts.tar.gz) SHA-256:
`83cce1ac212ac8966d6fb1f7c3b2546890487096003e67a663047717d957284a`.
The archive's `comparison/commands.txt` and `comparison/runs.txt` record all
12 original train/evaluate invocations and exits (all 0). A read-only audit
(`audit.py` in the earlier proof root, exit 0) recomputed the aggregate
outcome, verified archive JSON against local originals and matched all 33
recorded model/package/checkpoint hashes (`original-artifacts.sha256` in that
root); it did not execute training or load those checkpoints. Both learners
trained seeds 17/43/97, each with 131,072
transitions, 128 rollouts and 1,024 minibatch updates. All 384 held-out episode
rows ended at the 120-step timeout; success was 0/96 at each checkpoint for
each learner. Final mean joint error: 0.1152 rad native vs 0.1259 rad NexuML;
median warm throughput: 5,247 vs 3,828 frames/s (73.0%). The 20 percentage-point
success improvement target did not pass. The original quality decision
remains inconclusive; the bounded direct reruns cannot change it.
