# Task 5.2 — Direct native MuJoCo Warp reference

**Pass for direct, synchronous, headless single-GPU Cartpole only.**
`tests/fixtures/mujoco_warp_reference.py` publishes the existing
`EnvironmentDefinition`, `EnvironmentRuntime` and paired physical contract.
The unchanged frozen XML comes from `local-fit-receipts.tar.gz`. The fixture
uses upstream `put_model`, `make_data(nworld=N)`, two `step` calls per 60 Hz
decision, and `reset_data` with a CUDA mask. Torch and Warp share the caller's
CUDA stream; controls, positions and velocities are device aliases, while
returned observations own snapshots. The fixture keeps the CPU MuJoCo
survival reward, ±0.2 rad failure threshold, and 120-step time limit.

## Reproduce

Working-tree base `9cb3b1f7be91a36997455721fb1a96e54e740274`; fixture SHA-256
`e966ee91f765231f37e25645adb89691d040335f26b264da209aedc4bd908084`.
RTX 4080 SUPER (`cuda:0`, driver `610.43.03`), Python 3.12.16, MuJoCo
and MuJoCo Warp 3.12.0, Warp 1.17.0 (CUDA Toolkit 12.9), PyTorch
2.11.0+cu130, TorchRL 0.14.0, TensorDict 0.14.2, Lightning 2.6.5.

```bash
run=$(mktemp -d /tmp/opencode/nexuml-decouple-mjw-reference-XXXXXX)
tar -xOf openspec/changes/decouple-simulation-learning-and-evaluation/local-fit-receipts.tar.gz \
  pair/cartpole.xml > "$run/cartpole.xml"
flock -w 180 /tmp/opencode/nexuml-decouple-native-gpu.lock sh -c '
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader &&
  env TMPDIR=/tmp/opencode PYTHONPATH=tests/fixtures .venv/bin/python \
    -c "from mujoco_warp_reference import main; main()" "$1/cartpole.xml" "$1"
' sh "$run" > "$run/run.log" 2>&1
.venv/bin/ruff check tests/fixtures/mujoco_warp_reference.py
.venv/bin/ruff format --check tests/fixtures/mujoco_warp_reference.py
.venv/bin/ty check --python .venv tests/fixtures/mujoco_warp_reference.py
```

Import the fixture by module name, not as `__main__`: the existing export
records its registered definition. The verified archive explicitly includes
that definition's source; export and reload did not require an exporter edit.

## Observed

- Three worlds on a non-default CUDA stream: commands `[+0.5, -0.5, 0]`
  produced `[+50, -50, 0] N` and cart velocities
  `[+0.069356859, -0.069515556, -0.000079413] m/s`. All four state fields
  matched the CPU MuJoCo oracle after two substeps (`rtol=1e-4`, `atol=1e-6`).
  The masked reset cleared only the middle world's control and clock; the
  other rows' state, controls and clocks were bitwise unchanged, and the
  pre-reset snapshot remained intact.
- A scripted positive force terminated one row at step 13 (`0.230533 rad`,
  rewards `[0, 1, 1]`). Masking that row preserved its neighbors. At step 120
  the two uninterrupted rows truncated, not terminated, with clocks about
  2 s; the restarted row was still live. Double close dropped the fixture's
  owned model and data without error and the process exited normally.
- The canonical `NexuSession`/TorchRL direct collector and PPO ran with
  seed 31, 8 worlds, 128 frames per batch, one PPO epoch, minibatch 64:
  **1,024 frames, 8 batches, 16 PPO minibatch updates, 32 optimizer steps**.
  Rollout actions, observations, rewards, endings and learner losses were
  on CUDA; 21 terminal transitions retained their pre-reset observations and
  in-batch successors reflected row resets. No training truncations occurred.
  All eight batches had finite objective, critic and entropy losses
  (ranges `[-0.00142018, 0.00081321]`, `[3.90205, 5.87978]`,
  `[-0.00681437, -0.00589062]`). Actor parameter L2 delta was `0.08573348`;
  the session released its collector.
- The exported actor archive reloaded and produced bitwise-identical CUDA
  actions on two test observations (`0.06374016`, `0.04556442`). The
  standard evaluator used a fresh N=1 native runtime and the reloaded fixed
  actor: seeds 211–214 terminated after `[36, 31, 33, 33]` steps with returns
  `[35, 30, 32, 32]`. This is a correctness smoke, not a learning-quality gate.
- A separate import/contract check passed without loading native modules or
  initializing CUDA; CPU build, play, split-GPU training and Ray training were
  rejected. Targeted Ruff lint/format and ty checks passed.

Raw local logs, probe and training receipts, `receipt.json` and the package:
`/tmp/opencode/nexuml-decouple-mjw-reference-P6LFsC/`. SHA-256:

- `receipt.json`: `b62ec5bb3a5bbfd16420ad705cf5f75bbda64fd963552413c11eb196de5c30a4`
- `policy/pipeline.package`: `acc6565d5599ca1978a1b9022d1e6fa00d8d7d6fa00e81a79a223c6415d9170c`

No split-device, asynchronous, distributed, viewer, performance, cross-image,
or general CPU/GPU trajectory-equivalence claim. The existing Ray collector
rejects GPU placement before allocation. The later
[one-host Ray factory probe](ray-rollout-placement.md#native-warp-factory-lifecycle-32-one-host-slice)
tested native Warp worker cleanup, not distributed learning.
The CPU fixture, assets, learner, collector, package manifests and planning
status are unchanged by this slice.

### Installed binary-only profile

Rebuilt core and library wheels passed the archive check, then ran the same
native physics, PPO, export and replay probe in a fresh `uv` environment.
The binary-only install used the prepared CUDA 13.0 constraint set and
MuJoCo Warp 3.12.0. The first offline attempt lacked a cached GLFW wheel;
the retry used the public package indexes with the same constraints and
no source builds. `uv pip check` passed with 100 packages. Core and library
imports resolved to the installed wheels; only the reference fixture and
XML came from the checkout. Isaac, Ray, ROS, Newton, CARLA and LeRobot were
absent. This is a local installed profile, not a published image or wheel.

Receipts: `/tmp/opencode/nexuml-warp-installed-8vrmws/`. Probe SHA-256:
`511b53b1673ff7bf4c2bbe5ef01f7d27525d9fecf897c5054c431b2bb8939ba3`.
Actor SHA-256:
`905b489595f22328e97fac1f1bff3525d5c306892cf8650d3f169ffa40f001d8`.

## Task 5.4 — Matched-geometry measured profile

**Local state-only profile passed in all three lanes.** This is a comparison
of raw sampled throughput through the same NexuML direct TorchRL/PPO path,
not a simulator speed ranking or an upstream Isaac learner comparison.
Each lane ran one fresh job on the same host, with 8 worlds, 16 decisions per
world per rollout (128 physical frames), 8 rollouts (1,024 frames), and one
PPO epoch with minibatch 64. Initialization seed 17 reproduced the archived
actor SHA-256 `99899fb561134f94b18fcd1cccb2b8efc6b9b4a48c910b9f7e987e1ca588d4a2`;
all three loaded it strictly, then fit with seed 31. All ran the signed ±100 N,
60 Hz mapping and the 120-step horizon. CPU Gymnasium MuJoCo and GPU Warp used
the identical frozen XML; Isaac PhysX used the unchanged paired native USD
(`sha256:c98ce5dbb174876998052d486036fb79e07f52320851526a4ff61c60ed2db043`).

All three logged finite losses, 16 PPO minibatch updates and 32 optimizer
steps and released the collector. CPU MuJoCo / Warp / PhysX recorded
31 / 31 / 56 terminations, zero truncations, and actor L2 changes of
0.12547 / 0.09954 / 0.22849. Native reward sums were 993 / 993 / 13.82966.
Matching aggregate MuJoCo counts and rewards do not establish trajectory
parity: the CPU/GPU reset and policy RNG streams differ. Isaac also differs
in integration, dynamics, autoreset and reward. The short known-action
CPU/Warp oracle above is not a paired rollout oracle.

### Warm windows and phase boundaries

First two batches excluded. Windows are batches **3–4, 5–6 and 7–8**, each
256 frames, within one process per lane. Each cell is
**collect seconds / update seconds / frames per combined second**.
Collection includes actor inference, stepping, reset and batch handoff.
Update includes PPO, policy synchronization and step logging. Local
`perf_counter` brackets Lightning callbacks, with CUDA synchronized at each
boundary; this includes Python/launch overhead and is not kernel-only timing.

| Lane | Window 1 | Window 2 | Window 3 |
| --- | ---: | ---: | ---: |
| CPU Gymnasium MuJoCo | .0304 / .0117 / 6085 | .0314 / .0130 / 5766 | .0338 / .0129 / 5477 |
| GPU MuJoCo Warp | .9548 / .0396 / 257 | .7806 / .0295 / 316 | .6273 / .0259 / 392 |
| GPU Isaac PhysX | .1721 / .0283 / 1278 | .1772 / .0277 / 1249 | .4450 / .0282 / 541 |

| Lane | Child setup before fit (s) | Fit start to first collection (s) | Last batch end to fit return (s) | Total fit (s) | Child wall incl. lock and exit (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU Gymnasium MuJoCo | 3.710 | .067 | .016 | .274 | 4.925 |
| GPU MuJoCo Warp | 3.819 | .845 | .018 | 4.260 | 9.383 |
| GPU Isaac PhysX | 3.756 | 5.150 | .017 | 6.480 | 10.433 |

Peak process RSS before app exit, in the same order: 1,033,136 / 2,322,520 /
4,700,536 KiB. Torch CUDA allocator peaks: n/a / 18,429,952 / 18,448,896 bytes
(not total GPU memory; excludes Warp/PhysX/Kit-owned allocations). All children
exited 0; Isaac logged `SimulationContext cleared` and used its established
Kit fast process shutdown. The windows are not independent repeats; observed
variation, warm native caches, small batch width, own validation/synchronization
and shared host effects limit comparisons. No performance threshold or speedup
claim, and no benchmark of bigger batches, graphics, JIT cold start, or learning
quality. No network/distributed mode was exercised.

### Prepared package, ABI and renderer status

All children ran on Linux x86_64, glibc 2.39, CPython 3.12.16 (`cpython-312`),
Intel i7-14700K with Torch/OMP/MKL thread count 4, PyTorch 2.11.0+cu130
(CUDA build 13.0), TorchRL 0.14.0, TensorDict 0.14.2, Lightning 2.6.5, and
Gymnasium 1.3.0. The GPU lanes used RTX 4080 SUPER (`sm_89`), NVIDIA
610.43.03, logical `cuda:0`; Warp reported Toolkit 12.9 and driver API 13.3.
CPU MuJoCo was run with CUDA hidden. These are measured installed profiles,
not a new dependency lock, build recipe, or compatible version range.

| Lane | Native packages / ABI | Renderer readiness (outside timing) |
| --- | --- | --- |
| CPU Gymnasium MuJoCo | MuJoCo 3.12.0; CPython extension linked to bundled `libmujoco.so.3.12.0` | Explicit NVIDIA EGL produced two different 128×128 RGB frames on the frozen Cartpole (1,318 changed channel values; PyOpenGL 3.1.10). Explicit OSMesa exited 1 (`NoneType.glGetError`; loader found no `libOSMesa`). No fallback or OSMesa support claim. |
| GPU MuJoCo Warp | MuJoCo / MuJoCo Warp 3.12.0, Warp 1.17.0; working Torch CUDA aliases, no JAX needed | Not supported by this state-only fixture; Gymnasium EGL is not a Warp renderer. |
| GPU Isaac PhysX | Isaac Lab 3.0.0, `70ab24fec3eabf51222817db0cdb5a4e7133bd7e`; Isaac Sim 6.2.0.0, Warp 1.17.0 | Headless state-only in this run. See the existing [N=1 Kit PhysX recording](playback-evidence.md#bounded-headless-recording) for actual GIF/MP4 evidence; no viewer or rendering benchmark here. |

Classic CPU Gymnasium Humanoid/Reacher policy smokes still pass. The staged
Meta-World, CPU OSMesa/NVIDIA EGL media suite and GPU Playground/MJX/JAX
work stay with [the existing reference program](../add-isaac-physical-ai-backend/tasks.md);
this Cartpole profile does not qualify them, a prepared CI image, installation
on another ABI, or mixed/split/distributed placement.

### Reproduce and audit the opt-in

Profile run working-tree base `cbb910a84b780879de2de70eaa67c569e2b03116`;
`tests/fixtures/test_mujoco_warp_profile.py` SHA-256
`30a281be0216be76f82edca09f19cf2a9c5c33e3079b8b7e2a16d5cad780d6d2`.
The test uses existing `requires_simulator`, `requires_gpu` and optional
dependency markers, copies only the frozen XML from the archive into pytest's
output, and acquires the native `flock` for each child. Do not acquire the
same lock around the outer pytest command.

```bash
run=$(mktemp -d /tmp/opencode/nexuml-decouple-mjw-profile-XXXXXX)
env TMPDIR=/tmp/opencode NEXUML_RUN_SIMULATOR_TESTS=1 .venv/bin/python \
  -m pytest -q -ra tests/fixtures/test_mujoco_warp_profile.py \
  --basetemp="$run/pytest" -o cache_dir="$run/cache"
# Default lane: all three must skip before native work.
env TMPDIR=/tmp/opencode NEXUML_RUN_SIMULATOR_TESTS=0 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q -ra tests/fixtures/test_mujoco_warp_profile.py \
  --basetemp="$run/default" -o cache_dir="$run/cache"
# CPU simulator lane: classic MuJoCo and paired CPU smoke; GPU lanes must skip.
env TMPDIR=/tmp/opencode NEXUML_RUN_SIMULATOR_TESTS=1 SDL_VIDEODRIVER=dummy \
  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -ra \
  tests/core/test_simulation_scenarios.py::test_mujoco_policy_smoke \
  tests/fixtures/test_mujoco_warp_profile.py --basetemp="$run/cpu" -o cache_dir="$run/cache"
.venv/bin/ruff check tests/fixtures/test_mujoco_warp_profile.py
.venv/bin/ruff format --check tests/fixtures/test_mujoco_warp_profile.py
.venv/bin/ty check --python .venv tests/fixtures/test_mujoco_warp_profile.py
```

The prepared-profile command passed **3 in 25.58 s**. Default gate skipped
**3**; CPU-only passed **3** and skipped **2** for unavailable CUDA.
The existing `rl-simulators` CI job now selects the CPU lane. Its full
test command passed locally in the prepared environment (**196 passed,
1 skipped**), including two corrected assertions that validate standalone
replay's actual exported source frame. The opt-in GPU/CPU profile rerun
passed **3 in 25.78 s** with SHA-256
`e5a9440817a296790ce2f47758f192cc147e6ab9f345ab2c1ee0abb0e759b1c7`.
Rerun receipts: `/tmp/opencode/nexuml-mjw-ci-profile-1Dhl4r/`.
No hosted workflow was dispatched; the native hardware checks used the
prepared local interpreter.

Renderer probe, deliberately separate from the profile (no fallback):

```bash
for backend in egl osmesa; do
  flock -F -w 180 /tmp/opencode/nexuml-decouple-native-gpu.lock \
    env TMPDIR=/tmp/opencode MUJOCO_GL="$backend" PYOPENGL_PLATFORM="$backend" \
    .venv/bin/python - "$run/pytest/test_paired_cartpole_profile_m0/cartpole.xml" <<'PY'
import hashlib
import sys
import gymnasium as gym
import numpy as np

env = gym.make('InvertedPendulum-v5', xml_file=sys.argv[1], render_mode='rgb_array', width=128, height=128)
try:
    from OpenGL import GL

    env.reset(seed=23)
    first = env.render()
    for _ in range(4):
        env.step(np.array([0.5]))
    after = env.render()
    assert first.shape == after.shape == (128, 128, 3) and after.dtype == np.uint8
    assert np.any(after != first)
    print('vendor', GL.glGetString(GL.GL_VENDOR), 'renderer', GL.glGetString(GL.GL_RENDERER))
    print('changed_values', np.count_nonzero(first != after), 'sha256', hashlib.sha256(after.tobytes()).hexdigest())
finally:
    env.close()
PY
done
```

Raw profile receipts and logs:
`/tmp/opencode/nexuml-decouple-mjw-profile-Z7GnG9/`. `profile.json` SHA-256:

- CPU: `30ac5750f37b7c9e32542a761b03e826231d8b93500bf99613cbd6c700e9381d`
- Warp: `89c725da0eca3abbc56e6285f91ae5587e96a21eaef126ae33c301684d2ec9d2`
- Isaac: `6fe817c6a54c4577d6548298f27f540f08409f4fc662f037908c1870919d8c41`

Default and CPU smoke logs: `/tmp/opencode/nexuml-decouple-mjw-ci-smoke-omgld1/`.
Renderer logs: `/tmp/opencode/nexuml-decouple-mjw-profile-HX9lea/renderer-{egl,osmesa}.log`;
EGL RGB SHA-256 `fb676ae1959d82805fa983c4833a50d8b876666574223957422a2530e6acac89`.
