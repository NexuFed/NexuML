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
rejects GPU placement before allocation; no distributed Warp run was attempted.
The CPU fixture, assets, learner, collector, package manifests and planning
status are unchanged by this slice.
