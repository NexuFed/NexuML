# SO-101 local behavior cloning and Python handoff

This is the narrow NEX-234 Gate 0 reference, not federation, vial placement,
ACT/VLA, real-hardware readiness or the full Physical AI backend.

## Frozen native task

Use `IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach")` at Isaac Lab
`70ab24fec3eabf51222817db0cdb5a4e7133bd7e`, Sim 6.2, Python 3.12 and the
source-pinned Sim 6.2 SO-101 PhysX asset. The provider validates all 15 authored
asset-layer hashes, ordered joints, limits, actuator gains and fixed gripper
before action. Do not substitute the historical EA profile or six-action vial task.

- Actor input: `observation`, float32 width 15: five joint positions in radians,
  five velocities in radians/second, five task goals in radians.
- Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
- Action: five float32 values in [-1, 1], mapped to **absolute** joint targets
  `0.5 * action` radians, without home-pose offsets. Gripper stays fixed at
  0.2555162024919699 radians; it is not a sixth action.
- Clock: physics 120 Hz, four substeps per command, control 30 Hz.
- Reset: native seeded home pose and one of the existing three goals. Success
  requires every arm joint within 0.05 radians for eight consecutive control
  steps. Four seconds is a timeout, never success.
- Identity preprocessing: no fitted normalization or privileged student inputs.
  Train, validation and test contain different **episodes**, not unseen goal
  classes. This finite three-goal proof does not establish continuous-goal
  generalization.

Native execution needs the prepared isolated environment, CUDA device, approved
NVIDIA EULA and cached licensed assets. The state-only reference needs no camera
or graphics claim. Never rebuild the active container, replace `/env`, change
drivers or actuate hardware to run it. Preserve at least 40 GiB available disk.

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD/src:$PWD/library/src" \
  bash scripts/isaaclab/run.sh tests/fixtures/isaaclab_so101_probe.py
```

## Acceptance protocol

Record actual native expert transitions, using the actor-visible goal to command
absolute joint targets. Save before-action observations and executed commands,
native terminal observations, success/termination/timeout outcomes and timestamps
to a finalized local LeRobot v3 dataset. Never label a reset observation as the
terminal state. Fix the episode split before training (48 train, 12 validation,
12 test in the 72-episode reference); incomplete or unsuccessful expert episodes
are errors, not silently dropped training rows.

Fit the small ordinary regression actor through NexuML's compiled graph,
configured loss, optimizer and `NexuSession`. Export the entire `CompiledPolicy`,
including its action adapter and physical contract, not a pipeline alone.

Freeze evaluation **before** fitting: 60 fresh episodes starting at seed 10000,
with identical native reset distribution, horizon and success predicate for
expert, untrained, random and learned references. Require expert and learned
success at least 95%, learned improvement at least 30 percentage points over
untrained, and zero action/joint-limit violations. Report every failure and the
denominator; a lower offline loss is not acceptance. Reload the package in a new
process for learned evaluation. Do not tune on these evaluation seeds.

Zero-violation acceptance applies to the learned candidate and valid expert.
Retain baseline contract rejections as failed trials; report their counts
separately from native timeouts. Never clip unsafe baseline predictions merely
to hide those failures, or send a rejected command to the simulator.

Run each native stage in its own owned process. Choose **new** output directories;
the capture/fit commands refuse to overwrite existing evidence.

```bash
export PYTHONPATH="$PWD/src:$PWD/library/src"
export OMNI_KIT_ACCEPT_EULA=YES  # only under the operator's existing approval
DATA=/tmp/opencode/so101-episodes
RUN=/tmp/opencode/so101-fit
bash scripts/isaaclab/run.sh scripts/isaaclab/so101_imitation.py collect "$DATA" --episodes 72 --seed 0
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/so101_local_fit.py "$DATA" "$RUN" --steps 500 --seed 42
bash scripts/isaaclab/run.sh scripts/isaaclab/so101_imitation.py evaluate "$RUN/learned" "$RUN/learned.json"
bash scripts/isaaclab/run.sh scripts/isaaclab/so101_imitation.py evaluate "$RUN/untrained" "$RUN/untrained.json"
bash scripts/isaaclab/run.sh scripts/isaaclab/so101_imitation.py evaluate "$RUN/learned" "$RUN/expert.json" --reference expert
bash scripts/isaaclab/run.sh scripts/isaaclab/so101_imitation.py evaluate "$RUN/learned" "$RUN/random.json" --reference random
```

The fit script repeats both initial states, checks bitwise replay, saves numeric
Safetensors and exports separate untrained/learned complete packages. Keep its
`fit.json` and every per-trial evaluation JSON beside the artifacts. The authored
handoff schema is `tests/fixtures/so101_local_fit_contract.json`; actual source,
base-state, dataset and wheel hashes belong to the local run receipts.

## Python SDK trust boundary

```python
from nexuml.training import local_fit, local_fit_manifest

# Both objects come from local operator-owned code/config, never a remote request.
recipe = installed_recipe(approved_private_dataset)
manifest = local_fit_manifest(recipe, approved_initial_tensors)
result = local_fit(
    trusted_recipe=recipe,
    initial_tensors=approved_initial_tensors,
    model_manifest=manifest,
    local_data=recipe.data,
    budget={"steps": 50},  # alternatively {"epochs": 2}; exactly one positive integer
    seed=42,
    optimizer_mode="reset",
    device="cpu",  # default; no automatic GPU selection
)
```

`local_fit_manifest` binds the locally compiled architecture/source, physical
observation/action/processor identity, exact tensor names/shapes/dtypes/roles and
the initial numeric state's SHA256 (`base_version`). Build/check that expectation
locally; do not trust a remote manifest to select executable code. An incoming
round's tensors replace the previous round's model state exactly. Every fit owns
a fresh session and optimizer; optimizer continuation is rejected.

The NEX-234 CPU/reset reference remains the default. Fits are sequential and
single-process, with one complete on-disk episode source, fixed episode partitions,
a fixed batch size and the ordinary Torch loader. Optional `device="cuda:N"`
selects exactly one operator-approved available GPU by its process-visible index;
`auto`, unindexed `cuda`, indexed CPU and other accelerators are rejected before
data loading. Both paths use deterministic training and `32-true` precision through
the existing `NexuSession`; no mixed precision, distributed training or optimizer
continuation is supported. CPU and CUDA results need not be bitwise identical.

Single-CUDA execution requires a real optimizer-on-device proof on the approved
stack before calling it tested; the argument alone is not GPU acceptance or a new
native SO-101 quality result. With the GPU idle and approved for this check, set
the deterministic cuBLAS environment **before** CUDA initialization:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/pytest tests/core/test_local_fit.py -m requires_gpu -q
```

To update only an operator-selected subset, pass the same
`trainable_names=("exact.parameter.name", ...)` to **both** `local_fit_manifest`
and `local_fit`. The tuple must be nonempty, unique and contain only exact names
of parameters already eligible for gradients in the installed recipe. Buffers,
originally frozen parameters, unknown names and glob patterns are rejected before
data loading. `None` preserves the original full eligible scope and default recipe
hash; an explicit scope has its own recipe hash, even when selecting every eligible
parameter. Unselected tensors are frozen and the returned full state must preserve
them exactly. This is not a foundation-model, adapter or PEFT implementation.

Arbitrary window samplers, hidden preprocessing and persistent per-site state
remain unsupported here. Existing supervised/RL session APIs retain their own
broader capabilities.

The recipe's data binding is the local allowlist: `local_data` must match it
exactly before the source is built. A caller changing that operator-owned recipe
is privileged local code, not a remote authorization mechanism. Keep recipe
construction and data validation in the installed client, not in a network
request parser. Never deserialize a remote `torch.package` or pickle as an update.

Each call recompiles/revalidates the model manifest and scans all episode metadata
and numeric windows again; there is no admission cache over mutable paths. The
physical `describe()` contract is captured once per episode receipt, without
skipping any frame validation or changing receipt hashes. Keep the source stable
during validation and training: revalidation is not an atomic snapshot guarantee.
Future reuse requires content-versioned immutable model/data snapshots, explicit
invalidation and separate admission/round tests before any cache is safe.

Outputs are detached contiguous CPU named tensors plus approved finite scalar
metrics and opaque hashes. `update_kind="full"` means full state, **not a delta**.
Tensor roles identify the shared trainable scope; frozen tensors must remain
unchanged. Use Safetensors for numerical transport; executable inference packages
remain separately operator-trusted local artifacts.

`train_windows` is the number of distinct eligible training frames (one-step
windows), `episodes` the number of distinct training episodes. Neither is an
optimizer-step count or a padded-position count. `consumed_windows` counts actual
batch examples, including repeats across epochs; `optimizer_steps` is the actual
Lightning count. Data hashes cover split/clock metadata and numerical inputs and
targets, not private paths. Raw episodes, videos, dataset paths, sessions,
optimizer buffers and training checkpoints are never returned.
Provenance retains the selected-scope manifest and a `fit_config_hash` of the actual
training configuration, including device selection, precision, seed and budget.

## Handoff boundary

Keep datasets, run receipts, generated media, checkpoints and weights outside Git.
The narrowly scoped PR must identify its exact reviewed commit/wheel, numerical
contract fixture, runnable commands and local evidence locations. Do not close
NEX-234 until all eight Gate 0 tasks are verified. NexuFL integration is ready only
after that reviewed artifact is pinned; this reference implements no NexuFL,
REST, gRPC, FedAvg, Kubernetes or automatic promotion/merge.
