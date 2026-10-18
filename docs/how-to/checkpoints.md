# Checkpoints and weight reuse

NexuML exposes three different checkpoint-related operations. Keeping them separate avoids ambiguous "load checkpoint" behavior.

## 1. Create Lightning checkpoints

Checkpoint creation is a Lightning callback concern:

```python
from lightning.pytorch.callbacks import ModelCheckpoint

from nexuml import callback

callbacks = [
    callback(
        ModelCheckpoint,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
]
```

Set `dirpath` in the callback if the project needs a specific location. When it is omitted, Lightning writes checkpoints under the active logger's run directory, or under the trainer's `default_root_dir` when no logger is configured. Read `best_model_path` or `last_model_path` from the `ModelCheckpoint` callback after training instead of assuming a particular `version_0/...` path.

The Pendulum/CartPole references use native TensorBoard logging by default. For intermediate
RL checkpoints, add this callback to a resolved YAML config (no source edit needed):

```yaml
callbacks:
  - factory: lightning.pytorch.callbacks:ModelCheckpoint
    kwargs:
      every_n_train_steps: 1
      save_last: true
      save_top_k: -1
```

This saves at batch boundaries; PPO optimizer steps and collected frames are different
counters. With no `monitor`, no validation loss is required. Saving every batch is useful
for a tiny interruption/resume check, not a storage policy for a long run.

## 2. Resume a Lightning training run

```bash
nexuml train my-scenario --trainer-checkpoint /path/to/last.ckpt
```

This restores the Lightning trainer state: model state, optimizer/scheduler state, epoch/global step, and compatible callback state. A compatible NexuML scenario is persisted in the checkpoint and can be recovered for checkpoint-only resume.

Use this when the intent is **continue the same training run**.

RL restores the complete policy, critic, optimizers and frame/update counters, then rebuilds
environments/collectors. Remaining collection uses the saved frame budget; simulator state
and collected experience are not restored as live objects. Scheduled evaluation continues
at the next future frame interval, without repeating frame-zero media. A checkpoint already
at its budget has no additional updates; it is not a new run initialized at frame zero.

To find the actual latest checkpoint under your configured log root:

```bash
find "$NEXUML_LOGS_ROOT" -type f -name '*.ckpt' -print
CHECKPOINT=$(python -c 'import os; from pathlib import Path; print(max(Path(os.environ["NEXUML_LOGS_ROOT"]).rglob("*.ckpt"), key=lambda p: p.stat().st_mtime))')
nexuml train --trainer-checkpoint "$CHECKPOINT"
```

Choose a checkpoint from the intended run when the root contains multiple runs; the snippet
selects the most recently modified file. Use an intermediate checkpoint to recover an
interruption. The [RL round trip](reinforcement-learning.md#fast-cpu-round-trip) establishes
the log root and creates small checkpoints for both references.

## 3. Reuse selected model weights

Use `CheckpointLoadSpec` when the intent is **initialize a new scenario from previous weights**, not resume the old trainer:

```python
from nexuml.core.types import CheckpointLoadSpec

checkpoint = CheckpointLoadSpec(
    source="packages/pretrained",
    include=["stages.Encoder.*"],
    exclude=["*.Head.*"],
    allow_missing=True,
    allow_shape_mismatch=True,
    freeze_loaded=False,
)
```

`include` and `exclude` are glob patterns matched against the actual compiled pipeline `state_dict()` keys. They are not semantic stage aliases. Check the state-dict key names before writing a narrow pattern for a particular model.

Supported sources include exported package directories and compatible state/weight artifacts handled by the export loader.

## `CheckpointLoadSpec`

| Field | Meaning |
| --- | --- |
| `source` | artifact/checkpoint path |
| `include` | optional glob allow-list |
| `exclude` | optional glob deny-list |
| `allow_missing` | permit target keys with no loaded source value |
| `allow_shape_mismatch` | skip incompatible source shapes instead of failing |
| `freeze_loaded` | freeze parameters that were successfully loaded |

## Which one do I want?

| Goal | Mechanism |
| --- | --- |
| save best/last during training | `callback(ModelCheckpoint, ...)` |
| continue an interrupted run | `--trainer-checkpoint` |
| transfer learning / initialize a new architecture | `CheckpointLoadSpec` |
| distribute a trained model | [Export package](export.md) |
| replay/record a complete RL policy without training | `nexuml evaluate <policy-package>` |

## RL policy export and replay are not resume

```bash
nexuml export pendulum-ppo --checkpoint "$CHECKPOINT" --output /tmp/opencode/pendulum-policy
nexuml evaluate /tmp/opencode/pendulum-policy \
  -O training.evaluation.episodes=1 -O training.evaluation.max_steps_per_episode=16
```

Use the matching scenario (`cartpole-ppo` for a CartPole checkpoint). Python callers export
`result.policy`, not its raw neural `result.pipeline`. Complete RL policy packages preserve
the action adapter and interaction contract but exclude critic/optimizer/rollout/runtime
state; they cannot resume learning. This RL boundary does not change supervised packages'
existing training-state sidecars.

Replay loads the trusted executable package directly and creates a separate native
evaluation run, not a training session. Source frames use explicit `training_frames`
metadata or standalone step zero, never optimizer steps. See [recording and native logging](reinforcement-learning.md#record-gif-mp4-or-both)
for GIF/MP4/both, MLflow-only and optional rendering dependencies. Packages and checkpoints
can execute Python: load only artifacts you trust.

The [Coming from Lightning](../learn/from-lightning.md) page shows the same distinction in Lightning terms.
