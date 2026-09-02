# Checkpoints and weight reuse

NexuML exposes three different checkpoint-related operations. Keeping them separate avoids ambiguous "load checkpoint" behavior.

## 1. Create Lightning checkpoints

Checkpoint creation is a Lightning callback concern:

```python
from nexuml.core.types import CallbackSpec

callbacks = [
    CallbackSpec(
        type="checkpoint",
        params={
            "monitor": "val/loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": True,
        },
    )
]
```

Set `dirpath` in the callback if the project needs a specific location. Otherwise do not write code that assumes a particular `version_0/...` path.

## 2. Resume a Lightning training run

```bash
nexuml train my-scenario --trainer-checkpoint /path/to/last.ckpt
```

This restores the Lightning trainer state: model state, optimizer/scheduler state, epoch/global step, and compatible callback state. A compatible NexuML scenario is persisted in the checkpoint and can be recovered for checkpoint-only resume.

Use this when the intent is **continue the same training run**.

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
| save best/last during training | checkpoint `CallbackSpec` |
| continue an interrupted run | `--trainer-checkpoint` |
| transfer learning / initialize a new architecture | `CheckpointLoadSpec` |
| distribute a trained model | [Export package](export.md) |

The [Coming from Lightning](../learn/from-lightning.md) page shows the same distinction in Lightning terms.
