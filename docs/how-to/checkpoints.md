# Checkpoints

NexuML uses PyTorch Lightning's checkpoint system. This guide covers saving, resuming from, and selectively loading checkpoints.

## Prerequisites

- NexuML installed (`uv sync`)
- A scenario that has been trained (or is in progress)

## Where checkpoints are saved

Lightning checkpoints are saved automatically during training to:

```
.experiments/checkpoints/<scenario-name>/
├── last.ckpt
└── epoch=4-step=250.ckpt
```

The exact path depends on `NEXUML_LOGS_ROOT` and any `ModelCheckpoint` callback configuration. Add callbacks to the scenario to customize:

```python
from nexuml.core.types import ScenarioSpec, CallbackSpec

ScenarioSpec(
    name="my_scenario",
    callbacks=[
        CallbackSpec(
            type="checkpoint",
            params={
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 3,
                "filename": "{epoch:02d}-{val/loss:.4f}",
            },
        ),
    ],
    ...
)
```

## Resume training from a checkpoint

```bash
nexuml train my-scenario --trainer-checkpoint .experiments/checkpoints/my-scenario/last.ckpt
```

This resumes the full Lightning trainer state: epoch count, optimizer state, LR scheduler state, and model weights.

## Fine-tune from a package or checkpoint

To selectively load weights from a previously exported package or checkpoint (without resuming the trainer state), use `CheckpointLoadSpec`:

```python
from nexuml.core.types import ScenarioSpec, CheckpointLoadSpec

ScenarioSpec(
    name="fine_tuned",
    checkpoint=CheckpointLoadSpec(
        source="packages/pretrained/state_dict.pt",
        include=[],              # empty = load all layers
        exclude=["head"],        # skip the classification head (will train from scratch)
        allow_missing=True,      # new layers without pretrained weights are OK
        allow_shape_mismatch=True,   # mismatched shapes are logged and skipped
        freeze_loaded=False,     # set True to freeze pretrained weights during training
    ),
    ...
)
```

## `CheckpointLoadSpec` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `str \| None` | `None` | Path to a state dict `.pt` file or exported package directory |
| `include` | `list[str]` | `[]` | Stage/layer name prefixes to include (empty = all) |
| `exclude` | `list[str]` | `[]` | Stage/layer name prefixes to exclude |
| `allow_missing` | `bool` | `True` | If `True`, layers in the model with no matching key in the checkpoint are initialized normally |
| `allow_shape_mismatch` | `bool` | `True` | If `True`, keys with mismatched shapes are logged as warnings and skipped |
| `freeze_loaded` | `bool` | `False` | If `True`, parameters loaded from the checkpoint are frozen (no gradient) |

## Selective loading examples

### Load encoder only, train new head

```python
CheckpointLoadSpec(
    source="packages/pretrained/state_dict.pt",
    include=["encode"],    # load only the "encode" pipeline stage
    exclude=[],
    allow_missing=True,
)
```

### Transfer learning with frozen backbone

```python
CheckpointLoadSpec(
    source="packages/backbone/state_dict.pt",
    include=["backbone"],
    freeze_loaded=True,    # backbone weights frozen during fine-tuning
    allow_missing=True,
)
```

## Export after training

After training, export the pipeline to a portable package:

```bash
nexuml export my-scenario --checkpoint .experiments/checkpoints/my-scenario/best.ckpt
```

See [Model export and reload](export.md) for the full export API.

## Implementation map

- `src/nexuml/core/types.py` — `CheckpointLoadSpec`, `CallbackSpec`
- `src/nexuml/training/lightning.py` — Lightning checkpoint resume, selective loading
- `src/nexuml/cli/main.py` — `--trainer-checkpoint` flag on `train`
- `src/nexuml/core/export.py` — `load_package` for package-as-checkpoint-source

## See also

- [Train a model](train.md)
- [Model export and reload](export.md)
- [Tracking and logging](tracking.md)
