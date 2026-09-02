# TensorDict data flow

TensorDict is the common named data container flowing through a NexuML pipeline. It lets a pipeline expose intermediate representations without turning every model into a long positional-argument chain.

## Named tensors instead of one anonymous tensor

```python
from tensordict import TensorDict

td = TensorDict(
    {
        "waveform": waveform,
        "class": labels,
    },
    batch_size=[32],
)
```

A pipeline can then build a visible data flow:

```text
waveform
   ↓ feature extractor
spectrogram
   ↓ encoder
embedding
   ↓ head
logits
   ↓ loss / metrics
classification_loss, accuracy, f1
```

## Layer contracts

`LayerSpec` declares which TensorDict keys a component consumes and produces:

```python
LayerSpec(
    component=MyEncoder(width=128),
    keys_in=["spectrogram"],
    keys_out=["embedding"],
)
```

The runtime `PipelineLayer` reads/writes those keys as it executes. The compiler also uses the declared graph and dummy shape propagation to catch many missing-key/shape problems while materializing the pipeline rather than waiting for a long training run.

## Why this matters

- **Explicit flow** — model wiring is visible in the scenario and diagrams.
- **Reusable intermediates** — heads, losses, metrics, evaluation, or exports can consume a named representation.
- **Batch/device behavior** — TensorDict moves/slices related tensors together.
- **Less glue code** — components agree on named contracts instead of a project-specific tuple convention.

## x and y

NexuML data loaders generally produce `(x: TensorDict, y: TensorDict | None)`. The pipeline primarily transforms `x`, while labels can remain in `y` and be routed to components through `LayerSpec.label_key`/related routing options where needed.

## See also

- [Mental model](../learn/mental-model.md)
- [Architecture](architecture.md)
- [Pipeline diagrams](diagrams.md)
