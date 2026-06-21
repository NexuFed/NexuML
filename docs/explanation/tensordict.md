# TensorDict data flow

NexuML uses [TensorDict](https://github.com/pytorch/tensordict) as the universal data container flowing through every pipeline layer.

## What is a TensorDict?

A `TensorDict` is a dictionary-like object whose values are tensors. It supports batching, stacking, and device transfers like a single tensor.

```python
from tensordict import TensorDict

td = TensorDict({"audio": waveform, "label": labels}, batch_size=[32])
```

## How data flows through a pipeline

Each `PipelineLayer` receives a `TensorDict` and returns a `TensorDict`:

```
Input TensorDict ──► Layer 1 (feature extractor) ──► Layer 2 (head) ──► Loss
      {audio, label}      {audio, embedding, label}       {logits, label}
```

Layers are free to:
- **Add keys** — e.g., a feature extractor adds `embedding`
- **Keep keys** — pass-through keys unchanged
- **Remove keys** — prune keys no longer needed

## Key contracts

The compiler validates that every key consumed by a layer is produced by an earlier layer. This is the TensorDict equivalent of type-checking: if layer B requires `embedding` but no previous layer produces it, the build fails at resolve time — not at training time.

## Benefits

- **Explicit data flow** — the pipeline diagram shows exactly which keys travel between layers
- **No hidden state** — intermediate tensors are named, inspectable, and debuggable
- **Device-agnostic** — `td.to(device)` moves all tensors at once
- **Batch-safe** — slicing, stacking, and cat work across the whole dict simultaneously

## See also

- [Architecture](architecture.md)
- [Pipeline diagrams](diagrams.md)
- [`nexuml.core.pipeline`](../reference/api/nexuml/core/pipeline.md)
