# nexuml.evaluation

*module*

Evaluation module for NexuML.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/__init__.py#L1-L1)

## nexuml.evaluation.EvalAlgorithm

*class*

```python
class EvalAlgorithm(ABC)
```

Bases: `ABC`

Re-exported from: `nexuml.evaluation.algorithm`

Base class for post-training evaluation algorithms.

> **Flat batch lifecycle**
>
> - fit_batch(x, y) — accumulate statistics from one train batch (optional)
> - fit_end() — finalize fitting after all train batches
> - eval_batch(x, y) — process one test batch using fitted state
> - eval_end() — finalize evaluation after all test batches
> - results() — return computed metrics as a flat dict

Evaluation algorithms are consumer-only: they read from the pipeline output
TensorDict and never produce new score keys. Score-producing components
belong in the pipeline as PostTrainFitLayer subclasses.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/algorithm.py#L127-L159)

### nexuml.evaluation.EvalAlgorithm.eval_batch

*method*

```python
def eval_batch(x: TensorDict, y: TensorDict | None) -> None
```

Score/process one test batch using fitted state.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/algorithm.py#L148-L149)

### nexuml.evaluation.EvalAlgorithm.eval_end

*method*

```python
def eval_end() -> None
```

Finalize evaluation after all test batches.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/algorithm.py#L151-L152)

### nexuml.evaluation.EvalAlgorithm.fit_batch

*method*

```python
def fit_batch(x: TensorDict, y: TensorDict | None) -> None
```

Accumulate statistics from one train batch.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/algorithm.py#L142-L143)

### nexuml.evaluation.EvalAlgorithm.fit_end

*method*

```python
def fit_end() -> None
```

Finalize fitting after all train batches. Default: no-op.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/algorithm.py#L145-L146)

### nexuml.evaluation.EvalAlgorithm.results

*method* · *abstract*

```python
def results() -> dict[str, float]
```

Return computed metrics as a flat dict.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/algorithm.py#L157-L159)

### nexuml.evaluation.EvalAlgorithm.visualize

*method*

```python
def visualize(logger: Any) -> None
```

Produce visualizations and log them (optional).

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/algorithm.py#L154-L155)

## nexuml.evaluation.create_algorithm

*function*

```python
def create_algorithm(spec: EvalAlgorithmSpec) -> EvalAlgorithm
```

Re-exported from: `nexuml.evaluation.registry`

Materialize and validate an evaluation algorithm definition.

**Returns**

- (`EvalAlgorithm`) — Runtime evaluation algorithm.

**Raises**

- `TypeError` — If the definition builds the wrong runtime type.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/evaluation/registry.py#L10-L27)

**Modules**

- `nexuml.evaluation.algorithm`
- `nexuml.evaluation.registry`
- `nexuml.evaluation.storage`
- `nexuml.evaluation.utils`
