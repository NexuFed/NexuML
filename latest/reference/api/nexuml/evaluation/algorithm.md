# nexuml.evaluation.algorithm

*module*

Base class for post-training evaluation algorithms.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L1-L1)

## nexuml.evaluation.algorithm.ContractError

*class*

```python
class ContractError(KeyError)
```

Bases: `KeyError`

Raised when a declared key contract cannot be satisfied at runtime.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L12-L13)

## nexuml.evaluation.algorithm.DistanceEstimator

*class*

```python
class DistanceEstimator(ABC)
```

Bases: `ABC`

Abstract base for streaming distance estimators.

> **Lifecycle**
>
> 1. fit_batch(features, labels=None) — accumulate train statistics batch-by-batch
> 2. fit_end() — finalize (compute inverse covariance, fit GMM, etc.)
> 3. score(features, labels=None) — return per-sample anomaly scores

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L162-L181)

### nexuml.evaluation.algorithm.DistanceEstimator.fit_batch

*method* · *abstract*

```python
def fit_batch(features: torch.Tensor, labels: TensorDict | None = None) -> None
```

Accumulate statistics from a batch of features.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L171-L173)

### nexuml.evaluation.algorithm.DistanceEstimator.fit_end

*method* · *abstract*

```python
def fit_end() -> None
```

Finalize the estimator after all batches.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L175-L177)

### nexuml.evaluation.algorithm.DistanceEstimator.score

*method* · *abstract*

```python
def score(features: torch.Tensor, labels: TensorDict | None = None) -> torch.Tensor
```

Return per-sample anomaly scores (higher = more anomalous).

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L179-L181)

## nexuml.evaluation.algorithm.EvalAlgorithm

*class*

```python
class EvalAlgorithm(ABC)
```

Bases: `ABC`

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

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L127-L159)

### nexuml.evaluation.algorithm.EvalAlgorithm.eval_batch

*method*

```python
def eval_batch(x: TensorDict, y: TensorDict | None) -> None
```

Score/process one test batch using fitted state.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L148-L149)

### nexuml.evaluation.algorithm.EvalAlgorithm.eval_end

*method*

```python
def eval_end() -> None
```

Finalize evaluation after all test batches.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L151-L152)

### nexuml.evaluation.algorithm.EvalAlgorithm.fit_batch

*method*

```python
def fit_batch(x: TensorDict, y: TensorDict | None) -> None
```

Accumulate statistics from one train batch.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L142-L143)

### nexuml.evaluation.algorithm.EvalAlgorithm.fit_end

*method*

```python
def fit_end() -> None
```

Finalize fitting after all train batches. Default: no-op.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L145-L146)

### nexuml.evaluation.algorithm.EvalAlgorithm.results

*method* · *abstract*

```python
def results() -> dict[str, float]
```

Return computed metrics as a flat dict.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L157-L159)

### nexuml.evaluation.algorithm.EvalAlgorithm.visualize

*method*

```python
def visualize(logger: Any) -> None
```

Produce visualizations and log them (optional).

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L154-L155)

## nexuml.evaluation.algorithm.get_declared_axis

*function*

```python
def get_declared_axis(x: TensorDict, y: TensorDict | None, axis_spec: Any, *, metadata: Any = None, algorithm: str = '') -> Any
```

Resolve a declared axis key from x, y, or metadata per its provenance.

Enforces EXACT provenance — no silent fallback between sources.
axis_spec may be a string (shorthand, defaults source=y) or an AxisKeySpec.
``metadata`` is an optional DataFrame-like with per-sample columns.

**Returns**

- (`Any`) — The resolved axis value from the appropriate source.

**Raises**

- `ValueError` — If *axis_spec* references an unknown source.
- `ContractError` — If the declared key is missing from the resolved source.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L36-L92)

## nexuml.evaluation.algorithm.get_declared_tensor

*function*

```python
def get_declared_tensor(x: TensorDict, key: str, *, algorithm: str = '') -> Any
```

Return a declared tensor key from x; raise ContractError if absent.

``algorithm`` is included in diagnostics when provided.

**Returns**

- (`Any`) — The tensor stored under *key*.

**Raises**

- `ContractError` — If *key* is not present in *x*.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L16-L33)

## nexuml.evaluation.algorithm.get_fit_mask

*function*

```python
def get_fit_mask(x: TensorDict, y: TensorDict | None, fit_mask_key: str | None, batch_size: int, *, algorithm: str = '') -> torch.Tensor
```

Resolve fit mask to a bool tensor; returns all-True if fit_mask_key is None.

**Returns**

- (`torch.Tensor`) — Bool tensor of shape ``(batch_size,)``.

**Raises**

- `ContractError` — If *fit_mask_key* is not found in *y* or *x*.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/evaluation/algorithm.py#L95-L124)
