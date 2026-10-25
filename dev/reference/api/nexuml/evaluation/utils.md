# nexuml.evaluation.utils

*module*

Utilities for evaluation algorithms.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L1-L1)

## nexuml.evaluation.utils.FeatureStore

*class*

```python
class FeatureStore(ABC)
```

Bases: `ABC`

Collect fit features for batch-fit distance estimators.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L55-L72)

### nexuml.evaluation.utils.FeatureStore.append

*method* · *abstract*

```python
def append(features: torch.Tensor) -> None
```

Append a feature batch.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L58-L60)

### nexuml.evaluation.utils.FeatureStore.as_array

*method* · *abstract*

```python
def as_array() -> torch.Tensor | np.ndarray | None
```

Return stored features as a CPU-compatible matrix.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L66-L68)

### nexuml.evaluation.utils.FeatureStore.cleanup

*method* · *abstract*

```python
def cleanup() -> None
```

Close handles and remove temporary files when not retained.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L70-L72)

### nexuml.evaluation.utils.FeatureStore.finalize

*method* · *abstract*

```python
def finalize() -> None
```

Finalize storage for reading.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L62-L64)

## nexuml.evaluation.utils.MemmapFeatureStore

*class*

```python
class MemmapFeatureStore(FeatureStore)
```

Bases: `nexuml.evaluation.utils.FeatureStore`

Disk-backed feature storage that streams directly to a numpy memmap.

Features are written to disk during ``append()`` without buffering in RAM.
The memmap is pre-allocated to ``(max_samples, D)`` when ``max_samples`` is
provided; otherwise capacity starts at 1024 rows and doubles as needed.
``finalize()`` trims the file to the actual row count written.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L109-L199)

### nexuml.evaluation.utils.MemmapFeatureStore.filename

*attribute* · *instance attribute*

```python
filename = self.storage_path / features-{uuid.uuid4().hex}.dat
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L134-L134)

### nexuml.evaluation.utils.MemmapFeatureStore.max_samples

*attribute* · *instance attribute*

```python
max_samples = max_samples
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L126-L126)

### nexuml.evaluation.utils.MemmapFeatureStore.retain_storage

*attribute* · *instance attribute*

```python
retain_storage = retain_storage or storage_path is not None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L132-L132)

### nexuml.evaluation.utils.MemmapFeatureStore.storage_path

*attribute* · *instance attribute*

```python
storage_path = Path(storage_path) if storage_path is not None else Path(tempfile.mkdtemp(prefix='nexuml-features-'))
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L127-L131)

### nexuml.evaluation.utils.MemmapFeatureStore.append

*method*

```python
def append(features: torch.Tensor) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L159-L170)

### nexuml.evaluation.utils.MemmapFeatureStore.as_array

*method*

```python
def as_array() -> np.ndarray | None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L187-L192)

### nexuml.evaluation.utils.MemmapFeatureStore.cleanup

*method*

```python
def cleanup() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L194-L199)

### nexuml.evaluation.utils.MemmapFeatureStore.finalize

*method*

```python
def finalize() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L172-L185)

## nexuml.evaluation.utils.RAMFeatureStore

*class*

```python
class RAMFeatureStore(FeatureStore)
```

Bases: `nexuml.evaluation.utils.FeatureStore`

In-memory feature storage with optional reservoir sampling.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L75-L106)

### nexuml.evaluation.utils.RAMFeatureStore.max_samples

*attribute* · *instance attribute*

```python
max_samples = max_samples
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L79-L79)

### nexuml.evaluation.utils.RAMFeatureStore.append

*method*

```python
def append(features: torch.Tensor) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L84-L89)

### nexuml.evaluation.utils.RAMFeatureStore.as_array

*method*

```python
def as_array() -> torch.Tensor | None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L99-L102)

### nexuml.evaluation.utils.RAMFeatureStore.cleanup

*method*

```python
def cleanup() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L104-L106)

### nexuml.evaluation.utils.RAMFeatureStore.finalize

*method*

```python
def finalize() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L91-L97)

## nexuml.evaluation.utils.ReservoirSampler

*class*

```python
class ReservoirSampler
```

Online reservoir sampling (Vitter's Algorithm R).

Maintains a fixed-size random sample of items seen so far,
suitable for streaming large datasets without storing everything.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L16-L52)

### nexuml.evaluation.utils.ReservoirSampler.max_samples

*attribute* · *instance attribute*

```python
max_samples = max_samples
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L24-L24)

### nexuml.evaluation.utils.ReservoirSampler.n_sampled

*property*

```python
n_sampled: int
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L51-L52)

### nexuml.evaluation.utils.ReservoirSampler.n_seen

*property*

```python
n_seen: int
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L47-L48)

### nexuml.evaluation.utils.ReservoirSampler.add

*method*

```python
def add(items: torch.Tensor) -> None
```

Add a batch of items (first dim = batch size).

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L28-L38)

### nexuml.evaluation.utils.ReservoirSampler.get

*method*

```python
def get() -> torch.Tensor | None
```

Return sampled items as a tensor, or None if empty.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L40-L44)

## nexuml.evaluation.utils.create_feature_store

*function*

```python
def create_feature_store(backend: str = 'ram', *, max_samples: int | None = None, storage_path: str | Path | None = None, retain_storage: bool = False) -> FeatureStore
```

Create a feature store backend.

**Returns**

- (`FeatureStore`) — The requested ``FeatureStore`` instance.

**Raises**

- `ValueError` — If *backend* is not ``"ram"`` or ``"memmap"``.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/evaluation/utils.py#L202-L225)
