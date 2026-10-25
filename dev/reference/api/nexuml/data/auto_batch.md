# nexuml.data.auto_batch

*module*

Automatic CUDA batch-size resolution helpers.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L1-L1)

## nexuml.data.auto_batch.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L15-L15)

## nexuml.data.auto_batch.BatchProbeAttempt

*class* · *dataclass*

```python
class BatchProbeAttempt
```

Result of a single batch-size probe attempt.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L18-L28)

### nexuml.data.auto_batch.BatchProbeAttempt.batch_size

*attribute* · *instance attribute*

```python
batch_size: int
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L22-L22)

### nexuml.data.auto_batch.BatchProbeAttempt.error

*attribute* · *class attribute* · *instance attribute*

```python
error: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L24-L24)

### nexuml.data.auto_batch.BatchProbeAttempt.memory_fraction

*attribute* · *class attribute* · *instance attribute*

```python
memory_fraction: float | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L28-L28)

### nexuml.data.auto_batch.BatchProbeAttempt.memory_peak_bytes

*attribute* · *class attribute* · *instance attribute*

```python
memory_peak_bytes: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L26-L26)

### nexuml.data.auto_batch.BatchProbeAttempt.memory_total_bytes

*attribute* · *class attribute* · *instance attribute*

```python
memory_total_bytes: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L27-L27)

### nexuml.data.auto_batch.BatchProbeAttempt.seconds

*attribute* · *class attribute* · *instance attribute*

```python
seconds: float | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L25-L25)

### nexuml.data.auto_batch.BatchProbeAttempt.status

*attribute* · *instance attribute*

```python
status: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L23-L23)

## nexuml.data.auto_batch.BatchProbeResult

*class* · *dataclass*

```python
class BatchProbeResult
```

Aggregated result of an automatic batch-size probe run.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L31-L46)

### nexuml.data.auto_batch.BatchProbeResult.attempts

*attribute* · *class attribute* · *instance attribute*

```python
attempts: list[BatchProbeAttempt] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L36-L36)

### nexuml.data.auto_batch.BatchProbeResult.device

*attribute* · *class attribute* · *instance attribute*

```python
device: dict[str, Any] = field(default_factory=dict)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L38-L38)

### nexuml.data.auto_batch.BatchProbeResult.safety

*attribute* · *class attribute* · *instance attribute*

```python
safety: str = 'largest'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L37-L37)

### nexuml.data.auto_batch.BatchProbeResult.selected_batch_size

*attribute* · *instance attribute*

```python
selected_batch_size: int
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L35-L35)

### nexuml.data.auto_batch.BatchProbeResult.to_dict

*method*

```python
def to_dict() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L40-L46)

## nexuml.data.auto_batch.cuda_device_info

*function*

```python
def cuda_device_info() -> dict[str, Any]
```

Return CUDA device metadata, or ``{"available": False}`` if no GPU is present.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L100-L112)

## nexuml.data.auto_batch.generate_candidates

*function*

```python
def generate_candidates(config: AutoBatchSizeSpec) -> list[int]
```

Generate bounded power-of-two candidates in ascending order.

**Returns**

- (`list[int]`) — Sorted list of candidate batch sizes.

**Raises**

- `ValueError` — If the candidate strategy is not ``power_of_two``.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L49-L69)

## nexuml.data.auto_batch.is_cuda_oom

*function*

```python
def is_cuda_oom(exc: BaseException) -> bool
```

Return True if the exception is a CUDA out-of-memory error.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L94-L97)

## nexuml.data.auto_batch.resolve_with_probe

*function*

```python
def resolve_with_probe(config: AutoBatchSizeSpec, probe: Callable[[int], None]) -> BatchProbeResult
```

Try candidate batch sizes and return selected size plus observable metadata.

**Returns**

- (`BatchProbeResult`) — Aggregated probe result with selected batch size.

**Raises**

- `RuntimeError` — If no candidate succeeds because candidates hit CUDA OOM or exceed
the configured margin policy.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L127-L202)

## nexuml.data.auto_batch.select_batch_size

*function*

```python
def select_batch_size(successes: list[int], config: AutoBatchSizeSpec) -> int
```

Select the effective batch size from successful candidates.

**Returns**

- (`int`) — The selected batch size.

**Raises**

- `ValueError` — If ``successes`` is empty or the safety policy is unsupported.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/auto_batch.py#L72-L91)
