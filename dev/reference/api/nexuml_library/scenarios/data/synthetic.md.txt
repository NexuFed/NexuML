# nexuml_library.scenarios.data.synthetic

*module*

Synthetic data scenario fragments.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/data/synthetic.py#L1-L1)

## nexuml_library.scenarios.data.synthetic.synthetic_vector_data

*function*

```python
def synthetic_vector_data(feature_shape: tuple[int, ...] = (128), num_samples: int = 1000, noise_type: str = 'gaussian', num_clusters: int | None = None, targets: list[TargetSpec] | None = None, seed: int = 42, feature_key: str = 'features') -> DataSpec
```

Create a DataSpec for synthetic vector data.

**Returns**

- `DataSpec` (`DataSpec`) — Synthetic dataset specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/data/synthetic.py#L9-L55)
