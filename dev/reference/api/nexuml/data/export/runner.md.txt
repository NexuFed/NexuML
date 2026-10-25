# nexuml.data.export.runner

*module*

Loader-driven dataset export utilities.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/runner.py#L1-L1)

## nexuml.data.export.runner.BatchTransform

*attribute* · *module attribute*

```python
BatchTransform: TypeAlias = Callable[[TensorDict, TensorDict | None], tuple[TensorDict, TensorDict | None]]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/runner.py#L28-L31)

## nexuml.data.export.runner.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/runner.py#L26-L26)

## nexuml.data.export.runner.export_data_module

*function*

```python
def export_data_module(data_module: NexuDataModule, path: str | Path, *, backend: str | WriterSpec = 'numpy', splits: Sequence[str] | None = None, transform: BatchTransform | None = None, x_keys: Sequence[str] | None = None, y_keys: Sequence[str] | None = None, include_labels: bool = True, label_prefix: str = 'label__', dtype: object | None = None, device: torch.device | str | None = None, **backend_kwargs: object) -> Path | str
```

Export the data exactly as seen through a configured data module.

**Returns**

- (`Path | str`) — Local export path, or the original S3 URI for a remote WebDataset export.

**Raises**

- `ValueError` — If there are no samples or an unsupported remote backend is requested.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/runner.py#L70-L150)
