# nexuml.tracking.logger

*module*

Logger factory for NexuML experiment tracking.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L1-L1)

## nexuml.tracking.logger.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L21-L21)

## nexuml.tracking.logger.create_loggers

*function*

```python
def create_loggers(logging_spec: 'LoggingSpec | None', run_name: str | None = None) -> list[Any]
```

Build a list of Lightning loggers from a LoggingSpec.

Returns an empty list (disables logging) if spec is None.
Each backend is imported lazily so missing optional deps don't crash the import.

**Parameters**

- `logging_spec` (`'LoggingSpec | None'`) — LoggingSpec instance or None.
- `run_name` (`str | None`) (default: `None`) — Override run name; falls back to logging_spec.run_name.

**Returns**

- (`list[Any]`) — List of Lightning logger instances ready for Trainer(logger=...).

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L238-L296)

## nexuml.tracking.logger.get_temp_artifact_root

*function*

```python
def get_temp_artifact_root() -> Path
```

Return the directory used for temporary artifact staging.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L40-L44)

## nexuml.tracking.logger.iter_loggers

*function*

```python
def iter_loggers(logger_obj: Any) -> list[Any]
```

Normalize Lightning logger containers into a flat list.

**Returns**

- (`list[Any]`) — Flat list of non-None logger instances.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L422-L437)

## nexuml.tracking.logger.log_artifact

*function*

```python
def log_artifact(logger_obj: Any, source_path: str | Path, artifact_path: str | None = None) -> None
```

Log or copy a file artifact to all configured logger backends.

**Raises**

- `FileNotFoundError` — If *source_path* does not exist.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L440-L477)

## nexuml.tracking.logger.log_image

*function*

```python
def log_image(logger_obj: Any, tag: str, image: Any, step: int | None = None, artifact_path: str | None = None) -> None
```

Log an image to all configured logger backends.

**Parameters**

- `logger_obj` (`Any`) — Lightning logger or list of loggers.
- `tag` (`str`) — Name/tag for the image (e.g. ``"val/reconstruction"``). Slashes are used as subdirectory separators for file-based backends.
- `image` (`Any`) — Image data — numpy array (HWC, uint8 or float [0,1]), PIL Image, or torch tensor (CHW, uint8 or float [0,1]).
- `step` (`int | None`) (default: `None`) — Global step for time-series backends (TensorBoard).
- `artifact_path` (`str | None`) (default: `None`) — Optional subdirectory within the run's artifact store (MLflow / file-based backends only).

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L519-L580)

## nexuml.tracking.logger.log_text_artifact

*function*

```python
def log_text_artifact(logger_obj: Any, text: str, artifact_name: str, artifact_path: str | None = None) -> None
```

Write text content to a temporary file and log it as an artifact.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L480-L490)

## nexuml.tracking.logger.print_service_info

*function*

```python
def print_service_info(trainer_loggers: list[Any], scenario_name: str, log_dir: str | Path = '.experiments', logging_spec: Any | None = None, tuning_spec: Any | None = None, data_backend: str | None = None, training_backend: str | None = None) -> None
```

Print actionable commands for all active monitoring services.

Called once at the start of training so the user knows how to
inspect metrics, tune hyperparameters, etc.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L706-L821)

## nexuml.tracking.logger.staged_artifact_path

*function*

```python
def staged_artifact_path(artifact_name: str, *, prefix: str = 'nexuml_artifact_') -> Generator[Path, None, None]
```

Yield a temporary filesystem path for building an artifact before logging it.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tracking/logger.py#L47-L58)
