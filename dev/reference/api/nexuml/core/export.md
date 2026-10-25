# nexuml.core.export

*module*

Export, reload, and selective checkpoint loading for trained pipelines.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L1-L1)

## nexuml.core.export.CHECKPOINT_SIDECAR

*attribute* · *module attribute*

```python
CHECKPOINT_SIDECAR = 'lightning.ckpt'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L40-L40)

## nexuml.core.export.LEGACY_PACKAGE_PICKLE_NAME

*attribute* · *module attribute*

```python
LEGACY_PACKAGE_PICKLE_NAME = 'pipeline.pkl'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L38-L38)

## nexuml.core.export.LEGACY_PACKAGE_PICKLE_PACKAGE

*attribute* · *module attribute*

```python
LEGACY_PACKAGE_PICKLE_PACKAGE = 'model'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L37-L37)

## nexuml.core.export.PACKAGE_FILENAME

*attribute* · *module attribute*

```python
PACKAGE_FILENAME = 'pipeline.package'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L34-L34)

## nexuml.core.export.PACKAGE_PICKLE_NAME

*attribute* · *module attribute*

```python
PACKAGE_PICKLE_NAME = 'artifact.pkl'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L36-L36)

## nexuml.core.export.PACKAGE_PICKLE_PACKAGE

*attribute* · *module attribute*

```python
PACKAGE_PICKLE_PACKAGE = 'nexuml_export'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L35-L35)

## nexuml.core.export.REQUIREMENTS_FILENAME

*attribute* · *module attribute*

```python
REQUIREMENTS_FILENAME = 'requirements.txt'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L39-L39)

## nexuml.core.export.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L32-L32)

## nexuml.core.export.LoadReport

*class* · *dataclass*

```python
class LoadReport
```

Selective load result for package/state reuse.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L114-L131)

### nexuml.core.export.LoadReport.excluded

*attribute* · *class attribute* · *instance attribute*

```python
excluded: list[str] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L121-L121)

### nexuml.core.export.LoadReport.matched

*attribute* · *class attribute* · *instance attribute*

```python
matched: list[str] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L118-L118)

### nexuml.core.export.LoadReport.missing

*attribute* · *class attribute* · *instance attribute*

```python
missing: list[str] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L119-L119)

### nexuml.core.export.LoadReport.shape_mismatched

*attribute* · *class attribute* · *instance attribute*

```python
shape_mismatched: list[str] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L122-L122)

### nexuml.core.export.LoadReport.unexpected

*attribute* · *class attribute* · *instance attribute*

```python
unexpected: list[str] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L120-L120)

### nexuml.core.export.LoadReport.to_dict

*method*

```python
def to_dict() -> dict[str, list[str]]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L124-L131)

## nexuml.core.export.TrainingReload

*class* · *dataclass*

```python
class TrainingReload
```

Reloaded package prepared for current-codebase training.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L134-L142)

### nexuml.core.export.TrainingReload.lightning_module

*attribute* · *instance attribute*

```python
lightning_module: Any
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L139-L139)

### nexuml.core.export.TrainingReload.metadata

*attribute* · *instance attribute*

```python
metadata: dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L141-L141)

### nexuml.core.export.TrainingReload.pipeline

*attribute* · *instance attribute*

```python
pipeline: CompiledPipeline
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L138-L138)

### nexuml.core.export.TrainingReload.report

*attribute* · *instance attribute*

```python
report: LoadReport
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L142-L142)

### nexuml.core.export.TrainingReload.scenario

*attribute* · *instance attribute*

```python
scenario: ScenarioSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L140-L140)

## nexuml.core.export.export_onnx

*function*

```python
def export_onnx(pipeline: CompiledPipeline, path: Path, input_key: str | None = None, output_key: str = 'reconstructed', opset_version: int = 18) -> Path
```

Export an inference-only ONNX graph for single-input pipelines.

**Returns**

- (`Path`) — Path to the created ``.onnx`` file.

**Raises**

- `ImportError` — If the ``onnxscript`` package is not installed.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L963-L1003)

## nexuml.core.export.export_package

*function*

```python
def export_package(pipeline: CompiledPipeline, path: Path, metadata: dict[str, Any] | None = None, lightning_module: Any | None = None, trainer: Any | None = None, checkpoint_path: str | Path | None = None, include_modules: list[str] | None = None, source_metadata: dict[str, Any] | None = None) -> Path
```

Export a trained pipeline as a rich package-backed artifact directory.

**Parameters**

- `pipeline` (`CompiledPipeline`) — Compiled pipeline to export.
- `path` (`Path`) — Destination directory.
- `metadata` (`dict[str, Any] | None`) (default: `None`) — Optional provenance metadata merged into the artifact.
- `lightning_module` (`Any | None`) (default: `None`) — Optional Lightning module for checkpoint sidecars.
- `trainer` (`Any | None`) (default: `None`) — Optional Lightning trainer for training-state sidecars.
- `checkpoint_path` (`str | Path | None`) (default: `None`) — Optional source Lightning checkpoint to preserve.
- `include_modules` (`list[str] | None`) (default: `None`) — Optional glob patterns for additional source modules to intern (useful for dynamic imports invisible to torch.package).
- `source_metadata` (`dict[str, Any] | None`) (default: `None`) — Optional metadata describing the export source (e.g. CLI checkpoint path). Merged into *metadata*.

**Returns**

- (`Path`) — Path to the created export directory.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L501-L623)

## nexuml.core.export.export_safetensors

*function*

```python
def export_safetensors(pipeline: CompiledPipeline, path: Path, include: list[str] | None = None, exclude: list[str] | None = None, metadata: dict[str, Any] | None = None) -> Path
```

Export pipeline weights as SafeTensors plus a JSON manifest.

**Returns**

- (`Path`) — Path to the created ``.safetensors`` file.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L917-L947)

## nexuml.core.export.infer

*function*

```python
def infer(pipeline: CompiledPipeline, x: TensorDict, y: TensorDict | None = None) -> TensorDict
```

Run inference on a pipeline (eval mode, no grad).

**Returns**

- (`TensorDict`) — Output TensorDict from the pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L1006-L1019)

## nexuml.core.export.load_inference_package

*function*

```python
def load_inference_package(path: Path) -> tuple[CompiledPipeline, ResolvedConfig, dict[str, Any]]
```

Load the packaged pipeline object directly from the torch.package artifact.

**Returns**

- (`tuple[CompiledPipeline, ResolvedConfig, dict[str, Any]]`) — Tuple of ``(pipeline, resolved_config, metadata)``.

**Raises**

- `FileNotFoundError` — If no package artifact exists at *path*.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L857-L877)

## nexuml.core.export.load_package

*function*

```python
def load_package(path: Path) -> tuple[CompiledPipeline, ResolvedConfig, dict[str, Any]]
```

Reload an exported pipeline into the current codebase.

**Returns**

- (`tuple[CompiledPipeline, ResolvedConfig, dict[str, Any]]`) — Tuple of ``(pipeline, resolved_config, metadata)``.

**Raises**

- `ValueError` — If no scenario config is found in the artifact.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L830-L854)

## nexuml.core.export.load_package_for_training

*function*

```python
def load_package_for_training(path: Path, scenario: ScenarioSpec | None = None, checkpoint: CheckpointLoadSpec | None = None) -> TrainingReload
```

Reload a package into the current codebase for resume or fine-tuning.

**Returns**

- (`TrainingReload`) — TrainingReload with pipeline, lightning module, scenario, and load report.

**Raises**

- `ValueError` — If no scenario is provided and the artifact has no packaged config.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L880-L914)

## nexuml.core.export.load_weights

*function*

```python
def load_weights(pipeline: CompiledPipeline, source: str | Path, checkpoint: CheckpointLoadSpec | None = None, *, include: list[str] | None = None, exclude: list[str] | None = None, allow_missing: bool | None = None, allow_shape_mismatch: bool | None = None, freeze_loaded: bool | None = None) -> LoadReport
```

Selectively load weights into an already-compiled pipeline.

**Returns**

- (`LoadReport`) — LoadReport summarising matched, missing, and excluded keys.

**Raises**

- `ValueError` — On shape mismatch or unexpected missing keys when not allowed.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/export.py#L761-L827)
