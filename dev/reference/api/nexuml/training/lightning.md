# nexuml.training.lightning

*module*

Lightning-based training backend for NexuML.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L1-L1)

## nexuml.training.lightning.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('lightning.pytorch.nexuml.training')
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L37-L37)

## nexuml.training.lightning.LightningFeatureExtractor

*class*

```python
class LightningFeatureExtractor(torch.nn.Module)
```

Bases: `torch.nn.Module`

Partial pipeline wrapper for dataset preprocessing/export.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L533-L564)

### nexuml.training.lightning.LightningFeatureExtractor.lightning_module

*attribute* · *instance attribute*

```python
lightning_module = lightning_module
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L544-L544)

### nexuml.training.lightning.LightningFeatureExtractor.x_keys

*attribute* · *instance attribute*

```python
x_keys = list(x_keys or [])
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L545-L545)

### nexuml.training.lightning.LightningFeatureExtractor.y_keys

*attribute* · *instance attribute*

```python
y_keys = list(y_keys or [])
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L546-L546)

### nexuml.training.lightning.LightningFeatureExtractor.forward

*method*

```python
def forward(x: TensorDict, y: TensorDict | None = None) -> tuple[TensorDict, TensorDict | None]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L548-L558)

### nexuml.training.lightning.LightningFeatureExtractor.on_predict_end

*method*

```python
def on_predict_end() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L563-L564)

### nexuml.training.lightning.LightningFeatureExtractor.on_predict_start

*method*

```python
def on_predict_start() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L560-L561)

## nexuml.training.lightning.NexuLightningModule

*class*

```python
class NexuLightningModule(L.LightningModule)
```

Bases: `L.LightningModule`

Lightning module wrapping a CompiledPipeline.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L44-L530)

### nexuml.training.lightning.NexuLightningModule.pipeline

*attribute* · *instance attribute*

```python
pipeline = pipeline
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L61-L61)

### nexuml.training.lightning.NexuLightningModule.scenario

*attribute* · *instance attribute*

```python
scenario = scenario
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L62-L62)

### nexuml.training.lightning.NexuLightningModule.evaluation_results

*property*

```python
evaluation_results: dict[str, float]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L231-L232)

### nexuml.training.lightning.NexuLightningModule.test_result_eval_metrics

*property*

```python
test_result_eval_metrics: dict[str, float]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L238-L243)

### nexuml.training.lightning.NexuLightningModule.configure_optimizers

*method*

```python
def configure_optimizers()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L182-L185)

### nexuml.training.lightning.NexuLightningModule.forward

*method*

```python
def forward(x: TensorDict, y: TensorDict | None = None) -> tuple[TensorDict, TensorDict | None]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L75-L78)

### nexuml.training.lightning.NexuLightningModule.forward_until

*method*

```python
def forward_until(x: TensorDict, y: TensorDict | None = None, *, x_keys: Sequence[str] | None = None, y_keys: Sequence[str] | None = None) -> tuple[TensorDict, TensorDict | None]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L80-L88)

### nexuml.training.lightning.NexuLightningModule.get_stage_metric_results

*method*

```python
def get_stage_metric_results(stage: str) -> dict[str, float]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L234-L235)

### nexuml.training.lightning.NexuLightningModule.load_from_checkpoint

*method* · *classmethod*

```python
def load_from_checkpoint(checkpoint_path: Any, map_location: Any = None, hparams_file: str | Path | None = None, strict: bool | None = None, weights_only: bool | None = None, allow_unsafe_globals: bool = True, **kwargs)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L283-L312)

### nexuml.training.lightning.NexuLightningModule.on_fit_end

*method*

```python
def on_fit_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L190-L191)

### nexuml.training.lightning.NexuLightningModule.on_fit_start

*method*

```python
def on_fit_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L187-L188)

### nexuml.training.lightning.NexuLightningModule.on_load_checkpoint

*method*

```python
def on_load_checkpoint(checkpoint: dict[str, Any]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L264-L281)

### nexuml.training.lightning.NexuLightningModule.on_predict_end

*method*

```python
def on_predict_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L222-L223)

### nexuml.training.lightning.NexuLightningModule.on_predict_start

*method*

```python
def on_predict_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L219-L220)

### nexuml.training.lightning.NexuLightningModule.on_save_checkpoint

*method*

```python
def on_save_checkpoint(checkpoint: dict[str, Any]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L245-L262)

### nexuml.training.lightning.NexuLightningModule.on_test_end

*method*

```python
def on_test_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L211-L212)

### nexuml.training.lightning.NexuLightningModule.on_test_epoch_end

*method*

```python
def on_test_epoch_end() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L225-L228)

### nexuml.training.lightning.NexuLightningModule.on_test_start

*method*

```python
def on_test_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L207-L209)

### nexuml.training.lightning.NexuLightningModule.on_train_epoch_end

*method*

```python
def on_train_epoch_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L193-L194)

### nexuml.training.lightning.NexuLightningModule.on_validation_end

*method*

```python
def on_validation_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L204-L205)

### nexuml.training.lightning.NexuLightningModule.on_validation_epoch_end

*method*

```python
def on_validation_epoch_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L200-L202)

### nexuml.training.lightning.NexuLightningModule.on_validation_start

*method*

```python
def on_validation_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L196-L198)

### nexuml.training.lightning.NexuLightningModule.predict_step

*method*

```python
def predict_step(batch, batch_idx, dataloader_idx=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L214-L217)

### nexuml.training.lightning.NexuLightningModule.test_step

*method*

```python
def test_step(batch, batch_idx)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L163-L180)

### nexuml.training.lightning.NexuLightningModule.training_step

*method*

```python
def training_step(batch, batch_idx)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L119-L145)

### nexuml.training.lightning.NexuLightningModule.validation_step

*method*

```python
def validation_step(batch, batch_idx)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L147-L161)

## nexuml.training.lightning.NexuSession

*class*

```python
class NexuSession
```

Thin orchestration layer for building and running a NexuML session.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L591-L977)

### nexuml.training.lightning.NexuSession.accelerator

*attribute* · *instance attribute*

```python
accelerator = accelerator
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L609-L609)

### nexuml.training.lightning.NexuSession.devices

*attribute* · *instance attribute*

```python
devices = devices
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L610-L610)

### nexuml.training.lightning.NexuSession.enable_loggers

*attribute* · *instance attribute*

```python
enable_loggers = enable_loggers
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L613-L613)

### nexuml.training.lightning.NexuSession.enable_progress_bar

*attribute* · *instance attribute*

```python
enable_progress_bar = enable_progress_bar
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L612-L612)

### nexuml.training.lightning.NexuSession.log_dir

*attribute* · *instance attribute*

```python
log_dir = resolve_logs_root(log_dir)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L611-L611)

### nexuml.training.lightning.NexuSession.num_nodes

*attribute* · *instance attribute*

```python
num_nodes = num_nodes
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L620-L620)

### nexuml.training.lightning.NexuSession.run_name

*attribute* · *instance attribute*

```python
run_name = run_name
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L617-L617)

### nexuml.training.lightning.NexuSession.scenario

*attribute* · *instance attribute*

```python
scenario = self._resolve_scenario(scenario)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L621-L621)

### nexuml.training.lightning.NexuSession.trainer_checkpoint

*attribute* · *instance attribute*

```python
trainer_checkpoint = Path(trainer_checkpoint) if trainer_checkpoint is not None else None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L614-L616)

### nexuml.training.lightning.NexuSession.data_module

*property*

```python
data_module: NexuDataModule
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L674-L675)

### nexuml.training.lightning.NexuSession.lightning_module

*property*

```python
lightning_module: NexuLightningModule
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L670-L671)

### nexuml.training.lightning.NexuSession.pipeline

*property*

```python
pipeline: CompiledPipeline
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L666-L667)

### nexuml.training.lightning.NexuSession.runtime

*property*

```python
runtime: RuntimeArtifacts
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L662-L663)

### nexuml.training.lightning.NexuSession.trainer

*property*

```python
trainer: L.Trainer
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L678-L679)

### nexuml.training.lightning.NexuSession.trainer_callbacks

*property*

```python
trainer_callbacks: list[Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L687-L689)

### nexuml.training.lightning.NexuSession.trainer_loggers

*property*

```python
trainer_loggers: list[Any] | bool
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L682-L684)

### nexuml.training.lightning.NexuSession.build_runtime

*method*

```python
def build_runtime() -> RuntimeArtifacts
```

Create runtime artifacts lazily.

**Returns**

- (`RuntimeArtifacts`) — The ``RuntimeArtifacts`` (creating them on first call).

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L701-L715)

### nexuml.training.lightning.NexuSession.build_trainer

*method*

```python
def build_trainer() -> L.Trainer
```

Create the Lightning Trainer lazily.

**Returns**

- (`L.Trainer`) — The ``lightning.Trainer`` instance (creating it on first call).

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L717-L773)

### nexuml.training.lightning.NexuSession.fit

*method*

```python
def fit() -> 'NexuSession'
```

Run Trainer.fit() for the current session.

**Returns**

- (`'NexuSession'`) — The same ``NexuSession`` instance (for chaining).

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L775-L786)

### nexuml.training.lightning.NexuSession.from_scenario

*method* · *classmethod*

```python
def from_scenario(scenario: ScenarioSpec, **kwargs) -> 'NexuSession'
```

Create a session from a scenario definition.

**Returns**

- (`'NexuSession'`) — New ``NexuSession`` instance.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L630-L641)

### nexuml.training.lightning.NexuSession.from_trainer_checkpoint

*method* · *classmethod*

```python
def from_trainer_checkpoint(trainer_checkpoint: str | Path, scenario: ScenarioSpec | None = None, **kwargs) -> 'NexuSession'
```

Create a session from a Lightning Trainer checkpoint.

**Returns**

- (`'NexuSession'`) — New ``NexuSession`` instance restored from the checkpoint.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L643-L659)

### nexuml.training.lightning.NexuSession.predict

*method*

```python
def predict(dataloaders: Any = None, datamodule: NexuDataModule | None = None, return_predictions: bool = False) -> Any
```

Run Trainer.predict() with session defaults unless explicitly overridden.

**Returns**

- (`Any`) — Prediction outputs from ``Trainer.predict()``, or ``None`` when
- (`Any`) — *return_predictions* is ``False``.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L805-L824)

### nexuml.training.lightning.NexuSession.run

*method*

```python
def run() -> TrainResult
```

Execute the standard fit → validate → post-train fit → test flow.

**Returns**

- (`TrainResult`) — ``TrainResult`` containing the pipeline, module, trainer, and
- (`TrainResult`) — metric results.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L854-L883)

### nexuml.training.lightning.NexuSession.setup

*method*

```python
def setup() -> 'NexuSession'
```

Build runtime objects and trainer.

**Returns**

- (`'NexuSession'`) — The same ``NexuSession`` instance (for chaining).

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L691-L699)

### nexuml.training.lightning.NexuSession.test

*method*

```python
def test(dataloaders: Any = None, datamodule: NexuDataModule | None = None) -> list[dict[str, float]]
```

Run Trainer.test() with session defaults unless explicitly overridden.

**Returns**

- (`list[dict[str, float]]`) — List of metric dictionaries from the test stage.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L826-L852)

### nexuml.training.lightning.NexuSession.validate

*method*

```python
def validate() -> list[dict[str, float]]
```

Run Trainer.validate() with the session datamodule.

**Returns**

- (`list[dict[str, float]]`) — List of metric dictionaries from the validation stage.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L788-L803)

## nexuml.training.lightning.RuntimeArtifacts

*class* · *dataclass*

```python
class RuntimeArtifacts
```

Compiled runtime objects shared by training and dataset export.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L581-L588)

### nexuml.training.lightning.RuntimeArtifacts.data_module

*attribute* · *instance attribute*

```python
data_module: NexuDataModule
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L587-L587)

### nexuml.training.lightning.RuntimeArtifacts.lightning_module

*attribute* · *instance attribute*

```python
lightning_module: NexuLightningModule
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L586-L586)

### nexuml.training.lightning.RuntimeArtifacts.load_report

*attribute* · *class attribute* · *instance attribute*

```python
load_report: dict[str, list[str]] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L588-L588)

### nexuml.training.lightning.RuntimeArtifacts.pipeline

*attribute* · *instance attribute*

```python
pipeline: CompiledPipeline
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L585-L585)

## nexuml.training.lightning.TrainResult

*class* · *dataclass*

```python
class TrainResult
```

Result of a training run.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L567-L578)

### nexuml.training.lightning.TrainResult.artifacts

*attribute* · *class attribute* · *instance attribute*

```python
artifacts: list[dict[str, str]] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L578-L578)

### nexuml.training.lightning.TrainResult.eval_algorithm_results

*attribute* · *class attribute* · *instance attribute*

```python
eval_algorithm_results: dict[str, float] = field(default_factory=dict)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L577-L577)

### nexuml.training.lightning.TrainResult.lightning_module

*attribute* · *instance attribute*

```python
lightning_module: NexuLightningModule
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L572-L572)

### nexuml.training.lightning.TrainResult.load_report

*attribute* · *class attribute* · *instance attribute*

```python
load_report: dict[str, list[str]] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L576-L576)

### nexuml.training.lightning.TrainResult.pipeline

*attribute* · *instance attribute*

```python
pipeline: CompiledPipeline
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L571-L571)

### nexuml.training.lightning.TrainResult.test_results

*attribute* · *class attribute* · *instance attribute*

```python
test_results: list[dict[str, float]] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L575-L575)

### nexuml.training.lightning.TrainResult.trainer

*attribute* · *instance attribute*

```python
trainer: L.Trainer
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L573-L573)

### nexuml.training.lightning.TrainResult.validation_results

*attribute* · *class attribute* · *instance attribute*

```python
validation_results: list[dict[str, float]] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L574-L574)

## nexuml.training.lightning.create_data_module_from_spec

*function*

```python
def create_data_module_from_spec(scenario: ScenarioSpec) -> NexuDataModule
```

Create a LightningDataModule from a ScenarioSpec.

**Returns**

- (`NexuDataModule`) — A ``NexuDataModule`` configured for the scenario.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L992-L1014)

## nexuml.training.lightning.create_dataset_from_spec

*function*

```python
def create_dataset_from_spec(scenario: ScenarioSpec) -> NexuDataset
```

Create a dataset from a ScenarioSpec's data configuration.

**Returns**

- (`NexuDataset`) — A ``NexuDataset`` built from the scenario's data config.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L980-L989)

## nexuml.training.lightning.create_runtime_artifacts

*function*

```python
def create_runtime_artifacts(scenario: ScenarioSpec, apply_selective_checkpoint: bool = True) -> RuntimeArtifacts
```

Compile the pipeline and create the matching Lightning/DataModule runtime.

**Returns**

- (`RuntimeArtifacts`) — ``RuntimeArtifacts`` containing the compiled pipeline, Lightning module,
- (`RuntimeArtifacts`) — data module, and optional load report.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L1106-L1161)

## nexuml.training.lightning.create_runtime_artifacts_from_trainer_checkpoint

*function*

```python
def create_runtime_artifacts_from_trainer_checkpoint(checkpoint_path: str | Path, scenario: ScenarioSpec | None = None) -> RuntimeArtifacts
```

Rebuild runtime objects from a Lightning Trainer checkpoint.

**Returns**

- (`RuntimeArtifacts`) — ``RuntimeArtifacts`` reconstructed from the checkpoint.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L1327-L1348)

## nexuml.training.lightning.load_scenario_from_trainer_checkpoint

*function*

```python
def load_scenario_from_trainer_checkpoint(checkpoint_path: str | Path, fallback: ScenarioSpec | None = None) -> ScenarioSpec
```

Load serialized scenario metadata from a Lightning Trainer checkpoint.

**Returns**

- (`ScenarioSpec`) — ``ScenarioSpec`` deserialized from the checkpoint, or *fallback* if
- (`ScenarioSpec`) — the checkpoint contains no scenario metadata and *fallback* is provided.

**Raises**

- `ValueError` — If the checkpoint lacks scenario metadata and no
*fallback* is given.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L1301-L1324)

## nexuml.training.lightning.materialize_preprocessed_dataset

*function*

```python
def materialize_preprocessed_dataset(scenario: ScenarioSpec) -> Path
```

Materialize the configured preprocessing view and return its export path.

**Returns**

- (`Path`) — ``Path`` to the directory containing the materialized dataset.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L1029-L1085)

## nexuml.training.lightning.resolve_preprocessing_path

*function*

```python
def resolve_preprocessing_path(scenario: ScenarioSpec) -> Path
```

Resolve the configured preprocessing output directory.

**Returns**

- (`Path`) — Resolved ``Path`` for the preprocessing output directory.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L1088-L1103)

## nexuml.training.lightning.train

*function*

```python
def train(scenario: ScenarioSpec | None, accelerator: str = 'auto', devices: int | str = 'auto', log_dir: str | Path = '.experiments', enable_progress_bar: bool = True, trainer_checkpoint: str | Path | None = None, run_name: str | None = None) -> TrainResult
```

Compatibility wrapper around ``NexuSession.run()``.

**Returns**

- (`TrainResult`) — ``TrainResult`` from the session run.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/lightning.py#L1275-L1298)
