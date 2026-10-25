# nexuml.training

*module*

Training runtime exports.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/__init__.py#L1-L1)

## nexuml.training.NexuSession

*class*

```python
class NexuSession
```

Re-exported from: `nexuml.training.lightning`

Thin orchestration layer for building and running a NexuML session.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L591-L977)

### nexuml.training.NexuSession.accelerator

*attribute* · *instance attribute*

```python
accelerator = accelerator
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L609-L609)

### nexuml.training.NexuSession.devices

*attribute* · *instance attribute*

```python
devices = devices
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L610-L610)

### nexuml.training.NexuSession.enable_loggers

*attribute* · *instance attribute*

```python
enable_loggers = enable_loggers
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L613-L613)

### nexuml.training.NexuSession.enable_progress_bar

*attribute* · *instance attribute*

```python
enable_progress_bar = enable_progress_bar
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L612-L612)

### nexuml.training.NexuSession.log_dir

*attribute* · *instance attribute*

```python
log_dir = resolve_logs_root(log_dir)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L611-L611)

### nexuml.training.NexuSession.num_nodes

*attribute* · *instance attribute*

```python
num_nodes = num_nodes
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L620-L620)

### nexuml.training.NexuSession.run_name

*attribute* · *instance attribute*

```python
run_name = run_name
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L617-L617)

### nexuml.training.NexuSession.scenario

*attribute* · *instance attribute*

```python
scenario = self._resolve_scenario(scenario)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L621-L621)

### nexuml.training.NexuSession.trainer_checkpoint

*attribute* · *instance attribute*

```python
trainer_checkpoint = Path(trainer_checkpoint) if trainer_checkpoint is not None else None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L614-L616)

### nexuml.training.NexuSession.data_module

*property*

```python
data_module: NexuDataModule
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L674-L675)

### nexuml.training.NexuSession.lightning_module

*property*

```python
lightning_module: NexuLightningModule
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L670-L671)

### nexuml.training.NexuSession.pipeline

*property*

```python
pipeline: CompiledPipeline
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L666-L667)

### nexuml.training.NexuSession.runtime

*property*

```python
runtime: RuntimeArtifacts
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L662-L663)

### nexuml.training.NexuSession.trainer

*property*

```python
trainer: L.Trainer
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L678-L679)

### nexuml.training.NexuSession.trainer_callbacks

*property*

```python
trainer_callbacks: list[Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L687-L689)

### nexuml.training.NexuSession.trainer_loggers

*property*

```python
trainer_loggers: list[Any] | bool
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L682-L684)

### nexuml.training.NexuSession.build_runtime

*method*

```python
def build_runtime() -> RuntimeArtifacts
```

Create runtime artifacts lazily.

**Returns**

- (`RuntimeArtifacts`) — The ``RuntimeArtifacts`` (creating them on first call).

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L701-L715)

### nexuml.training.NexuSession.build_trainer

*method*

```python
def build_trainer() -> L.Trainer
```

Create the Lightning Trainer lazily.

**Returns**

- (`L.Trainer`) — The ``lightning.Trainer`` instance (creating it on first call).

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L717-L773)

### nexuml.training.NexuSession.fit

*method*

```python
def fit() -> 'NexuSession'
```

Run Trainer.fit() for the current session.

**Returns**

- (`'NexuSession'`) — The same ``NexuSession`` instance (for chaining).

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L775-L786)

### nexuml.training.NexuSession.from_scenario

*method* · *classmethod*

```python
def from_scenario(scenario: ScenarioSpec, **kwargs) -> 'NexuSession'
```

Create a session from a scenario definition.

**Returns**

- (`'NexuSession'`) — New ``NexuSession`` instance.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L630-L641)

### nexuml.training.NexuSession.from_trainer_checkpoint

*method* · *classmethod*

```python
def from_trainer_checkpoint(trainer_checkpoint: str | Path, scenario: ScenarioSpec | None = None, **kwargs) -> 'NexuSession'
```

Create a session from a Lightning Trainer checkpoint.

**Returns**

- (`'NexuSession'`) — New ``NexuSession`` instance restored from the checkpoint.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L643-L659)

### nexuml.training.NexuSession.predict

*method*

```python
def predict(dataloaders: Any = None, datamodule: NexuDataModule | None = None, return_predictions: bool = False) -> Any
```

Run Trainer.predict() with session defaults unless explicitly overridden.

**Returns**

- (`Any`) — Prediction outputs from ``Trainer.predict()``, or ``None`` when
- (`Any`) — *return_predictions* is ``False``.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L805-L824)

### nexuml.training.NexuSession.run

*method*

```python
def run() -> TrainResult
```

Execute the standard fit → validate → post-train fit → test flow.

**Returns**

- (`TrainResult`) — ``TrainResult`` containing the pipeline, module, trainer, and
- (`TrainResult`) — metric results.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L854-L883)

### nexuml.training.NexuSession.setup

*method*

```python
def setup() -> 'NexuSession'
```

Build runtime objects and trainer.

**Returns**

- (`'NexuSession'`) — The same ``NexuSession`` instance (for chaining).

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L691-L699)

### nexuml.training.NexuSession.test

*method*

```python
def test(dataloaders: Any = None, datamodule: NexuDataModule | None = None) -> list[dict[str, float]]
```

Run Trainer.test() with session defaults unless explicitly overridden.

**Returns**

- (`list[dict[str, float]]`) — List of metric dictionaries from the test stage.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L826-L852)

### nexuml.training.NexuSession.validate

*method*

```python
def validate() -> list[dict[str, float]]
```

Run Trainer.validate() with the session datamodule.

**Returns**

- (`list[dict[str, float]]`) — List of metric dictionaries from the validation stage.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L788-L803)

## nexuml.training.TrainResult

*class* · *dataclass*

```python
class TrainResult
```

Re-exported from: `nexuml.training.lightning`

Result of a training run.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L567-L578)

### nexuml.training.TrainResult.artifacts

*attribute* · *class attribute* · *instance attribute*

```python
artifacts: list[dict[str, str]] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L578-L578)

### nexuml.training.TrainResult.eval_algorithm_results

*attribute* · *class attribute* · *instance attribute*

```python
eval_algorithm_results: dict[str, float] = field(default_factory=dict)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L577-L577)

### nexuml.training.TrainResult.lightning_module

*attribute* · *instance attribute*

```python
lightning_module: NexuLightningModule
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L572-L572)

### nexuml.training.TrainResult.load_report

*attribute* · *class attribute* · *instance attribute*

```python
load_report: dict[str, list[str]] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L576-L576)

### nexuml.training.TrainResult.pipeline

*attribute* · *instance attribute*

```python
pipeline: CompiledPipeline
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L571-L571)

### nexuml.training.TrainResult.test_results

*attribute* · *class attribute* · *instance attribute*

```python
test_results: list[dict[str, float]] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L575-L575)

### nexuml.training.TrainResult.trainer

*attribute* · *instance attribute*

```python
trainer: L.Trainer
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L573-L573)

### nexuml.training.TrainResult.validation_results

*attribute* · *class attribute* · *instance attribute*

```python
validation_results: list[dict[str, float]] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L574-L574)

## nexuml.training.train

*function*

```python
def train(scenario: ScenarioSpec | None, accelerator: str = 'auto', devices: int | str = 'auto', log_dir: str | Path = '.experiments', enable_progress_bar: bool = True, trainer_checkpoint: str | Path | None = None, run_name: str | None = None) -> TrainResult
```

Re-exported from: `nexuml.training.lightning`

Compatibility wrapper around ``NexuSession.run()``.

**Returns**

- (`TrainResult`) — ``TrainResult`` from the session run.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/training/lightning.py#L1275-L1298)

**Modules**

- `nexuml.training.callbacks`
- `nexuml.training.lightning`
