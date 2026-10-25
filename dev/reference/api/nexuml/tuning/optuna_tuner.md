# nexuml.tuning.optuna_tuner

*module*

Optuna-based hyperparameter tuning for NexuML.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tuning/optuna_tuner.py#L1-L1)

## nexuml.tuning.optuna_tuner.DEFAULT_SEARCH_SPACE

*attribute* · *module attribute*

```python
DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {'training.lr': {'type': 'float', 'low': 1e-05, 'high': 0.01, 'log': True}, 'training.batch_size': {'type': 'categorical', 'choices': [32, 64, 128]}}
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tuning/optuna_tuner.py#L16-L19)

## nexuml.tuning.optuna_tuner.OptunaDirection

*attribute* · *module attribute*

```python
OptunaDirection = Literal['minimize', 'maximize']
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tuning/optuna_tuner.py#L14-L14)

## nexuml.tuning.optuna_tuner.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tuning/optuna_tuner.py#L13-L13)

## nexuml.tuning.optuna_tuner.build_objective

*function*

```python
def build_objective(base_scenario: ScenarioSpec, search_space: dict[str, dict[str, Any]], metric_key: str = 'val/loss', enable_progress_bar: bool = False, build_factory: Callable[..., ScenarioSpec] | None = None) -> Callable
```

Build an Optuna objective function from a base scenario and search space.

**Parameters**

- `base_scenario` (`ScenarioSpec`) — The base ScenarioSpec to tune.
- `search_space` (`dict[str, dict[str, Any]]`) — Dict mapping dotted-path param names to Optuna suggest kwargs. Each entry must have a "type" key (float, int, categorical) and the remaining keys are forwarded to the Optuna suggest method. Example::     {         "training.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},         "training.max_epochs": {"type": "int", "low": 5, "high": 50},         "training.batch_size": {"type": "categorical", "choices": [32, 64, 128]},     }
- `metric_key` (`str`) (default: `'val/loss'`) — Lightning logged metric to optimise (must be logged with self.log()).
- `enable_progress_bar` (`bool`) (default: `False`) — Show Lightning progress bar during each trial.
- `build_factory` (`Callable[..., ScenarioSpec] | None`) (default: `None`) — Optional factory to rebuild the scenario from sampled params.

**Returns**

- `Callable` (`Callable`) — objective(trial) -> float | list[float]

**Raises**

- `ImportError` — If the ``optuna`` package is not installed.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tuning/optuna_tuner.py#L41-L112)

## nexuml.tuning.optuna_tuner.tune

*function*

```python
def tune(scenario: ScenarioSpec, search_space: dict[str, dict[str, Any]], tuning_spec: TuningSpec | None = None, metric_key: str = 'val/loss', enable_progress_bar: bool = False, build_factory: Callable[..., ScenarioSpec] | None = None) -> Any
```

Run Optuna hyperparameter search.

**Parameters**

- `scenario` (`ScenarioSpec`) — Base ScenarioSpec to tune.
- `search_space` (`dict[str, dict[str, Any]]`) — Dict mapping dotted-path param names to Optuna suggest kwargs.
- `tuning_spec` (`TuningSpec | None`) (default: `None`) — TuningSpec controlling n_trials, storage, pruning, etc.
- `metric_key` (`str`) (default: `'val/loss'`) — Logged metric to optimise.
- `enable_progress_bar` (`bool`) (default: `False`) — Show progress bar during each trial.
- `build_factory` (`Callable[..., ScenarioSpec] | None`) (default: `None`) — Optional factory to rebuild the scenario from sampled params.

**Returns**

- (`Any`) — Completed ``optuna.Study``.

**Raises**

- `ImportError` — If the ``optuna`` package is not installed.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/tuning/optuna_tuner.py#L233-L325)
