# nexuml_library.scenarios.tune.conv_ae_asd

*module*

Optuna tuning file for the convolutional AE ToyCarEmu ASD scenario.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/tune/conv_ae_asd.py#L1-L1)

## nexuml_library.scenarios.tune.conv_ae_asd.SEARCH_SPACE

*attribute* · *module attribute*

```python
SEARCH_SPACE = {'channel_schedule': {'type': 'categorical', 'choices': [[16, 32, 64], [32, 64, 128], [16, 32, 64, 128]]}, 'latent_dim': {'type': 'int', 'low': 16, 'high': 128}, 'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]}, 'training.lr': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True}}
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/tune/conv_ae_asd.py#L8-L21)

## nexuml_library.scenarios.tune.conv_ae_asd.TUNING_SPEC

*attribute* · *module attribute*

```python
TUNING_SPEC = TuningSpec(n_trials=20, directions=['maximize'], metric_key='anomaly_eval/omega', storage='.experiments/optuna/conv_ae_asd.log', prune=False)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/tune/conv_ae_asd.py#L23-L29)

## nexuml_library.scenarios.tune.conv_ae_asd.build

*function*

```python
def build(channel_schedule: list[int], latent_dim: int, batch_size: int) -> ScenarioSpec
```

Build a scenario from sampled architectural hyperparameters.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with tuning spec attached.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/tune/conv_ae_asd.py#L41-L54)

## nexuml_library.scenarios.tune.conv_ae_asd.scenario

*function*

```python
def scenario() -> ScenarioSpec
```

Return the default base scenario for tuning.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/tune/conv_ae_asd.py#L32-L38)
