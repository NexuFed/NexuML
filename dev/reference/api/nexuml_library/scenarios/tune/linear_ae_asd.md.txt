# nexuml_library.scenarios.tune.linear_ae_asd

*module*

Optuna tuning file for the linear AE ToyCarEmu ASD scenario.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/tune/linear_ae_asd.py#L1-L1)

## nexuml_library.scenarios.tune.linear_ae_asd.SEARCH_SPACE

*attribute* · *module attribute*

```python
SEARCH_SPACE = {'hidden_dims': {'type': 'categorical', 'choices': [[64, 32], [128, 64], [128, 64, 32]]}, 'latent_dim': {'type': 'int', 'low': 4, 'high': 32}, 'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]}, 'training.lr': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True}}
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/tune/linear_ae_asd.py#L8-L21)

## nexuml_library.scenarios.tune.linear_ae_asd.TUNING_SPEC

*attribute* · *module attribute*

```python
TUNING_SPEC = TuningSpec(n_trials=20, directions=['maximize'], metric_key='anomaly_eval/omega', storage='.experiments/optuna/linear_ae_asd.log', prune=False)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/tune/linear_ae_asd.py#L23-L29)

## nexuml_library.scenarios.tune.linear_ae_asd.build

*function*

```python
def build(hidden_dims: list[int], latent_dim: int, batch_size: int) -> ScenarioSpec
```

Build a scenario from sampled architectural hyperparameters.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with tuning spec attached.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/tune/linear_ae_asd.py#L41-L54)

## nexuml_library.scenarios.tune.linear_ae_asd.scenario

*function*

```python
def scenario() -> ScenarioSpec
```

Return the default base scenario for tuning.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/tune/linear_ae_asd.py#L32-L38)
