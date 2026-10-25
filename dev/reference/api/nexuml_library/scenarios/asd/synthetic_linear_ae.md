# nexuml_library.scenarios.asd.synthetic_linear_ae

*module*

Composed scenarios combining data + model + training + evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/synthetic_linear_ae.py#L1-L1)

## nexuml_library.scenarios.asd.synthetic_linear_ae.synthetic_linear_ae_multiclass

*function*

```python
def synthetic_linear_ae_multiclass(feature_shape: tuple[int, ...] = (128), num_samples: int = 1000, num_classes: int = 5, hidden_dims: list[int] | None = None, latent_dim: int = 8, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 10) -> ScenarioSpec
```

Synthetic vector reconstruction + multiclass classification.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/synthetic_linear_ae.py#L63-L104)

## nexuml_library.scenarios.asd.synthetic_linear_ae.synthetic_linear_ae_multilabel

*function*

```python
def synthetic_linear_ae_multilabel(feature_shape: tuple[int, ...] = (128), num_samples: int = 1000, num_classes: int = 5, hidden_dims: list[int] | None = None, latent_dim: int = 8, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 10) -> ScenarioSpec
```

Synthetic vector reconstruction + multilabel classification.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/synthetic_linear_ae.py#L107-L147)

## nexuml_library.scenarios.asd.synthetic_linear_ae.synthetic_linear_ae_reconstruction

*function*

```python
def synthetic_linear_ae_reconstruction(feature_shape: tuple[int, ...] = (128), num_samples: int = 1000, hidden_dims: list[int] | None = None, latent_dim: int = 8, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 10) -> ScenarioSpec
```

Synthetic vector reconstruction with linear autoencoder.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/synthetic_linear_ae.py#L24-L60)

## nexuml_library.scenarios.asd.synthetic_linear_ae.synthetic_linear_ae_regression

*function*

```python
def synthetic_linear_ae_regression(feature_shape: tuple[int, ...] = (128), num_samples: int = 1000, num_outputs: int = 3, hidden_dims: list[int] | None = None, latent_dim: int = 8, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 10) -> ScenarioSpec
```

Synthetic vector reconstruction + regression.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/synthetic_linear_ae.py#L150-L191)
