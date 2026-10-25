# nexuml_library.scenarios.asd.synthetic_conv_ae

*module*

Synthetic convolutional autoencoder scenarios.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/asd/synthetic_conv_ae.py#L1-L1)

## nexuml_library.scenarios.asd.synthetic_conv_ae.synthetic_conv_ae_anomaly

*function*

```python
def synthetic_conv_ae_anomaly(input_shape: tuple[int, int, int] = (1, 64, 64), num_samples: int = 160, anomaly_fraction: float = 0.2, latent_dim: int = 64, channel_schedule: list[int] | None = None, lr: float = 0.001, batch_size: int = 16, max_epochs: int = 2) -> ScenarioSpec
```

Synthetic spectrogram convolutional AE anomaly workflow.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/asd/synthetic_conv_ae.py#L12-L53)

## nexuml_library.scenarios.asd.synthetic_conv_ae.synthetic_conv_cvae_anomaly

*function*

```python
def synthetic_conv_cvae_anomaly(input_shape: tuple[int, int, int] = (1, 64, 64), num_samples: int = 160, anomaly_fraction: float = 0.2, encoder_dim: int = 128, latent_dim: int = 32, beta: float = 0.001, channel_schedule: list[int] | None = None, lr: float = 0.001, batch_size: int = 16, max_epochs: int = 2) -> ScenarioSpec
```

Synthetic spectrogram convolutional VAE anomaly workflow.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/asd/synthetic_conv_ae.py#L56-L101)
