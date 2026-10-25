# nexuml_library.scenarios.asd.dcase_conv_ae_baseline

*module*

DCASE convolutional autoencoder scenarios.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/asd/dcase_conv_ae_baseline.py#L1-L1)

## nexuml_library.scenarios.asd.dcase_conv_ae_baseline.dcase_conv_ae

*function*

```python
def dcase_conv_ae(machine_types: list[str] | None = None, machine_specs: list[MachineSpec] | None = None, download: bool = False, n_mels: int = 128, hop_length: int = 512, clip_num_samples: int = 160000, latent_dim: int = 64, channel_schedule: list[int] | None = None, lr: float = 0.001, batch_size: int = 16, max_epochs: int = 25) -> ScenarioSpec
```

DCASE convolutional autoencoder anomaly workflow.

Machine year and data_type come from the built-in DCASE catalog when
``machine_types`` is given. Use ``machine_specs`` for explicit multi-year
control. ``time_frames`` is computed from ``clip_num_samples / hop_length``.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/asd/dcase_conv_ae_baseline.py#L15-L64)

## nexuml_library.scenarios.asd.dcase_conv_ae_baseline.dcase_conv_cvae

*function*

```python
def dcase_conv_cvae(machine_types: list[str] | None = None, machine_specs: list[MachineSpec] | None = None, download: bool = False, n_mels: int = 128, hop_length: int = 512, clip_num_samples: int = 160000, encoder_dim: int = 128, latent_dim: int = 32, beta: float = 0.001, channel_schedule: list[int] | None = None, lr: float = 0.001, batch_size: int = 16, max_epochs: int = 25) -> ScenarioSpec
```

DCASE convolutional variational autoencoder anomaly workflow.

Machine year and data_type come from the built-in DCASE catalog when
``machine_types`` is given. ``time_frames`` is computed automatically.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/asd/dcase_conv_ae_baseline.py#L67-L119)
