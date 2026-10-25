# nexuml_library.scenarios.asd.conv_ae_toycaremu

*module*

Convolutional autoencoder ASD scenario on the ToyCarEmu DCASE 2026 machine.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/asd/conv_ae_toycaremu.py#L1-L1)

## nexuml_library.scenarios.asd.conv_ae_toycaremu.conv_ae_toycaremu

*function*

```python
def conv_ae_toycaremu(n_mels: int = 128, hop_length: int = 512, clip_num_samples: int = 160000, latent_dim: int = 64, channel_schedule: list[int] | None = None, lr: float = 0.001, batch_size: int = 32, max_epochs: int = 25) -> ScenarioSpec
```

Convolutional AE anomaly detector trained on DCASE 2026 ToyCarEmu.

This is a narrow, single-machine didactic preset with an explicit decision
rule in the loss stage.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/asd/conv_ae_toycaremu.py#L18-L68)
