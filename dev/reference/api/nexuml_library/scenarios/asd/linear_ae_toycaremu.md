# nexuml_library.scenarios.asd.linear_ae_toycaremu

*module*

Linear autoencoder ASD scenario on the ToyCarEmu DCASE 2026 machine.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/linear_ae_toycaremu.py#L1-L1)

## nexuml_library.scenarios.asd.linear_ae_toycaremu.linear_ae_toycaremu

*function*

```python
def linear_ae_toycaremu(n_mels: int = 128, hop_length: int = 512, clip_num_samples: int = 160000, hidden_dims: list[int] | None = None, latent_dim: int = 8, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 25) -> ScenarioSpec
```

Linear AE anomaly detector trained on DCASE 2026 ToyCarEmu.

A decision rule is appended to the loss stage so the pipeline emits binary
decisions alongside the anomaly score.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/linear_ae_toycaremu.py#L18-L68)
