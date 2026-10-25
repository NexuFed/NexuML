# nexuml_library.scenarios.asd.audioset_conv_ae_clshead

*module*

Convolutional autoencoder on AudioSet with a latent classification head.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/audioset_conv_ae_clshead.py#L1-L1)

## nexuml_library.scenarios.asd.audioset_conv_ae_clshead.audioset_conv_ae_clshead

*function*

```python
def audioset_conv_ae_clshead(data_root: str = 'audioset_hf/full', download: bool = False, sample_rate: int = 16000, clip_num_samples: int = 160000, n_mels: int = 128, hop_length: int = 512, latent_dim: int = 64, channel_schedule: list[int] | None = None, num_classes: int = 527, lr: float = 0.001, batch_size: int = 128, max_epochs: int = 10, validate_layout: bool = False) -> ScenarioSpec
```

Conv AE trained on AudioSet waveforms with reconstruction + classification.

A ``LatentClassificationHead`` is added on top of the encoder latent space,
and both reconstruction and classification losses are optimized jointly.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/asd/audioset_conv_ae_clshead.py#L16-L93)
