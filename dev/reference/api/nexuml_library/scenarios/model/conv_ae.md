# nexuml_library.scenarios.model.conv_ae

*module*

Convolutional autoencoder model scenario fragments.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/model/conv_ae.py#L1-L1)

## nexuml_library.scenarios.model.conv_ae.conv_ae_lmbe

*function*

```python
def conv_ae_lmbe(sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 512, time_frames: int = 128, latent_dim: int = 64, channel_schedule: list[int] | None = None, activation: str = 'relu') -> PipelineSpec
```

Waveform -> LMBE -> convolutional AE.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Full waveform-to-reconstruction pipeline with LMBE feature extraction and convolutional autoencoder.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/model/conv_ae.py#L137-L176)

## nexuml_library.scenarios.model.conv_ae.conv_cvae_lmbe

*function*

```python
def conv_cvae_lmbe(sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 512, time_frames: int = 128, encoder_dim: int = 128, latent_dim: int = 32, beta: float = 1.0, channel_schedule: list[int] | None = None, activation: str = 'relu') -> PipelineSpec
```

Waveform -> LMBE -> convolutional CVAE.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Full waveform-to-reconstruction pipeline with LMBE feature extraction and convolutional variational autoencoder.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/model/conv_ae.py#L179-L222)

## nexuml_library.scenarios.model.conv_ae.spectrogram_conv_ae

*function*

```python
def spectrogram_conv_ae(input_shape: tuple[int, int, int] = (1, 64, 64), latent_dim: int = 64, channel_schedule: list[int] | None = None, activation: str = 'relu', score_reduction: str = 'mean') -> PipelineSpec
```

2D convolutional autoencoder with reconstruction loss and anomaly score.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Pipeline with encoder, decoder, reconstruction loss and anomaly score layers.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/model/conv_ae.py#L16-L71)

## nexuml_library.scenarios.model.conv_ae.spectrogram_conv_cvae

*function*

```python
def spectrogram_conv_cvae(input_shape: tuple[int, int, int] = (1, 64, 64), encoder_dim: int = 128, latent_dim: int = 32, beta: float = 1.0, channel_schedule: list[int] | None = None, activation: str = 'relu', score_reduction: str = 'mean') -> PipelineSpec
```

2D convolutional variational autoencoder with anomaly score.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Pipeline with encoder, variational latent, decoder, reconstruction loss and anomaly score layers.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/model/conv_ae.py#L74-L134)
