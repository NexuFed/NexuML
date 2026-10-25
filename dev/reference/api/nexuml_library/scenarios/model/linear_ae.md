# nexuml_library.scenarios.model.linear_ae

*module*

Linear autoencoder model scenario fragments.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/model/linear_ae.py#L1-L1)

## nexuml_library.scenarios.model.linear_ae.linear_ae_anomaly_detection

*function*

```python
def linear_ae_anomaly_detection(input_dim: int = 128, hidden_dims: list[int] | None = None, latent_dim: int = 8, activation: str = 'torch.nn.ReLU', feature_key: str = 'features', score_reduction: str = 'mean') -> PipelineSpec
```

Linear autoencoder with reconstruction loss and an anomaly score.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Pipeline with linear encoder, decoder, reconstruction loss and anomaly score layers.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/model/linear_ae.py#L69-L97)

## nexuml_library.scenarios.model.linear_ae.linear_ae_lmbe

*function*

```python
def linear_ae_lmbe(sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 512, time_frames: int = 128, hidden_dims: list[int] | None = None, latent_dim: int = 8, activation: str = 'torch.nn.ReLU', score_reduction: str = 'mean') -> PipelineSpec
```

Waveform -> LMBE -> linear autoencoder with an anomaly score.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Full waveform-to-reconstruction pipeline with LMBE feature extraction and linear autoencoder.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/model/linear_ae.py#L100-L142)

## nexuml_library.scenarios.model.linear_ae.linear_ae_multiclass

*function*

```python
def linear_ae_multiclass(input_dim: int = 128, hidden_dims: list[int] | None = None, latent_dim: int = 8, activation: str = 'torch.nn.ReLU') -> PipelineSpec
```

Create a PipelineSpec for a linear AE with reconstruction + classification.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Pipeline with encoder, decoder, classification head, reconstruction loss, classification loss and metrics layers.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/model/linear_ae.py#L145-L211)

## nexuml_library.scenarios.model.linear_ae.linear_ae_multilabel

*function*

```python
def linear_ae_multilabel(input_dim: int = 128, hidden_dims: list[int] | None = None, latent_dim: int = 8, activation: str = 'torch.nn.ReLU') -> PipelineSpec
```

Create a PipelineSpec for a linear AE with reconstruction + multilabel classification.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Pipeline with encoder, decoder, multilabel head, reconstruction loss and multilabel loss layers.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/model/linear_ae.py#L214-L274)

## nexuml_library.scenarios.model.linear_ae.linear_ae_reconstruction

*function*

```python
def linear_ae_reconstruction(input_dim: int = 128, hidden_dims: list[int] | None = None, latent_dim: int = 8, activation: str = 'torch.nn.ReLU', feature_key: str = 'features') -> PipelineSpec
```

Create a PipelineSpec for a linear autoencoder with reconstruction loss.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Pipeline with linear encoder, decoder and reconstruction loss layers.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/model/linear_ae.py#L17-L66)

## nexuml_library.scenarios.model.linear_ae.linear_ae_regression

*function*

```python
def linear_ae_regression(input_dim: int = 128, hidden_dims: list[int] | None = None, latent_dim: int = 8, num_outputs: int = 3, activation: str = 'torch.nn.ReLU') -> PipelineSpec
```

Create a PipelineSpec for a linear AE with reconstruction + regression.

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Pipeline with encoder, decoder, regression head, reconstruction loss and regression loss layers.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/model/linear_ae.py#L277-L338)
