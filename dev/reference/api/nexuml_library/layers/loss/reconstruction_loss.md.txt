# nexuml_library.layers.loss.reconstruction_loss

*module*

Reconstruction loss layer.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L1-L1)

## nexuml_library.layers.loss.reconstruction_loss.ReconstructionLoss

*class*

```python
class ReconstructionLoss(LayerDefinition)
```

Bases: `LayerDefinition`

Computes MSE reconstruction loss between original and reconstructed features.

keys_in should be [original_key, reconstructed_key].
keys_out should be [loss_key].

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L17-L33)

### nexuml_library.layers.loss.reconstruction_loss.ReconstructionLoss.inverse_mask

*attribute* · *class attribute* · *instance attribute*

```python
inverse_mask: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L29-L29)

### nexuml_library.layers.loss.reconstruction_loss.ReconstructionLoss.match_min_length

*attribute* · *class attribute* · *instance attribute*

```python
match_min_length: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L26-L26)

### nexuml_library.layers.loss.reconstruction_loss.ReconstructionLoss.norm_pix_loss

*attribute* · *class attribute* · *instance attribute*

```python
norm_pix_loss: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L25-L25)

### nexuml_library.layers.loss.reconstruction_loss.ReconstructionLoss.num_tokens

*attribute* · *class attribute* · *instance attribute*

```python
num_tokens: int = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L30-L30)

### nexuml_library.layers.loss.reconstruction_loss.ReconstructionLoss.patch_size

*attribute* · *class attribute* · *instance attribute*

```python
patch_size: int | tuple[int, int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L27-L27)

### nexuml_library.layers.loss.reconstruction_loss.ReconstructionLoss.use_mask_loss

*attribute* · *class attribute* · *instance attribute*

```python
use_mask_loss: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L28-L28)

### nexuml_library.layers.loss.reconstruction_loss.ReconstructionLoss.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/reconstruction_loss.py#L32-L33)
