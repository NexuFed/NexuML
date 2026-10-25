# nexuml.core.post_train_layer

*module*

PostTrainFitLayer — pipeline layer with a post-training accumulate/fit lifecycle.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L1-L1)

## nexuml.core.post_train_layer.PostTrainFitLayer

*class*

```python
class PostTrainFitLayer(PipelineLayer)
```

Bases: `nexuml.core.base_layer.PipelineLayer`

Abstract pipeline layer that requires a post-training fit pass.

Lifecycle (managed by NexuSession._fit_post_train_layers):
  1. Orchestrator arms exactly one unfitted layer per predict pass: ``layer._armed = True``
  2. During that predict pass, ``forward()`` calls ``collect_batch(x, y)`` and returns
     ``(x, y)`` unchanged (no output key written yet).
  3. ``on_predict_end()`` calls ``finalize_fit()`` and sets ``_fitted = True``.
  4. On subsequent calls, ``forward()`` calls ``_transform_forward(x, y)`` (fitted mode).
  5. Other unfitted layers in the same pass are not armed — they pass through silently.

Fitted state is persisted via Lightning checkpoint hooks so loading a checkpoint
restores a fully fitted pipeline without re-running fit passes.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L17-L98)

### nexuml.core.post_train_layer.PostTrainFitLayer.delay_epochs

*attribute* · *instance attribute*

```python
delay_epochs = delay_epochs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L64-L64)

### nexuml.core.post_train_layer.PostTrainFitLayer.epoch

*attribute* · *instance attribute*

```python
epoch = 0
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L71-L71)

### nexuml.core.post_train_layer.PostTrainFitLayer.fit_mode

*attribute* · *instance attribute*

```python
fit_mode = False
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L72-L72)

### nexuml.core.post_train_layer.PostTrainFitLayer.input_sizes

*attribute* · *instance attribute*

```python
input_sizes = input_sizes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L61-L61)

### nexuml.core.post_train_layer.PostTrainFitLayer.keys_in

*attribute* · *instance attribute*

```python
keys_in = keys_in
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L56-L56)

### nexuml.core.post_train_layer.PostTrainFitLayer.keys_out

*attribute* · *instance attribute*

```python
keys_out = keys_out
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L57-L57)

### nexuml.core.post_train_layer.PostTrainFitLayer.label_in_x

*attribute* · *instance attribute*

```python
label_in_x = label_in_x
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L59-L59)

### nexuml.core.post_train_layer.PostTrainFitLayer.label_key

*attribute* · *instance attribute*

```python
label_key = label_key
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L58-L58)

### nexuml.core.post_train_layer.PostTrainFitLayer.lightning_mode

*attribute* · *instance attribute*

```python
lightning_mode = LightningMode.NONE
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L73-L73)

### nexuml.core.post_train_layer.PostTrainFitLayer.num_classes

*attribute* · *instance attribute*

```python
num_classes = num_classes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L60-L60)

### nexuml.core.post_train_layer.PostTrainFitLayer.output_sizes

*attribute* · *instance attribute*

```python
output_sizes = output_sizes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L62-L62)

### nexuml.core.post_train_layer.PostTrainFitLayer.shared_inputs

*attribute* · *instance attribute*

```python
shared_inputs = shared_inputs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L69-L69)

### nexuml.core.post_train_layer.PostTrainFitLayer.shared_memory

*attribute* · *instance attribute*

```python
shared_memory = shared_memory
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L67-L67)

### nexuml.core.post_train_layer.PostTrainFitLayer.shared_outputs

*attribute* · *instance attribute*

```python
shared_outputs = shared_outputs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L68-L68)

### nexuml.core.post_train_layer.PostTrainFitLayer.update_every_n_epochs

*attribute* · *instance attribute*

```python
update_every_n_epochs = update_every_n_epochs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L65-L65)

### nexuml.core.post_train_layer.PostTrainFitLayer.collect_batch

*method* · *abstract*

```python
def collect_batch(x: TensorDict, y: TensorDict | None) -> None
```

Accumulate statistics from one train batch during the fit predict pass.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L38-L40)

### nexuml.core.post_train_layer.PostTrainFitLayer.finalize_fit

*method* · *abstract*

```python
def finalize_fit() -> None
```

Finalize fitting after all train batches have been collected.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L42-L44)

### nexuml.core.post_train_layer.PostTrainFitLayer.forward

*method*

```python
def forward(x: TensorDict, y: TensorDict | None = None) -> tuple[TensorDict, TensorDict | None]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L52-L69)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_load_checkpoint

*method*

```python
def on_load_checkpoint(checkpoint: dict[str, Any]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L86-L91)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_predict_end

*method*

```python
def on_predict_end() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L71-L76)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_save_checkpoint

*method*

```python
def on_save_checkpoint(checkpoint: dict[str, Any]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L80-L84)

### nexuml.core.post_train_layer.PostTrainFitLayer.check_update

*method*

```python
def check_update() -> bool
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L91-L99)

### nexuml.core.post_train_layer.PostTrainFitLayer.forward_tensor

*method*

```python
def forward_tensor(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L149-L150)

### nexuml.core.post_train_layer.PostTrainFitLayer.get_label

*method*

```python
def get_label(x: TensorDict | torch.Tensor, y: TensorDict | None) -> torch.Tensor | None
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L101-L109)

### nexuml.core.post_train_layer.PostTrainFitLayer.merge_td

*method*

```python
def merge_td(x: TensorDict | torch.Tensor) -> torch.Tensor
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

Concatenate multiple keys_in tensors along the token (dim=1).

**Returns**

- (`torch.Tensor`) — Concatenated tensor from the TensorDict keys, or *x* unchanged
- (`torch.Tensor`) — when it is already a plain tensor.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L75-L89)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_fit_end

*method*

```python
def on_fit_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L155-L156)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_fit_start

*method*

```python
def on_fit_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L152-L153)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_predict_start

*method*

```python
def on_predict_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L179-L180)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_test_end

*method*

```python
def on_test_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L176-L177)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_test_start

*method*

```python
def on_test_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L173-L174)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_train_end

*method*

```python
def on_train_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L164-L165)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_train_epoch_end

*method*

```python
def on_train_epoch_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L161-L162)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_train_start

*method*

```python
def on_train_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L158-L159)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_validation_end

*method*

```python
def on_validation_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L170-L171)

### nexuml.core.post_train_layer.PostTrainFitLayer.on_validation_start

*method*

```python
def on_validation_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/base_layer.py#L167-L168)

## nexuml.core.post_train_layer.PostTrainLayerNotFittedError

*class*

```python
class PostTrainLayerNotFittedError(RuntimeError)
```

Bases: `RuntimeError`

Raised when a PostTrainFitLayer is asked to transform before fitting.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/post_train_layer.py#L13-L14)
