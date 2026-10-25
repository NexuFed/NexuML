# nexuml.core

*module*

Core NexuML runtime: pipeline, layers, registry, compiler, and types.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/__init__.py#L1-L1)

## nexuml.core.NnModuleLayer

*class* · *pydantic model*

```python
class NnModuleLayer(LayerDefinition)
```

Bases: `nexuml.core.components.LayerDefinition`

Re-exported from: `nexuml.core.torch_adapter`

Portable definition for an importable one-input/one-output PyTorch module.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L22-L69)

### nexuml.core.NnModuleLayer.args

*attribute* · *pydantic field*

```python
args: list[object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L27-L27)

### nexuml.core.NnModuleLayer.factory

*attribute* · *pydantic field*

```python
factory: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L26-L26)

### nexuml.core.NnModuleLayer.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L28-L28)

### nexuml.core.NnModuleLayer.kind

*attribute* · *pydantic field*

```python
kind = 'layer'
```

Inherited from: `nexuml.core.components.LayerDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L124-L124)

### nexuml.core.NnModuleLayer.requires_post_train_fit

*attribute* · *class attribute*

```python
requires_post_train_fit: bool = False
```

Inherited from: `nexuml.core.components.LayerDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L125-L125)

### nexuml.core.NnModuleLayer.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.core.NnModuleLayer.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.core.NnModuleLayer.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

### nexuml.core.NnModuleLayer.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L46-L69)

## nexuml.core.PostTrainFitLayer

*class*

```python
class PostTrainFitLayer(PipelineLayer)
```

Bases: `nexuml.core.base_layer.PipelineLayer`

Re-exported from: `nexuml.core.post_train_layer`

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

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/post_train_layer.py#L17-L98)

### nexuml.core.PostTrainFitLayer.delay_epochs

*attribute* · *instance attribute*

```python
delay_epochs = delay_epochs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L64-L64)

### nexuml.core.PostTrainFitLayer.epoch

*attribute* · *instance attribute*

```python
epoch = 0
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L71-L71)

### nexuml.core.PostTrainFitLayer.fit_mode

*attribute* · *instance attribute*

```python
fit_mode = False
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L72-L72)

### nexuml.core.PostTrainFitLayer.input_sizes

*attribute* · *instance attribute*

```python
input_sizes = input_sizes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L61-L61)

### nexuml.core.PostTrainFitLayer.keys_in

*attribute* · *instance attribute*

```python
keys_in = keys_in
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L56-L56)

### nexuml.core.PostTrainFitLayer.keys_out

*attribute* · *instance attribute*

```python
keys_out = keys_out
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L57-L57)

### nexuml.core.PostTrainFitLayer.label_in_x

*attribute* · *instance attribute*

```python
label_in_x = label_in_x
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L59-L59)

### nexuml.core.PostTrainFitLayer.label_key

*attribute* · *instance attribute*

```python
label_key = label_key
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L58-L58)

### nexuml.core.PostTrainFitLayer.lightning_mode

*attribute* · *instance attribute*

```python
lightning_mode = LightningMode.NONE
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L73-L73)

### nexuml.core.PostTrainFitLayer.num_classes

*attribute* · *instance attribute*

```python
num_classes = num_classes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L60-L60)

### nexuml.core.PostTrainFitLayer.output_sizes

*attribute* · *instance attribute*

```python
output_sizes = output_sizes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L62-L62)

### nexuml.core.PostTrainFitLayer.shared_inputs

*attribute* · *instance attribute*

```python
shared_inputs = shared_inputs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L69-L69)

### nexuml.core.PostTrainFitLayer.shared_memory

*attribute* · *instance attribute*

```python
shared_memory = shared_memory
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L67-L67)

### nexuml.core.PostTrainFitLayer.shared_outputs

*attribute* · *instance attribute*

```python
shared_outputs = shared_outputs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L68-L68)

### nexuml.core.PostTrainFitLayer.update_every_n_epochs

*attribute* · *instance attribute*

```python
update_every_n_epochs = update_every_n_epochs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L65-L65)

### nexuml.core.PostTrainFitLayer.collect_batch

*method* · *abstract*

```python
def collect_batch(x: TensorDict, y: TensorDict | None) -> None
```

Accumulate statistics from one train batch during the fit predict pass.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/post_train_layer.py#L38-L40)

### nexuml.core.PostTrainFitLayer.finalize_fit

*method* · *abstract*

```python
def finalize_fit() -> None
```

Finalize fitting after all train batches have been collected.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/post_train_layer.py#L42-L44)

### nexuml.core.PostTrainFitLayer.forward

*method*

```python
def forward(x: TensorDict, y: TensorDict | None = None) -> tuple[TensorDict, TensorDict | None]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/post_train_layer.py#L52-L69)

### nexuml.core.PostTrainFitLayer.on_load_checkpoint

*method*

```python
def on_load_checkpoint(checkpoint: dict[str, Any]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/post_train_layer.py#L86-L91)

### nexuml.core.PostTrainFitLayer.on_predict_end

*method*

```python
def on_predict_end() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/post_train_layer.py#L71-L76)

### nexuml.core.PostTrainFitLayer.on_save_checkpoint

*method*

```python
def on_save_checkpoint(checkpoint: dict[str, Any]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/post_train_layer.py#L80-L84)

### nexuml.core.PostTrainFitLayer.check_update

*method*

```python
def check_update() -> bool
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L91-L99)

### nexuml.core.PostTrainFitLayer.forward_tensor

*method*

```python
def forward_tensor(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L149-L150)

### nexuml.core.PostTrainFitLayer.get_label

*method*

```python
def get_label(x: TensorDict | torch.Tensor, y: TensorDict | None) -> torch.Tensor | None
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L101-L109)

### nexuml.core.PostTrainFitLayer.merge_td

*method*

```python
def merge_td(x: TensorDict | torch.Tensor) -> torch.Tensor
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

Concatenate multiple keys_in tensors along the token (dim=1).

**Returns**

- (`torch.Tensor`) — Concatenated tensor from the TensorDict keys, or *x* unchanged
- (`torch.Tensor`) — when it is already a plain tensor.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L75-L89)

### nexuml.core.PostTrainFitLayer.on_fit_end

*method*

```python
def on_fit_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L155-L156)

### nexuml.core.PostTrainFitLayer.on_fit_start

*method*

```python
def on_fit_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L152-L153)

### nexuml.core.PostTrainFitLayer.on_predict_start

*method*

```python
def on_predict_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L179-L180)

### nexuml.core.PostTrainFitLayer.on_test_end

*method*

```python
def on_test_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L176-L177)

### nexuml.core.PostTrainFitLayer.on_test_start

*method*

```python
def on_test_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L173-L174)

### nexuml.core.PostTrainFitLayer.on_train_end

*method*

```python
def on_train_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L164-L165)

### nexuml.core.PostTrainFitLayer.on_train_epoch_end

*method*

```python
def on_train_epoch_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L161-L162)

### nexuml.core.PostTrainFitLayer.on_train_start

*method*

```python
def on_train_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L158-L159)

### nexuml.core.PostTrainFitLayer.on_validation_end

*method*

```python
def on_validation_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L170-L171)

### nexuml.core.PostTrainFitLayer.on_validation_start

*method*

```python
def on_validation_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L167-L168)

## nexuml.core.PostTrainLayerNotFittedError

*class*

```python
class PostTrainLayerNotFittedError(RuntimeError)
```

Bases: `RuntimeError`

Re-exported from: `nexuml.core.post_train_layer`

Raised when a PostTrainFitLayer is asked to transform before fitting.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/post_train_layer.py#L13-L14)

## nexuml.core.TorchModuleAdapter

*class*

```python
class TorchModuleAdapter(PipelineLayer)
```

Bases: `nexuml.core.base_layer.PipelineLayer`

Re-exported from: `nexuml.core.torch_adapter`

Wraps any torch.nn.Module to conform to PipelineLayer interface.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L95-L116)

### nexuml.core.TorchModuleAdapter.module

*attribute* · *instance attribute*

```python
module = module
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L107-L107)

### nexuml.core.TorchModuleAdapter.delay_epochs

*attribute* · *instance attribute*

```python
delay_epochs = delay_epochs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L64-L64)

### nexuml.core.TorchModuleAdapter.epoch

*attribute* · *instance attribute*

```python
epoch = 0
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L71-L71)

### nexuml.core.TorchModuleAdapter.fit_mode

*attribute* · *instance attribute*

```python
fit_mode = False
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L72-L72)

### nexuml.core.TorchModuleAdapter.input_sizes

*attribute* · *instance attribute*

```python
input_sizes = input_sizes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L61-L61)

### nexuml.core.TorchModuleAdapter.keys_in

*attribute* · *instance attribute*

```python
keys_in = keys_in
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L56-L56)

### nexuml.core.TorchModuleAdapter.keys_out

*attribute* · *instance attribute*

```python
keys_out = keys_out
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L57-L57)

### nexuml.core.TorchModuleAdapter.label_in_x

*attribute* · *instance attribute*

```python
label_in_x = label_in_x
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L59-L59)

### nexuml.core.TorchModuleAdapter.label_key

*attribute* · *instance attribute*

```python
label_key = label_key
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L58-L58)

### nexuml.core.TorchModuleAdapter.lightning_mode

*attribute* · *instance attribute*

```python
lightning_mode = LightningMode.NONE
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L73-L73)

### nexuml.core.TorchModuleAdapter.num_classes

*attribute* · *instance attribute*

```python
num_classes = num_classes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L60-L60)

### nexuml.core.TorchModuleAdapter.output_sizes

*attribute* · *instance attribute*

```python
output_sizes = output_sizes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L62-L62)

### nexuml.core.TorchModuleAdapter.shared_inputs

*attribute* · *instance attribute*

```python
shared_inputs = shared_inputs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L69-L69)

### nexuml.core.TorchModuleAdapter.shared_memory

*attribute* · *instance attribute*

```python
shared_memory = shared_memory
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L67-L67)

### nexuml.core.TorchModuleAdapter.shared_outputs

*attribute* · *instance attribute*

```python
shared_outputs = shared_outputs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L68-L68)

### nexuml.core.TorchModuleAdapter.update_every_n_epochs

*attribute* · *instance attribute*

```python
update_every_n_epochs = update_every_n_epochs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L65-L65)

### nexuml.core.TorchModuleAdapter.forward_tensor

*method*

```python
def forward_tensor(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L109-L116)

### nexuml.core.TorchModuleAdapter.check_update

*method*

```python
def check_update() -> bool
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L91-L99)

### nexuml.core.TorchModuleAdapter.forward

*method*

```python
def forward(x: TensorDict | torch.Tensor, y: TensorDict | None = None) -> tuple[TensorDict | torch.Tensor, TensorDict | None]
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L111-L147)

### nexuml.core.TorchModuleAdapter.get_label

*method*

```python
def get_label(x: TensorDict | torch.Tensor, y: TensorDict | None) -> torch.Tensor | None
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L101-L109)

### nexuml.core.TorchModuleAdapter.merge_td

*method*

```python
def merge_td(x: TensorDict | torch.Tensor) -> torch.Tensor
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

Concatenate multiple keys_in tensors along the token (dim=1).

**Returns**

- (`torch.Tensor`) — Concatenated tensor from the TensorDict keys, or *x* unchanged
- (`torch.Tensor`) — when it is already a plain tensor.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L75-L89)

### nexuml.core.TorchModuleAdapter.on_fit_end

*method*

```python
def on_fit_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L155-L156)

### nexuml.core.TorchModuleAdapter.on_fit_start

*method*

```python
def on_fit_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L152-L153)

### nexuml.core.TorchModuleAdapter.on_predict_end

*method*

```python
def on_predict_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L182-L183)

### nexuml.core.TorchModuleAdapter.on_predict_start

*method*

```python
def on_predict_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L179-L180)

### nexuml.core.TorchModuleAdapter.on_test_end

*method*

```python
def on_test_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L176-L177)

### nexuml.core.TorchModuleAdapter.on_test_start

*method*

```python
def on_test_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L173-L174)

### nexuml.core.TorchModuleAdapter.on_train_end

*method*

```python
def on_train_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L164-L165)

### nexuml.core.TorchModuleAdapter.on_train_epoch_end

*method*

```python
def on_train_epoch_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L161-L162)

### nexuml.core.TorchModuleAdapter.on_train_start

*method*

```python
def on_train_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L158-L159)

### nexuml.core.TorchModuleAdapter.on_validation_end

*method*

```python
def on_validation_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L170-L171)

### nexuml.core.TorchModuleAdapter.on_validation_start

*method*

```python
def on_validation_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L167-L168)

## nexuml.core.callback

*function*

```python
def callback(factory: Callable[P, Callback], *args: P.args, **kwargs: P.kwargs) -> CallbackSpec
```

Re-exported from: `nexuml.core.factory`

Configure an importable Lightning callback factory.

**Returns**

- (`CallbackSpec`) — Portable callback specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L127-L135)

## nexuml.core.nn_module

*function*

```python
def nn_module(factory: Callable[P, nn.Module], *args: P.args, **kwargs: P.kwargs) -> NnModuleLayer
```

Re-exported from: `nexuml.core.torch_adapter`

Create a portable direct-module definition from an importable factory.

**Returns**

- (`NnModuleLayer`) — Validated universal layer definition.

**Raises**

- `TypeError` — If *factory* is a live module instance.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L75-L92)

## nexuml.core.optimizer

*function*

```python
def optimizer(factory: Callable[Concatenate[Any, P], Optimizer], *args: P.args, **kwargs: P.kwargs) -> OptimizerSpec
```

Re-exported from: `nexuml.core.factory`

Configure an optimizer factory; model parameters are supplied at runtime.

**Returns**

- (`OptimizerSpec`) — Portable optimizer specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L138-L148)

## nexuml.core.scheduler

*function*

```python
def scheduler(factory: Callable[Concatenate[Optimizer, P], LRScheduler], *args: P.args, **kwargs: P.kwargs) -> SchedulerSpec
```

Re-exported from: `nexuml.core.factory`

Configure a scheduler factory; its optimizer is supplied at runtime.

**Returns**

- (`SchedulerSpec`) — Portable scheduler specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L151-L163)

## nexuml.core.strategy

*function*

```python
def strategy(factory: Callable[P, Strategy], *args: P.args, **kwargs: P.kwargs) -> StrategySpec
```

Re-exported from: `nexuml.core.factory`

Configure an importable Lightning strategy factory.

**Returns**

- (`StrategySpec`) — Portable strategy specification.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L166-L174)

## nexuml.core.writer

*function*

```python
def writer(factory: Callable[P, ExportBackend], *args: P.args, **kwargs: P.kwargs) -> WriterSpec
```

Re-exported from: `nexuml.core.factory`

Configure an importable dataset export backend factory.

**Returns**

- (`WriterSpec`) — Portable writer specification.

**Raises**

- `TypeError` — If *factory* is not an export backend class.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/factory.py#L177-L192)

**Modules**

- `nexuml.core.backends`
- `nexuml.core.base_layer`
- `nexuml.core.compiler`
- `nexuml.core.components`
- `nexuml.core.config`
- `nexuml.core.diagram`
- `nexuml.core.discovery`
- `nexuml.core.export`
- `nexuml.core.factory`
- `nexuml.core.log_paths`
- `nexuml.core.pipeline`
- `nexuml.core.post_train_layer`
- `nexuml.core.provenance`
- `nexuml.core.registry`
- `nexuml.core.scenario_loader`
- `nexuml.core.scenario_registry`
- `nexuml.core.schema`
- `nexuml.core.serialization`
- `nexuml.core.storage`
- `nexuml.core.torch_adapter`
- `nexuml.core.types`
