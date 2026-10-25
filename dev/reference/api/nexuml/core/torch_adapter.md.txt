# nexuml.core.torch_adapter

*module*

Adapter wrapping plain torch.nn.Modules for TensorDict pipeline compatibility.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L1-L1)

## nexuml.core.torch_adapter.P

*attribute* · *module attribute*

```python
P = ParamSpec('P')
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L72-L72)

## nexuml.core.torch_adapter.NnModuleLayer

*class* · *pydantic model*

```python
class NnModuleLayer(LayerDefinition)
```

Bases: `nexuml.core.components.LayerDefinition`

Portable definition for an importable one-input/one-output PyTorch module.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L22-L69)

### nexuml.core.torch_adapter.NnModuleLayer.args

*attribute* · *pydantic field*

```python
args: list[object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L27-L27)

### nexuml.core.torch_adapter.NnModuleLayer.factory

*attribute* · *pydantic field*

```python
factory: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L26-L26)

### nexuml.core.torch_adapter.NnModuleLayer.kwargs

*attribute* · *pydantic field*

```python
kwargs: dict[str, object]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L28-L28)

### nexuml.core.torch_adapter.NnModuleLayer.kind

*attribute* · *pydantic field*

```python
kind = 'layer'
```

Inherited from: `nexuml.core.components.LayerDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L124-L124)

### nexuml.core.torch_adapter.NnModuleLayer.requires_post_train_fit

*attribute* · *class attribute*

```python
requires_post_train_fit: bool = False
```

Inherited from: `nexuml.core.components.LayerDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L125-L125)

### nexuml.core.torch_adapter.NnModuleLayer.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L34-L34)

### nexuml.core.torch_adapter.NnModuleLayer.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L35-L35)

### nexuml.core.torch_adapter.NnModuleLayer.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/components.py#L25-L31)

### nexuml.core.torch_adapter.NnModuleLayer.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L46-L69)

## nexuml.core.torch_adapter.TorchModuleAdapter

*class*

```python
class TorchModuleAdapter(PipelineLayer)
```

Bases: `nexuml.core.base_layer.PipelineLayer`

Wraps any torch.nn.Module to conform to PipelineLayer interface.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L95-L116)

### nexuml.core.torch_adapter.TorchModuleAdapter.module

*attribute* · *instance attribute*

```python
module = module
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L107-L107)

### nexuml.core.torch_adapter.TorchModuleAdapter.delay_epochs

*attribute* · *instance attribute*

```python
delay_epochs = delay_epochs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L64-L64)

### nexuml.core.torch_adapter.TorchModuleAdapter.epoch

*attribute* · *instance attribute*

```python
epoch = 0
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L71-L71)

### nexuml.core.torch_adapter.TorchModuleAdapter.fit_mode

*attribute* · *instance attribute*

```python
fit_mode = False
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L72-L72)

### nexuml.core.torch_adapter.TorchModuleAdapter.input_sizes

*attribute* · *instance attribute*

```python
input_sizes = input_sizes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L61-L61)

### nexuml.core.torch_adapter.TorchModuleAdapter.keys_in

*attribute* · *instance attribute*

```python
keys_in = keys_in
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L56-L56)

### nexuml.core.torch_adapter.TorchModuleAdapter.keys_out

*attribute* · *instance attribute*

```python
keys_out = keys_out
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L57-L57)

### nexuml.core.torch_adapter.TorchModuleAdapter.label_in_x

*attribute* · *instance attribute*

```python
label_in_x = label_in_x
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L59-L59)

### nexuml.core.torch_adapter.TorchModuleAdapter.label_key

*attribute* · *instance attribute*

```python
label_key = label_key
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L58-L58)

### nexuml.core.torch_adapter.TorchModuleAdapter.lightning_mode

*attribute* · *instance attribute*

```python
lightning_mode = LightningMode.NONE
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L73-L73)

### nexuml.core.torch_adapter.TorchModuleAdapter.num_classes

*attribute* · *instance attribute*

```python
num_classes = num_classes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L60-L60)

### nexuml.core.torch_adapter.TorchModuleAdapter.output_sizes

*attribute* · *instance attribute*

```python
output_sizes = output_sizes
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L62-L62)

### nexuml.core.torch_adapter.TorchModuleAdapter.shared_inputs

*attribute* · *instance attribute*

```python
shared_inputs = shared_inputs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L69-L69)

### nexuml.core.torch_adapter.TorchModuleAdapter.shared_memory

*attribute* · *instance attribute*

```python
shared_memory = shared_memory
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L67-L67)

### nexuml.core.torch_adapter.TorchModuleAdapter.shared_outputs

*attribute* · *instance attribute*

```python
shared_outputs = shared_outputs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L68-L68)

### nexuml.core.torch_adapter.TorchModuleAdapter.update_every_n_epochs

*attribute* · *instance attribute*

```python
update_every_n_epochs = update_every_n_epochs
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L65-L65)

### nexuml.core.torch_adapter.TorchModuleAdapter.forward_tensor

*method*

```python
def forward_tensor(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L109-L116)

### nexuml.core.torch_adapter.TorchModuleAdapter.check_update

*method*

```python
def check_update() -> bool
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L91-L99)

### nexuml.core.torch_adapter.TorchModuleAdapter.forward

*method*

```python
def forward(x: TensorDict | torch.Tensor, y: TensorDict | None = None) -> tuple[TensorDict | torch.Tensor, TensorDict | None]
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L111-L147)

### nexuml.core.torch_adapter.TorchModuleAdapter.get_label

*method*

```python
def get_label(x: TensorDict | torch.Tensor, y: TensorDict | None) -> torch.Tensor | None
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L101-L109)

### nexuml.core.torch_adapter.TorchModuleAdapter.merge_td

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

### nexuml.core.torch_adapter.TorchModuleAdapter.on_fit_end

*method*

```python
def on_fit_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L155-L156)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_fit_start

*method*

```python
def on_fit_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L152-L153)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_predict_end

*method*

```python
def on_predict_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L182-L183)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_predict_start

*method*

```python
def on_predict_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L179-L180)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_test_end

*method*

```python
def on_test_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L176-L177)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_test_start

*method*

```python
def on_test_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L173-L174)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_train_end

*method*

```python
def on_train_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L164-L165)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_train_epoch_end

*method*

```python
def on_train_epoch_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L161-L162)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_train_start

*method*

```python
def on_train_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L158-L159)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_validation_end

*method*

```python
def on_validation_end()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L170-L171)

### nexuml.core.torch_adapter.TorchModuleAdapter.on_validation_start

*method*

```python
def on_validation_start()
```

Inherited from: `nexuml.core.base_layer.PipelineLayer`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/base_layer.py#L167-L168)

## nexuml.core.torch_adapter.nn_module

*function*

```python
def nn_module(factory: Callable[P, nn.Module], *args: P.args, **kwargs: P.kwargs) -> NnModuleLayer
```

Create a portable direct-module definition from an importable factory.

**Returns**

- (`NnModuleLayer`) — Validated universal layer definition.

**Raises**

- `TypeError` — If *factory* is a live module instance.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/torch_adapter.py#L75-L92)
