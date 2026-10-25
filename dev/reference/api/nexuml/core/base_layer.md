# nexuml.core.base_layer

*module*

Base pipeline layer definitions for NexuML.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L1-L1)

## nexuml.core.base_layer.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('model.' + __name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L13-L13)

## nexuml.core.base_layer.LightningMode

*class*

```python
class LightningMode(Enum)
```

Bases: `Enum`

Lightning trainer phase reported to pipeline layers.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L19-L26)

### nexuml.core.base_layer.LightningMode.NONE

*attribute* · *class attribute* · *instance attribute*

```python
NONE = 'none'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L22-L22)

### nexuml.core.base_layer.LightningMode.PREDICTING

*attribute* · *class attribute* · *instance attribute*

```python
PREDICTING = 'predicting'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L26-L26)

### nexuml.core.base_layer.LightningMode.TESTING

*attribute* · *class attribute* · *instance attribute*

```python
TESTING = 'testing'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L25-L25)

### nexuml.core.base_layer.LightningMode.TRAINING

*attribute* · *class attribute* · *instance attribute*

```python
TRAINING = 'training'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L23-L23)

### nexuml.core.base_layer.LightningMode.VALIDATING

*attribute* · *class attribute* · *instance attribute*

```python
VALIDATING = 'validating'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L24-L24)

## nexuml.core.base_layer.PipelineLayer

*class*

```python
class PipelineLayer(torch.nn.Module)
```

Bases: `torch.nn.Module`

Base class for all NexuML pipeline layers.

Subclasses implement ``forward_tensor`` and optionally override lifecycle
hooks (``on_fit_start``, ``on_fit_end``).  The base ``forward`` method
routes each ``keys_in`` key through ``forward_tensor`` and writes results
to the corresponding ``keys_out`` key in the output ``TensorDict``.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L29-L183)

### nexuml.core.base_layer.PipelineLayer.delay_epochs

*attribute* · *instance attribute*

```python
delay_epochs = delay_epochs
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L64-L64)

### nexuml.core.base_layer.PipelineLayer.epoch

*attribute* · *instance attribute*

```python
epoch = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L71-L71)

### nexuml.core.base_layer.PipelineLayer.fit_mode

*attribute* · *instance attribute*

```python
fit_mode = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L72-L72)

### nexuml.core.base_layer.PipelineLayer.input_sizes

*attribute* · *instance attribute*

```python
input_sizes = input_sizes
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L61-L61)

### nexuml.core.base_layer.PipelineLayer.keys_in

*attribute* · *instance attribute*

```python
keys_in = keys_in
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L56-L56)

### nexuml.core.base_layer.PipelineLayer.keys_out

*attribute* · *instance attribute*

```python
keys_out = keys_out
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L57-L57)

### nexuml.core.base_layer.PipelineLayer.label_in_x

*attribute* · *instance attribute*

```python
label_in_x = label_in_x
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L59-L59)

### nexuml.core.base_layer.PipelineLayer.label_key

*attribute* · *instance attribute*

```python
label_key = label_key
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L58-L58)

### nexuml.core.base_layer.PipelineLayer.lightning_mode

*attribute* · *instance attribute*

```python
lightning_mode = LightningMode.NONE
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L73-L73)

### nexuml.core.base_layer.PipelineLayer.num_classes

*attribute* · *instance attribute*

```python
num_classes = num_classes
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L60-L60)

### nexuml.core.base_layer.PipelineLayer.output_sizes

*attribute* · *instance attribute*

```python
output_sizes = output_sizes
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L62-L62)

### nexuml.core.base_layer.PipelineLayer.shared_inputs

*attribute* · *instance attribute*

```python
shared_inputs = shared_inputs
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L69-L69)

### nexuml.core.base_layer.PipelineLayer.shared_memory

*attribute* · *instance attribute*

```python
shared_memory = shared_memory
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L67-L67)

### nexuml.core.base_layer.PipelineLayer.shared_outputs

*attribute* · *instance attribute*

```python
shared_outputs = shared_outputs
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L68-L68)

### nexuml.core.base_layer.PipelineLayer.update_every_n_epochs

*attribute* · *instance attribute*

```python
update_every_n_epochs = update_every_n_epochs
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L65-L65)

### nexuml.core.base_layer.PipelineLayer.check_update

*method*

```python
def check_update() -> bool
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L91-L99)

### nexuml.core.base_layer.PipelineLayer.forward

*method*

```python
def forward(x: TensorDict | torch.Tensor, y: TensorDict | None = None) -> tuple[TensorDict | torch.Tensor, TensorDict | None]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L111-L147)

### nexuml.core.base_layer.PipelineLayer.forward_tensor

*method*

```python
def forward_tensor(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L149-L150)

### nexuml.core.base_layer.PipelineLayer.get_label

*method*

```python
def get_label(x: TensorDict | torch.Tensor, y: TensorDict | None) -> torch.Tensor | None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L101-L109)

### nexuml.core.base_layer.PipelineLayer.merge_td

*method*

```python
def merge_td(x: TensorDict | torch.Tensor) -> torch.Tensor
```

Concatenate multiple keys_in tensors along the token (dim=1).

**Returns**

- (`torch.Tensor`) — Concatenated tensor from the TensorDict keys, or *x* unchanged
- (`torch.Tensor`) — when it is already a plain tensor.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L75-L89)

### nexuml.core.base_layer.PipelineLayer.on_fit_end

*method*

```python
def on_fit_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L155-L156)

### nexuml.core.base_layer.PipelineLayer.on_fit_start

*method*

```python
def on_fit_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L152-L153)

### nexuml.core.base_layer.PipelineLayer.on_predict_end

*method*

```python
def on_predict_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L182-L183)

### nexuml.core.base_layer.PipelineLayer.on_predict_start

*method*

```python
def on_predict_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L179-L180)

### nexuml.core.base_layer.PipelineLayer.on_test_end

*method*

```python
def on_test_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L176-L177)

### nexuml.core.base_layer.PipelineLayer.on_test_start

*method*

```python
def on_test_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L173-L174)

### nexuml.core.base_layer.PipelineLayer.on_train_end

*method*

```python
def on_train_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L164-L165)

### nexuml.core.base_layer.PipelineLayer.on_train_epoch_end

*method*

```python
def on_train_epoch_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L161-L162)

### nexuml.core.base_layer.PipelineLayer.on_train_start

*method*

```python
def on_train_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L158-L159)

### nexuml.core.base_layer.PipelineLayer.on_validation_end

*method*

```python
def on_validation_end()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L170-L171)

### nexuml.core.base_layer.PipelineLayer.on_validation_start

*method*

```python
def on_validation_start()
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/base_layer.py#L167-L168)
