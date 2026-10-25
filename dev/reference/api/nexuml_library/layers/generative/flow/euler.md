# nexuml_library.layers.generative.flow.euler

*module*

Euler integration layer for continuous flows.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/generative/flow/euler.py#L1-L1)

## nexuml_library.layers.generative.flow.euler.EulerIntegrator

*class*

```python
class EulerIntegrator(PipelineLayer)
```

Bases: `PipelineLayer`

Fixed-step Euler integrator over a learned vector field.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/generative/flow/euler.py#L12-L53)

### nexuml_library.layers.generative.flow.euler.EulerIntegrator.last_trajectory

*attribute* · *instance attribute*

```python
last_trajectory: torch.Tensor | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/generative/flow/euler.py#L28-L28)

### nexuml_library.layers.generative.flow.euler.EulerIntegrator.num_steps

*attribute* · *instance attribute*

```python
num_steps = num_steps
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/generative/flow/euler.py#L26-L26)

### nexuml_library.layers.generative.flow.euler.EulerIntegrator.record_trajectory

*attribute* · *instance attribute*

```python
record_trajectory = record_trajectory
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/generative/flow/euler.py#L27-L27)

### nexuml_library.layers.generative.flow.euler.EulerIntegrator.velocity

*attribute* · *instance attribute*

```python
velocity = velocity
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/generative/flow/euler.py#L25-L25)

### nexuml_library.layers.generative.flow.euler.EulerIntegrator.forward_tensor

*method*

```python
def forward_tensor(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/generative/flow/euler.py#L30-L53)
