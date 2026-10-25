# nexuml_library.scenarios.model.flow_matching

*module*

Flow Matching model fragment.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/model/flow_matching.py#L1-L1)

## nexuml_library.scenarios.model.flow_matching.mlp_flow_matching

*function*

```python
def mlp_flow_matching(feature_key: str = 'features', hidden_dims: list[int] | None = None, num_steps: int = 100, record_trajectory: bool = False) -> PipelineSpec
```

Build the basic MLP Flow Matching pipeline.

**Returns**

- (`PipelineSpec`) — Linear path sampling, flow model, and Flow Matching loss pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/model/flow_matching.py#L11-L50)

## nexuml_library.scenarios.model.flow_matching.unet_flow_matching

*function*

```python
def unet_flow_matching(feature_key: str = 'features', base_channels: int = 32, time_embedding_dim: int = 128, num_steps: int = 100) -> PipelineSpec
```

Build an image Flow Matching pipeline with a U-Net vector field.

**Returns**

- (`PipelineSpec`) — Linear path sampling, U-Net flow model, and Flow Matching loss pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/model/flow_matching.py#L53-L92)
