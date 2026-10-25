# nexuml_library.scenarios.model.resnet_classifier

*module*

ResNet classification model scenario fragments.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/model/resnet_classifier.py#L1-L1)

## nexuml_library.scenarios.model.resnet_classifier.resnet_classifier

*function*

```python
def resnet_classifier(resnet_type: ResNetType = 'resnet18', pretrained: bool = False, cifar_stem: bool | None = None, label_key: str = 'class_labels') -> PipelineSpec
```

Create a PipelineSpec for a ResNet image classifier.

Pipeline stages:
- Encoder: ResNet backbone producing embeddings
- Head: LatentClassificationHead producing logits
- Loss: ClassificationLoss + ClassificationMetrics

**Returns**

- `PipelineSpec` (`PipelineSpec`) — Pipeline with ResNet encoder, classification head and loss layers.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/model/resnet_classifier.py#L12-L64)
