# nexuml_library.scenarios.model.resnet_classifier

*module*

ResNet classification model scenario fragments.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/model/resnet_classifier.py#L1-L1)

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

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/scenarios/model/resnet_classifier.py#L12-L64)
