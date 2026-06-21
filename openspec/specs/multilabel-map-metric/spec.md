## Purpose

Defines the `MultiLabelMAPMetrics` pipeline layer for computing mean Average Precision (mAP) on multi-label classification tasks. This is the standard evaluation metric for AudioSet and similar multi-label audio classification benchmarks.

## Requirements

### Requirement: Layer Registration
The system SHALL register a pipeline layer with type key `MultiLabelMAPMetrics` via the `@layer` decorator.

#### Scenario: Layer discovery
- **WHEN** the layer registry scans for available layers
- **THEN** `MultiLabelMAPMetrics` is discoverable and instantiable via `LayerSpec(type_key="MultiLabelMAPMetrics")`.

### Requirement: Multi-label mAP Computation
The layer SHALL compute mAP using `torchmetrics.classification.MultilabelAveragePrecision` over accumulated batches.

#### Scenario: Valid mAP for random predictions
- **WHEN** the layer receives random logits `[B, num_labels]` and multi-hot targets `[B, num_labels]`
- **THEN** it produces a scalar mAP value in `[0, 1]`.

#### Scenario: Perfect predictions
- **WHEN** logits perfectly rank positive labels above negative labels for all samples
- **THEN** mAP equals 1.0.

### Requirement: Sigmoid Activation
The layer SHALL apply sigmoid to raw logits before passing to the metric, converting logits to probability estimates.

#### Scenario: Logit to probability conversion
- **WHEN** raw logits are received
- **THEN** `torch.sigmoid(logits)` is applied before metric update.

### Requirement: Epoch-level Aggregation
The layer SHALL implement `get_epoch_metrics()` to return the accumulated mAP at epoch end.

#### Scenario: Epoch metric retrieval
- **WHEN** `get_epoch_metrics("val")` is called at the end of a validation epoch
- **THEN** it returns `{"mAP": <accumulated_mAP_tensor>}`.

### Requirement: Reset on Epoch Start
The layer SHALL reset metric state at the start of each validation and test epoch.

#### Scenario: Validation reset
- **WHEN** `on_validation_start()` is called
- **THEN** the internal metric state is reset for fresh accumulation.

### Requirement: Label Key Configuration
The layer SHALL read multi-hot labels from the y TensorDict using a configurable `label_key` (default `"class_logits"`).

#### Scenario: Custom label key
- **WHEN** the layer is configured with `label_key="my_labels"`
- **THEN** it reads multi-hot targets from `y["my_labels"]`.

### Requirement: Missing Labels Handling
The layer SHALL gracefully handle missing labels by emitting zero placeholder tensors.

#### Scenario: No labels available
- **WHEN** `y` is None or `label_key` is not present in `y`
- **THEN** the layer outputs zero tensors for metric keys and does not update the metric state.
