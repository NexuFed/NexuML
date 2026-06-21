# framework-test-coverage Specification

## Purpose
Ensure core NexuML framework modules that fall outside registry conformance tests (training orchestration, data module/loader construction, evaluation storage, data export backends, tracking integrations) have deterministic, hermetic regression coverage in the default pytest suite, using synthetic data and local fakes instead of real datasets, GPUs, or external services.

## Requirements

### Requirement: Framework modules have deterministic regression tests
NexuML SHALL provide default pytest coverage for core framework modules that are not fully covered by registry conformance tests, using synthetic data and local temporary files.

#### Scenario: Training entry point is tested without expensive training
- **WHEN** the default test suite runs without `slow`, `requires_data`, or `requires_gpu` markers
- **THEN** at least one test SHALL exercise the public training entry point (`NexuSession`) with mocked or minimal dependencies and verify orchestration behavior without long-running model training. The single existing `NexuSession` test is marked `slow`, so this scenario requires a separate test runnable in the default suite, not relaxing that marker.

#### Scenario: Data module behavior is tested with synthetic inputs
- **WHEN** the default test suite runs
- **THEN** data module setup and loader construction behavior SHALL be tested for at least the default split plus one additional split (val or test) and one constructor-error path, using synthetic or temporary data, and SHALL NOT require external datasets

#### Scenario: Evaluation storage persists and reads batches
- **WHEN** `nexuml.evaluation.storage` (memory or memmap TensorDict storage, or its appendable/reservoir buffers) receives one or more synthetic batches
- **THEN** tests SHALL verify the stored data can be finalized and read back with expected keys, shapes, and values. This scenario targets `nexuml.evaluation.storage`, which has no dedicated tests today; it does not cover `nexuml.core.storage.SharedStorage`, which already has dedicated coverage.

#### Scenario: Data export roundtrip is validated across backends
- **WHEN** a small synthetic data module is exported via `export_data_module`
- **THEN** tests SHALL verify the artifact can be loaded or inspected enough to prove the export is usable for the `numpy_mmap`, `torch`, `tensordict_memmap`, and `webdataset` backends, in addition to the already-tested `numpy` backend. This scenario does not cover pipeline export (`export_package`/`export_safetensors`/`export_onnx`), which already has dedicated coverage.

#### Scenario: Tracking integrations are bounded by local fakes
- **WHEN** a minimal training or evaluation call emits metrics or artifacts through a fake or monkeypatched logger in the default suite
- **THEN** tests SHALL verify the fake logger captured the expected emitted values and SHALL NOT require external tracking services. This scenario targets the call boundary where metrics are emitted, distinct from the existing pure-utility tests (URI normalization, color helpers, git metadata) already covered.

### Requirement: Default pytest suite remains fast and hermetic
The default pytest suite SHALL avoid real datasets, GPUs, network access, external service credentials, and expensive optimization/training runs unless tests are explicitly marked opt-in.

#### Scenario: Optional integration remains marked
- **WHEN** a test requires real data, GPU, network, or long runtime
- **THEN** it SHALL be marked with the appropriate opt-in marker and excluded by the standard fast-suite selector

#### Scenario: Fast suite command covers new tests
- **WHEN** `pytest tests/ -m "not slow and not requires_data and not requires_gpu"` is run
- **THEN** all new default coverage tests SHALL be selected and runnable in a local developer environment
