# fix-nexuml-self-contained-package-export Tasks

## 1. Reproduce and isolate the failure

- [x] 1.1 Add a regression test that exports `cifar-resnet` and fails when `nexuml_library.**` is externalized.
- [x] 1.2 Add a regression test for the observed "interning only `nexuml_library.**` still fails" path so the real import/dataclass/module failure is captured.
- [x] 1.3 Add a test helper that launches a Python subprocess without `PYTHONPATH` pointing at the NexuML repo.
- [x] 1.4 Add a diagnostic helper that prints `PackageImporter.file_structure()` and exporter dependency decisions when package load fails.

## 2. Implement package policy

- [x] 2.1 Refactor `_extern_runtime_dependencies()` in `src/nexuml/core/export.py` into an explicit package policy helper.
- [x] 2.2 Ensure runtime dependencies are externalized:
  - `torch`, `torch.**`
  - `torchvision`, `torchvision.**`
  - `tensordict`, `tensordict.**`
  - `pydantic`, `pydantic.**`
  - `lightning`, `lightning.**`
  - `torchmetrics`, `torchmetrics.**`
  - `numpy`, `numpy.**`
  - `pandas`, `pandas.**`
  - optional runtime libs already supported by NexuML.
- [x] 2.3 Intern `nexuml.**`.
- [x] 2.4 Intern `nexuml_library.**`.
- [x] 2.5 Add best-effort discovery of concrete pipeline layer module packages from `pipeline.iter_layers()`.
- [x] 2.6 Add optional explicit package include patterns to the export API for dynamic custom modules.
- [x] 2.7 Add validation that no unexpected `nexuml` or `nexuml_library` module remains external.
- [x] 2.8 Add actionable error messages for modules that cannot be interned because they are non-source modules.

## 3. Stabilize package payload

- [x] 3.1 Keep `nexuml_export/artifact.pkl` as the primary entry.
- [x] 3.2 Keep legacy `model/pipeline.pkl` as a compatibility entry.
- [x] 3.3 Ensure `payload["pipeline"]` is a CPU `torch.nn.Module` with loaded state.
- [x] 3.4 Ensure `payload["resolved_config"]` is a plain JSON/YAML-safe dict.
- [x] 3.5 Ensure `payload["metadata"]` is a plain JSON-safe dict.
- [x] 3.6 Ensure `payload["training_state"]` is present even when empty.
- [x] 3.7 Ensure `load_inference_package()` and `_load_packaged_payload()` handle the updated payload without registry reconstruction.

## 4. Preserve checkpoint/training provenance

- [x] 4.1 Extend `export_package()` to accept checkpoint/source metadata from CLI export.
- [x] 4.2 When exporting from a `.ckpt`, load top-level Lightning checkpoint metadata:
  - `epoch`
  - `global_step`
  - `state_dict`
  - optimizer states
  - scheduler states
  - callbacks
  - loops
  - `hyper_parameters`
  - NexuML-specific checkpoint entries.
- [x] 4.3 Normalize checkpoint metadata into `metadata["checkpoint"]`.
- [x] 4.4 Preserve validation metrics when present in callback/logged metric state.
- [x] 4.5 Preserve ModelCheckpoint metadata when present:
  - best model path
  - best model score
  - monitor
  - mode.
- [x] 4.6 Add `lightning.ckpt` sidecar when a checkpoint is provided.
- [x] 4.7 Generate `lightning.ckpt` when a live trainer/lightning module is available and no checkpoint path was passed.
- [x] 4.8 Keep `state_dict.pt`, `training_state.pt`, `resolved_config.yaml`, and `metadata.json`.

## 5. Generate external dependency manifest

- [x] 5.1 Collect actual external modules used by the packaged payload after export dependency resolution.
- [x] 5.2 Normalize external module names to distribution names.
- [x] 5.3 Resolve installed versions with `importlib.metadata.version`.
- [x] 5.4 Exclude stdlib modules from the install manifest.
- [x] 5.5 Write `requirements.txt` with only actually-used external distributions.
- [x] 5.6 Add structured `metadata["external_dependencies"]`.
- [x] 5.7 Add tests for dependency manifest contents using CIFAR ResNet export.
- [x] 5.8 Document that `requirements.txt` is an export-time runtime snapshot.

## 6. Clean-env smoke tests

- [x] 6.1 Add a CIFAR ResNet package export/load test.
- [x] 6.2 In a clean subprocess, load `nexuml_export/artifact.pkl` with `PackageImporter`.
- [x] 6.3 Assert `payload["pipeline"]` is trainable and has parameters.
- [x] 6.4 Run one dummy CIFAR forward using TensorDict.
- [x] 6.5 Assert loss keys produce a scalar loss.
- [x] 6.6 Run one optimizer step.
- [x] 6.7 Assert `payload["metadata"]["checkpoint"]` exists when exporting from a checkpoint.
- [x] 6.8 Assert `requirements.txt` exists and excludes `nexuml` / `nexuml_library`.

## 7. Custom code packaging test

- [x] 7.1 Add a tiny custom package outside `src/nexuml` and `library/src/nexuml_library` in a test fixture.
- [x] 7.2 Define a custom `PipelineLayer` in that package.
- [x] 7.3 Build/export a tiny pipeline using the custom layer.
- [x] 7.4 Load it in a clean subprocess without the custom source path available.
- [x] 7.5 Assert the custom layer class comes from the package importer and can train.
- [x] 7.6 Add a dynamic-import variant if feasible and prove explicit include patterns work.

## 8. NexuFL validation

- [x] 8.1 Add or update a NexuFL-side fixture that consumes a NexuML `pipeline.package`.
- [x] 8.2 Validate `src/nexufl/integrations/nexumodular/package_loader.py` prefers `nexuml_export/artifact.pkl`.
- [x] 8.3 Validate the loaded pipeline works with `model_adapter.py` TrainerV1-style train batch.
- [x] 8.4 Ensure NexuFL does not import from a local NexuML workspace during this test.
- [x] 8.5 Validate runtime dependency errors mention `requirements.txt` and `metadata.external_dependencies`.

## 9. CLI and docs

- [x] 9.1 Update `nexuml export` to pass checkpoint path/source metadata into `export_package()`.
- [x] 9.2 Update `docs/how-to/export.md` with the new artifact layout.
- [x] 9.3 Document `nexuml_export/artifact.pkl` payload contract.
- [x] 9.4 Document generated `requirements.txt`.
- [x] 9.5 Document `lightning.ckpt` and normalized checkpoint metadata.
- [x] 9.6 Add the exact re-export command:

```
nexuml export cifar-resnet --checkpoint /workspaces/NexuFederated/.experiments/checkpoints/cifar-resnet/last.ckpt -o /workspaces/NexuFederated/.experiments/model/cifar-resnet
```

## 10. Validation commands

- [x] 10.1 Run targeted NexuML export tests.
- [x] 10.2 Run the clean-env subprocess tests.
- [x] 10.3 Run the custom layer package tests.
- [x] 10.4 Run the NexuFL package loader validation.
- [x] 10.5 Run `ruff`/type checks required by the repo.
- [x] 10.6 Run `openspec validate --strict`.
