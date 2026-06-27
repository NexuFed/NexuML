# fix-nexuml-self-contained-package-export

## Why

NexuML currently exports `.experiments/model/cifar-resnet/pipeline.package`, and NexuFL loads it through `torch.package.PackageImporter`. Loading `nexuml_export/artifact.pkl` fails in clean NexuFL environments that have runtime dependencies installed but do not have the NexuML workspace source tree installed.

The old package policy externalized `nexuml_library.**`, which makes built-in library pipelines fail with `ModuleNotFoundError` in NexuFL. Simply switching `nexuml_library.**` from `extern` to `intern` is not enough: loading then fails with `torch.package` import/dataclass/module issues involving modules such as `nexuml.core.discovery` and `nexuml_library.layers.model.resnet`.

NexuFL must be able to consume arbitrary NexuML pipelines as portable training artifacts without requiring clients or servers to install the whole NexuML workspace source tree. The artifact also needs enough checkpoint/training provenance for downstream TrainerV1 adapters to continue training rather than only run inference.

## What Changes

- Fix NexuML `torch.package` export so `pipeline.package` is self-contained for NexuML-owned, `nexuml_library`, and pipeline-defining custom Python code.
- Keep heavy/runtime third-party dependencies external and emit a generated `requirements.txt` plus structured dependency metadata listing only externalized modules actually referenced by the package.
- Make `nexuml_export/artifact.pkl` the stable primary entry and keep legacy `model/pipeline.pkl` compatibility.
- Preserve richer checkpoint/training metadata, including Lightning checkpoint information when available.
- Keep existing sidecars (`state_dict.pt`, `resolved_config.yaml`, `metadata.json`) and add `lightning.ckpt` when a Lightning checkpoint is available or can be generated.
- Add clean-environment smoke tests that export and load built-in CIFAR ResNet plus a tiny custom layer outside `nexuml` / `nexuml_library`.
- Add NexuFL validation coverage proving `PackageImporter(...).load_pickle("nexuml_export", "artifact.pkl")` works without installing the NexuML source workspace.
- Update docs to describe the package contract, dependency manifest, checkpoint metadata, and exact re-export command.

## Capabilities

### New

- `self-contained-nexuml-package-export`
- `exported-runtime-dependency-manifest`
- `exported-checkpoint-provenance`

### Modified

- `pipeline-export`
- `nexufl-nexuml-package-loading`

## Impact

- Primary files:
  - `src/nexuml/core/export.py`
  - `src/nexuml/core/compiler.py`
  - `src/nexuml/core/pipeline.py`
  - `src/nexuml/core/discovery.py`
  - `library/src/nexuml_library/layers/model/resnet.py`
  - `src/nexuml/cli/main.py`
  - `src/nexuml/training/lightning.py`
  - `docs/how-to/export.md`
  - `docs/reference/backends.md`
  - `pyproject.toml`
- Tests:
  - Add NexuML package export/load smoke tests.
  - Add clean Python subprocess tests with no `PYTHONPATH` pointing at the NexuML source tree.
  - Add custom external layer packaging test.
  - Add NexuFL validation fixture/test for TrainerV1 loading.
- Related downstream repo:
  - `NexuFL`, especially `src/nexufl/integrations/nexumodular/package_loader.py` and `src/nexufl/integrations/nexumodular/model_adapter.py`.
- Linear:
  - NEX-187
- Non-goals:
  - Do not require NexuFL to install the full NexuML workspace source tree.
  - Do not remove `state_dict.pt`, `resolved_config.yaml`, or `metadata.json`.
  - Do not replace all export formats with `torch.package`.
  - Do not implement Hugging Face export in this change; keep export metadata/backend seams reusable by `add-huggingface-transformers-export-backend`.
