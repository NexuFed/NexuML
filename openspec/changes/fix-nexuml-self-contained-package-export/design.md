# fix-nexuml-self-contained-package-export Design

## Context

Current NexuML package export is implemented in `src/nexuml/core/export.py` using `torch.package.PackageExporter`. The package constants are:

- `PACKAGE_FILENAME = "pipeline.package"`
- `PACKAGE_PICKLE_PACKAGE = "nexuml_export"`
- `PACKAGE_PICKLE_NAME = "artifact.pkl"`
- legacy package/name: `model/pipeline.pkl`

The current payload shape already points in the right direction:

```
{
  "pipeline": packaged_pipeline,
  "resolved_config": pipeline.resolved_config.model_dump(mode="json"),
  "metadata": metadata,
  "state_dict": {k: v.detach().cpu() for k, v in pipeline.state_dict().items()},
  "training_state": training_state or {},
}
```

The failing part is the packaging/import policy and the validation path. The current exporter interns `nexuml.**`, externalizes `nexuml_library.**`, externalizes runtime dependencies, and saves the payload. This breaks clean NexuFL loading because built-in library code is needed to unpickle the pipeline object.

PyTorch `torch.package` dependency management model:

- `intern`: package source modules and load them hermetically from the package.
- `extern`: leave modules outside the archive and import them from the runtime Python environment.
- `PackageExporter.save_pickle(...)` discovers dependencies through pickle globals and import analysis.
- Only Python source modules can be interned; extension/bytecode/runtime packages must be externed.

This change treats the packaged trainable pipeline object as the authoritative runtime artifact and treats config/metadata as portable contract data.

Relevant current files and roles:

- `src/nexuml/core/export.py`
  - `export_package`
  - `_extern_runtime_dependencies`
  - `_package_payload`
  - `_load_packaged_payload`
  - `load_package`
  - `load_inference_package`
  - `load_package_for_training`
- `src/nexuml/core/compiler.py`
  - builds `CompiledPipeline` from `ScenarioSpec` and registry.
- `src/nexuml/core/pipeline.py`
  - `CompiledPipeline` is the trainable `torch.nn.Module`.
  - creates optimizer/scheduler from pipeline specs.
- `src/nexuml/core/discovery.py`
  - dynamic discovery/decorators are needed before export, but should not be required to reconstruct the packaged object at load time.
- `library/src/nexuml_library/layers/model/resnet.py`
  - built-in CIFAR ResNet proof path.
- `src/nexuml/training/lightning.py`
  - `NexuLightningModule` already saves scenario/runtime metadata via `save_hyperparameters`.
  - checkpoint hooks preserve NexuML eval and post-train state.
- `src/nexufl/integrations/nexumodular/package_loader.py`
  - downstream consumer loads `pipeline.package` via `PackageImporter`.
- `src/nexufl/integrations/nexumodular/model_adapter.py`
  - downstream TrainerV1 adapter trains the loaded pipeline.

External references:

- PyTorch `torch.package` documentation:
  - `PackageExporter` packages code, pickled Python data, and resources.
  - `PackageImporter` loads packaged code hermetically, except modules marked external.
  - `intern` should be used for model code; `extern` should be used for modules expected in the runtime environment.
- Lightning checkpoint documentation:
  - Lightning checkpoints contain current epoch, global step, LightningModule state dict, optimizer states, scheduler states, callback/datamodule/loop state, and hyperparameters when saved.

## Goals & Non-Goals

### Goals

- Export packages that NexuFL can load in a clean runtime environment with runtime deps installed but without the NexuML workspace source tree.
- Include all Python source modules needed to unpickle and continue training the packaged pipeline:
  - `nexuml.**`
  - `nexuml_library.**`
  - custom modules that define pipeline layers, transforms, losses, heads, metrics, scenario support code, and other pipeline-defining classes/functions.
- Keep heavyweight/runtime third-party packages external:
  - `torch`
  - `torchvision`
  - `tensordict`
  - `pydantic`
  - `lightning`
  - `torchmetrics`
  - `numpy`
  - `pandas`
  - `safetensors`
  - `timm`
  - `transformers`
  - `librosa`
  - `torchaudio`
  - `sklearn`
  - `matplotlib`
  - `optuna`
  - `mlflow`
  - and similar runtime libraries.
- Emit a generated dependency manifest for only actually-used external modules.
- Preserve Lightning checkpoint richness where available.
- Keep package and sidecar formats backward compatible where practical.
- Prove export/load in subprocesses that cannot import from the NexuML source tree.

### Non-Goals

- Do not make NexuFL install the full NexuML workspace source tree.
- Do not vendor large third-party runtime dependencies into `pipeline.package`.
- Do not make registry/discovery scanning required for NexuFL package load.
- Do not remove existing sidecars.
- Do not implement Hugging Face export in this change.

## Core decisions

### D1 — The packaged pipeline object is the primary runtime object

`nexuml_export/artifact.pkl` SHALL contain a fully constructed, trainable `torch.nn.Module` pipeline under `payload["pipeline"]`.

NexuFL SHALL be able to use this object directly:

```
imp = torch.package.PackageImporter("pipeline.package")
payload = imp.load_pickle("nexuml_export", "artifact.pkl")
pipeline = payload["pipeline"]
```

The loaded pipeline SHALL:

- be an instance of `torch.nn.Module`;
- expose trainable parameters;
- have weights loaded;
- accept the expected TensorDict input/output contract;
- run forward on dummy CIFAR input for the CIFAR ResNet proof case;
- produce a scalar training loss through the existing loss-key path;
- support one optimizer step through downstream TrainerV1-style code.

Rationale: reconstructing from registry/discovery at NexuFL load time is fragile and reintroduces the workspace dependency.

### D2 — Config is portable metadata, not the load-time reconstruction mechanism

`payload["resolved_config"]` SHALL be JSON/YAML-safe plain data.

It MAY be loaded into `ResolvedConfig` by NexuML tooling when NexuML is installed, but NexuFL package loading SHALL NOT require rebuilding the pipeline from `ResolvedConfig`.

Sidecar `resolved_config.yaml` SHALL remain.

Rationale: `ResolvedConfig`/Pydantic/discovery objects can pull in broad source dependencies and fail under `PackageImporter`. The config is still valuable for provenance, debugging, downstream metadata, and fine-tuning workflows.

### D3 — Explicit package policy separates source modules from runtime modules

The exporter SHALL apply packaging actions in an order that prevents runtime deps from being accidentally interned and source modules from being accidentally externalized.

Policy:

```
extern runtime/third-party modules
intern nexuml.**
intern nexuml_library.**
intern discovered pipeline custom modules
optionally intern user-specified include modules
deny known workspace/dev-only modules if they appear unexpectedly
```

The implementation MAY use `PackageExporter.extern(...)`, `intern(...)`, hooks, export graph inspection, and post-export validation, but the observable contract is:

- NexuML-owned source modules are packaged.
- Built-in `nexuml_library` source modules are packaged.
- Custom source modules referenced by the pickled pipeline are packaged.
- Runtime dependencies are external and reported.

Custom code selection:

- Default: rely on `torch.package` dependency discovery from the fully constructed pipeline object.
- Add a best-effort helper to inspect the pipeline layers/classes and intern their top-level source module packages.
- Add optional explicit include patterns as an escape hatch for dynamic imports that are not visible to pickle/import analysis.

Rationale: users should not need to manually list every custom layer for normal pipelines, but dynamic import patterns need an override.

### D4 — Runtime dependency manifest is generated from actual extern usage

Export SHALL write:

```
requirements.txt
```

and add structured metadata:

```
{
  "external_dependencies": [
    {
      "module": "torch",
      "distribution": "torch",
      "version": "2.x.y",
      "specifier": "torch==2.x.y",
      "reason": "extern"
    }
  ]
}
```

Rules:

- Include only externalized modules actually referenced by the packaged payload, not every dependency in `pyproject.toml`.
- Normalize module names to distribution names where known:
  - `torch.*` -> `torch`
  - `torchvision.*` -> `torchvision`
  - `tensordict.*` -> `tensordict`
  - `pydantic.*` -> `pydantic`
  - `lightning.*` -> `lightning`
  - `torchmetrics.*` -> `torchmetrics`
  - `numpy.*` -> `numpy`
  - `pandas.*` -> `pandas`
- Resolve versions using `importlib.metadata.version(...)` where possible.
- If a version cannot be resolved, include an unpinned line and record the unresolved reason in metadata.
- Exclude stdlib modules from `requirements.txt`.

Rationale: the package should not vendor runtime deps, but NexuFL and users need a precise environment contract.

### D5 — Preserve Lightning checkpoint richness as a sidecar and normalized package metadata

The export directory SHALL preserve or generate:

```
lightning.ckpt
```

when a Lightning checkpoint is available or when a live `NexuLightningModule`/trainer can save one.

The export SHALL continue to write:

```
state_dict.pt
training_state.pt
metadata.json
```

when applicable.

`payload["metadata"]["checkpoint"]` SHALL include normalized checkpoint/training provenance when present:

```
{
  "source": ".../last.ckpt",
  "epoch": 12,
  "global_step": 3456,
  "validation_metrics": {
    "val/loss": 0.42,
    "val/accuracy": 0.87
  },
  "best_model_score": 0.42,
  "best_model_path": "...",
  "monitor": "val/loss",
  "mode": "min",
  "hyper_parameters": {
    "scenario": {},
    "runtime_metadata": {}
  }
}
```

`payload["training_state"]` SHALL preserve machine resume state where available:

- trainer epoch/global step;
- optimizer states;
- scheduler states;
- loop/callback/datamodule state where safely captured;
- NexuML eval and post-train layer state when present.

Rationale: Lightning checkpoints carry more training context than a raw state dict. NexuFL needs normalized metadata for TrainerV1 adapters, and NexuML users need the rich sidecar for resume/fine-tuning.

### D6 — Keep legacy package entry compatibility

The exporter SHALL continue writing:

```
exporter.save_pickle("model", "pipeline.pkl", payload["pipeline"])
```

unless this is proven impossible under the corrected packaging policy.

The loader SHALL prefer:

```
nexuml_export/artifact.pkl
```

and then fall back to:

```
model/pipeline.pkl
```

Rationale: existing NexuFL loaders already check both entries.

### D7 — Validate real clean-environment load path

Tests SHALL include subprocess-based validation:

1. Export built-in `cifar-resnet` to a temp directory.
2. Launch a fresh Python subprocess with:
   - no `PYTHONPATH` pointing to the NexuML repo source tree;
   - runtime dependencies importable;
   - no editable NexuML/NexuML library workspace dependency.
3. Run:

```
from torch.package import PackageImporter
imp = PackageImporter("pipeline.package")
payload = imp.load_pickle("nexuml_export", "artifact.pkl")
pipeline = payload["pipeline"]
```

4. Assert:
   - payload contains `pipeline`, `resolved_config`, `metadata`, and `training_state`;
   - pipeline is trainable;
   - parameters exist;
   - dummy CIFAR TensorDict forward succeeds;
   - a scalar loss is produced;
   - an optimizer step succeeds.
5. Repeat with a tiny custom layer defined outside `nexuml` and `nexuml_library`.

Rationale: importing inside the same editable workspace hides the failure mode.

## Risks & trade-offs

### R1 — `torch.package` source-only limitation

Only Python source modules can be interned. Some dependencies are extension modules or generated/bytecode modules and must remain external. The exporter must treat third-party libraries as runtime dependencies and not try to intern them.

### R2 — Dynamic imports may be missed

`torch.package` discovers dependencies from pickle globals and source import analysis. Dynamic imports may be invisible. Mitigation:

- inspect concrete pipeline layer classes;
- allow explicit include patterns;
- fail with actionable diagnostics when a missing module is discovered during clean-load validation.

### R3 — Discovery/global registry code can be fragile under PackageImporter

The load path should avoid requiring discovery scans or global registry population. The package contains the constructed pipeline object and class definitions, not an instruction to rebuild from registry.

### R4 — Lightning checkpoint can reintroduce source coupling

`lightning.ckpt` is valuable as a sidecar but should not become the NexuFL primary load path. The portable contract remains `pipeline.package` + normalized metadata.

### R5 — requirements.txt pinning can be too strict

Pinned versions maximize reproducibility but may be overly strict for downstream environments. Mitigation:

- write exact pinned `requirements.txt`;
- also write structured dependency metadata so consumers can implement compatibility ranges later;
- document that the generated file is the export-time environment snapshot.
