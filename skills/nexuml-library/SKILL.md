---
name: nexuml-library
description: >-
  Create or extend a user's external NexuML library: package discovery, typed
  dataset/layer/loss/evaluation definitions, ordinary PyTorch modules, scenario
  recipes, portable configs and focused contract tests. Use for new NexuML
  algorithms, custom components, library packaging or missing-component work,
  including components needed by paper reproduction or architecture research.
  Do not assume access to NexuML's source repository or modify its core.
compatibility: Python >=3.12 with compatible installed NexuML; filesystem and shell access.
---

# Author an external NexuML library

The deliverable belongs to the user's library, not the framework's base library.
Keep algorithms reusable and experiment settings in recipes/configs.

## Locate and reuse

1. Inspect the user's package metadata, layout, public imports, dependencies,
   existing definitions, recipes and tests. Respect flat or `src/` layouts rather
   than moving an existing package to match an example.
2. Select its Python environment explicitly and inspect installed NexuML's
   version, schemas/help and discovery diagnostics. A uv-tool environment cannot
   use dependencies installed only in the project `.venv`.
3. Search the user and installed libraries for reusable components. Use
   `nn_module(importable_factory, ...)` for ordinary one-input/one-output PyTorch
   behavior. Add a typed definition when NexuML context, labels, metadata,
   fitting/lifecycle or richer routing makes one necessary.
4. Identify the precise contract and smallest numerical example before coding.
   Check trust, data/compute authorization and output paths before execution.

For example, ordinary dropout needs only these public imports and routing:

```python
import torch
from nexuml import nn_module
from nexuml.core.types import LayerSpec

dropout = LayerSpec(
    component=nn_module(torch.nn.Dropout, p=0.2),
    keys_in=["features"], keys_out=["regularized"],
)
```

Use verified import paths, not a guessed `specs` module; executable examples
should include their imports so they can be checked in the selected runtime.
Prefer distinct output keys: this preserves upstream tensors for residuals,
losses and later consumers. Reuse an input key only for an intentional,
documented overwrite; route downstream layers to the new output explicitly.

Read [component contracts](references/contracts.md) for the public interfaces.
The [offline starter](assets/minimal-library/README.md) demonstrates a standalone
package with no base-library dependency. It is a learning asset, not a template
to copy wholesale into every existing project.

## Package the user's code

Declare a compatible `nexuml` dependency and only algorithm-owned dependencies.
For published libraries, express the supported compatibility range; for research
record the exact resolved environment as well. Optional dataset integrations
should not make unrelated components unimportable.

Expose the importable package through:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

For development, install the user package editably into the selected environment.
Use the existing uv project workflow, or `uv pip install --python ... -e .` in an
existing environment. A local root registered by `nexuml library add` is an
alternative: it is user-level configuration, supplies no dependencies, and can
cause duplicate registrations if the same source is imported under another name.
Never require editable installation of NexuML itself.

## Separate definitions from runtime state

- Public immutable Pydantic definitions store semantic, serializable parameters.
- Private runtime objects own tensors, trainable modules, datasets, fitted state
  and evaluation accumulators. Forward runtime context, do not duplicate its
  routing or shapes as definition fields.
- Give registered definitions descriptive stable identities and versions. Avoid
  conflicting identities across libraries; incompatible persisted behavior needs
  an explicit version/identity decision, not import-order precedence.
- Compose Python recipes with concrete imports and definitions. Resolved YAML
  uses `type/version/params`; ordinary `model_dump()` is not portable persistence.
- Keep stage order, keys, labels, metadata, placement and metrics in the enclosing
  specs. Put only algorithm-specific options on the definition.

For a new algorithm, map its mathematics to a minimal model/loss/fitting/evaluator
combination. Preserve gradient flow and train-only fitting. An evaluation
algorithm consumes pipeline outputs; a fitted component that produces score
tensors uses a pipeline `PostTrainFitLayer`, not a reporting `EvalAlgorithm`.
Read [score production and fitting](references/contracts.md#evaluation-and-fitting)
for its lifecycle and placement constraints. Do not replace the training loop.

## Prove the contract

Use the nearest existing test style or one small runnable check. Cover the
actual risk, not just import success:

1. Validate parameters, including meaningful invalid values.
2. Check dataset sample/batch keys, shapes, labels and partition identities.
3. Check a numerical reference, finite outputs and gradients where applicable.
4. Check evaluator aggregation across unequal batches; do not average batch
   means when the metric requires sample/element weighting.
5. Discover the definitions in a fresh process and round-trip a scenario through
   `ResolvedConfig`; compile with declared shapes.
6. Run a small authorized lifecycle check when needed, recording its evidence.

Build/import success is not algorithm parity or convergence. If the installed
framework fails its documented contract, isolate the failure and report it;
do not edit site-packages, monkeypatch core behavior or clone core as a workaround.

## Deliver

Report component identities/imports, dependencies, recipe/config paths, exact
checks and limitations. Promote reusable components into the user's normal
library only after their relevant checks pass; keep unsuccessful experiment
evidence and avoid overwriting previous baselines.
