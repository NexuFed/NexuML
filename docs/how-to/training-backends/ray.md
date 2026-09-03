# Ray training

Ray is an optional execution backend for the existing NexuML Lightning session. Use it when one scenario should use multiple GPUs/nodes or shared cluster capacity.

Install the Ray dependency. Add `s3` when using shared S3 storage:

```bash
uv sync --extra ray --extra s3
```

## Runtime and version ownership

NexuML defines the Ray integration API, not a specific cluster environment. The `ray` extra declares the Ray releases supported by this NexuML version, while the consuming project chooses the exact environment that matches its target cluster.

Ray requires the application environment to use compatible Ray and Python versions with the target cluster. Keep those exact versions in the consuming project's environment and lockfile rather than in NexuML configuration or documentation.

For a uv-managed application, the Ray worker can reuse that locked project environment:

```yaml
execution:
  kind: ray
  target:
    kind: cluster
    py_executable: uv run --locked python
```

The separation is intentional:

- NexuML `pyproject.toml` declares supported Python and Ray API ranges.
- The consuming project's `pyproject.toml`, `.python-version` if desired, and `uv.lock` pin its exact Python and Ray environment.
- Ray/KubeRay infrastructure selects the matching Ray image, Python version, CUDA/GPU image variant, node selectors, volumes, and other platform settings.
- `ScenarioSpec` contains only execution placement needed by NexuML; it does not model container images, CUDA versions, Kubernetes policy, or cluster-specific caches.

## Configure an existing cluster

Ray configuration describes placement only. Training behavior remains in `training`:

```yaml
training:
  strategy: ddp

execution:
  kind: ray
  target:
    kind: cluster
    address: ray://ray.example.org:10001
    working_dir: .
    py_executable: uv run --locked python
  workers: 4
  resources_per_worker:
    CPU: 4
    GPU: 1
  storage_path: s3://my-bucket/nexuml/runs
```

`workers` may also be an elastic `[min, max]` range supported directly by Ray Train:

```yaml
execution:
  kind: ray
  workers: [2, 8]
```

Running the normal command connects to the configured cluster and executes a `TorchTrainer`. Each worker reconstructs the scenario, creates the real `NexuSession`, and calls `NexuSession.run()`.

```bash
uv run --locked nexuml train --config configs/my-scenario.yaml
```

## Detached Ray Jobs

NexuML intentionally does not wrap the Ray Jobs lifecycle. Use Ray's native job CLI when the driver should run remotely and remain detached from the submitting shell:

```bash
ray job submit --working-dir . -- \
  uv run --locked nexuml train --config configs/my-scenario.yaml
```

`--working-dir .` lets Ray upload the current project to the cluster. `uv run` then uses the project environment described by that project's `pyproject.toml` and `uv.lock`, so code and dependency changes do not require rebuilding a custom image for every run.

Use Ray's own commands/API for job status, logs, stopping, and other lifecycle operations rather than a NexuML-specific job handle.

## DDP, FSDP, and DeepSpeed

The strategy has one source of truth: `training.strategy`.

DDP:

```yaml
training:
  strategy: ddp
```

FSDP:

```yaml
training:
  strategy: fsdp
  strategy_params: {}
```

DeepSpeed:

```yaml
training:
  strategy: deepspeed
  strategy_params: {}
```

Under Ray, NexuML maps these values to Ray's official `RayDDPStrategy`, `RayFSDPStrategy`, or `RayDeepSpeedStrategy`, plus `RayLightningEnvironment`, `RayTrainReportCallback`, and `prepare_trainer`. There is no NexuML-specific implementation of these distributed strategies.

DeepSpeed itself must be available in the consuming project/worker environment when that strategy is selected.

## Post-train fitted pipeline layers

Ray currently rejects scenarios containing a `PostTrainFitLayer`. NexuML normally fits these layers after gradient training by running a predict pass over the full training set. Under distributed DALI loading, each worker sees only its rank shard, so fitting independently would silently create different fitted state on different workers.

The backend therefore fails before Ray allocation instead of changing model semantics. Keep those scenarios local until NexuML implements global post-train finalization and fitted-state synchronization.

## Evaluation algorithms

Ray also currently rejects scenarios with `evaluation.algorithms`. Those algorithms accumulate arbitrary state from test batches, so averaging their final scalar results across rank-local shards is not generally equivalent to evaluating the full dataset once. Scalar Lightning and pipeline metrics such as validation/test loss, accuracy, and F1 remain supported and are reduced across workers.

Keep stateful evaluation algorithms disabled for Ray until NexuML has a globally correct state aggregation/finalization contract.

## Shared datasets with S3 WebDataset

Ray workers need a shared data location. NexuML's distributed data path is:

```text
TensorDict
  → WebDataset export (.npy components)
  → .tar + DALI .idx
  → S3-compatible storage
  → DALI WebDataset reader on each Ray worker
  → TensorDict
```

Export directly to S3:

```python
from nexuml.data.export import export_data_module

export_data_module(
    data_module,
    "s3://my-bucket/datasets/my-dataset",
    backend="webdataset",
    samples_per_shard=2048,
)
```

The exporter keeps only the active shard locally. When a shard closes it generates its DALI `.idx`, uploads the `.tar`/`.idx` pair, removes the local pair, and continues with the next shard.

WebDataset components are stored as lossless NumPy `.npy` values with pickle disabled. NexuML does not reconstruct tensors into PNG/WAV/MP4 during the generic export path.

Load the exported dataset through the normal data source:

```python
from nexuml.data.exported import ExportedDataset

dataset = ExportedDataset("s3://my-bucket/datasets/my-dataset")
```

For S3 exports, `ExportedDataset` loads `config.yaml`, metadata, and the small DALI `.idx` files. Tensor payloads remain in S3 and are read directly by DALI rather than downloaded through Python/boto3. Remote exports also avoid the per-sample Python WebDataset index; the tar/index lists in `config.yaml` are sufficient for DALI.

NexuML passes each `s3://...tar` path directly to the existing DALI WebDataset reader. DALI's WebDataset index parser requires local files, so NexuML stages only the corresponding `.idx` files in a worker-local temporary directory. DALI still owns sharding through `shard_id=global_rank` and `num_shards=world_size`.

!!! warning "Verify direct S3 tar support in the target DALI environment"
    DALI supports S3 in several readers, but current `readers.webdataset` documentation does not explicitly guarantee cloud URLs. NexuML therefore keeps this boundary deliberately thin: direct tar URLs are the intended path, and a real target-environment integration test must confirm them before production use. If that combination needs a fallback, add the smallest bounded adapter demonstrated by that test rather than a second cache/capability framework.

S3 credentials are resolved by boto3's normal AWS/provider credential chain and are not stored in scenario configuration.

## Temporary KubeRay clusters

Temporary Ray clusters are intentionally treated as infrastructure rather than a second execution framework. A KubeRay integration should use an infrastructure-owned `RayJob` template containing the matching Ray/Python/CUDA image plus node selectors, queues, tolerations, service accounts, volumes, and autoscaling policy. NexuML only needs to provide the code working-directory URI and entrypoint.

Until that small template adapter is implemented, use the KubeRay `RayJob` manifest directly with the same `uv run --locked nexuml train ...` entrypoint.

## Troubleshooting

**Ray imports are missing**

Install `nexuml[ray]` (or `uv sync --extra ray`) in the driver and worker runtime.

**Ray reports a version mismatch**

Pin the consuming project's Ray and Python versions to match the target cluster. For KubeRay, also ensure the head and worker images use the same Ray/Python combination. Do not solve this by pinning NexuML itself to one deployment environment.

**A `PostTrainFitLayer` scenario is rejected**

This is intentional until the post-train fit pass can aggregate the complete training set and synchronize one fitted state across all Ray workers. Run that scenario locally for now.

**A scenario with `evaluation.algorithms` is rejected**

This is intentional until those algorithms can aggregate their underlying state globally before finalization. Keep scalar pipeline metrics enabled and disable stateful evaluation algorithms for the Ray run.

**DeepSpeed strategy fails to initialize**

Install DeepSpeed in the consuming project/worker environment. NexuML intentionally does not make it a base dependency.

**Workers cannot read the S3 dataset**

First verify the normal AWS/provider credential chain in the Ray worker environment. For S3-compatible endpoints, also verify the endpoint configuration expected by DALI itself; boto3 export settings do not automatically configure DALI's native readers.

**`wds2idx` is not found during export**

Install the DALI extra/tooling in the environment performing the export, or explicitly disable index generation if startup-time index inference is acceptable.
