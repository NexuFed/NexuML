# Ray training

Ray is an optional execution backend for the existing NexuML Lightning session. Use it when one scenario should use multiple GPUs/nodes or shared cluster capacity.

Install the optional dependency:

```bash
uv sync --extra ray
```

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
uv run nexuml train --config configs/my-scenario.yaml
```

## Detached Ray Jobs

NexuML intentionally does not wrap the Ray Jobs lifecycle. Use Ray's native job CLI when the driver should run remotely and remain detached from the submitting shell:

```bash
ray job submit --working-dir . -- \
  uv run nexuml train --config configs/my-scenario.yaml
```

`--working-dir .` lets Ray upload the current project to the cluster. `uv run` then uses the project environment described by `pyproject.toml` and `uv.lock`, so changing NexuML code does not require rebuilding a custom image for every run.

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

DeepSpeed itself must be available in the Ray worker environment when that strategy is selected.

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

For S3 exports, `ExportedDataset` loads only `config.yaml` and metadata. Tensor payloads are intentionally read by DALI, not one-by-one through Python/boto3.

NexuML passes the `s3://...tar` and `s3://...idx` paths directly to the existing DALI WebDataset reader and preserves DALI sharding with `shard_id=global_rank` and `num_shards=world_size`.

!!! warning "Verify direct S3 WebDataset support in the target DALI environment"
    DALI supports S3 in several readers, but current `readers.webdataset` documentation does not explicitly guarantee cloud URLs and NVIDIA still tracks cloud-WebDataset support publicly. NexuML therefore keeps this boundary deliberately thin: direct URLs are the intended path, and a real target-environment integration test must confirm them before production use. If that combination needs a fallback, add the smallest worker-local shard materializer rather than a second cache/capability framework.

S3 credentials are resolved by boto3's normal AWS/provider credential chain and are not stored in scenario configuration.

## Temporary KubeRay clusters

Temporary Ray clusters are intentionally treated as infrastructure rather than a second execution framework. A KubeRay integration should use an infrastructure-owned `RayJob` template containing image, node selectors, queues, tolerations, service accounts, and autoscaling policy. NexuML only needs to provide the code working-directory URI and entrypoint.

Until that small template adapter is implemented, use the KubeRay `RayJob` manifest directly with the same `uv run nexuml train ...` entrypoint.

## Troubleshooting

**Ray imports are missing**

Install `nexuml[ray]` (or `uv sync --extra ray`) in the driver and worker runtime.

**DeepSpeed strategy fails to initialize**

Install DeepSpeed in the Ray worker environment. NexuML intentionally does not make it a base dependency.

**Workers cannot read the S3 dataset**

First verify the normal AWS/provider credential chain in the Ray worker environment. For S3-compatible endpoints, also verify the endpoint configuration expected by DALI itself; boto3 export settings do not automatically configure DALI's native readers.

**`wds2idx` is not found during export**

Install the DALI extra/tooling in the environment performing the export, or explicitly disable index generation if startup-time index inference is acceptable.
