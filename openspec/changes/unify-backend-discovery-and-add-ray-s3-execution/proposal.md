# simplify-ray-s3-training

## Why

NexuML needs to run the same scenario locally and on Ray without creating a second training stack. The current NEX-154 implementation grew into a general distributed platform with custom execution contracts, job lifecycle, preflight, recovery, storage publication, observability, strategy adapters, Tune, KubeRay/Kueue abstractions, and broad backend inventory. That complexity does not fit NexuML's design.

The useful product path is much smaller: keep `NexuSession.run()` and the existing Lightning/data abstractions authoritative, use Ray Train only for distributed placement, and use S3-backed WebDataset shards so workers can read shared data through the existing DALI loader.

## What Changes

- Add one optional `ScenarioSpec.execution` choice: local by default or Ray.
- Keep training semantics in `TrainingSpec`. `training.strategy` remains the single strategy setting and supports `auto`, `ddp`, `fsdp`, and `deepspeed` for both local Lightning and Ray Lightning execution.
- Add a thin Ray adapter built on current Ray Train APIs: `TorchTrainer`, `ScalingConfig`, `RunConfig`, `RayDDPStrategy`, `RayFSDPStrategy`, `RayDeepSpeedStrategy`, `RayLightningEnvironment`, `RayTrainReportCallback`, and `prepare_trainer`.
- Every Ray worker constructs the real `NexuSession` and calls `run()`. No fit/validate/test logic is duplicated in Ray code.
- Support direct connection to an existing Ray cluster. Detached submission is intentionally delegated to the Ray CLI and documented as `ray job submit --working-dir . -- uv run ...`; NexuML does not reimplement Ray Jobs.
- Keep temporary KubeRay execution as a small template-based boundary: infrastructure owns the RayJob YAML; NexuML only fills the working directory/entrypoint and submits it. Kueue policy and Kubernetes topology remain outside the NexuML model.
- Extend the existing `webdataset` export backend instead of introducing a new data framework. The canonical shard payload is NumPy `.npy` for tensors, written with `allow_pickle=False`.
- Emit one DALI `.idx` per tar shard with DALI's bundled `wds2idx` utility. NexuML does not implement its own DALI index format.
- Add a small S3-compatible helper using boto3. Credentials remain in the standard AWS/provider chain and never enter scenario configuration.
- Allow WebDataset export to stream closed `.tar` + `.idx` pairs to S3 so local disk usage is bounded to the current shard.
- Allow an exported S3 WebDataset to expose `s3://` shard/index paths to the existing DALI WebDataset loader. DALI continues to own rank/world-size sharding.
- Add focused documentation under `docs/how-to/training-backends/`, including local-vs-Ray concepts, existing-cluster execution, Ray Jobs via `--working-dir . -- uv run`, DDP/FSDP/DeepSpeed, S3 data, and KubeRay templates.

## Capabilities

### New

- `ray-training-backend` — Thin Ray Train execution of the canonical NexuML session with DDP/FSDP/DeepSpeed and existing-cluster/KubeRay targets.
- `s3-webdataset-dali` — NumPy-based WebDataset export with DALI `.idx` files, S3 publication, and DALI loading/sharding from shared object storage.

## Non-goals

- No generic `ExecutionRequest`/`ExecutionResult`/job-handle framework.
- No `LocalExecutionBackend`; local execution remains the existing direct `NexuSession.run()` path.
- No custom DDP/FSDP/DeepSpeed implementations, profiles, symbol pinning, report bridges, or process-group ownership.
- No custom Ray Jobs lifecycle, bundling, detach/reattach, status, log, or cancellation APIs.
- No generic storage provider abstraction, conditional publication protocol, generation pointers, multipart framework, cache framework, or storage preflight system.
- No generic capability/preflight/observability/security subsystem for this feature.
- No Ray Tune in NEX-154. Existing Optuna tuning is unchanged.
- No Ray Data requirement in the training path.
- No Kueue model or scheduler policy inside NexuML.
- No legacy compatibility layer for the discarded NEX-154 implementation.

## Impact

The intended implementation is small and concentrated in existing extension points: scenario types, the Lightning trainer seam, a thin Ray module, the existing WebDataset exporter, the existing DALI loader, a small S3 helper, focused tests, and documentation. The default local training path and existing dataset formats remain unchanged.
