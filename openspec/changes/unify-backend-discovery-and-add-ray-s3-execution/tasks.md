# Tasks

## 1. Replace the oversized NEX-154 architecture

- [x] 1.1 Reset the feature branch implementation to the original base while preserving this rewritten OpenSpec change.
- [x] 1.2 Remove the generic execution backend/result/job-handle framework and keep local training as direct `NexuSession.run()`.
- [x] 1.3 Remove custom Ray Jobs lifecycle/bundling/preflight/recovery/observability/security layers.
- [x] 1.4 Remove custom Ray DDP/FSDP/DeepSpeed strategy implementations and Ray Tune from this change.
- [x] 1.5 Remove generic storage publication/cache/generation abstractions that are not required by S3 WebDataset transport.

## 2. Add the thin Ray training backend

- [x] 2.1 Add local/Ray execution configuration to `ScenarioSpec` with local as the default.
- [x] 2.2 Keep strategy selection in `TrainingSpec`; add `strategy_params` and support `auto`, `ddp`, `fsdp`, `deepspeed` without duplicating the strategy in Ray config.
- [ ] 2.3 Add a small common Lightning Trainer-kwargs seam so local and Ray construction share NexuML defaults.
- [x] 2.4 Implement `src/nexuml/execution/ray.py` using Ray 2.58 `TorchTrainer`, `ScalingConfig`, and `RunConfig`.
- [x] 2.5 Map Lightning strategies to official `RayDDPStrategy`, `RayFSDPStrategy`, and `RayDeepSpeedStrategy`, plus `RayLightningEnvironment`, `RayTrainReportCallback`, and `prepare_trainer`.
- [x] 2.6 Ensure each Ray worker reconstructs `ScenarioSpec`, creates the real `NexuSession`, and calls `run()` once.
- [x] 2.7 Support existing-cluster `ray.init(address=...)` and fixed/elastic worker counts.
- [x] 2.8 Keep native Ray `Result` as the return value; do not add another result/handle schema.
- [ ] 2.9 Implement globally correct `PostTrainFitLayer` finalization/state synchronization and remove the temporary Ray safety guard.
- [x] 2.10 Reject stateful `evaluation.algorithms` under Ray until their underlying state can be aggregated globally; keep scalar Lightning/pipeline metrics distributed normally.
- [ ] 2.11 Keep KubeRay as a small infrastructure-template adapter only if it can remain infrastructure-agnostic.

## 3. Simplify WebDataset and add `.idx`

- [x] 3.1 Keep the existing `webdataset` export backend; do not add another export framework.
- [x] 3.2 Make `.npy` with `allow_pickle=False` the canonical payload for tensor components.
- [x] 3.3 Remove generic tensor-to-PNG/WAV/MP4 re-encoding from the default WebDataset path.
- [x] 3.4 Generate one sibling `.idx` for every closed tar with DALI's bundled `wds2idx` utility.
- [x] 3.5 Record tar/index paths in the existing export metadata only; do not add NexuML-specific index profiles or integrity protocols.

## 4. Add slim S3 transport

- [x] 4.1 Add optional `boto3` S3 support with endpoint, region, and AWS profile configuration but no credentials in scenario/config serialization.
- [x] 4.2 Support `s3://` output for WebDataset export by writing one local shard, indexing it, uploading the tar/index pair, and deleting the local pair before proceeding.
- [x] 4.3 Upload `config.yaml` and metadata after export completion.
- [x] 4.4 Add an S3-backed exported-dataset path that downloads only config/metadata and exposes shard/index `s3://` URLs.

## 5. Connect S3 WebDataset to DALI

- [x] 5.1 Reuse the existing DALI WebDataset loader and decoder path.
- [x] 5.2 Pass S3 tar/index URLs directly to `fn.readers.webdataset`.
- [x] 5.3 Preserve DALI rank sharding through `shard_id=global_rank` and `num_shards=world_size`.
- [ ] 5.4 Add one optional real DALI + S3-compatible integration test; do not add a speculative cache/capability framework.

## 6. Documentation

- [x] 6.1 Add `docs/how-to/training-backends/index.md` explaining local vs Ray execution while Lightning remains the training engine.
- [x] 6.2 Add `docs/how-to/training-backends/ray.md` covering existing clusters, worker resources, DDP/FSDP/DeepSpeed, S3 WebDataset data, KubeRay, and troubleshooting.
- [x] 6.3 Prominently document detached execution with `ray job submit --working-dir . -- uv run ...` and state that NexuML intentionally does not wrap Ray Jobs.
- [x] 6.4 Document WebDataset `.npy` payloads and automatic `.idx` generation.

## 7. Focused validation

- [x] 7.1 Add compact unit tests for config, strategy mapping, Ray worker/session reuse, scaling config, WebDataset indexing, and S3 publication.
- [ ] 7.2 Add at most one optional Ray integration smoke test and one optional S3+DALI integration test.
- [ ] 7.3 Run the existing test suite plus Ruff and Ty; fix regressions without adding compatibility shims for the discarded NEX-154 implementation.
