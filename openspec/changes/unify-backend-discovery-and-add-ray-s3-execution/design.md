# Design: thin Ray execution with S3 WebDataset data

## Design principles

1. **One training lifecycle.** `NexuSession.run()` remains the only implementation of fit → validate → post-train fit → test.
2. **Use NexuML's existing seams.** Extend `TrainingSpec`, `ExportBackend`, `ExportedDataset`, and the DALI loader rather than adding parallel frameworks.
3. **Delegate platform behavior.** Ray owns workers, retries/checkpoints/results, Ray Jobs, process groups, and Lightning integration. DALI owns WebDataset reading and rank sharding. boto3 owns S3 transport and credentials.
4. **Keep infrastructure out of config.** Exact Ray/Python/CUDA images, Kubernetes worker images, node selectors, Kueue, service accounts, tolerations, volumes, caches, and cluster topology live in the consuming project or Ray/KubeRay manifests, not NexuML models.
5. **No compatibility debt.** The existing oversized NEX-154 implementation is replaced rather than wrapped or preserved.

## Target structure

```text
src/nexuml/
├── core/
│   └── types.py
├── training/
│   └── lightning.py
├── execution/
│   ├── __init__.py
│   ├── ray.py
│   └── kuberay.py              # small optional template adapter
├── data/
│   ├── exported.py
│   ├── export/
│   │   ├── backend.py
│   │   ├── runner.py
│   │   └── webdataset.py
│   └── loaders/
│       ├── dali_backend.py
│       └── dali_multimodal.py
├── storage/
│   ├── __init__.py
│   └── s3.py
└── cli/
    └── main.py

docs/how-to/training-backends/
├── index.md
└── ray.md

tests/
├── execution/
│   └── test_ray.py
└── data/
    └── test_s3_webdataset.py
```

Files absent from this structure are not to be recreated merely because the discarded branch contained them.

## Configuration

Local remains the default and requires no wrapper:

```yaml
execution:
  kind: local
```

Ray describes placement only:

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

Large-model strategies stay in `training`:

```yaml
training:
  strategy: fsdp
  strategy_params: {}
```

or:

```yaml
training:
  strategy: deepspeed
  strategy_params: {}
```

Elastic worker counts use Ray's native representation instead of a separate topology enum:

```yaml
execution:
  kind: ray
  workers: [2, 8]
```

## Runtime/version ownership

NexuML declares compatibility, not a deployment image.

- NexuML's package metadata declares the supported Python range and a bounded Ray API range.
- A consuming project pins the exact Ray and Python versions that match its target cluster, normally through its own `pyproject.toml`, optional `.python-version`, and `uv.lock`.
- Ray/KubeRay infrastructure selects the exact Ray/Python/CUDA image and all platform scheduling/storage details.
- `RayClusterTarget.py_executable` is an optional generic Ray runtime hook. It can point at `uv run --locked python` so workers use the consuming project's locked environment; it must not encode a NexuML-owned Python patch release or image tag.

The repository lockfile may resolve one exact Ray version for reproducible NexuML development/CI. That resolved version is not part of the public deployment contract as long as it lies within NexuML's declared compatibility range.

## Ray execution

`src/nexuml/execution/ray.py` is the complete normal Ray execution boundary.

### Driver

The driver:

1. optionally calls `ray.init(address=...)` for an existing cluster;
2. creates a `TorchTrainer`;
3. passes the serialized `ScenarioSpec` through `train_loop_config`;
4. maps `workers` and `resources_per_worker` to `ScalingConfig`;
5. maps `storage_path` to `RunConfig`;
6. returns Ray's native `Result`.

No provider-neutral result wrapper is introduced.

### Worker

Each worker:

1. validates the received scenario with `ScenarioSpec.model_validate`;
2. constructs the real `NexuSession`;
3. injects the Ray-aware Lightning trainer construction seam;
4. calls `session.run()` exactly once.

No Ray module implements training/evaluation steps.

### Lightning integration

Ray-specific Trainer construction uses the official Ray Lightning integration:

- `RayDDPStrategy` for `auto`/`ddp`;
- `RayFSDPStrategy` for `fsdp`;
- `RayDeepSpeedStrategy` for `deepspeed`;
- `RayLightningEnvironment`;
- one `RayTrainReportCallback`;
- `prepare_trainer`.

The common Lightning Trainer kwargs remain defined once by NexuML. The Ray seam changes only the fields that genuinely differ under Ray.

### Existing cluster and Ray Jobs

Interactive/direct execution connects with `ray.init(address=...)` and runs `TorchTrainer`.

Detached execution is intentionally external to the NexuML API:

```bash
ray job submit --working-dir . -- uv run --locked nexuml train <scenario>
```

Ray uploads the working directory and `uv run` uses the consuming project's environment. NexuML does not duplicate working-directory manifests, hashing, staging, job handles, lifecycle calls, or log/status APIs.

### KubeRay

Temporary cluster execution uses one user/infrastructure-owned RayJob template. NexuML may substitute only the code working-directory URI and entrypoint and submit the resulting resource. The template owns the exact Ray/Python/CUDA image, Kubernetes topology, storage mounts, queues, and scheduling policy. This adapter must stay small; if it needs a platform model hierarchy, it is out of scope.

## Distributed semantic guards

Some NexuML post-training behavior cannot yet be reproduced correctly from independent rank shards.

- `PostTrainFitLayer` scenarios are rejected under Ray until one globally correct fit/finalization state can be produced and synchronized.
- `evaluation.algorithms` are rejected under Ray until their underlying accumulated state can be combined globally before finalization.
- Scalar Lightning/pipeline metrics remain distributed normally and may use Lightning/TorchMetrics synchronization.

Failing explicitly is preferable to returning plausible but rank-local results.

## WebDataset format

WebDataset represents the **tensor view exported by NexuML**, not a reconstruction of the original media files.

Canonical component encoding is `.npy`:

```text
00000000.features.npy
00000000.label__machine.npy
00000000.label__condition.npy
00000001.features.npy
...
```

All NumPy serialization uses `allow_pickle=False`.

Reasons:

- lossless and self-describing shape/dtype;
- independent of PyTorch pickle formats;
- one representation for audio/image/video/features/labels;
- no PIL/soundfile/FFmpeg re-encoding path;
- directly decodable inside DALI.

Native compressed media may be added later only as an explicit optimization backed by benchmark evidence; the generic exporter must not decode and re-encode media to emulate source formats.

## DALI indexes

Every closed tar shard produces a sibling `.idx` using DALI's bundled `wds2idx` utility. NexuML does not implement or version-pin the `.idx` format.

```text
shard-000000.tar
shard-000000.idx
```

Local export calls `wds2idx` after closing each shard. S3 export does the same before uploading the pair.

## S3 export

S3 support is intentionally small and boto3-compatible.

`src/nexuml/storage/s3.py` provides only operations needed by this feature: parse an `s3://` URI, create a lazy client, upload/download bytes or files, and list objects. Endpoint/profile/region may be configured; credentials remain in the normal AWS/provider chain.

The WebDataset backend streams shard publication:

```text
write current tar locally
→ close tar
→ generate .idx
→ upload tar + idx
→ delete local pair
→ continue next shard
→ upload config.yaml + metadata.parquet at completion
```

Local temporary usage is therefore bounded by approximately one shard.

## S3 exported dataset and DALI

An S3-backed exported dataset loads its small `config.yaml`, metadata, and DALI `.idx` files locally/in memory. Tar references stay as `s3://...` URLs so workers do not download the tensor payloads.

The existing DALI WebDataset loader receives those paths and continues to pass:

```python
shard_id=global_rank
num_shards=world_size
```

DALI therefore remains the sharding authority.

The implementation sends direct `s3://` WebDataset tar paths to DALI and stages only `.idx` files because DALI's WebDataset index parser uses local file I/O. A real DALI + S3-compatible integration test is the acceptance check. No speculative direct/cache capability framework is added. If a supported DALI environment cannot consume the tar paths, resolve that integration limitation with the smallest bounded adapter rather than prebuilding a cache subsystem.

## Dependencies

- `ray[default,train]>=2.57,<2.59` is the optional Ray compatibility range for this NexuML release. Consuming projects pin the exact cluster-compatible Ray version themselves.
- `boto3` remains the optional `s3` extra.
- DALI remains optional; exact CUDA/image selection belongs to the consuming runtime/infrastructure rather than `ScenarioSpec`.
- `all` includes `ray` and `s3`.

## Tests

Keep tests focused on NexuML-owned behavior:

- execution config validation;
- strategy selection maps to the official Ray strategy class;
- worker calls canonical `NexuSession.run()`;
- fixed and elastic `ScalingConfig` mapping;
- existing-cluster `ray.init` mapping;
- distributed-semantic guards;
- WebDataset writes `.npy` components with pickle disabled;
- one `.idx` is generated per tar by invoking `wds2idx`;
- S3 export uploads closed tar/index pairs and final metadata without retaining all shards locally;
- S3 exported dataset resolves shard/index URLs;
- DALI receives S3 paths plus correct `shard_id`/`num_shards`.

Add at most one optional real Ray smoke test and one optional real S3+DALI integration test. Do not rebuild a large conformance matrix.
