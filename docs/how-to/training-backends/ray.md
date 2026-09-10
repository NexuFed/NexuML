# Ray execution

Ray is an optional placement backend for the existing NexuML Lightning session. Use it when a scenario should run on an existing Ray cluster. NexuML does not maintain a parallel Ray-specific model or training loop.

## Install

```bash
uv pip install "nexuml[ray]"
```

Add `nexuml[s3]` when Ray run storage or datasets use S3-compatible storage. Add the DALI integration separately when the selected data loader requires it.

The driver and workers need compatible Python/Ray environments. Keep exact environment/image/CUDA choices in the consuming project or cluster configuration, not in `ScenarioSpec`.

## Configure an existing cluster

```python
from nexuml.core.types import RayClusterTarget, RayExecutionSpec

execution = RayExecutionSpec(
    target=RayClusterTarget(
        address="ray://ray.example.org:10001",
        working_dir=".",
        py_executable="uv run --locked python",
    ),
    workers=4,
    resources_per_worker={"CPU": 4, "GPU": 1},
    storage_path="s3://my-bucket/nexuml/runs",
)
```

Attach it to the scenario and use the normal command:

```bash
nexuml train my-scenario
```

Each worker restores the typed scenario, creates the normal `NexuSession`, and executes its lifecycle with a Ray-prepared Lightning trainer.

### Ray Client UV packaging

NexuML disables Ray's driver-side UV runtime-environment hook before importing
Ray. The development-runtime configs instead pass Ray's native `working_dir`,
`excludes`, and `uv` runtime environment explicitly. Ray uploads only the
bounded source/metadata allowlist, then creates the worker environment with
editable local NexuML packages plus the pinned third-party requirements.

The previous `working_dir=None` setup stopped source/dependency propagation; a
`.python-version` file would only select Python and would not deliver either.
The fix is explicit runtime-environment propagation, not a fatal Python patch
warning. If another importer loads Ray before `_connect`, set this before
`uv run`:

```bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run nexuml train my-scenario
```

This only prevents the legacy automatic UV hook. The native `runtime_env` is
still applied by Ray. Top-level `working_dir` and `py_executable` take
precedence over same-named nested values; other nested Ray runtime settings are
forwarded unchanged.

## Lightning strategies

`training.strategy` remains the single source of truth. Ray currently maps `auto`/`ddp`, `fsdp`, and `deepspeed` to Ray's official Lightning strategy integrations.

```python
TrainingSpec(strategy="ddp")
```

For strategy-specific settings, reference Ray's real strategy class:

```python
from ray.train.lightning import RayFSDPStrategy

from nexuml import strategy

TrainingSpec(strategy=strategy(RayFSDPStrategy, state_dict_type="full"))
```

The class and its constructor parameters remain navigable and statically checkable. Required third-party strategy packages still have to exist in the worker environment.

## Ray Jobs

NexuML intentionally does not wrap the Ray Jobs lifecycle. If the driver itself should run remotely/detached, use Ray's native CLI:

```bash
ray job submit --working-dir . -- \
  uv run --locked nexuml train -c configs/my-scenario.yaml
```

Use Ray's own status/log/stop tooling for the job lifecycle.

## Distributed semantic restrictions

Ray currently rejects two kinds of stateful post-training work rather than silently changing their meaning:

- scenarios with `evaluation.algorithms`;
- components whose definition declares `requires_post_train_fit`.

Both need global state across rank-sharded data. Until NexuML implements correct cross-worker finalization, keep those scenarios local. Ordinary Lightning validation/test metrics can still be reduced by the distributed trainer.

## Shared datasets and S3 WebDataset

A supported shared-data workflow is:

```text
NexuML dataset
  → WebDataset export
  → S3-compatible storage
  → ExportedDataset
  → direct tar streaming and worker-local index staging
  → DALI WebDataset reader
```

Export to S3 with the normal dataset-export command or Python API, then use the base-library `ExportedDataset` definition for the remote root.

For S3 exports, `ExportedDataset` loads `config.yaml`, metadata, and the small DALI `.idx` files. Tensor payloads remain in S3 and are read directly by DALI rather than downloaded through Python/boto3. Remote exports also avoid the per-sample Python WebDataset index; the tar/index lists in `config.yaml` are sufficient for DALI.

NexuML passes each `s3://...tar` path directly to the existing DALI WebDataset reader. DALI's WebDataset index parser requires local files, so NexuML stages only the corresponding `.idx` files in a worker-local temporary directory. DALI still owns sharding through `shard_id=global_rank` and `num_shards=world_size`.

### S3 endpoints and explicit TLS exceptions

An S3 URI is parsed as a bucket authority plus an object key. For example,
`s3://rglitza/data/shard.tar` means bucket `rglitza` and key
`data/shard.tar`; the URI does not require HTTP path-style addressing. DALI
extracts the authority and key in its
[`s3_client_manager.h`](https://github.com/NVIDIA/DALI/blob/main/dali/util/s3_client_manager.h)
and [`s3_filesystem.cc`](https://github.com/NVIDIA/DALI/blob/main/dali/util/s3_filesystem.cc)
URI handling, then its SDK chooses the HTTP addressing style. If virtual-host
addressing is selected, DNS must resolve the actual bucket host.

The secure default is unchanged. For an authorized temporary exception to the
public endpoint, keep the endpoint HTTPS and opt out only for the clients that
need it:

```bash
# boto3/NexuML S3 client; the URL scheme is required by boto3.
export AWS_ENDPOINT_URL=https://s3.ika.rub.de
export NEXUML_S3_VERIFY_SSL=0
```

When `NEXUML_S3_VERIFY_SSL` is present, NexuML derives DALI's inverse
`DALI_S3_NO_VERIFY_SSL` immediately before a native DALI reader is created. If
the shared flag is absent, an existing direct-DALI `DALI_S3_NO_VERIFY_SSL`
override is left untouched. DALI initializes its S3 client manager on first
native-reader use, so changing this setting requires a fresh process; NexuML
does not create a second DALI controller or alter the normal SDK cache.

The same boto3 setting can be selected directly in Python with
`boto3.client("s3", endpoint_url="https://s3.ika.rub.de", verify=False)` or
with `S3Client(endpoint_url="https://s3.ika.rub.de", verify=False)`. These are
temporary, user-authorized exceptions; they do not disable TLS verification by
default and do not change credentials or bucket paths.

For a per-config Ray setting, use the external config helper rather than
exporting the shared flag globally:

```python
from ray_runtime import ray_runtime_env

target = RayClusterTarget(
    address="ray://ray.example.org:10001",
    runtime_env=ray_runtime_env(verify_ssl=False),
)
```

`AWS_ENDPOINT_URL_S3` is an independent override that NexuML's Ray run-storage
configuration chooses before `AWS_ENDPOINT_URL`; it does not configure the
shared `S3Client`. PyArrow 24's `S3FileSystem` exposes `tls_ca_file_path`, not a
`verify=False` switch, so use a trusted CA there. If an already-authorized
internal `AWS_ENDPOINT_URL_S3` HTTP route is used for Ray run storage, configure
that explicitly; NexuML does not silently downgrade HTTPS or patch PyArrow.

DALI reads `AWS_ENDPOINT_URL`, while boto3/NexuML and Ray run storage can use
their own endpoint settings. Do not blindly assign the same public HTTPS URL to
`AWS_ENDPOINT_URL_S3` when its certificate is invalid. Credentials/endpoints
come from the normal AWS/provider environment and supported S3 options, not
from persisted secrets in the scenario. TLS verification opt-out cannot fix DNS
or network reachability.

!!! note "Verified direct S3 tar path"
    On the existing Ray head with UV/DALI 2.3, native DALI streamed one CPU
    AudioSet sample (`float32[160000]`) from
    `s3://rglitza/data/audioset/data/shards/val/shard-000000.tar`; only the
    72,113-byte local `.idx` was staged and no local tar copy was made. This
    verifies the single-sample read path only; a complete GPU epoch remains
    untested.

For an existing in-cluster SeaweedFS deployment, resolve the service ClusterIP
at launch from the authorized kubeconfig context rather than hard-coding an IP:

```bash
S3_IP="$(kubectl -n seaweedfs get svc seaweedfs-cluster-s3 -o jsonpath='{.spec.clusterIP}')" &&
export AWS_ENDPOINT_URL="http://${S3_IP}:8333" &&
export AWS_ENDPOINT_URL_S3="http://seaweedfs-cluster-s3.seaweedfs.svc.cluster.local:8333"
```

Service recreation can assign a new ClusterIP. `AWS_ENDPOINT_URL` is the global
native-DALI route: the DALI SDK uses HTTP path-style requests through that IP,
so no bucket DNS wildcard is needed. `AWS_ENDPOINT_URL_S3` is the separate
service-DNS HTTP route for the Boto/Ray-storage client. Both routes are
internal HTTP, so no TLS bypass setting is needed. This split is specific to
the Ray-cluster workflow; it is not a public-ingress or standalone local-DALI
solution.

An external Ray Client driver may use the same endpoint exports and the same
train command when the config factory only reads the existing S3 root; the Ray
server/workers perform the S3 I/O remotely:

```bash
export RAY_ADDRESS='ray://ray.ika.rub.de:10001'
nexuml train my-scenario
```

The internal service route is not a claim that standalone local Arrow/export
operations can reach the cluster network. Keep those local commands on a
localhost port-forward or another endpoint reachable from the local driver.

## Temporary KubeRay clusters

Cluster creation is infrastructure policy, not a second NexuML execution model. Use an infrastructure-owned KubeRay `RayJob`/cluster template to define images, CUDA, node selectors, queues, tolerations, service accounts, volumes, and autoscaling. Its entrypoint can remain the same `nexuml train ...` command.

## Troubleshooting

- **Ray import error** → install `nexuml[ray]` in the driver/worker environment.
- **Version mismatch** → align Ray/Python versions between the project and cluster.
- **Post-train/evaluation rejection** → current distributed global-finalization semantics are intentionally unsupported; run that scenario locally.
- **S3 failure** → verify credentials, endpoint/region settings, and worker access to the S3 service.
- **DALI failure** → install a compatible DALI build separately and verify `python -c "import nvidia.dali"` in the worker environment.
