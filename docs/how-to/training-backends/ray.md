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
from nexuml.execution.definitions import RayClusterTarget, RayClusterExecution

execution = RayClusterExecution(
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

`address="auto"` resolves an existing connection (or validated `RAY_ADDRESS`); discovery
never initializes Ray. A caller-owned already initialized Ray runtime must be shut down
before launch so an unverified target cannot replace the reviewed selection. Set
`target.dashboard_address` explicitly for bounded read-only node/resource discovery.

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

## Ray Jobs: `ray_job`

NexuML's `RayJobExecution` means the **KubeRay RayJob resource**, not Ray Jobs API.
Its infrastructure-owned template chooses a new job-scoped or an existing Ray cluster.
For generic Ray Jobs API submissions, use Ray's native CLI:

Select `RayJobExecutionSpec` to submit the **whole normal CLI process** remotely
through Ray's `JobSubmissionClient`. Unlike `RayExecutionSpec`, this is an
asynchronous submission backend, not a Ray Train worker group. It supports the
current single-process RL/Isaac lifecycle without enabling distributed RL.

### Python defaults: no profile file required

```python
from nexuml.core.types import RayJobExecutionSpec

spec.execution = RayJobExecutionSpec()
```

This defaults to `uv sync --locked`, a fresh worker-local venv **per submission**,
direct venv Python execution, one GPU, and these environment settings:

```text
UV_PROJECT_ENVIRONMENT=/tmp/ray/nexuml-envs/{run_id}
UV_CACHE_DIR=/tmp/ray/uv-cache
UV_LINK_MODE=hardlink
UV_HTTP_TIMEOUT=600
UV_CONCURRENT_DOWNLOADS=4
```

`{run_id}` is resolved in the Python command and environment values for each
submission, even when reusing a spec. The caller's spec is not mutated.
Source upload defaults to `working_dir="."`; relative paths use the submission
directory, not a profile file's directory. Address, CA and pool are not hardcoded.

The project's lockfile owns dependencies and Python compatibility. To enable
extras, replace `setup_command`; for this checkout's ordinary library scenarios:

```python
spec.execution = RayJobExecutionSpec(
    address="https://ray.example.org",
    verify="/path/to/ca-bundle.pem",
    setup_command=["uv", "sync", "--locked", "--python", "3.12",
                   "--extra", "library", "--extra", "ray"],
    entrypoint_resources={"pool_gpu_workers": 1},
)
```

These defaults intentionally replace the former no-install/zero-GPU behavior.
For an already-prepared environment use
`RayJobExecutionSpec(setup_command=None, py_executable="/prepared/bin/python")`;
no uv environment settings are added in that mode. For CPU jobs set
`entrypoint_num_gpus=0` and choose CPU training settings in the scenario.
Explicit interpreter and environment values take precedence over generated
defaults. If you override the venv path, keep setup and Python paths aligned.
Local execution remains the default for scenarios that do not select `ray_job`.

### Reuse a project execution profile

For repeated CLI submissions, the optional `configs/execution/ray.yaml` adds
this checkout's Python 3.12, `library`/`ray` extras and IKA `pool_gpu_workers` tag.
It does not duplicate the class's cache, venv or GPU defaults. From the repo root:

```bash
export RAY_API_SERVER_ADDRESS=https://ray.example.org
nexuml train --scenario-file my_scenario.py \
  --execution-profile configs/execution/ray.yaml
```

Profiles are plain `RayJobExecutionSpec` YAML mappings with `kind: ray_job`.
Only use trusted files: their setup commands execute remotely. YAML does not
expand shell/environment variables. Keep secrets out of uploaded files.

Precedence is **scenario → profile replaces execution → `-O` overrides**;
other scenario settings are unchanged. Copy the profile for different dependency
extras or pools, or override fields per run. For example:

- `-O 'execution.entrypoint_resources={"pool_dgx_workers":1}'` selects another pool.
- `-O execution.verify=/path/to/ca-bundle.pem` selects a private CA.
- `-O 'execution.setup_command=["uv","sync","--locked","--extra","isaac","--extra","ray"]'`
  selects Isaac dependencies. Extras are project-owned, not inferred from components.

For CPU use, clear both the GPU reservation and the profile's pool tag:
`-O execution.entrypoint_num_gpus=0 -O 'execution.entrypoint_resources={}'`.
Also select CPU training in the scenario; reservations do not rewrite its devices.

Python callers can assign `load_ray_job_profile(Path("configs/execution/ray.yaml"))`
from `nexuml.execution` to `spec.execution` before calling `submit_ray_job(spec)`.
Profiles are neither selected automatically nor reloaded remotely.

### Submission and paths

For a scenario that already sets its execution, use the normal command:

```bash
nexuml train --scenario-file scenario_job.py
```

The CLI and `nexuml.execution.submit_ray_job(spec)` return a **submission ID**,
not a training result. Neither starts a local learner/simulator, waits, tails logs
or retries. Use Ray's native status/log/stop tools afterwards.

The backend sends the resolved scenario after overrides, resets execution to
`LocalExecutionSpec`, and invokes the normal remote CLI. It does not rerun the
factory, compile locally, write local staging files or recursively submit.
`--artifact-dir` snapshots remain local and include the submission ID.

Datasets, importable components and checkpoints must be worker-accessible.
`--trainer-checkpoint` is rejected rather than implicitly transported;
selective `scenario.checkpoint.source` must also exist remotely. Set unique
absolute `exports[].output` paths on durable storage. `NEXUML_LOGS_ROOT` redirects
relative logs, **not policy exports**. Nothing infers persistent output storage.

### Remote dependency setup

`setup_command` is a nonempty **argv list**, or `None` to skip setup. It runs once
remotely before Python, with shell-quoted arguments and `runtime_env` variables.
A nonzero exit prevents training and preserves failure; success is followed by
`exec` of `py_executable`. Custom installers remain supported but must create the
selected environment or supply a matching Python command. A POSIX shell is required.

Separate setup and direct Python execution avoid the cold `uv run` SIGPIPE
handoff in [uv #21935](https://github.com/astral-sh/uv/issues/21935). No signal
masking, worker-image rebuild or arbitrary `uv run` rewriting is involved.

The default cache is shared only within a worker; mutable venvs are private.
Hardlinks require cache and venv on the same filesystem. Both disappear on worker
cleanup and consume local disk until then. Avoid unpacked caches on shared FUSE
storage such as JuiceFS: per-file metadata can stall even cached installs, and
copy mode or a shared hardlinked venv does not solve that cost. Use mounted storage
for results. Explicit environment paths/values and SDK packaging/exclusions are
preserved; only missing uv settings are filled when setup is enabled.

Workers still need uv, Git, compatible Python/Ray, host drivers/system libraries
and enough physical disk. The backend does not enforce wall-time/free-disk limits
or guarantee available GPU capacity. Resource fields describe **logical
reservations**, not CPU/RAM utilization limits or Ray Train worker resources.

`address=None` uses Ray's native environment-based endpoint selection. Even an
explicit address is overridden by `RAY_API_SERVER_ADDRESS`/`RAY_ADDRESS` according
to the SDK's rules; do not leave an ingress setting active when switching to a
forwarded localhost endpoint. TLS verification defaults to `True`; supply the CA
bundle for a private HTTPS endpoint. Connection settings and runtime-env values
are not embedded in the remote scenario, but setup/Python commands and the
scenario itself are visible in the job entrypoint: do not put secrets in those
commands or model/data configuration.

For a complete single-GPU robotics example with locked uv dependencies, TLS,
worker-pool selection, detached submission and private viewer forwarding, see
[Isaac Cartpole via Ray Jobs](../isaac-cartpole.md#run-remotely-with-ray-jobs).
That launches one remote local-execution CLI job, not distributed RL.

## Distributed semantic restrictions

These restrictions apply to Ray Train (`ray`), not whole-process Ray Jobs (`ray_job`).
Ray Train rejects two kinds of stateful post-training work rather than silently changing their meaning:

- scenarios with `evaluation.algorithms`;
- components whose definition declares `requires_post_train_fit`.

Both need global state across rank-sharded data. Until cross-worker finalization
is supported, use local execution or Ray Jobs for those scenarios. Ordinary
Lightning validation/test metrics can still be reduced by the distributed trainer.

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

!!! warning "Verify direct S3 tar support in the target DALI environment"
    DALI supports S3 in several readers, but current `readers.webdataset` documentation does not explicitly guarantee cloud URLs. NexuML therefore keeps this boundary deliberately thin: direct tar URLs are the intended path, and a real target-environment integration test must confirm them before production use. If that combination needs a fallback, add the smallest bounded adapter demonstrated by that test rather than a second cache/capability framework.

Credentials/endpoints come from the normal AWS/provider environment and supported S3 options, not from persisted secrets in the scenario.

## Temporary KubeRay clusters

Cluster creation is infrastructure policy, not a second NexuML execution model. Use an infrastructure-owned KubeRay `RayJob`/cluster template to define images, CUDA, node selectors, queues, tolerations, service accounts, volumes, and autoscaling. Its entrypoint can remain the same `nexuml train ...` command.

## Troubleshooting

- **Ray import error** → install `nexuml[ray]` in the driver/worker environment.
- **Version mismatch** → align Ray/Python versions between the project and cluster.
- **Ray Train post-train/evaluation rejection** → use local execution or Ray Jobs; distributed global finalization is unsupported.
- **S3 failure** → verify credentials, endpoint/region settings, and worker access to the S3 service.
- **DALI failure** → install a compatible DALI build separately and verify `python -c "import nvidia.dali"` in the worker environment.
