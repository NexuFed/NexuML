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

## KubeRay: `ray-job`

NexuML's `RayJobExecution` means the **KubeRay RayJob resource**, not Ray Jobs API.
Its infrastructure-owned template chooses a new job-scoped or an existing Ray cluster;
see [native jobs](jobs.md). Ray Train remains `RayClusterExecution` (`ray-cluster`).

<a id="ray-jobs-ray_job"></a>

## Ray Jobs SDK: `ray-jobs-sdk`

Select `RayJobsSDKExecution` to submit the **whole normal CLI process** remotely
through Ray's `JobSubmissionClient`. Unlike `RayClusterExecution`, this is an
asynchronous submission backend, not a Ray Train worker group. It supports the
single-process RL/Isaac lifecycle without enabling distributed RL. The registered
routes are implemented; remote lifecycle qualification remains pending. These
recipes alone are not runtime evidence.

### Python defaults: no profile file required

```python
from nexuml.execution.definitions import RayJobsSDKExecution

spec.execution = RayJobsSDKExecution()
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

The project's lockfile owns dependencies and Python compatibility. Select extras
through `setup_command`; the optional profile below supplies this checkout's values.

For an already-prepared environment use
`RayJobsSDKExecution(setup_command=None)`; the implicit interpreter is `python`
and no uv environment settings are added. An explicit `py_executable` may select
another prepared interpreter. For CPU jobs set
`entrypoint_num_gpus=0` and choose CPU training settings in the scenario.
Explicit interpreter and environment values take precedence over generated
defaults. If you override the venv path, keep setup and Python paths aligned.
`LocalExecution` (`local`) remains the default for scenarios that do not select a backend.

### Reuse a project execution profile

For repeated CLI submissions, the optional `configs/execution/ray.yaml` adds
this checkout's Python 3.12, `library`/`ray` extras and IKA `pool_gpu_workers` tag.
It does not duplicate the class's cache, venv or GPU defaults. From the repo root:

```bash
export RAY_API_SERVER_ADDRESS=https://ray.example.org
nexuml train --scenario-file my_scenario.py \
  --execution-profile configs/execution/ray.yaml
```

Profiles use the same registered representation as persisted execution:

```yaml
type: ray-jobs-sdk
version: "1"
params:
  setup_command: [uv, sync, --locked, --python, "3.12", --extra, library, --extra, ray]
  entrypoint_resources: {pool_gpu_workers: 1}
```

The old `kind` schema is rejected.
Only use trusted files: their setup commands execute remotely. YAML does not
expand shell/environment variables. Keep secrets out of uploaded files.

Precedence is **scenario → profile replaces execution → `-O` overrides**;
other scenario settings are unchanged. Copy the profile for different dependency
extras or pools, or override fields per run. For example:

- `-O 'execution.entrypoint_resources={"pool_gpu_workers":1}'` sets the pool reservation.
- `-O execution.verify=/path/to/ca-bundle.pem` selects a private CA.
- `-O 'execution.setup_command=["uv","sync","--locked","--extra","isaac","--extra","ray"]'`
  selects Isaac dependencies. Extras are project-owned, not inferred from components.

Field overrides address Python model attributes (`execution.entrypoint_resources`),
not `execution.params`. Replacing the whole execution with `-O` requires a registered
dictionary, e.g. `-O 'execution={"type":"ray-jobs-sdk","version":"1","params":{}}'`.

For CPU use, clear both the GPU reservation and the profile's pool tag:
`-O execution.entrypoint_num_gpus=0 -O 'execution.entrypoint_resources={}'`.
Also select CPU training in the scenario; reservations do not rewrite its devices.

Python callers can assign `load_ray_job_profile(Path("configs/execution/ray.yaml"))`
from `nexuml.execution` to `spec.execution` before calling `run(spec)`.
Profiles are neither selected automatically nor reloaded remotely.

### Submission and paths

For a scenario that already sets its execution, use the normal command:

```bash
nexuml train --scenario-file scenario_job.py
```

Shared `nexuml.execution.run(spec)` dispatch and the public default-training
convenience `submit_ray_job(spec)` return a **submission ID**,
not a training result. Neither starts a local learner/simulator, waits, tails logs
or retries. Use Ray's native status/log/stop tools afterwards.

The backend sends the resolved scenario after overrides, resets execution to
`LocalExecution`, and invokes the normal remote CLI. It does not rerun the
factory, compile locally, write local staging files or recursively submit.
`--artifact-dir` snapshots remain local and include the submission ID.

Datasets, importable components, trainer checkpoints and selective
`scenario.checkpoint.source` must be worker-accessible; they are not implicitly
transported. Set unique
absolute `exports[].output` paths on durable storage. `NEXUML_LOGS_ROOT` redirects
relative logs, **not policy exports**. Nothing infers persistent output storage.

Public lifecycle routes (**remote qualification pending**) use
trusted submitter configs selecting `ray-jobs-sdk` and worker-accessible paths:

```bash
nexuml train --config CASE_JOB.yaml
nexuml train --config CASE_JOB.yaml --trainer-checkpoint /mnt/data/run/model.ckpt
nexuml export --config CASE_JOB.yaml --checkpoint /mnt/data/run/model.ckpt --output /mnt/data/run/policy
nexuml evaluate /mnt/data/run/policy --config EVALUATION_JOB.yaml --device cuda:0
```

Remote resume needs an explicit scenario/config; checkpoint-only local resume is
unchanged. Evaluation `--config` supplies only allowed evaluation/data/logging/environment
settings, never replacement packaged weights. Dependent native stages run sequentially
in fresh processes; submission alone does not establish completion or qualification.

Remote continuation checks the requested pipeline, learner, task, policy and
evaluation semantics against the checkpoint on the worker before native allocation.
Changed semantics fail rather than silently replacing the saved run. Saved scenario
metadata remains authoritative, including its output paths: choose isolated durable
paths before the original training run. A resume does not implicitly relocate them.

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

These restrictions apply to Ray Train (`ray-cluster`), not whole-process SDK Jobs (`ray-jobs-sdk`).
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
