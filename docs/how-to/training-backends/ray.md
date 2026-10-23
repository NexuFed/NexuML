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

The backend substitutes one fresh `{run_id}` in `py_executable` and environment
variable values for every submission, even when reusing the same spec. It does
not mutate the caller's spec. Source uploads still default to `working_dir="."`.
No cluster address, private CA or pool tag is hardcoded. Configure those normally;
`address=None` uses Ray's native environment-based address selection.

The project's lockfile owns dependencies and Python compatibility. To enable
extras, replace `setup_command`; for this checkout's ordinary library scenarios:

```python
spec.execution = RayJobExecutionSpec(
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

Keep remote setup and placement in one YAML file instead of repeating them in
every scenario. From this checkout's root:

```bash
export RAY_API_SERVER_ADDRESS=https://ray.example.org
nexuml train --scenario-file my_scenario.py \
  --execution-profile configs/execution/ray.yaml
```

The optional `configs/execution/ray.yaml` only adds project-specific Python 3.12,
`library`/`ray` extras and the IKA `pool_gpu_workers` tag to the class defaults.
Change that tag for another cluster. Cache, venv and GPU defaults live in Python,
not in the YAML file.
The worker still needs uv, Git, compatible system libraries/drivers and disk;
Python/Ray must match the cluster. No worker image is built or changed.

The file is a plain `RayJobExecutionSpec` YAML mapping with `kind: ray_job`.
`{run_id}` in the Python command and environment values is resolved at submission,
keeping the venv path and Python command aligned without sharing mutable
environments. Cache is shared only within a worker. Relative
paths, including `working_dir`, are relative to the **submission directory**,
not the YAML file's directory. Profile setup commands execute remotely; use only
trusted profiles. The loader does not expand shell variables or environment
variables in YAML. Keep secrets out of profiles and uploaded project files.

Precedence is **scenario → profile replaces execution → CLI `-O` overrides**.
Other scenario settings are unchanged. For example, select another pool, a private
CA and durable logs without duplicating the setup configuration:

```bash
nexuml train --scenario-file my_scenario.py \
  --execution-profile configs/execution/ray.yaml \
  -O 'execution.entrypoint_resources={"pool_dgx_workers":1}' \
  -O execution.verify=/path/to/ca-bundle.pem \
  -O execution.runtime_env.env_vars.NEXUML_LOGS_ROOT=/mnt/data/my-unique-run
```

For a CPU job, override both `execution.entrypoint_num_gpus=0` and
`execution.entrypoint_resources={}`. Resource selection does not rewrite the
scenario's training accelerator or device settings.

Copy the profile once per dependency profile and edit its `setup_command` extras
(for example, add `--extra isaac`, `--extra mujoco` or `--extra tracking`). The
framework cannot infer arbitrary component dependencies. For a one-off run,
`-O 'execution.setup_command=["uv","sync","--locked","--python","3.12","--extra","isaac","--extra","ray"]'`
replaces that argv list. Do not change the venv environment path without also
updating `py_executable`.

Python callers can set
`spec.execution = nexuml.execution.load_ray_job_profile(Path("configs/execution/ray.yaml"))`
before `submit_ray_job(spec)`. The profile
is not selected automatically and is not reloaded by the remote CLI, preventing
recursive submission. A profile is optional: selecting `RayJobExecutionSpec`
directly already enables the setup/cache/GPU defaults above.

Datasets/checkpoints still need worker-accessible paths. Set absolute durable
`exports[].output` paths explicitly: `NEXUML_LOGS_ROOT` redirects logs, **not policy
exports**. Unique venvs live until worker cleanup and consume disk; the profile
does not enforce a wall-time/free-disk guard or guarantee available GPU capacity.
Submission returns a job ID, not training success. This profile mechanism is
tested without launching a cluster job; it does not qualify arbitrary workloads.

### Configure execution directly

Attach this execution spec to the scenario returned by your trusted scenario file:

```python
from nexuml.core.types import RayJobExecutionSpec

spec.execution = RayJobExecutionSpec(
    address="https://ray.example.org",
    verify="/path/to/ca-bundle.pem",
    setup_command=["uv", "sync", "--locked"],
    runtime_env={"working_dir": "."},
)
```

Choose setup flags/extras in the consuming project; for example, this checkout's
Isaac profile adds `--python 3.12 --extra isaac --extra ray` to `setup_command`.
uv is the default installer, but a custom argv command remains supported. Python
versions and dependency extras are project-owned, not inferred from components.

Then use the normal CLI from the project root:

```bash
nexuml train --scenario-file scenario_job.py
```

The command returns a **submission ID**, not a training result. It does not compile
or start a learner/simulator locally, wait for training, tail logs or retry a
failed submission. Python callers can use `nexuml.execution.submit_ray_job(spec)`
for the same submission ID. Use Ray's native status/log/stop APIs afterwards.
`nexuml backend list training` includes `ray_job`.

The backend serializes the fully resolved scenario after CLI overrides, resets
its remote execution to `LocalExecutionSpec`, and invokes the normal CLI with a
temporary remote YAML config. It does not rerun the scenario factory, create
staging files in the local project, or recursively submit another job. Existing
training/evaluation/export semantics stay with the remote local CLI. Optional
`--artifact-dir` snapshots remain local and record the submission ID. The
`--trainer-checkpoint` flag is rejected rather than assuming a submitter-local
checkpoint exists remotely; selective `scenario.checkpoint.source` still needs
to be accessible to the remote process.

### Remote dependency setup

`setup_command` is an optional, nonempty **argv list**. It runs once in the remote
job's working directory, with its `runtime_env` environment, before Python starts.
Arguments are shell-quoted, not interpreted as shell operators. If setup exits
nonzero, Python does not start and Ray records the failure without a NexuML retry.
After successful setup, the waiting shell `exec`s `py_executable`; the installer
is no longer Python's parent. Output remains in Ray's normal job logs.

By default, setup installs the locked project into the private local venv above.
Use `setup_command=None` to preserve prepared environments; the implicit Python
then stays `python`. A custom setup command must create the selected environment,
or specify its own matching `py_executable`. The backend does not rewrite an
arbitrary `uv run` command.
Use a project-specific environment, not an installer that changes the base worker
environment shared by other Ray processes. Concurrent jobs with different
dependency profiles must use distinct venv paths and matching `py_executable`
values; only the package download/build cache should be shared.

For large uv profiles on stock KubeRay workers, keep the cache and private venv
on the worker-local filesystem. For example, add these project-owned values to
`runtime_env["env_vars"]`:

```python
{"UV_CACHE_DIR": "/tmp/ray/uv-cache", "UV_LINK_MODE": "hardlink"}
```

This reuses cached packages across jobs on the same worker without sharing their
mutable environments. Cache and venv must be on the same filesystem for hardlinks;
worker cleanup removes that cache, so other workers download independently.
An unpacked cache on shared FUSE storage can instead stall on per-file metadata
and copying, even when downloads are already cached. Use mounted persistent
storage for results; explicit cache paths are never relocated by the backend.

Separating locked `uv sync` from direct Python execution avoids the cold
`uv run` SIGPIPE handoff reported in
[uv #21935](https://github.com/astral-sh/uv/issues/21935), without changing signal
handling or requiring a custom image. The same setup mechanism works for other
project-owned installation commands and for non-Isaac jobs. It requires a POSIX
worker shell, like the existing Ray Jobs entrypoint quoting.

`runtime_env` retains working-directory packaging, native exclusions and explicit
environment values; missing uv settings are filled when setup is enabled and
`{run_id}` is resolved in environment values before submission. Importable
components/factories, their dependencies, data
and checkpoint paths must be available remotely. Relative paths retain their
meaning under Ray's remote working directory; no persistent output storage is
inferred.

`address=None` uses Ray's native environment-based endpoint selection. Even an
explicit address is overridden by `RAY_API_SERVER_ADDRESS`/`RAY_ADDRESS` according
to the SDK's rules; do not leave an ingress setting active when switching to a
forwarded localhost endpoint. TLS verification defaults to `True`; supply the CA
bundle for a private HTTPS endpoint. Connection settings and runtime-env values
are not embedded in the remote scenario, but setup/Python commands and the
scenario itself are visible in the job entrypoint: do not put secrets in those
commands or model/data configuration.

`entrypoint_num_cpus`, `entrypoint_num_gpus`, `entrypoint_memory` and
`entrypoint_resources` describe the **entrypoint's logical resource reservations**,
not CPU/RAM utilization limits or resources for a separate Ray Train worker group.
The GPU count defaults to one; the pool tag remains an explicit cluster-specific
choice. A custom image is not
required; uv still cannot supply missing host drivers or system libraries.

### Native CLI / SDK alternative

Ray's own CLI and Python SDK remain supported alternatives to this backend:

```bash
ray job submit --working-dir . -- bash -c \
  'uv sync --locked && exec .venv/bin/python -m nexuml.cli.main train -c configs/my-scenario.yaml'
```

Use Ray's own status/log/stop tooling for the job lifecycle.

For a complete single-GPU robotics example with locked uv dependencies, TLS,
worker-pool selection, detached submission and private viewer forwarding, see
[Isaac Cartpole via Ray Jobs](../isaac-cartpole.md#run-remotely-with-ray-jobs).
That launches one remote local-execution CLI job, not distributed RL.

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

!!! warning "Verify direct S3 tar support in the target DALI environment"
    DALI supports S3 in several readers, but current `readers.webdataset` documentation does not explicitly guarantee cloud URLs. NexuML therefore keeps this boundary deliberately thin: direct tar URLs are the intended path, and a real target-environment integration test must confirm them before production use. If that combination needs a fallback, add the smallest bounded adapter demonstrated by that test rather than a second cache/capability framework.

Credentials/endpoints come from the normal AWS/provider environment and supported S3 options, not from persisted secrets in the scenario.

## Temporary KubeRay clusters

Cluster creation is infrastructure policy, not a second NexuML execution model. Use an infrastructure-owned KubeRay `RayJob`/cluster template to define images, CUDA, node selectors, queues, tolerations, service accounts, volumes, and autoscaling. Its entrypoint can remain the same `nexuml train ...` command.

## Troubleshooting

- **Ray import error** → install `nexuml[ray]` in the driver/worker environment.
- **Version mismatch** → align Ray/Python versions between the project and cluster.
- **Post-train/evaluation rejection** → current distributed global-finalization semantics are intentionally unsupported; run that scenario locally.
- **S3 failure** → verify credentials, endpoint/region settings, and worker access to the S3 service.
- **DALI failure** → install a compatible DALI build separately and verify `python -c "import nvidia.dali"` in the worker environment.
