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

Attach this execution spec to the scenario returned by your trusted scenario file:

```python
from nexuml.core.types import RayJobExecutionSpec

spec.execution = RayJobExecutionSpec(
    address="https://ray.example.org",
    verify="/path/to/ca-bundle.pem",
    setup_command=["uv", "sync", "--locked"],
    py_executable=".venv/bin/python",
    runtime_env={"working_dir": "."},
)
```

Choose setup flags/extras in the consuming project; for example, this checkout's
Isaac profile adds `--python 3.12 --extra isaac --extra ray` to `setup_command`.
The backend does not hardcode a package manager, Python version or optional extra.

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

By default, `setup_command=None` and `py_executable="python"` keep prepared
environments unchanged: no installation or interpreter selection is inferred.
When setup creates a venv, explicitly select its Python, as above. The backend
does not infer working-directory paths or rewrite an arbitrary `uv run` command.
Use a project-specific environment, not an installer that changes the base worker
environment shared by other Ray processes. Concurrent jobs with different
dependency profiles must use distinct venv paths and matching `py_executable`
values; only the package download/build cache should be shared.

Separating locked `uv sync` from direct Python execution avoids the cold
`uv run` SIGPIPE handoff reported in
[uv #21935](https://github.com/astral-sh/uv/issues/21935), without changing signal
handling or requiring a custom image. The same setup mechanism works for other
project-owned installation commands and for non-Isaac jobs. It requires a POSIX
worker shell, like the existing Ray Jobs entrypoint quoting.

`runtime_env` is passed unchanged to the SDK, including working-directory packaging
and native exclusions. Importable components/factories, their dependencies, data
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
The GPU and pool tag are explicit cluster-specific choices. A custom image is not
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
