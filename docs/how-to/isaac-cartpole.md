# Train state-only Isaac Cartpole

Use the prepared robotics environment. This example needs one CUDA GPU and the
pinned Isaac stack; it does not install packages, start a public server or use a
different trainer. It learns continuous cart effort from four native state values,
not camera pixels or Gymnasium's categorical CartPole actions.

## Train and export

From the repository root:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nexuml train --scenario-file scripts/isaaclab/cartpole.py
```

The file only declares a `scenario()`. The CLI owns training, native lifetime,
logger cleanup and export. On completion it prints a unique complete policy path:
`rl-runs/cartpole-<id>/policy`. Keep that path for evaluation; reruns do not reuse it.

The private read-only spectator is at **http://127.0.0.1:8080**, with at most four
visible environments. An occupied port fails rather than replacing another server.
For a remote machine, use a private SSH tunnel/port forward, not a public bind.
The viewer closes when the job finishes. Its controls do not actuate the task.

This is the separate `isaac-cartpole-learning-ppo` reference:

- Native `Isaac-Cartpole`, **Newton mjwarp**, unchanged native reward/reset rules
  and five-second horizon (300 control steps at 60 Hz).
- 512 environments, 8,192 frames/batch: **16 steps per environment**, not 8,192
  sequential steps on one cart.
- 2,097,152 total frames, training seed 0; 32×32 actor and critic.
- PPO: five update epochs, four 2,048-sample minibatches, actor/critic learning
  rate 0.001, gamma 0.99, GAE lambda 0.95 and entropy coefficient 0.005.

These are scenario-local settings. The ordinary PPO defaults and existing
state/sensor smoke recipes are unchanged. The tiny four-step sensor example is
not a replacement for this learning reference.

## Run remotely with Ray Jobs

Submit the whole CLI process to an existing GPU worker. With the native Jobs
CLI below, leave the scenario's `execution.kind="local"`: simulation and learning
run in that remote process, not as distributed RL. A custom worker image is
optional. Run locked dependency setup first, then execute the worker venv's
Python directly. This does not upload or alter the submitter's local `.venv`.

Alternatively, set [RayJobExecutionSpec](training-backends/ray.md#ray-jobs-ray_job)
in your own scenario file and invoke `nexuml train` locally. That backend uses the
Jobs Python SDK, sends the resolved scenario and resets its remote execution to
local automatically; you do not need the outer `ray job submit` command.

To select the backend without editing `cartpole.py`, reuse the
[project execution profile](training-backends/ray.md#reuse-a-project-execution-profile)
with the Jobs address and CA exports defined below. Only the Isaac dependency
extras and per-run paths need overriding:

```bash
nexuml train --scenario-file scripts/isaaclab/cartpole.py \
  --execution-profile configs/execution/ray.yaml \
  -O 'execution.setup_command=["uv","sync","--locked","--python","3.12","--extra","isaac","--extra","ray"]' \
  -O "execution.verify=$RAY_CA_BUNDLE" \
  -O "execution.runtime_env.env_vars.NEXUML_LOGS_ROOT=$RUN_ROOT" \
  -O "exports=[{\"output\":\"$RUN_ROOT/policy\"}]" \
  -O interaction.environment.viewer.port=8081
```

This reads `RAY_API_SERVER_ADDRESS` through the SDK and returns a submission ID
without waiting or tailing logs. Set a unique `RUN_ROOT` as below before running
it. Running it submits a new training job. For repeated Isaac runs, copy the
profile and add the Isaac extra there once instead of repeating the setup override.

The following example uses the IKA cluster's Jobs ingress, CA bundle, worker-pool
tags and `/mnt/data` mount. Adapt those infrastructure-owned values for another
cluster. Workers need uv, Git, compatible GPU drivers/system libraries and enough
real free disk for the large Isaac installation. Keep Python/Ray compatible with
the cluster; this lockfile selects Python 3.12 and Ray 2.57.0.

From the repository root, with the local Ray CLI available:

```bash
export RAY_API_SERVER_ADDRESS=https://ray.ika.rub.de
export RAY_CA_BUNDLE=/workspaces/ika-k8s-cluster/platform/apps/cert-manager/rub-certificate/acme/cacerts.pem

ray job list --verify "$RAY_CA_BUNDLE"
RUN_ID="cartpole-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="/mnt/data/nexuml-runs/$RUN_ID"
ray job submit \
  --verify "$RAY_CA_BUNDLE" \
  --submission-id "$RUN_ID" \
  --working-dir=. \
  --entrypoint-num-gpus=1 \
  --entrypoint-resources='{"pool_gpu_workers":1}' \
  --runtime-env-json="{\"env_vars\":{\"UV_CACHE_DIR\":\"/tmp/ray/uv-cache\",\"UV_LINK_MODE\":\"hardlink\",\"UV_HTTP_TIMEOUT\":\"600\",\"UV_CONCURRENT_DOWNLOADS\":\"4\",\"NEXUML_LOGS_ROOT\":\"$RUN_ROOT\"}}" \
  --no-wait \
  -- bash -c '
set -euo pipefail
source_dir="$PWD"
mkdir -p "$NEXUML_LOGS_ROOT"
{
  cd "$NEXUML_LOGS_ROOT" &&
  UV_PROJECT_ENVIRONMENT="$source_dir/.venv" uv sync \
    --project "$source_dir" --locked --python 3.12 --extra isaac --extra ray &&
  exec "$source_dir/.venv/bin/python" -u -X faulthandler -m nexuml.cli.main train \
    --scenario-file "$source_dir/scripts/isaaclab/cartpole.py" \
    -O interaction.environment.viewer.port=8081
} 2>&1 | tee "$NEXUML_LOGS_ROOT/driver.log"
'
```

Both the Python backend's `setup_command` and the native CLI example deliberately
separate installation from execution. The stock
worker's uv 0.12.1 forwards SIGPIPE to its child; the upstream
[uv #21935](https://github.com/astral-sh/uv/issues/21935) reports a network-signal
race that can kill Python before its first output, returning **141** immediately
after installation. Executing Python only after `uv sync` exits avoids that
handoff without masking signals, changing the lockfile or rebuilding an image.
The existing `run.sh` remains installer-free for already-prepared environments;
no Isaac-specific Ray launcher is required.

For a four-batch startup/training/export smoke rather than the full learning
budget, append `-O training.total_frames=32768` and
`-O training.algorithm.update_epochs=1` to the remote `train` command above.
This is not a convergence or held-out-quality test. A short smoke may finish
before you connect a browser.

`RAY_API_SERVER_ADDRESS` is Ray's native Jobs setting. `RAY_JOBS_ADDRESS` is not
needed. Jobs uses HTTPS port 443 here, not the distinct Ray Client port 10001.
Keep TLS verification enabled with the actual CA bundle. Source upload must fit
the ingress body limit; this cluster uses `64m`.

The custom resource chooses `gpu-workers`; use `pool_dgx_workers` only with a
compatible, available GPU in that pool. A pool tag is placement, not CUDA/driver
compatibility. The job reserves one GPU, not a CPU/RAM hard limit. Keep the uv
cache and private venv on the worker's local `/tmp/ray` filesystem. Cached packages
can be hardlinked into separate environments without copying their large payloads;
the cache can be reused by jobs on that worker, but disappears on worker cleanup.
This trades cross-worker download reuse for predictable installation I/O.

Do not put this profile's unpacked uv cache on the shared `/mnt/data` JuiceFS
mount. Its many small-file metadata operations can stall installation even when
the packages are cached and free disk is ample. `UV_LINK_MODE=copy` only handles
cross-filesystem installation; it does not fix a slow shared cache. Putting a
hardlinked venv on that same shared mount still pays its metadata cost. Keep
mounted storage for durable logs and exports, not this installation cache.

Cold setup also downloads multi-GiB native wheels. The examples allow a 600-second
uv HTTP timeout and four concurrent downloads: the stock 30-second HTTP timeout
can abort the 5.1-GiB Isaac extension-cache wheel. These settings affect package
downloads, not Ray's supervisor startup timeout or GPU availability.

The [uv HTTP timeout](https://docs.astral.sh/uv/reference/environment/#uv_http_timeout)
limits stalled reads, not total download duration. At a worker transfer rate of
6 MB/s, the two large Isaac extension-cache wheels alone need about 20 minutes;
the remaining dependencies add to that. Size any overall setup deadline using
worker throughput, not the submitter's network speed. Quiet installer output is
not proof of a hang: inspect network/disk progress and the eventual exit status.

The native CLI command runs from the unique mounted `RUN_ROOT`: its
`driver.log`, TensorBoard logs and `rl-runs/cartpole-<id>/policy` survive worker
cleanup. `tee` mirrors installer/training output into Ray logs while retaining
the full file, and `pipefail` preserves installation/training failures. Redirecting
only to `driver.log` and running `tail` afterwards would hide live output.

!!! warning "SDK submission does not infer persistent outputs"
    The `RayJobExecutionSpec` example above still runs in Ray's temporary working
    directory. For durable SDK runs, set `NEXUML_LOGS_ROOT` in `env_vars` for
    relative logs and choose a unique absolute `ExportSpec.output` under mounted
    storage. The logs-root setting alone does not redirect policy exports.

### Status, logs and progress

`--no-wait` returns after submission rather than opening a WebSocket log stream.
Save the printed submission ID; for an already-submitted job, inspect that ID
instead of submitting again:

```bash
export JOB_ID=raysubmit_REPLACE_WITH_YOUR_SUBMISSION_ID
ray job status --verify "$RAY_CA_BUNDLE" "$JOB_ID"
ray job logs --verify "$RAY_CA_BUNDLE" "$JOB_ID"
```

These are snapshots; omit `--follow`. `PENDING` can mean worker provisioning,
resource waiting or runtime-environment setup, so an empty log is not a failure.
`RUNNING` also includes uv installation and native initialization, not just
optimizer updates. Use Ray's final status and actual training metrics to assess
the job.

`WebSocket connection closed unexpectedly with close code 1006` after submission
does not by itself stop the job. The IKA ingress's 60-second read/send timeouts can
disconnect an idle log stream while provisioning or installing dependencies.
The snapshots above avoid that stream. For live streaming, the infrastructure
owner can raise the Ray ingress's `nginx.ingress.kubernetes.io/proxy-read-timeout`
and `nginx.ingress.kubernetes.io/proxy-send-timeout`, for example to `"3600"`
seconds, through GitOps. `--no-wait` does not change those timeouts or report the
job's eventual success/failure.

The CLI enables Lightning progress, but Ray Jobs captures non-interactive output;
Rich generally does not redraw a live progress bar there. TensorBoard receives
`train/rl/frames`, `train/rl/reward` and optimizer metrics after rollout updates.
One batch here is 8,192 frames across 512 environments, not one simulator step.

### Private viewer access

The override selects **8081** because KubeRay uses **8080** for Ray metrics. An
occupied viewer port fails startup; it does not silently select another port or
fall back to headless training.

Find the worker hosting **your job**, not the head/dashboard service. List pool
pods and match the job's Ray driver node to its worker when several are present:

```bash
kubectl -n ray-cluster get pods -l ray.io/group=gpu-workers -o wide
export WORKER_POD=REPLACE_WITH_YOUR_JOB_WORKER_POD
kubectl -n ray-cluster port-forward "pod/$WORKER_POD" \
  8081:8081 --address=127.0.0.1
```

Open **http://127.0.0.1:8081** after native viewer startup. Keep the forward running;
the server remains loopback-only inside the worker. If kubectl runs on a remote
development host, privately forward that host's port to your browser machine too.
No public ingress or `0.0.0.0` bind is needed. The viewer closes when the job ends;
a healthy Ray worker alone does not mean a viewer or training loop is running.

## Evaluate the exported policy

In a fresh process, replace the placeholder with the path printed by training:

```bash
nexuml evaluate rl-runs/cartpole-<id>/policy --device cuda:0 \
  -O training.evaluation.episodes=32 \
  -O training.evaluation.seed=10000
```

This uses the same native task, complete exported policy and held-out reset seeds
10000–10031. Do **not** add `max_steps_per_episode`: let native termination or the
full five-second horizon end each episode. In-fit evaluation is disabled so it
does not compete with the training runtime.

Look at `eval/rl/return_mean`, `length_mean`, `terminated` and `truncated`.
The native reward is integrated over simulation time: near-ideal five-second
returns approach **5**, not 300. A zero-weight `success_rate` reward plot remains
zero and is **not** a learning score. Mean per-transition training reward is also
not the same as an episode return.

Training viewing does not automatically enable viewing during evaluation. To
watch a bounded replay explicitly:

```bash
nexuml evaluate rl-runs/cartpole-<id>/policy --device cuda:0 \
  --scenario-file scripts/isaaclab/cartpole.py \
  -O training.evaluation.episodes=128 -O training.evaluation.seed=20000 \
  -O 'interaction.evaluation_environment.viewer.purposes=["eval"]'
```

Simulation runs faster than real time; 128 episodes leave time to open the browser.
The server still closes at the end of this finite replay.

See [private Viser ownership](isaac-read-only-viser.md) and
[native configuration](isaac-native-configuration.md) for customization. Keep the
operator's monitored **4 GiB free-disk stop** during native jobs.

This reference must be assessed against an untrained policy on identical held-out
seeds. A single training seed does not establish robustness across seeds, backend
equivalence, or learning from mixed sensor inputs.
