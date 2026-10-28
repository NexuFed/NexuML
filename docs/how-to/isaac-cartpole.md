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

Submit the normal CLI through [Ray Jobs SDK](training-backends/ray.md#ray-jobs-sdk-ray-jobs-sdk),
not distributed RL. The backend installs locked dependencies into a private
worker-local venv, then runs training; no custom image or submitter-venv change
is needed. The [optional IKA profile](training-backends/ray.md#reuse-a-project-execution-profile)
selects `gpu-workers`. Override its extras for Isaac and set durable per-run paths.
This is the migration target, not evidence that the migrated code or remote runtime
is qualified. Planned remote resume/export/evaluate routes are listed in the
[SDK lifecycle recipe](training-backends/ray.md#submission-and-paths); implementation
and qualification remain pending.

From the repository root, with the local Ray CLI and `nexuml[ray]` installed:

```bash
export RAY_API_SERVER_ADDRESS=https://ray.ika.rub.de
export RAY_CA_BUNDLE=/workspaces/ika-k8s-cluster/platform/apps/cert-manager/rub-certificate/acme/cacerts.pem
RUN_ID="cartpole-$(python -c 'from uuid import uuid4; print(uuid4().hex)')"
RUN_ROOT="/mnt/data/nexuml-runs/$RUN_ID"

nexuml train --scenario-file scripts/isaaclab/cartpole.py \
  --execution-profile configs/execution/ray.yaml \
  -O 'execution.setup_command=["uv","sync","--locked","--python","3.12","--extra","isaac","--extra","ray"]' \
  -O "execution.verify=$RAY_CA_BUNDLE" \
  -O "execution.runtime_env.env_vars.NEXUML_LOGS_ROOT=$RUN_ROOT" \
  -O "exports=[{\"output\":\"$RUN_ROOT/policy\"}]" \
  -O interaction.environment.viewer.port=8081
```

This returns a **submission ID**, not training success, without waiting or tailing
logs. It resets the remote scenario to local execution and does not need an outer
`ray job submit`. For repeated Isaac runs, copy the profile and add the Isaac extra
once instead of repeating the setup override. You can also configure
`RayJobsSDKExecution` from `nexuml.execution.definitions` directly in a scenario;
YAML is not required.

Adapt the Jobs address, local CA path, pool and mounted storage for another
cluster. Workers need uv, Git, compatible drivers/system libraries and physical
free disk; this lock selects Python 3.12 and Ray 2.57.0. Pool selection does not
prove GPU readiness. The accepted campaign uses only `pool_gpu_workers`, one whole
GPU per job and at most four in-flight jobs (including pending/setup), further
limited by qualified capacity. No DGX, separate CPU pool or local fallback is used.
The one-GPU reservation is not a CPU/RAM hard limit. Jobs uses HTTPS port 443 here,
not Ray Client port 10001; keep TLS verification enabled. `RAY_JOBS_ADDRESS` is
unnecessary. Source uploads must fit this ingress's `64m` body limit.
Monitor actual physical free disk on the worker throughout setup/native execution;
stop owned work at the **4 GiB** floor without deleting another run's data.

For a four-batch startup/training/export smoke rather than the full learning
budget, append `-O training.total_frames=32768` and
`-O training.algorithm.update_epochs=1` to the `nexuml train` command above.
This is not a convergence or held-out-quality test. A short smoke may finish
before you connect a browser.

The [class defaults](training-backends/ray.md#python-defaults-no-profile-file-required)
keep the unpacked cache and venv local, avoiding JuiceFS metadata stalls. Only
logs/exports belong on mounted storage. Setup followed by direct Python execution
also avoids the `uv run` SIGPIPE handoff; `run.sh` stays installer-free.

Cold setup downloads multi-GiB wheels. The default 600-second
[uv HTTP timeout](https://docs.astral.sh/uv/reference/environment/#uv_http_timeout)
limits stalled reads, not total transfer time or Ray supervisor startup. At
6 MB/s the two large extension-cache wheels alone take about 20 minutes, before
other dependencies. Budget overall setup using worker throughput, not the
submitter's network speed; quiet logs are not proof of a hang.

!!! warning "Keep both logs and exports durable"
    The command explicitly sets `NEXUML_LOGS_ROOT` for relative logs and a unique
    absolute `exports[].output` for the policy. The logs-root setting alone does
    not redirect exports. Driver output stays in Ray Jobs logs; retain it with the
    log snapshots below before cluster cleanup.

### Status, logs and progress

Save the printed submission ID; inspect that ID instead of submitting again:

```bash
export JOB_ID=raysubmit_REPLACE_WITH_YOUR_SUBMISSION_ID
ray job status --verify "$RAY_CA_BUNDLE" "$JOB_ID"
ray job logs --verify "$RAY_CA_BUNDLE" "$JOB_ID"
ray job logs --verify "$RAY_CA_BUNDLE" "$JOB_ID" > "$JOB_ID.log"
```

These are snapshots; omit `--follow`. The last command saves driver output on the
submitter. `PENDING` can mean worker provisioning,
resource waiting or runtime-environment setup, so an empty log is not a failure.
`RUNNING` also includes uv installation and native initialization, not just
optimizer updates. Use Ray's final status and actual training metrics to assess
the job.

`WebSocket connection closed unexpectedly with close code 1006` after submission
does not by itself stop the job. The IKA ingress's 60-second read/send timeouts can
disconnect an idle log stream while provisioning or installing dependencies.
The snapshots above avoid that stream. Persistent streaming requires the
infrastructure owner to adjust ingress read/send timeouts through GitOps;
submission options do not change them or report eventual training success.

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

For a completed RL result with a training viewer configured, CLI completion now
labels the **configured training viewer** as belonging to a finished run. This is
not proof that a viewer actually opened (an exhausted resume can allocate none),
and it does not keep physics running or start replay automatically. The exact
produced-policy evaluation command remains separate; watching requires the
explicit replay viewer opt-in below and an approved private worker route.

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

## Frozen remote learning qualification

The fixed acceptance case uses the existing **2,097,152-frame / 512-environment /
8,192-frame-batch / seed-0** recipe unchanged. Use a trusted registered SDK worker
profile with locked Python 3.12 all-workspace/all-extras setup, one GPU and
`pool_gpu_workers`, plus a new evidence directory:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/physical_ai_acceptance.py \
  --case learning --profile QUALIFIED_PROFILE.yaml --ca "$RAY_CA_BUNDLE" \
  --output "$EVIDENCE_ROOT" --wall-seconds 3600 --queue-seconds 900 \
  --concurrency 1 --submit
```

This case has no shortened-budget option. Before training, it freezes checkpoint
selection at 8,192, 1,048,576 and 2,097,152 physical frames. After native training
exits, it copies the actual saved checkpoints, exports them and evaluates each
plus the matched untrained policy in fresh sequential processes: 32 reset seeds
10000–10031, deterministic actions, native five-second/300-step horizon and no
evaluator step cap. The retained protocol, configuration, checkpoint/policy hashes,
commands and logs identify the run; check terminal status and retrieval hashes.

The frozen final thresholds are mean return ≥4.0, mean length ≥285, at most one
native early termination and improvement over the matched untrained return.
Check native episode endings rather than treating evaluator caps as full-horizon
success. This is a single-training-seed **scalar learning protocol**, not multi-seed
robustness, camera learning, RGB recording, browser playback or live-view qualification. This Newton
case does not enable the adapter's unsupported Newton `rgb_array` replay or switch
to PhysX to obtain pictures. Keep run-specific receipts, measurements and artifacts
in the campaign's private evidence store, outside source commits.
