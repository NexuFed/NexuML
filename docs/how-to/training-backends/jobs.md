# Native jobs and capacity review

Install `nexuml[kubernetes]` into the selected Python installation. This change tests
Kubernetes Python client **36.x**, KubeRay `ray.io/v1` RayJob, and Training Operator
`kubeflow.org/v1` PyTorchJob. A served API is not proof of a healthy controller.
Newer Kubeflow **TrainJob is not PyTorchJob** and is not substituted.

## Infrastructure-owned templates

Start from [Job](templates/job.yaml), [RayJob](templates/ray-job.yaml), or
[PyTorchJob](templates/pytorch-job.yaml). These are **reviewable examples**, not ready
deployments: replace images with your reviewed NexuML/library installation, provision
the referenced shared volume and service account, and configure remote data/output
access. Images, code, mounts, secrets, selectors, tolerations and cleanup stay yours.
NexuML never builds images, installs packages or bundles your directory for job handoff.
Use Secret references/normal credential chains, never literal credentials in templates
or execution settings. Do not apply these templates directly: NexuML binds the config
placeholder and unique invocation name only after review.

For an existing ordinary scenario YAML, replace its execution section:

```yaml
execution:
  type: kubernetes-job
  version: "1"
  params:
    context: explicitly-selected-context
    namespace: default
    template: docs/how-to/training-backends/templates/job.yaml
    handoff:
      destination: shared/configs
      worker_directory: /shared/configs
    worker_group: training
    container: training
    resources: {cpu: "2", memory: "2Gi"}
```

The client path `shared/configs` and the pod path `/shared/configs` must be the **same
shared storage**, with suitable permissions. Files are uniquely named and exclusively
created. S3 alternatively uses `handoff: {destination: "s3://your-bucket/configs"}`;
the client and remote images need `nexuml[s3]` and their own authorized credential chains.
S3 writes use conditional `If-None-Match: *`; stores must support this non-overwrite contract.
No credential, kubeconfig or certificate is uploaded.

`{{NEXUML_CONFIG}}` is permitted only in command/args/entrypoint fields. Kubernetes
Job/PyTorchJob workers receive an explicit **Local** execution configuration; RayJob
drivers receive **RayCluster** settings. Training/data/components remain unchanged.
RayJob's nested `ray` setting uses ordinary registered `type/version/params`; its default
worker `working_dir` is null because code must already be in the template-owned image.
For a RayJob targeting an existing cluster, replace `rayClusterSpec` with the operator's
`clusterSelector` and explicitly retain a cleanup policy that does not remove that
shared cluster. NexuML does not change cleanup or infer cluster ownership.

PyTorchJob requires one `Master`, optional `Worker` replicas, a named `pytorch`
container, equal explicit `--nproc-per-node` values, and the operator's
`WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT` node-launch arguments shown in the template.
`torchrun` supplies process ranks/world size to Lightning. Tested support is external
DDP (`training.strategy: auto` or `ddp`), not independently trained replicas. Stateful
evaluation and post-training fitting remain rejected. Device lists/custom strategies,
heterogeneous local process counts and standalone launchers are not supported here.
The CPU two-rank check does **not** certify a particular operator, GPU or multi-host setup.

## Discovery and preflight (no launch)

```sh
nexuml backend catalog
nexuml backend discover --config scenario.yaml --timeout 10
nexuml backend preflight --config scenario.yaml
```

Catalog reads list installed definitions, schemas and missing-dependency diagnostics;
they do not contact clusters or execute kubeconfig auth helpers. Kubernetes discovery
lists local context identities and probes only the selected/current context and namespace,
without changing global kubeconfig context. Namespace/KubeRay candidates are read where
authorized. Ray discovery reads only a configured HTTP(S) dashboard; it does not call
`ray.init`, scan networks, create clusters or start port forwarding. Ray node totals are
distinct from Kubernetes node inventory; immediate Ray availability is unknown when the
supported State API cannot expose it.
Ray's optional `target.context`/`target.namespace` adds a bounded selected-scope
KubeRay candidate read when the Kubernetes extra is installed. Those are cluster
identities, not assumed reachable endpoints or invented live Ray capacity; explicitly
configure a reachable execution/dashboard address. A failed candidate read preserves
independent Ray observations.

Snapshots include `source`, `observed_at`, scope and `complete`/diagnostics. `null` means
unknown, `0` an observed zero. Local reports process affinity/cgroup-v2 limits separately
from host totals; unobservable platform/device limits remain diagnosed. Kubernetes
reports allocatable, request-based estimated unallocated resources, namespace quota,
visible nodes and per-role matching nodes. Effective requests include init/sidecars,
pod overhead and observable pod-level requests. CPU is cores; memory is bytes;
extended GPU/MIG/shared resource names remain separate **resource units**, not a combined
physical-GPU count. Device model is shown only when observed.

Denied node/pod reads hide capacity, not create permission. Truncated enumeration is
partial; missing pod accounting makes unallocated capacity unknown. Matching nodes use
readiness, cordon, required selectors/affinity, taints and **per-node** request fit, not
aggregate worker-count arithmetic. Volumes, gang/inter-pod scheduling, controller health,
autoscaling and actual device allocation are unresolved. No snapshot reserves capacity.
Known quota violations/access/admission failures block preflight; unknown/free-capacity
shortfalls may still lead to native Pending. Server dry-run reviews the exact manifest
without persisting a workload and does not promise scheduling.

## Explicit operator smoke procedure

**Do not run the following without separate operator authorization.** Choose a disposable
test namespace/context and reviewed image/volume/service account. First run the three
read-only commands above and inspect the exact worker derivation/template revision.
Then explicitly launch only that selected test scenario:

```sh
nexuml train --config scenario.yaml
# Save the returned native_reference object to reference.json.
nexuml backend inspect reference.json
```

Record context, namespace, API/kind, generated name and **UID**, dry-run evidence,
actual native terminal result and available logs. Submission/driver exit is not success.
If acknowledgement is lost, inspect the exact generated identity; never blindly rerun.
For Job/PyTorchJob, Python `nexuml.execution.cancel(reference)` deletes only the
UID-verified workload with foreground propagation and waits for termination. RayJob
cancellation/log tails are not advertised in this slice; use authorized native operator
tooling and do not mistake submitter termination for training cancellation.

Clean up only the recorded UID-verified disposable workload and its invocation-specific
config/artifacts. Confirm names have not been reused before native cleanup; never delete
a shared Ray cluster, namespace, PVC or service account. This implementation's acceptance
tests use fake native APIs plus real local CPU training/torchrun; **no real cluster smoke
run is implied**. Remote mount/code/data access, operator versions, multi-host networking
and device behavior require this separately authorized environment check.
