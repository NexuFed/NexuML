# Design

## Context

See `proposal.md` for motivation and `specs/` for acceptance behavior. The user's central constraint is that backends evolve with the selected NexuML Python installation, not with FastAPI or Studio.

Relevant current seams:

| Existing code | Present behavior / reuse |
|---|---|
| `core/components.py`, `core/registry.py`, `core/discovery.py` | Typed definitions, stable identity/schema lookup, resilient installed-library and local-root scanning |
| `core/serialization.py`, `core/config.py` | Generic registered `type`/`version`/`params` lowering and restoration |
| `core/types.py` | Closed `LocalExecutionSpec | RayExecutionSpec` union |
| `core/backends.py` | Shared CLI/API descriptions, not execution or capacity discovery |
| `execution/ray.py` | Existing Ray Train allocation, canonical session reuse and distributed-semantic guards |
| `cli/main.py`, `api/worker.py` | Separate Local/Ray branches; API catalog only checks whether Ray is installed |
| `api/operations.py` | One active invocation, frozen YAML, sequenced disk observation, local process ownership; remote capabilities inferred from `execution.kind` |
| `studio/components/studio.tsx` | Existing Base UI Run dialog and schema-backed execution settings |

The in-flight Studio/Ray changes are prerequisites, not artifacts to rewrite. This follow-up replaces their closed dispatch selection and extends native remote observation; it preserves one ML lifecycle, infrastructure-owned templates and honest distributed restrictions. The API remains optional and loopback-authenticated.

## Goals / Non-Goals

**Goals:**

- Extend the existing typed discovery/serialization model once, so backend additions work in Python and CLI before API/UI integration.
- Deliver useful target/capacity inspection and real native submission for all five initial modes without owning a scheduler.
- Keep platform policy in templates, placement in execution settings, training semantics in `training`, and current capacity out of saved configuration.
- Reuse the existing dialog, form controls and observation records; keep partial/unknown information usable.

**Non-Goals:**

- No separate execution plugin registry, plugin RPC service, provider-neutral training result hierarchy or long-running resource inventory service.
- No network scanning, automatic port forwarding, provisioning, deployment-policy editor, environment installation, whole-directory packaging, queue manipulation or resource reservation.
- No legacy execution aliases, new database, monitoring dashboard or guaranteed workload scheduling/model-memory prediction.
- No generic Ray Jobs API backend or Kubeflow TrainJob backend in this change. RayJob is KubeRay; PyTorchJob is the specifically discovered Training Operator resource.

## Decisions

### D1. Reuse typed component discovery for execution

Add an execution definition role to `ComponentDefinition`, the existing decorator helpers and the component registry scan. Core built-ins are registered through a small explicit core definitions module; do not recursively import every NexuML execution/runtime module. Scan the existing `nexuml.libraries` packages and configured local roots for contributed definitions.

Built-in stable identities are `local`, `ray-cluster`, `ray-job`, `kubernetes-job` and `kubernetes-pytorchjob`, version `1`. Display names are metadata, not hardcoded Studio identifiers. Optional dependency imports occur in probes/run calls, not during definition/schema registration. Preserve existing registry conflict/error collection; failures must not silently overwrite another definition.

The definition owns typed settings and the narrow operations required by real consumers: dependency/feature description, target discovery, preflight, execution, and supported native status/log/cancel inspection. Advertise capabilities explicitly rather than infer them from backend names. Small shared discovery/reference records are data contracts, not a scheduler or replacement for native training results. Return the existing local/Ray native results; backend-owned observation summaries provide safe transport data where needed.

`core/backends.py` continues serving the general backend description command, but its execution entries come from the registered definitions. New Python catalog/discovery/preflight/dispatch functions live under `nexuml.execution`; they must not import `nexuml.api` or FastAPI. Keep package exports lazy so listing Local works without Ray/Kubernetes.

**Alternative rejected:** an API-owned provider mapping or a new parallel entry-point/registry system. Both duplicate installed discovery and leave Python/CLI behind.

### D2. One registered execution representation

Change `ScenarioSpec.execution` to the registered execution definition role. Reuse existing lowering/restoration and registry-backed per-definition schemas instead of growing a static union each time a backend is installed. Default remains the registered Local definition.

```yaml
execution:
  type: local
  version: "1"
  params: {}
```

Existing Ray placement settings become parameters of `ray-cluster`; worker counts, resources per worker, storage and runtime environment settings retain their current meaning. Strategy is never duplicated in execution. Job definitions add explicit context/namespace/template and configuration-handoff references, not Kubernetes topology models. Parameters must be JSON-safe and never contain credentials.

Remove Local/Ray type-name dispatch from CLI and API. Update repository fixtures, examples and affected checkpoint/config serialization tests to the new representation. Old `execution.kind` data fails with an actionable conversion diagnostic; no hidden coercion or second legacy parser. Historical raw configuration/log files remain readable as files, not overwritten or falsely validated as current configurations.

**Alternative rejected:** preserve the old union and add an extension union branch. It creates two persistence/schema paths and still requires special frontend selection handling.

### D3. Separate definition discovery from target probing

Catalog reads enumerate definitions, schemas and dependency support; they do not contact clusters. Target discovery is an explicit read-only call for the selected backend/scope, run in the selected Python runtime outside the API event loop.

- **Local:** inspect the current runtime's process/affinity/cgroup limits and supported visible accelerators. Use platform/stdlib probes first, existing optional system libraries where available, and explicit unknown diagnostics when a reliable platform measurement is unavailable. Never add an API dependency merely to obtain local inventory.
- **RayCluster:** obtain candidate addresses from the scenario, `RAY_ADDRESS`, an explicitly entered address and discoverable KubeRay resources in the selected context/namespace. Inspect known cluster endpoints using supported read-only Ray state/dashboard APIs. Do not call an auto-starting `ray.init()` to discover capacity; missing inspection endpoints remain a diagnosable limitation even when execution connectivity differs. Do not automatically start port forwarding.
- **Kubernetes modes:** list context identities from the normal runtime kubeconfig, then probe only the selected/current context and chosen namespace. Build a client for that context explicitly; never mutate kubeconfig's global current context. List KubeRay cluster resources and served job APIs where authorized. No all-cluster fan-out or network scan.

Use a lazy optional `kubernetes` dependency extra with a compatibility range verified during implementation; keep the existing Ray range unchanged. Catalog listing and Local execution must work without either extra or the base library. Normal credential chains stay in Python. Kubeconfig exec authentication helpers are part of the trusted selected-context probe, never ordinary catalog enumeration. Endpoint validation rejects credential-bearing URLs and restricts API probes to explicitly supplied/discovered targets.

Each discovery request has a finite overall budget, bounded per-call network timeout, pagination/result bounds and cancellation/cleanup. Return independent successes with timed-out/forbidden/unsupported diagnostics; mark truncated counts incomplete. Fresh queries reuse ordinary discovery and do not introduce a persistent registry/resource cache.

**Alternative rejected:** probe every kubeconfig context at startup or treat installed dependencies as operational availability. Both make routine Studio use slow and misleading.

### D4. Capacity is a sourced estimate, not a reservation

Use a small Python-owned snapshot contract shared by CLI/API: backend identity, safe target identity/scope, observation time, completeness, diagnostics, node/worker summaries and named resource quantities with units. Unknown is nullable/explicit, never zero. Avoid merging unlike GPU/MIG/shared resource names into a physical-device total.

| Source | Capacity interpretation |
|---|---|
| Local | Effective process-visible CPU/memory limits, device type/count, observable free memory; physical totals separately if available |
| Existing Ray cluster | Live Ray nodes and scheduling resources; available Ray resources separately from total resources and Kubernetes node inventory |
| Kubernetes jobs | Node allocatable quantities minus effective nonterminal pod requests, constrained by the selected template/request and remaining namespace quota |
| New RayJob cluster | Prospective Kubernetes fit plus requested head/worker/submitter footprint, not invented running Ray capacity |

Kubernetes accounting must use native quantity semantics and effective pod requests, including init/sidecar behavior, pod overhead and observable pod-level requests for the supported API version; summing ordinary containers alone is insufficient. Check known readiness/cordon, required selectors/affinity, taints/tolerations and per-role resource fit. Include head/master/submitter requests, not only training workers. Keep runtime utilization distinct from scheduling requests. GPU type depends on observable device metadata/labels; otherwise show the resource name/count with type unknown.

Report matching nodes and per-worker fit, not a fabricated exact maximum worker count from aggregate resources. Advanced scheduler plugins, gang scheduling, inter-pod constraints, volume feasibility, device allocation and autoscaler capacity may remain unknown; label that ceiling. No scheduler simulator, model-memory heuristic or `kubectl top` requirement. Add a `ponytail:` comment at the estimator describing the scheduler-feasibility ceiling.

Forbidden node/pod/metrics reads degrade capacity visibility, not submission authorization. Served CRDs show API support, not verified operator health. Schema/semantic errors and known quota/admission failures block launch; transient lack of immediately free capacity normally warns and allows native Pending/Queued behavior.

**Alternative rejected:** a single green Available flag or total-GPU counter. Neither represents per-node placement, quota or present access accurately.

### D5. Thin native implementations for five modes

| Mode | Execution boundary |
|---|---|
| Local | Existing direct `NexuSession.run()` path and local checkpoint/export behavior |
| RayCluster | Existing `run_ray` / Ray Train path against an explicitly resolved existing cluster |
| RayJob | Submit the selected KubeRay RayJob template; the template chooses `rayClusterSpec` or existing-cluster selection and cleanup policy |
| Kubernetes Job | Submit one training-pod Job template; Lightning still owns intra-pod device/strategy execution |
| Kubernetes PyTorchJob | Submit a served `kubeflow.org` PyTorchJob template with a verified native distributed launcher/environment |

Share one small Kubernetes access/submission/inspection helper across the three job modes, not three client implementations. The reviewed template owns images, mounts, service account, queues, tolerations, selectors and cleanup. Only allowed identity/config/entrypoint substitutions and explicit replica/resource bindings are patched. Identify worker group, replica role and training container by reviewed names; ambiguous multi-group/multi-container templates require explicit binding, never positional guesses. Resource controls derive from those bindings and the definition schema, not arbitrary JSON patch input from the browser.

Preflight uses native discovery/access checks and a server dry-run of the exact generated manifest when permitted, without side-effecting job creation. Dry-run success is not a controller-health or scheduling guarantee. Unsupported dry-run/visibility is reported distinctly; actual missing create permissions or known admission rejection blocks launch. Template changes between review and launch require renewed review.

A Kubernetes Job is one training workload, not independent parallel learners disguised as distributed training. Reject unsupported completions/parallelism/indexed topology. PyTorchJob rank/world-size/launcher handling must be verified against the target templates and supported Lightning environment; supported strategies/devices are reported only after that integration is tested. Reuse/extract Ray's existing distributed-semantic guard where it applies to other rank-sharded execution rather than duplicate or relax it.

**Alternative rejected:** synthesize universal Kubernetes/Ray topology from a form or implement another distributed loop. Templates/native launchers already own those concerns.

### D6. Make remote configuration handoff explicit

Preflight produces the reviewed immutable launch configuration plus the effective worker configuration/manifest. The outer selection identifies how to submit; the worker selection is derived explicitly as Local inside Kubernetes Job/PyTorchJob or existing-cluster Ray inside RayJob, avoiding recursive submission. This derivation changes placement only, preserves training/data/component settings and is inspectable before launch.

Use a configured shared filesystem mapping or explicit S3 config URI through the existing storage utilities to publish the small validated worker configuration under a unique invocation location. The template owns code/environment delivery and remote read credentials. Do not invent a source-code bundler, container builder, dependency installer or storage framework. Validate the handoff destination and template's reference mapping; remote mount/code availability not verifiable from the client stays an explicit assumption, not a successful readiness claim.

Persist the original launch YAML, effective worker YAML, template revision and safe backend reference in the existing invocation observation directory. Capture the reviewed template content privately before dispatch so the worker does not reread a changed path. Return allowlisted manifest summaries, not raw credential-bearing kubeconfig/client objects; templates must reference secrets rather than embed cluster credentials. New shared-config writes use unique locations and never overwrite an unrelated artifact. External checkpoints/output URIs stay references unless the existing authorized download boundary can serve an actual local file.

**Alternative rejected:** pass a laptop-only path to a remote pod or silently upload the entire working directory. Neither preserves the reviewed inputs or credential boundary.

### D7. Backend observation; existing API bookkeeping

Backend calls use optional caller-supplied observation hooks without importing API models. Local callbacks and Ray driver/final results remain as today. Native job adapters emit submitted/pending/running/terminal observations and available logs from their actual workload; scalar metrics are shown only if an existing configured source supplies them. Native results are not converted into a new ML result hierarchy.

Record safe remote references as soon as submission is acknowledged: backend identity/version, context, namespace, resource API/name/UID and Ray job identity when applicable. Submit each invocation under one unique name/correlation identity. If creation acknowledgement is lost, inspect that exact identity before retrying; never blindly create a second job. Inspect/status/cancel must refuse a same-named UID replacement.

Job `run` observes the native workload through completion while the API continues lightweight requests. A submitter process exit is not terminal training evidence. Bound log/event delivery using existing observation mechanisms; prefer status polling and finite log reads over adding a watch/retry framework. Supported cancellation calls native APIs for only the submitted resource; confirm termination before publishing Cancelled, and preserve template/shared-cluster ownership. RayCluster direct cancellation stays unsupported unless the existing provider can actually stop training.

Retain the existing single-active-operation guard. On API shutdown or loss of control, detach remote observation without claiming worker cancellation. After restart, recorded references can be explicitly inspected through the backend; do not promise automatic monitor restoration, resubmission or training recovery. Unverified work remains ownership-unknown. Preserve current browser reconnect/event replay behavior.

API routes expose catalog, selected-scope discovery and execution preflight from Python, with safe typed JSON inputs/results. Existing `/train` and operation routes consume frozen execution definitions and actual capabilities. No API-side worker/resource estimator or backend dispatch switch. Discovery is not subject to the resource-consuming operation slot, but remains bounded and subprocess-isolated.

The configuration/catalog/operation contract changes require incrementing the local interface major and updating its existing launcher/client handshake together. Reject old-client/new-runtime mismatches; do not add a compatibility proxy.

**Alternative rejected:** mark `kubectl apply` success as training success or stop only the local submitter. Both misrepresent the remote workload.

### D8. Extend Run review, not the product shell

Reuse the existing Base UI dialog, semantic tokens, controlled schema fields and query client. The catalog supplies labels, schemas, safe target options and minimal presentation hints for target/namespace/template/resource roles; no frontend switch on the five identities. Backends lacking detailed hints still receive usable schema-driven controls and diagnostics.

```text
Run on
  [Backend choices and unavailable reasons]
  Target / context      [selected destination]
  Namespace / template  [where applicable]

  Capacity                          Refresh
  Visible nodes / Ray workers · matching nodes
  Allocatable · estimated unallocated · quota
  Observed timestamp · partial/unknown warnings

  Request: backend-labeled workers/replicas,
           CPUs · GPUs/type · memory
  > Matching nodes / constraints
  > Expert: effective worker settings / manifest

  Source revision · output references
                    Cancel   Run on <selection>
```

Keep dialog edits in a staged copy. Cancel discards them. Change of backend/target/template/resource inputs immediately invalidates earlier preflight/eligibility; key queries by the full selection/revision and ignore out-of-order responses. Debounce ordinary form edits, do not debounce explicit Refresh. Show indeterminate discovery, source age, no-target/setup guidance and diagnostic-specific retries. Do not label a partial snapshot Ready.

On confirm, revalidate, bind template content/revision and freeze the staged configuration. Commit the accepted execution settings to the ordinary draft as one undoable edit without an automatic file save; active runs keep their frozen source. If draft/template changed, require renewed review. Prevent duplicate confirmation. Existing Training execution fields consume the same definition schema/settings, and results continue in Execution rather than a new dashboard.

Keep advanced infrastructure details collapsed; do not add a new Compute page or persistent endpoint-profile store. Scenario/manual Ray addresses and normal kubeconfig already provide target sources. Remembered selection is ordinary saved execution configuration, not hidden browser state.

Apply the verified UI/UX guidance: controlled inputs; an error summary that receives focus after failed submit and links to inline, `aria-describedby`-connected errors; readable status labels and restrained live announcements. Reuse the dialog's keyboard focus management/restoration, visible focus and reduced motion. Layout uses two columns only when space permits, otherwise one internally scrollable column with reachable actions. Retain `#101010`, Montserrat/Geist, existing radii and `#188FD5` actions with black text; no new palette/design system/dependency.

**Alternative rejected:** a provider wizard, separate cluster dashboard or YAML-only backend selection. The existing Run review and schema forms cover the requested interaction.

## Risks / Trade-offs

- **Breaking configuration/interface change** → explicit conversion examples, affected fixture updates, interface-major mismatch rejection, and untouched historical files. Do not hide it behind aliases.
- **Controller versions differ** → discover served APIs; verify supported client/API versions during implementation. PyTorchJob is not interchangeable with newer TrainJob.
- **Restricted RBAC or offline endpoints** → partial snapshots and separate create/read capabilities; unknown capacity does not equal an unusable target.
- **Stale/heterogeneous capacity and advanced scheduling** → per-node fit, request/quota accounting, resource units and honest unresolved constraints; native scheduler remains authoritative.
- **Remote configuration/code unavailable** → explicit handoff/template contract and declared unverifiable assumptions; no automatic provisioning workaround.
- **Ambiguous submission or resource-name reuse** → unique invocation identity, inspect-before-retry and UID-verified controls; report ownership-unknown if verification fails.
- **Template or probe secrets** → server-side credential chains, secret references, allowlisted summaries/errors, private frozen artifacts and existing file authorization.
- **Backend extension overgrowth** → existing registry/serializer, five real adapters, one small Kubernetes helper, finite observations and no generic platform framework.

## Migration Plan

1. Add the execution role/catalog and prove core-only discovery plus an installed-library test backend. Replace the union and update affected Local/Ray fixtures/examples without changing their ML behavior.
2. Introduce discovery/preflight and template adapters in Python; verify CLI parity and native integration before API/UI coupling.
3. Update API dispatch, safe job references/capabilities and the interface-major handshake; then connect the existing Run dialog and Execution view.
4. Document the new syntax, optional extras, known-target discovery, three minimal template/handoff examples and limitations. Run targeted Python/Studio tests and package checks; real cluster tests require explicit opt-in and exact cleanup ownership.
5. Rollback pairs the previous Python/Studio versions with retained pre-conversion configurations. New configurations are not automatically down-converted. Historical run records and artifacts are never rewritten or deleted.

## Reference Checks

- [Ray available resources](https://docs.ray.io/en/latest/ray-core/api/doc/ray.available_resources.html): current available resources differ from total resources and may become stale.
- [KubeRay RayJob](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html): job-scoped cluster or existing-cluster selection; cleanup remains template-owned.
- [Kubernetes resource management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/): scheduling requests/allocatable are not utilization.
- [PyTorchJob documentation](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/pytorch/): specifically Training Operator v1; newer Kubeflow Trainer uses a different resource. Support the requested served API, not its replacement by assumption.

Exact optional client ranges and launcher/environment compatibility are implementation verification gates, not claims that every cluster/version is supported.
