# Tasks

## 1. Registered execution definitions and configuration

- [x] 1.1 Add the execution definition role, decorator and explicit lazy core registration to the existing component registry; verify core-only catalog loading, an installed-library test backend, local-root discovery, conflict diagnostics and absent Ray/Kubernetes/FastAPI imports with focused discovery tests.
- [x] 1.2 Replace the Local/Ray execution union with registered execution definitions and the existing lowering/restoration path; verify Local default, all built-in schema identities, contributed-backend round-trip, invalid parameter locations, unknown versions and explicit old-syntax rejection in configuration tests.
- [x] 1.3 Expose Python-owned backend descriptions, capability metadata, discovery/preflight/dispatch entrypoints and safe snapshot/native-reference records; verify a test backend is callable without API modules and that snapshots distinguish zero from unknown and never serialize credentials.
- [x] 1.4 Update affected repository execution fixtures/examples and document the new execution representation and registration seam; verify examples load/round-trip, existing unrelated component configuration tests pass and historical files are not rewritten.

## 2. Preserve Local and RayCluster execution

- [x] 2.1 Implement the Local definition through the existing session path and optional caller observation hooks; verify lifecycle order, Trainer checkpoint resume, configured local exports and CPU execution without cluster dependencies using the existing focused execution tests.
- [x] 2.2 Adapt the existing Ray execution to the registered RayCluster definition without replacing its training implementation; verify fixed/elastic placement, resource mappings, runtime settings and training strategy behavior against the existing Ray tests.
- [x] 2.3 Expose preflight restrictions before allocation and share the existing rank-sharded semantic guard where required; verify stateful evaluation/post-train fitting, unsupported strategies and resume/export restrictions fail before Ray connection or any job create call.
- [x] 2.4 Update Local/Ray Python usage and target-selection documentation; verify the documented configuration examples resolve through the shared dispatcher and retain the current optional Ray compatibility range.

## 3. Read-only target and resource discovery

- [x] 3.1 Implement scoped discovery budgets, source timestamps/completeness and partial diagnostics; verify catalog reads have no network/auth-helper side effects, timed-out probes cannot hang, result truncation is explicit and one failing probe does not erase other results.
- [x] 3.2 Implement effective local CPU/memory/accelerator probes with supported platform fallbacks; verify affinity/cgroup limits, unknown platform/device readings and visible GPU resource labels using bounded probe tests without training allocations.
- [x] 3.3 Implement read-only inspection of explicit/configured Ray endpoints and selected-scope KubeRay cluster candidates; verify no local Ray auto-start/port-forward occurs and live Ray node/resources remain distinct from Kubernetes inventory and prospective RayJob capacity.
- [x] 3.4 Add the lazy optional Kubernetes dependency and explicit-context client helper; verify served APIs, safe context/namespace identities, create-versus-read access, absent PyTorchJob versus TrainJob, missing dependency and unchanged global kubeconfig context with stubbed API tests.
- [x] 3.5 Implement Kubernetes quantity/effective-pod-request accounting and workload eligibility summaries; verify init/sidecar/overhead accounting, readiness/cordon, selectors/required affinity, taints, per-node fit, all job roles, quotas, heterogeneous/shared GPUs and forbidden node/pod reads with small fixture-based tests. Mark unresolved scheduler feasibility with the planned `ponytail:` ceiling comment.
- [x] 3.6 Document discovery scope, permissions, resource units, freshness and advisory/unknown capacity behavior; verify the documented examples match snapshot fixtures and no example equates CRD presence, utilization or aggregate resources with guaranteed scheduling.

## 4. Native job adapters and remote configuration handoff

- [x] 4.1 Implement authorized template loading, content revision capture, named resource bindings and exact manifest preflight; verify only allowed fields change, ambiguous bindings fail, unauthorized/symlink paths are rejected, credentials are not exposed and native dry-run never creates a resource.
- [x] 4.2 Implement the small shared-filesystem/S3 worker-config handoff using existing storage utilities and explicit outer-to-worker execution derivation; verify immutable reviewed configuration, unique non-overwriting destinations, no recursive job submission, missing setup errors and no directory bundling/package installation.
- [x] 4.3 Implement KubeRay RayJob submission against new-cluster and existing-cluster templates; verify one native create call, existing Ray worker dispatch, untouched infrastructure/cleanup fields, head/worker footprint and explicit unsupported API/access errors.
- [x] 4.4 Implement the single-training-pod Kubernetes Job adapter; verify canonical lifecycle entrypoint, exact resource binding and rejection of unsupported completions/parallelism/indexed-job topologies before submission.
- [x] 4.5 Implement PyTorchJob submission and the supported native Lightning launcher/environment integration; verify replica/rank/world-size and strategy/device mapping through focused launcher tests and reject unsupported distributed phases/topologies rather than treating replicas as independent learners.
- [x] 4.6 Implement native job status/log inspection, safe UID-bearing references and supported cancellation; verify submitted/pending/running/terminal transitions, submitter-exit independence, same-name UID replacement refusal, acknowledgement-loss inspect-before-retry and cancellation confirmation without deleting a shared cluster.
- [x] 4.7 Provide one minimal infrastructure-owned template/handoff example per job mode and explicitly opted-in native smoke procedures; verify templates against the supported served APIs and ensure smoke procedures submit only named test resources, record actual terminal results and clean up only their own resources. Real cluster execution requires explicit operator authorization.

## 5. Shared CLI and thin API integration

- [x] 5.1 Replace CLI Local/Ray branching with shared definition dispatch and expose catalog/discovery/preflight/native-reference inspection through the existing backend command area; verify a contributed backend and Local/Ray behave the same in Python and CLI without the API extra, and update CLI usage documentation in this task.
- [x] 5.2 Replace API-only backend availability/dispatch logic with selected-runtime calls to the same Python functions; verify backend catalog/schema parity, bounded selected-scope discovery, field-addressable preflight, runtime/environment isolation and responsive status reads during training using API tests.
- [x] 5.3 Adapt frozen operation records and observation to backend-reported capabilities, captured template/worker configuration and safe native references; verify one-active-operation behavior, real remote states/logs, supported cancel, browser replay, unknown ownership after shutdown/restart and no resubmission on reconnect.
- [x] 5.4 Extend API security checks to target/template/handoff inputs and allowlisted discovery/job diagnostics; verify origin/token/file protections, credential-bearing URL rejection, unauthorized probes and no credentials in serialized configs, logs, responses or events.
- [x] 5.5 Increment the interface major and update the existing launcher/client handshake and API documentation together; verify old/new mismatch rejection, matching version attachment and production launcher behavior without adding compatibility aliases.

## 6. Data-driven Run modal and execution view

- [x] 6.1 Extend existing catalog/client/schema types and execution field rendering to registered definitions and runtime-supplied presentation hints; verify a sixth test backend renders/selects its schema with no provider-specific frontend branch and unavailable choices retain diagnostics.
- [x] 6.2 Extend the existing Run dialog with staged backend/target/context/namespace/template/resource selection and sourced capacity summaries; verify cancellation preserves the prior draft, settings share the ordinary execution configuration and known/zero/unknown/partial/stale states render accurately in model/component tests.
- [x] 6.3 Bind discovery/preflight to the full staged selection/revision and freeze reviewed launch inputs; verify out-of-order responses cannot override the selected target, request edits invalidate eligibility, draft/template changes require renewed review, double confirmation submits once and accepted execution edits form one undoable transaction.
- [x] 6.4 Update the existing Execution view for native references, job pending states and backend-supported controls/artifacts; verify no fake telemetry, unsupported Stop/resume remain explained and external artifact references are not treated as authorized local downloads.
- [x] 6.5 Apply existing corporate styling and accessible dialog/error behavior; verify keyboard selection, focusable error links/inline errors, focus restoration, responsive scrolling/reachable actions, status text/announcements and reduced motion in focused browser tests. Document the Run workflow and capacity limitations alongside the Studio usage guide.

## 7. Cross-interface acceptance and stop condition

- [x] 7.1 Run the targeted discovery/configuration/execution/CLI/API Python suites plus relevant existing regressions; verify all five definitions and an installed-library backend pass the same acceptance contracts, Local/Ray semantics remain unchanged and no test launches an unapproved cluster job.
- [x] 7.2 Run Studio typecheck, lint, focused unit/browser tests and production build/package smoke checks; verify a real CPU Local launch/reconnect and stubbed remote create/pending/terminal/cancel behavior through the actual Run modal, including narrow-screen and keyboard checks.
- [x] 7.3 Validate the OpenSpec change and review the implementation diff against the three new capability specs and library-discovery delta; verify no second registry/ML loop, cluster-policy editor, legacy schema path, persistent resource cache or unrequested dependency was introduced. Record native smoke evidence or explicit unverified environment limitations; stop when scoped acceptance is met.

Evidence and unverified real-cluster limitations: [verification.md](verification.md).
