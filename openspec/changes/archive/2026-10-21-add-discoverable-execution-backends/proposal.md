# Proposal

## Why

NexuML Studio can review and launch Local/Ray training, but its backend availability means only that a Python package is installed; users cannot choose discovered execution targets or assess their usable capacity. Execution backends, configuration schemas, resource discovery and compatibility checks must belong to the selected NexuML Python installation so Python, CLI and Studio share the same behavior without requiring FastAPI.

## What Changes

- Register execution backend definitions through NexuML's existing component/library discovery machinery, with stable identities, schemas, dependency diagnostics and supported controls. Keep optional runtime imports lazy and permit installed libraries to add backends without API/frontend provider code.
- Supply Local, RayCluster, KubeRay RayJob, Kubernetes Job and Kubernetes PyTorchJob execution modes. Reuse `NexuSession.run()`, the existing Ray Train implementation and native platform submission/status/log APIs; do not create additional training loops.
- Add Python-owned discovery of local effective resources, known Ray endpoints, kubeconfig contexts, accessible namespaces/operators and eligible cluster capacity. Distinguish installation support, target accessibility, scenario compatibility, total/allocatable capacity and estimated currently unallocated resources.
- Use user/infrastructure-owned job templates, explicit target selection and bounded read-only discovery. Show partial/unknown results and actionable failures rather than provisioning infrastructure or claiming scheduling guarantees.
- **BREAKING**: Replace the closed Local/Ray execution union with registered execution definitions using the existing `type`/`version`/`params` configuration representation. Update affected repository configurations and examples; do not maintain a second legacy execution schema.
- Expose the same Python discovery/preflight/dispatch functions through the CLI and thin authenticated local API. Adapt existing operation observation to backend-reported capabilities and native job references rather than hardcoded Local/Ray branches.
- Extend the existing Run review modal with schema-backed backend/target/resource selection, live capacity summaries, explicit unavailable reasons, accessible diagnostics and an exact frozen launch review. Retain the approved Studio appearance and existing execution/results views.

## Capabilities

### New Capabilities

- `nexuml-execution-backends`: Installation-owned execution definitions, shared Python/CLI/API dispatch, five native execution modes and honest lifecycle capabilities.
- `execution-resource-discovery`: Bounded target/capacity discovery, workload eligibility, permissions, freshness, partial results and credential boundaries.
- `studio-run-target-selection`: Data-driven Run modal selection and review over the authoritative Python backend catalog and resource snapshots.

### Modified Capabilities

- `library-discovery`: Extend decorated discovery to registered execution backend definitions while preserving existing package/root discovery behavior.

## Impact

- Python: `core/components.py`, `core/discovery.py`, `core/registry.py`, `core/types.py`, `core/backends.py`, `execution/`, CLI dispatch and narrow API catalog/operation integration. Reuse existing serialization, selected-runtime subprocesses, log paths and observation records.
- Dependencies: existing optional Ray/API support; a lazy optional Kubernetes client extra. No new frontend dependencies or mandatory cluster dependencies for local use.
- Studio: existing catalog/client types, schema fields, Run dialog, execution controls and focused browser checks; no new dashboard or design system.
- Integration: build on the in-flight Studio, typed-configuration and thin-Ray changes without editing their artifacts. This follow-up explicitly extends their closed execution selection and limited remote observation boundaries; infrastructure ownership and distributed-semantic guards remain intact.
- Non-goals: cluster/operator installation, cluster administration, generated deployment policy, resource reservation, queue management, new job database, hosted authentication, universal training telemetry or globally correct distributed post-training finalization.
