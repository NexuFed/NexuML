# Verification evidence

## Scope

Five built-in definitions and a contributed sixth definition use the existing registry,
registered serialization and shared Python preflight/dispatch. API/CLI transport those
operations; Studio renders runtime-owned schemas and capabilities. Interface major is
2; `/api/v1` remains the transport prefix. Old `execution.kind` inputs are rejected,
not aliased or silently changed to Local.

## Passed checks

| Gate | Evidence |
| --- | --- |
| Python regression gate | `tests/execution`, `tests/api`, `tests/cli`, `tests/core`, `tests/packaging`, excluding `slow`: final rerun **280 passed, 3 skipped, 2 deselected** |
| Expanded execution/API fixture gate | 92 passed, including actual two-rank CPU `torchrun` |
| Last native-context/resource guard checks | 35 passed after pinning the resolved current context without hiding changed resource requests |
| Studio unit tests | 22 passed across 6 files |
| Studio typecheck and lint | `npm run typecheck`, `npm run lint` passed |
| Python lint and whitespace | Targeted Ruff checks and `git diff --check` passed |
| Wheel build/startup | `uv build --wheel`; extracted built-wheel API/CLI handshake test passed |
| Production Studio package | `npm run test:package` built/packed/installed the frontend and passed all 7 browser workflows against the selected built-wheel runtime |
| OpenSpec | `openspec validate add-discoverable-execution-backends --strict` passed |
| Documentation | `mkdocs build` passed; strict mode limitation is recorded below |

The package smoke uses a private npm install and an extracted NexuML wheel with existing
runtime dependencies and the worktree's base library. It does not claim a clean Python
tool/project-environment installation. It verifies offline fonts/assets, selected
interpreter identity, ordinary YAML save/build, CPU launch and live/finalized metrics,
reconnect/export, unchanged Python/npm installations, owned API shutdown and attached
API preservation. Browser traces are disabled because they could contain bearer headers.

## Acceptance mapping

- Registry/configuration: core-only lazy loading; installed-package/local-root contribution;
  duplicate diagnostics; built-in and contributed round-trip; unknown version/identity,
  old syntax and invalid field paths fail without replacement.
- Local/Ray: canonical session observation/resume/export and actual CPU training;
  fixed/elastic named Ray resources and native strategy integration; pre-allocation
  stateful-evaluation/post-fit/strategy/resume/export guards; initialized Ray cannot
  silently reuse an unverified target. Ray lifecycle checks use fixtures, not a cluster.
- Discovery: finite budgets/result bounds, source/time/unknown-versus-zero, affinity and
  cgroup limits, GPU timeout fallback, read-only Ray node totals, selected-scope KubeRay
  identities, explicit Kubernetes context, partial/denied reads, API-versus-controller
  support and TrainJob-versus-PyTorchJob separation. No runtime startup or port forwarding.
- Capacity: native quantities, sidecar/init/overhead/pod-level requests, readiness/cordon,
  selectors/required affinity/taints and per-node fit, named accelerator units, per-role
  footprints, quotas and absent pod visibility. Unselected multi-role overall fit stays
  unknown; individually measured role fits remain available.
- Native jobs: reviewed authorized templates and named bindings, exact dry-run calls,
  template/config revisions, explicit Local/Ray worker derivation, exclusive filesystem
  and conditional S3 config writes, one create, acknowledgement-loss inspection, actual
  state/log observations, UID replacement refusal and verified foreground cancellation.
  Three documented templates are checked with fake served APIs; those are not live
  admission/controller tests. Current-context resolution is pinned into the reviewed
  ordinary execution configuration before launch.
- API/Studio: safe inputs/diagnostics, one active operation, discovery outside its slot,
  frozen records, native references and actual terminal statuses, shutdown/restart unknown
  ownership, explicit inspection without resubmission, matching interface handshake;
  schema-driven sixth-backend rendering, cancellation of staged edits, stale-response and
  stale-review handling, double-submit guard and one undoable accepted edit. Browser
  checks cover focus/error links, mobile scrolling/actions, reduced motion, pending/run/
  cancel states and external artifacts without unauthorized download controls.

## Explicit limitations and stop condition

**No real Ray or Kubernetes cluster was probed or submitted to by acceptance checks.**
Real-cluster smoke remains unverified and needs separate operator authorization using
`docs/how-to/training-backends/jobs.md`. Controller versions/health, remote mounts and
credentials, multi-host networking, GPU execution and real scheduling/admission are not
certified by fake API tests or two local CPU ranks. Ray available capacity remains unknown
when the supported read-only API exposes totals only; no snapshot is a reservation.

PyTorchJob is the served `kubeflow.org/v1` resource with explicit operator-connected
`torchrun` DDP, not TrainJob or independently trained replicas. Distributed stateful
evaluation/post-training fitting remain unsupported. RayCluster/RayJob cancellation is
not advertised; Job/PyTorchJob cancellation is UID-verified. No automatic recovery,
resubmission, provisioning, policy editor, persistent resource cache or second ML loop
was introduced. The lockfile adds Kubernetes and its transitive dependencies only; Ray's
existing optional compatibility range is unchanged.

Strict documentation build fails on the existing missing return annotation in
`src/nexuml/api/observation.py:87`; that unrelated file was not changed. Normal docs build
succeeds. Slow/data/clean-install integration gates are not implied by targeted acceptance.
No archive, push or real-cluster launch is implied by these checks.

All 34 implementation tasks are checked. The smallest final context/resource correction
was additionally rechecked with the 35-test native/discovery gate; no frontend code changed
after the successful package smoke. Stop here unless the operator authorizes cluster smoke
or the user requests a commit/archive.
