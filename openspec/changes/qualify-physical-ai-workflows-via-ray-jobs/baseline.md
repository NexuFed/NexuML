# Apply baseline — 2026-10-22

Task 1.1 evidence only. No current-candidate application acceptance is claimed.

## Candidate and source ownership

- Local HEAD and published PR #15 head: `a812e54e1bd7ac4b52d52ce04f739064d5fe620a`.
- Fetched PR base `origin/dev`: `f8d5490ab1446177aa26ba529124310f0b8041f2`.
- PR: draft, conflicting, no checks attached to this head at inspection.
- Initial worktree: only this untracked OpenSpec change; no tracked application edits.
- The project `.venv/bin/python` is CPython 3.12.13. Static import discovery resolves
  `nexuml` to this checkout's `src/nexuml` and `nexuml_library` to `library/src`.
  Ray, Torch and pytest resolve inside this checkout's `.venv`.
- Installed metadata: NexuML/library 0.2.1, Ray 2.57.0, Torch 2.11.0+cu130,
  pytest 9.1.1, Pydantic 2.13.4, Typer 0.25.1, NumPy 2.3.1. These are inventory,
  not runtime compatibility or GPU qualification results.
- `/env/bin/python3` lacks NexuML/library, Ray and Torch. Do not mistake it for the
  project interpreter or install dependencies into that shared environment.
- Source inventory used AST parsing; interpreter inventory used `find_spec` and
  distribution metadata with `CUDA_VISIBLE_DEVICES=''`. Neither imported Torch,
  created a native runtime nor ran an application workload.

## Current-candidate gate ledger

All gate entries below are **unverified**: no remote jobs have been submitted for
this apply. C = CLI, V = visual, L = learning, R = recovery, D = distribution.
Required coverage and representative versus per-recipe distinctions remain in
[verification-matrix.md](verification-matrix.md); this ledger does not waive them.

| Matrix case | Discovered recipe or workflow | C | V | L | R | D |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | `cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| G1 | `pendulum-ppo` | unverified | unverified | unverified | unverified | unverified |
| G2 | `visual-cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| M1 | `mujoco-humanoid-ppo` | unverified | unverified | unverified | unverified | unverified |
| M1 | `mujoco-reacher-ppo` | unverified | unverified | unverified | unverified | unverified |
| I1 | `isaac-cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| I1 | `isaac-cartpole-learning-ppo` | unverified | unverified | unverified | unverified | unverified |
| I1 | `isaac-franka-reach-ppo` | unverified | unverified | unverified | unverified | unverified |
| I1 | `isaac-so101-joint-reach-ppo` | unverified | unverified | unverified | unverified | unverified |
| I1 | `isaac-custom-cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| I2 | `isaac-rgb-cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| I2 | `isaac-depth-cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| I2 | `isaac-segmentation-cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| I2 | `isaac-mixed-cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| I2 | `isaac-categorical-cartpole-ppo` | unverified | unverified | unverified | unverified | unverified |
| A1 | `isaac-pendulum-ippo` | unverified | unverified | unverified | unverified | unverified |
| A1 | `isaac-pendulum-mappo` | unverified | unverified | unverified | unverified | unverified |
| A1 | `isaac-handover-ippo` | unverified | unverified | unverified | unverified | unverified |
| A1 | `isaac-handover-mappo` | unverified | unverified | unverified | unverified | unverified |
| R1 | `ros2-gazebo-joint-reach-ppo` | unverified | unverified | unverified | unverified | unverified |
| C1 | `carla-lane-following-ppo` (external boundary) | unverified | unverified | unverified | unverified | unverified |
| S1 | SO-101 capture / `scripts/so101_local_fit.py` / SDK / replay | unverified | unverified | unverified | unverified | unverified |
| B1 | Synthetic supervised scenario / exported-model evaluation | unverified | unverified | unverified | unverified | unverified |

AST discovery found exactly 21 registered RL recipes and verified that each occurs
in the planned matrix. Original datasets, checkpoints and historical reports were
not changed or reclassified.

## Historical CI diagnostics, not current-head results

Most recent inspected run:
[39781691221](https://github.com/NexuFed/NexuML/actions/runs/39781691221),
created 2026-10-22, associated head `a5011a09024a7905835d15c99772ce5b0d34084b`.
Its failures must not be attributed to an untested final candidate.

- **Static:** lint/format succeeded; the final gate failed after type diagnostics,
  including invalid `forward` overrides and unresolved optional native/LeRobot
  imports. The step's displayed conclusion alone hides the continued-on-error
  type failure; consult the actual log.
- **Python 3.12:** `test_evaluate_help` failed its literal `--override` assertion;
  the synthetic smoke trained but failed during package export with a
  `PackagingError` involving imported media/protobuf modules.
- **Python 3.13:** the same CLI/smoke failures, plus native-runtime fake/admission
  tests failing the pinned-profile Python check and a numerical-body AST hash
  mismatch. These are diagnostics to reproduce, not permission to skip the minor.
- **Python 3.14:** install attempted to build pinned NumPy 2.3.1 from source and
  failed without an available C compiler; tests never ran.
- **Distributions:** wheel-discovery test ran in a pytest-only environment and
  failed on missing `tensordict` through an eager core import. Wheel construction
  succeeded, but installed/rebuilt-wheel checks were skipped after this failure.
- Optional simulator/Isaac jobs were skipped; they provide no native acceptance.

The fetched failure log is at
`/tmp/opencode/nexuml-pr15-ci-39781691221-failed.log` for local investigation only.
It is not durable release evidence and is not checked into Git.

## Historical task 1.2 blocker: conflicting execution contracts

Read-only `git merge-tree --write-tree --name-only HEAD origin/dev` found conflicts
in 15 files: `.github/workflows/ci.yml`, `.gitignore`,
`docs/how-to/training-backends/index.md`, `docs/how-to/training-backends/ray.md`,
`docs/reference/scenario-spec.md`, `pyproject.toml`, `src/nexuml/cli/main.py`,
`src/nexuml/core/components.py`, `src/nexuml/core/discovery.py`,
`src/nexuml/core/registry.py`, `src/nexuml/core/types.py`,
`src/nexuml/execution/__init__.py`, `src/nexuml/training/lightning.py`,
`tests/packaging/test_ci_workflow.py` and `uv.lock`.
This was not an index/worktree merge; no conflict markers were applied to source.

More importantly, the base has a contract the proposal did not reconcile:

- Base `openspec/specs/nexuml-execution-backends/spec.md` requires registered
  `type/version/params` execution, explicitly forbids retaining `execution.kind`
  as a second schema, and requires shared Python/CLI/API backend dispatch.
- Base `src/nexuml/core/types.py` enforces that rule; its
  `ScenarioSpec.execution` is an `ExecutionBackendDefinition`.
- Base `src/nexuml/execution/definitions.py` registers `ray-job` as the KubeRay
  RayJob CRD, explicitly distinct from generic Ray Jobs SDK submission. Its
  remote worker uses Ray Train placement, not the single-process RL lifecycle.
- This feature branch and proposed D2 instead depend on `RayJobExecutionSpec`,
  `execution.kind='ray_job'` and direct CLI transport routing.

Before source integration, amend the change coherently to register the SDK Jobs
backend in the base's execution catalog, keep it distinct from KubeRay and route
operations through shared backend dispatch, or explicitly choose a different
execution scope. Do not resolve by retaining two configuration schemas, silently
substituting KubeRay/Ray Train, or overwriting the base's execution infrastructure.
The requested remote-only execution, bounded parallelism and local-GPU exclusion
remain unchanged. No application code, jobs, commits or PR status were changed.

## Refreshed baseline and approved integration direction

Read-only inspection for the resumed apply found:

- Local HEAD and published PR #15 head: `851a9bc031645f98a2a33f2683304b6f65c50d89`.
- Fetched `origin/dev` and PR base remain `f8d5490ab1446177aa26ba529124310f0b8041f2`.
- PR #15 remains open, draft and conflicting (`mergeStateStatus: DIRTY`), with no checks attached to the current head.
- Git reports 59 feature-only commits and one `dev`-only commit. This is a history inventory, not a completed rebase or integration result.
- The 19 previously reviewed files remain staged: tracking (five), export provenance (three) and this planning change (eleven). No unstaged changes were present before this plan revision.
- Direct inspection of `dev`'s canonical execution spec, registered definitions and service confirms that the schema/dispatch discrepancy above still exists. No index/worktree merge or rebase was attempted.

The user resolved the integration choice: **rebase onto `dev` and port this branch's working SDK Jobs behavior into its execution structure**. They approved the distinct registered name `ray-jobs-sdk`, preserving `dev`'s existing KubeRay `ray-job` and rejecting a second legacy `execution.kind` schema. Preserve locked uv setup, one-GPU defaults, worker-local cache/download settings, private per-submission venvs, prepared-environment opt-out, configurable project settings and asynchronous submission results.

The approved campaign uses only `pool_gpu_workers`, never DGX or a separate CPU-worker pool, with at most four in-flight jobs constrained by qualified capacity. CPU controls remain CPU workloads inside default one-GPU jobs. Keep the monitored 4 GiB physical free-disk floor, durable per-attempt outputs and no-local-GPU boundary.

The rebase is explicitly authorized; source/config migration is now a planned implementation task, not permission to replace the SDK with KubeRay/Ray Train. Back up the pre-rebase ref and exact index/worktree state before rewriting history. No force-push, new implementation commit, external publication or PR/Linear status update is authorized by this plan correction.

This revision changes planning artifacts only. No rebase, application-source edit, dependency install, application/browser test or remote job was performed. All current-candidate C/V/L/R/D gates remain unverified and overall progress remains **2/36**; the earlier evidence retains its original revision and scope.
