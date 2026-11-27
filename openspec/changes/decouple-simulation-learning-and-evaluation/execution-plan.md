# Autonomous continuation and acceptance ownership

Maintainer decision: keep TorchRL; primary native GPU PPO = co-located direct and DDP/NCCL; optional split/remote roles for memory and measured throughput. Do not stop a whole implementation campaign because an external acceptance gate cannot be exercised. Existing evidence and completion marks are immutable historical receipts; do not rerun all completed work, reset counts or declare 100% without required live evidence. This plan supersedes any instruction to wait on all nine prior outstanding gates before proceeding.

## Start from the agent's actual worktree

- Preserve local milestone commits, dirty changes and receipts. Fetch the remote revision and reconcile these specification-only changes with local implementation (ordinary merge/rebase only under existing authority, no reset, discard, force-push or implicit publication). Inspect actual code and tests, not just historical status.
- Start with a short inventory: code already present, testable locally, prepared runtime/allowed GPU resources, approved assets/data, actual blocker. Use read-only preflight for cluster roles and network; do not change NetBird, cluster settings, GitOps, driver, worker pool, credentials, active environment or unrelated workloads. Do not require user to provision anything before doing unrelated source work. Respect existing execution/resource authorization; no new job/download/hardware permissions are granted.

## Ordered implementation queue (complete all independently feasible items)

| Order | Work | Evidence possible without special hardware |
| --- | --- | --- |
| 1 | §2a.1: existing Lightning/Ray Train PPO lifecycle, DDP semantics, rank-owned native runtime, global budgets, seed/optimizer/checkpoint/metrics | Actual 2-process CPU/Gloo, mock native scope, deterministic parity, CLI/config/negative tests |
| 2 | §2a.2–2a.3: GPU co-located profiles, NCCL preflight, realistic native benchmark and targeted overhead fixes | Preflight/config/harness tests; existing qualified single GPU and any explicitly available approved rank placement; absent multi-GPU recorded separately |
| 3 | §2.2, 3.4–3.5, 2a.4: independent GPU learner, optional split/inference and Ray GPU workers, version barrier, memory/transport evidence | CPU/GPU single-device and explicit fake multi-device/placement tests, installed simulator-free learner, measured small visual/memory-pressure fixtures where possible; actual split/remote proof separate |
| 4 | §4.4, 4a.6: native independent snapshot eval and selected actual rank Viser; rank-zero lifecycle | Ownership, fresh process, snapshot, cancellation/failure, viewer rank/port and negative tests; live native overlap and rank spectator separate |
| 5 | §4a.1/4a.4, 6.3: retain scoped local Kit/TigerVNC and headless passes; WebRTC play launcher, optional Kit-less, immutable prepared runtime roles and version pins | Native launcher/selection/preflight/endpoint tests, local approved runtime tests, docs and CI build/test where inputs available; external client, X11, unavailable wheel/image or GPU distinct gates |
| 6 | Integrate focused regressions, strict OpenSpec validation, documented public CLI/SDK examples, accurate support matrix and software handoff | No need to wait for physical ROS data, unavailable GPU layout, remote client or public release |

Do not add another trainer, scheduler, middleware, config hierarchy or custom distributed protocol. First inspect actual maintained APIs and existing tests. No broad rewrite of completed foundation. Fix concrete root causes and keep guards until corresponding software semantics pass. Use existing approved resources only; missing GPU capability does not mean select CPU as actual requested GPU backend. Prefer small actual CPU multiprocess correctness tests to pure mock DDP, but label CPU tests CPU.

## Bounded failure workflow

For each task, attempt an inexpensive prerequisite check, implement and run targeted tests; on failure inspect original logs, isolate a minimal reproducer and apply a justified fix. Retry only after a concrete changed prerequisite or code fix (at most two equivalent live attempts before classifying a blocker; no repeated identical retries or polling). If prerequisite is unavailable, immediately record the exact missing role/input/version/endpoint and the already implemented/tested slices, a reproducible command or preflight result, and move to the next independent task. Local failures with a fixable code cause remain implementation work, not external blockers. If correctness is uncertain, keep fail-closed path and test it; never fake a pass. Do not request repeated user decisions, wait for an operator/client, or terminate the session merely because one lane is blocked. End only after independently feasible queue exhausted and integrated review is complete, or a genuine overall constraint prevents further useful work; report what is left precisely. Do not exceed approved compute/storage/time budgets. Do not promise background work.

## Acceptance matrix: independent lanes

| Lane | Software handoff requirement | Native/live acceptance prerequisite |
| --- | --- | --- |
| Primary co-located PPO/DDP | Real CPU/Gloo correctness and public lifecycle, single GPU fast path, rank semantics, bounded admission | Actual NCCL 2 GPU same-host and distinct-node evidence (distinct receipts); available authorized compatible GPUs |
| GPU split / Ray collectors | GPU-learner/worker implementation, no CPU-only guard for supported path, version/transfer/admission/cleanup tests | Compatible visible GPU pair or learner + 2 native collector GPUs with prepared dependencies |
| Visual and large models | Explicit model memory/placement semantics and stress/visual reference, unsupported mode errors | Actual approved large VLA, fit budget, compatible objective and accelerators; future separate model qualification, not a PPO gate |
| Native evaluation / selected training Viser | Independent snapshot and selected-rank ownership/cleanup tests | Separate GPU/native runtime and actual remote browser/rollout where applicable |
| Kit local / TigerVNC | Preserve exact already demonstrated profile and regressions | Requalify affected behavior only; forwarded Windows X11 independent optional profile |
| Native WebRTC play | Supported launcher/endpoint/graphics prerequisites and fail-closed tests, separate play worker | Approved prepared private client/host TCP+UDP media path, encoder, actual frames; optional independent profile |
| Worker packaging | Exact pins, image/CI recipe and installed boundaries, preflight diagnostics | Authorized immutable missing wheels/images, compatible GPU workers; no arbitrary install in actor startup |
| ROS 2 real experience | Existing local-fit and data validation remain tested without private input | Authorized matching real episodes/calibration and supervised approval; never invent or recapture automatically |

Report each slice separately as implementation status (implemented / partial / not started), software tests (pass/fail/not run), native/live status (pass/fail/blocked/not attempted), revision and evidence. For blocked slices specify exact missing prerequisite, how to rerun, and whether it gates a requested support claim. Keep old 35/44 evidence scoped; new tasks are not marked complete. Software handoff can be qualified with external live gates open; no archival/full readiness until required acceptance is actually satisfied or explicitly narrowed by maintainer. Real ROS data gate can be addressed by follow-up robotics integration; no real hardware dependency for co-located learning, Ray placement code or local playback.

## Definition of done for this continuation

All feasible implementation/tests/docs and existing-regression checks executed, exact supported and blocked profiles documented, strict validation run if available, local commits created only as authorized. Provide a compact final report with actual tests, remaining external prerequisites and exact commands; do not ask user to babysit each step. No claim that small MLP/CPU tests prove large-VLA or multi-node GPU quality.
