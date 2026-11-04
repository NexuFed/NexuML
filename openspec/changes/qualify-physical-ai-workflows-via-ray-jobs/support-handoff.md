# Physical AI qualification support handoff

## Status and authority

Partial support handoff, not a release approval. [tasks.md](tasks.md) remains the
authority: **21/36 complete**. Required blocked rows are not waived. No publication,
PR/Linear status change, merge or archive is authorized by this handoff.

- Published head: `01d3c0a7dcde55b5d4915e9484d6f069e946fe67`, based on actual
  `origin/main`; preserve Starlight and the existing rebase backups.
- Uncommitted executable snapshot:
  `dd466d1cc6d3ecd0ed2bebeb800947a8b3b726eabe4655d6be25347f4b90b3d5`.
  This includes test-owned viewer IPC changes. The current eight-test spectator
  suite passed remotely; full spectator acceptance and release readiness remain open.
- Production `src/` and `library/src/` remain unchanged from the published head.
  Historical evidence keeps its actual source/runtime identity, never the new hash.
- Published-head CI remains failed; exact-candidate CI requires separately authorized
  publication. Local checks and historical remote passes are not that CI result.

## Claim-to-evidence map

| Surface | Retained result and scope | Remaining qualification |
| --- | --- | --- |
| G1/G2 public CLI, I1 native, A1 multi-agent, M1 MuJoCo | Recorded train/resume/export/fresh-evaluation passes under their original snapshots. Bounded recipe runs are smoke, not convergence. | Final identity reconciliation and applicable recovery/resource matrix. No Humanoid GUI extrapolation. |
| I2 sensors/categorical | All nine processing-owner CLI journeys plus 13 native oracles; 810 files / 1,129,237,671 artifact bytes rehashed. Sources `43ae0b9c…` and `2e7c0da5…` remain distinct. | Does not qualify spectator viewing, universal renderer support or camera-policy convergence. |
| Frozen Cartpole L1 | Actual held-out native learning comparison passed under its recorded single-training-seed protocol. | No multiseed, other-task or hardware-safety claim. |
| Public SO-101 cloning | Job `d3e74059a63d4b0aafc58e6f7965e4a3`: 64 GPU optimizer steps, 40/5/5 public-data split and reduced held-out loss. | Not frozen S1/L2: different task, channels, rate, episode count and qualification contract. |
| Installed B1/tutorial | Job `56bbb15a360f4516ae6918a15a49023a`: exact installed-wheel controls and canonical tutorial passed; 20 files rehashed. | Approved-data S1 fits and full task5.6 remain open. |
| Four logger modes / SQLite and S3 | Job `d239a4d387474ce396f178e104351f3d`, source `43ae0b9c…`: 32 CLI stages, 16 FINISHED MLflow runs; 293 retained files verified. Actual downloads and same-rollout payload equality passed. | Not current MLflow UI/failed-run cleanup proof, private-bucket protection or SQLite metadata backup. |
| MLflow successful-run UI | Historical jobs `fa1f34d39aba40db8de1c412bad04d23` and `d63ba45d56794618a410ff3fec0ec1f1`, source `bb40b77c…`: selected native Overview rows, status, media and cleanup passed. | Current SQLite/S3 browser reconciliation, native failed-run tree and unchanged strict cancellation contract remain open. Do not discard these genuine earlier passes or relabel them current. |
| Isaac public-CLI spectator | Current job `82d65c3218a24ee0a8a7764dbd0f4021`, source `dd466d1c…`, passed all eight tests without skips. All164files /296,978,604artifact bytes verified and privately preserved. Full trajectory/PPO equality, actual WebSocket mutation denial, IPC cleanup and same-job private8081HTTP200 were reviewed. | Task6.3 still needs external cancellation of a spectator-active registered Ray job with terminal status and listener reuse. Existing cancellation self-SIGTERMs only the CLI child. Training-reconnected capture is cropped, not positive framing proof. Browser SwiftShader is not server RTX proof. |
| Failed viewer recovery | Failed `64fbe044…` and `0cfaa3d7…` retain complete recovered original files. Failed `2b85841e57cb4b13b3fa7cec9e625727` now has all49original files verified; original FAIL/guard1 is unchanged. | Recovery is not a successful rerun or qualification of the failed source. Earlier23/49records remain historical partial observations. |
| Post-worker persistence | Registered recovery `37366add3c4846279d29ea27d144335b` passed after the current viewer producer was removed. A verified owned reader retrieved all164original files. | Initial stdin-based tar transfers truncated despite kubectl0. Strict EOF/hash checks rejected them; a new stdin-free literal-argv transfer passed. This is not a universal transport-repair claim or completion of the whole recovery matrix. |
| Ordinary public CLI reporting | Verified-CA command submitted `raysubmit_hK9Mg9DDf2xnMe7N`, which FAILED before guard/setup/training: preparation omitted `runtime_env.working_dir`, so the worker lacked the relative guard script. Original config/lock/logs are retained. | Task4.1 remains open. Corrected approved-CA/upload/exclusion inputs passed mocked transport checks but are OFFLINE ONLY, not submitted. No failed application auto-retry. |
| Infrastructure | Authorized GPU-worker-only plasma fix `23c3410e…` was deployed; preflight `5424f937…` verified4GiB tmpfs plasma and zero disk-backed plasma. | Does not establish application throughput, cancellation storage health or safe two-GPU overlap. No other infrastructure changes authorized. |

Receipts and exact identities live in the existing private preservation bundles and
chronological `preparation.md` / `.tasks/todo.md`, not in Git-tracked weights or media.
The current durable bundle is `current-eight-complete-public-launch-diagnosed-s2_7wlr5`
under `/home/nexuadmin/.local/state/opencode/nexuml-preservation/scheduling-mlflow-20261029-jb0h_ud0/`:
772files /959,485,635bytes rehashed, including the complete current viewer evidence,
original failures/transports and public launch diagnosis. Credential-value scans were
clean; private execution configs were recorded by hash, not copied into that bundle.

## Next execution order

1. No application/controller is active. The hourly capacity trigger is disabled
   after accepted admission and remains disabled after the real public-job failure.
   Latest read-only inventory at2026-10-30T03:21:46Z found3eligible free GPUs and no
   Ray jobs/workers; this is time-bound, not parallel-memory qualification.
2. Preserve the completed current eight-test proof and all49original failed files;
   do not rerun them merely because the first byte transport failed. Do not reuse
   either public preparation's exclusive submission lock.
3. For any explicitly restarted attempt, refresh eligibility/reservations/jobs,
   physical disk and actual shared-memory admission, confirm no overlapping checkout
   writer, and record external source freeze through submission and terminal readback.
   Serial whole-GPU `pool_gpu_workers` only. Fingerprints detect changes but do not
   enforce filesystem immutability.
4. Public reporting's corrected inputs explicitly preserve the approved CA, uploaded
   checkout and data/venv/output exclusions.116hermetic checks/35subtests passed;
   no new application was submitted after diagnosis. A fresh explicit attempt is
   needed before claiming actual CLI completion. Never use the old checkpoint-export
   preparation, which would restore historical logger/callback paths. For6.3, obtain
   actual registered-Ray cancellation/cleanup/listener-reuse proof, not another
   CLI-child self-signal; keep the storage/profile gate below in force.
5. Reconcile MLflow UI evidence and remaining isolation, resource-cost, persistence
   and recovery coverage. The historical `c340934b…` browser cancellation failure
   had owned D-state threads in `jbd2_log_wait_commit` / `fdatasync`; its exact old
   FD/file/device is unknown. Require storage/profile qualification before repeating
   that known-failing journey unchanged. Do not relax limits, durability or sandboxing.
6. Demonstrate two real overlapping GPU allocations only after capacity and aggregate
   memory are qualified. Queued jobs are not overlap.

The user authorized an hourly capacity trigger: OpenChamber task
`16fbdcab-4401-4c43-9a6d-989fb73dd35a`, minute20, Europe/Berlin, first
2026-10-30 00:20CET, targeting the current session. It was disabled after the first
accepted owned preflight and must not become an application retry queue. Check
active controllers and latest checkpoint before any separately restarted launch.

## Executed procedure references and boundaries

- `docs/src/content/docs/how-to/training-backends/ray.mdx`: registered SDK schema,
  asynchronous submission/status and explicit worker paths; ordinary public-CLI
  current reporting capture is still pending4.1.
- `docs/src/content/docs/how-to/isaac-read-only-viser.mdx`: separate train/replay
  enabling, fixed eight-test command and KubeRay8081 private forwarding. The new
  IPC fixture has current eight-test remote proof, but external-Ray cancellation
  remains missing; do not label task6.3 complete.
- `docs/src/content/docs/how-to/tracking.mdx`: per-attempt SQLite metadata, separate
  S3 artifact prefix, verified TLS, explicit failures and no implicit rollback.
- `docs/src/content/docs/how-to/isaac-sensors-and-policy-state.mdx`: processing-owner
  and sensor/backend limits; spectator pictures do not prove policy input correctness.

## External prerequisites that remain

- Approved72successful SO-101 joint-reach episodes, fixed48/12/12 split and explicit
  current-runtime qualification, or an approved replacement frozen protocol. No
  automatic recapture or private-data upload.
- Prepared ROS/Gazebo world, robot/controllers, reset/clock bridge, isolated namespace
  and private GUI access. Fake transport and GUI startup are not substitutes.
- Approved worker display and private access for MuJoCo Reacher live native viewing.
- Storage/profile qualification for the recorded browser journal-sync boundary;
  unrelated GPU plasma repair is not proof of that repair.
- Owner credential review/rotation after the separately reported tool-output exposure.
  Do not reproduce values or copy the raw affected output into evidence/publication.
- Separate authorization for commits/push/external statuses, then fresh required CI;
  merge and archive remain independently gated.

## Source and data hygiene

Keep raw episodes, checkpoints, policies, media, logs, browser traces and private
runtime profiles outside commits. Preserve the mixed uncommitted `preparation.md`,
root `.venv`, datasets, old receipts and unrelated work. No driver/image/RAM/security
changes, other-user deletion, local/DGX fallback or hardware actuation is authorized.
CARLA live integration, NexuFL/federation consumption and later model/robot roadmap
items remain separate, unqualified work—not claims closed by this change.

Task8.4 remains open until final reconciliation; this draft deliberately exposes
missing evidence rather than presenting partial qualification as release readiness.
