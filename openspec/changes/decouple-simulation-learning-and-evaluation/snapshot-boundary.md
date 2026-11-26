# Task 4.1 — local snapshot seam

The existing `NexuRLLightningModule.training_step` finishes PPO optimization,
increments physical frames/batches and synchronizes the rollout policy before
the periodic evaluation check. `export_package` can already copy the complete
deployable actor at this boundary. Reuse it rather than designing a second
checkpoint or reconstructing a policy from optimizer state.

A bounded CPU Pendulum probe used a canonical `NexuSession` with three
32-transition updates. A batch-end callback captured the policy at **32**
physical transitions using `export_package` and launched one fresh CPU
evaluator. The learner reached **96** and its weights changed. The worker
reloaded all actor tensors exactly as captured, completed two 12-step
episodes at seeds 23 and 24, and attributed the report to frame **32**.
Neither optimizer state nor a Lightning checkpoint was in the snapshot;
the worker's training and collector dependencies were blocked. The snapshot
package digest remained unchanged when rechecked after training and replay:
`4656b5a97b70f0e2711a525db89f18758e4e9fb2ebbac49a54b0a793ddf6be14`.
Logs, exported tensors and report:
`/tmp/opencode/nexuml-decouple-snapshot-boundary-XH1goD/`.

## Configured blocking CPU mode

`training.evaluation.process` uses the same initial, periodic, and required
final hooks as local fit evaluation. At the completed update boundary the
learner exports an inference-only package to a private temporary directory.
The worker verifies the digest before loading it, uses the ordinary
evaluator, and returns metrics plus episode, policy, and source-frame
provenance. The parent validates the snapshot digest, frame and batch again,
then publishes through the existing logger. The trainer waits for at most one
worker, fails on an unsuccessful exit, and terminates a timed-out worker.

The CPU Pendulum integration configured 96 transitions, batches of 32,
periodic interval 64 and two four-step episodes per evaluation. Distinct
worker processes evaluated frame/batch pairs **0/0**, **64/2** and **96/3**
at seeds 23/24. Each snapshot had a distinct digest; live actor tensors were
unchanged by evaluation and all three receipts landed in TensorBoard's
artifact directory. A timeout test reaped the owned worker and removed the
temporary package. Existing in-process validation, policy export, evaluation
and playback regression: **247 passed, 14 optional cases skipped** in 41.34 s.
Logs and reports: `/tmp/opencode/nexuml-decouple-process-eval-final-wz56G5/`.
The public `nexuml train` override also passed at 0/0, 16/1 and 32/2:
`/tmp/opencode/nexuml-decouple-process-cli-56TeQU/`.

This completes the minimal placement and snapshot requirements in 4.1.

## Bounded overlap in a fresh CPU process

`process.scheduling: skip_if_busy` starts one snapshot worker at a time. A
busy worker causes an explicit skip metric at the requested frame, never a
queued backlog. Fit end waits for the pending receipt and evaluates the final
actor if that exact policy is not already evaluated. Errors fail the fit;
cancelled or timed-out snapshots are reaped without publishing a receipt.

A gated subprocess let the real Pendulum PPO learner reach frames **32, 64,
and 96** while the frame-zero worker was still running. The learner recorded
all three skipped requests, then finalized exactly two independent receipts
at **0/0** and **96/3**, with distinct actor digests and seeds 23/24. Worker
failure and a Lightning cancellation during final drain both reaped the worker
and removed its scratch artifacts without reporting success. **97 focused
tests passed**, including the unchanged native Isaac fit-evaluation rejection.
Video, GPU and Ray placement are not qualified by these tests.

Receipt: `/tmp/opencode/nexuml-cpu-snapshot-milestone-aJ4XdI/`. The source and
test digest (using the acceptance matrix's command) is
`18c25d3a47ed2ad3138dc7432750859d108d5ceaa5ed29cf3206486fd1014ffd`.
