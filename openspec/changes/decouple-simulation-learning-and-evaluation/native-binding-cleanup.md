# Task 1.2 — native scope and bindings

Keep the existing `isaaclab_process` owner for ordinary direct jobs. Its
scope already surrounds native construction, collector, learner,
checkpoint/export, logger finalization, and teardown in the required order.
The project-owned direct task reuses it. No second native process owner or
speculative remote lifecycle was added. Arena composition and remote worker
ownership remain separate, unverified tasks.

The shared Isaac runtime no longer checks the SO-101 task name to validate
controls or copy episode outcomes. The registered SO-101 builder verifies
its asset and permitted action term, closes a rejected instance, and lets
the ordinary typed manager binding verify the live implementation, joint
order, offset and scale. Native reward and termination terms publish their
pre-reset outcomes in step extras. The adapter validates and snapshots the
scalar batches identically for any task using this convention. Reward,
reset, measured outcomes and policy semantics did not change.

On the prepared runtime listed in [custom-direct-task.md](custom-direct-task.md):

- **4 native tests passed in 81.34 s:** signed SO-101 controls and held
  outcomes; SO-101 checkpoint/resume/export and fresh scalar replay;
  independently registered custom direct controls/RSL-RL replay; and
  policy-versus-script headless parity. Logs and artifacts:
  `/tmp/opencode/nexuml-decouple-binding-qdyJyT/`.
- **1 native spectator test passed in 15.30 s:** loopback Viser and
  headless trajectories matched exactly, with isolated render/control
  storage and owned threads stopped. Logs:
  `/tmp/opencode/nexuml-decouple-binding-viser-fBy1Tq/`.
- **355 hermetic tests passed, 19 live tests skipped.** The checks cover
  ownership/cancellation, wrong or malformed native controls and outcomes,
  cleanup on asset rejection, config/contract, and Arena policy protocol.
  Lint, scoped type checks and strict OpenSpec validation passed.

No new Arena, visual play, split GPU or distributed support is implied.
