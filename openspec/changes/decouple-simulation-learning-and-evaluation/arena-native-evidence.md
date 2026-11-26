# Pinned Arena runtime and native evaluator

The upstream `release/0.4.0` checkout at
`a2f421e918952ee139a9f1a7ef8f9b23047a99df` installed in a private environment
using the upstream Dockerfile's uv `0.12.9` and `--frozen` lock behavior.
The release source reports Arena package version `0.3.0`. Its own Lab
submodule is not the NexuML pin. That installation proved dependency
availability, not NexuML compatibility.

A separate resolved environment used the exact NexuML Lab 3.0.0 source
`70ab24fec3eabf51222817db0cdb5a4e7133bd7e`, Isaac Sim `6.2.0.0`, the Arena
release source, and cuRobo
`78fd485fa82d9b9a063fb4985e371814587e666a`. It retained CUDA Torch `2.11.0`,
TensorDict `0.14.2`, TorchRL `0.14.0`, and NexuML's `requests==2.34.2` pin.
The separate profile requires Arena's documented `websockets==16.1.1`
override (outside Sim's declared `<15` range) and exact upstream PyPI
routes for `mujoco-usd-converter` and `isaacteleop`. No shared lock or
worker image changed. This is not a clean upstream metadata graph or a
claim that Sim's WebRTC streaming works with this override.

The upstream noneditable Arena build omits its `isaaclab_arena.evaluation`
namespace; starting Kit succeeded but the public evaluator failed before
composition. The documented editable install from the same clean pinned
checkout exposed the native evaluator without replacing the dependency
set. It then built the official SO-101 PhysX scene and reached the native
rollout, which exposed an unresolved joint selector: Arena calls the
success predicate outside the manager that normally resolves its
`SceneEntityCfg`. The composition now resolves that same selector against
the actual loaded scene and reuses it in scoring and diagnostics. The five
policy arm joints, calibrated gripper, targets, timing, reward and held
success logic are unchanged.

## Public headless replay result

```bash
nexuml evaluate "$POLICY" --scenario-file scripts/isaaclab/so101_arena.py --device cuda:0
```

The seed-17 actor from the frozen 524,288-frame
[comparison replay](so101-placement-readiness.md) succeeded on all
**3/3** held-out Arena conditions, averaging **10.67** native steps.
Arena's policy runner and native recorder owned each episode and reported
the original source frame. All three simulation contexts cleared and the
job exited 0 without a display. The 17 focused Arena tests, Ruff and
product type checks also passed.

Receipts: `/tmp/opencode/arena-locked-runtime-W5JZJ6/`,
`/tmp/opencode/arena-combined-dlavTU/`,
`/tmp/opencode/arena-native-eval-WJEfRG/` (wheel import failure),
`/tmp/opencode/arena-source-eval-j6lXdv/` (unresolved selector),
and `/tmp/opencode/arena-held-out-VqZ0MN/` (passing run). The latter
records the hashes of the applied code, policy and effective dependency
inputs. This first run used an actor trained in stock Isaac Lab.

## Arena-composed training and continuation (1.3–1.4)

The trusted `IsaacLabEnvironment.arena_factory` serializes the installed
builder function without importing Arena on the submitter. The existing
Isaac runtime owns Kit, validates the composed native config and contract,
and removes only its own Gym registration. Arena supplies scene,
embodiment, reward, success criteria, and environment construction. The
reference `scripts/isaaclab/so101_arena_train.py` and the held-out suite
consume the same native builder; neither implements another trainer.

- A fresh public CLI fit with four worlds completed **128 physical frames
  and eight optimizer steps** and wrote a standalone inference package.
- A four-world scripted oracle (seed 23) checked the pinned loaded asset,
  native GPU actions, finite rewards, selected reset isolation, held
  success, and pre-reset terminal observations. All **4/4** worlds ended
  successfully by tick **11**. Auto-reset acknowledgement left other rows
  unchanged, terminal/reset observations were distinct, and the owned
  Gym registration was released.
- A new fit initialized its actor exactly from the hashed seed-17 source
  using `CheckpointLoadSpec`, with a fresh learner. It changed six actor
  tensors, produced optimizer state and intentionally interrupted at
  **128/256 frames and eight optimizer steps**. A fresh ordinary CLI resume
  restored trainer state and reached **256 frames and 16 optimizer steps**.
  The final export SHA-256 was
  `e599256073cab2632e69c02c9c1c53f823d9d4af66dc39f0be2150c2f12c65d5`.
- Separate native processes replayed that export for **24 ticks / 48
  transitions / four finished episodes**, then evaluated **3/3 held-out
  successes** through Arena at source frame **256**. This was a warm-start
  workflow proof, not evidence of from-scratch convergence or a gain over
  the initial actor.

Receipts: `/tmp/opencode/arena-composed-fit-851kOp/` (ordinary fit),
`/tmp/opencode/arena-control-e5doIh/` (native control and reset),
and `/tmp/opencode/arena-warm-fit-e42sE6/` (weight, checkpoint and export
hashes, interrupted fit, CLI resume, replay and suite). This qualifies
headless PhysX state-only joint reaching; no pick/place, live Arena,
physical robot or distributed training claim.
