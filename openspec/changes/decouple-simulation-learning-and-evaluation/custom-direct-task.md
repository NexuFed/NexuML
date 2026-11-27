# Task 1.6 — native direct task qualification

`nexuml_isaac_tasks.centering_cartpole` owns the Gym registration and Python
configuration for `NexuML-Cartpole-Direct-Centering`. It subclasses the native
`DirectRLEnv` Cartpole implementation, adds its own reward and track limits,
and uses the native asset, efforts and reset path. The native scripts import
no NexuML code. The scenario adds only the learning configuration and
portable bindings. No task-ID branch or replacement physics is used.

The adapter requires a trusted live control descriptor for the typed action
contract; unmanaged direct controls cannot be inferred from matching widths.
Static discovery, YAML restoration and compilation do not import the native
task. The descriptor rejects a changed effort implementation; generic tests
reject mismatched scale, an invalid result, use on a manager task and missing
verification before any reset or action.

## Qualified lane

Base `fcb4a81a150b0eb272be6708315ea3fa64e00813`, with the current candidate
sources. Python 3.12, Lab 3.0.0 revision
`70ab24fec3eabf51222817db0cdb5a4e7133bd7e`, Sim 6.2.0.0 installed,
Newton 1.6.1 / MJWarp (headless, no Kit), Torch 2.11.0+cu130, TorchRL 0.14.0,
TensorDict 0.14.2, Gymnasium 1.3.0, RSL-RL 5.5.1, RTX 4080 SUPER `cuda:0`
(driver 610.43.03). Native Cartpole USD SHA-256:
`c98ce5dbb174876998052d486036fb79e07f52320851526a4ff61c60ed2db043`.
Native task source SHA-256:
`c1268af7e243fcea1a42b9e757779a0f7b2305dc19f4edea8997f15489d6a8c8`.

```bash
NEXUML_RUN_SIMULATOR_TESTS=1 .venv/bin/python -m pytest -q \
  tests/core/test_isaaclab_integration.py::test_project_owned_direct_task_native_and_adapted_controls \
  'tests/core/test_isaaclab_integration.py::test_native_checkpoint_remaining_budget_export_and_scalar_replay[custom-direct]'
```

**2 passed in 60.13 s**. `P=/tmp/opencode/nexuml-decouple-custom-final-HdYxaF`
holds the process logs, trajectories, training receipts, and artifacts.

- **Controls:** Two fresh jobs used seed 23, eight native environments and
  60 ticks at 60 Hz. Action `+1` became `+100 N`, with cart velocity
  `+1.189976 m/s` at the first step. Row 0 terminated after crossing the
  track limit at tick 9; rows 1–7 truncated at tick 60. Native and adapted
  initial state, rewards, flags, pre-reset final observations and distinct
  post-reset observations matched. Native autoreset was acknowledged without
  resetting again. Arbitrary partial reset was rejected.
- **Native runner:** Isaac's RSL-RL config loader, version adapter and
  `RslRlVecEnvWrapper`/`OnPolicyRunner` ran 256 transitions, two finite PPO
  loss batches and four optimizer steps; actor and critic both changed.
  The exported TorchScript actor ran 60 control steps in a separate native
  process with finite feedback. No NexuML adapter was imported.
- **NexuML:** The ordinary direct runner collected 256 transitions in two
  `[8, 16]` batches and four optimizer steps per actor/critic. It restored
  the checkpoint at 128 frames and ran only the remaining 128. A second
  resume at completion did not start a simulator. The complete actor
  reproduced CPU inference with simulator and learner imports blocked, then
  ran two independently evaluated 8-step episodes (mean return 0.127419).

The native actor is `native-actor.pt` (`42f13c9e4c8950560dc22be89dec5268549991ecd967bea7ddb4ad32f754ad4b`).
The NexuML `resume/policy/pipeline.package` is
`9f928e778450a98fe90c489a292649f507ba8f21b8049a765d361c6d04b240f3`.
These are different trained actors, not a checkpoint conversion or a learning
quality comparison. No state-equivalence claim is made across other physics
or native versions. The wheel includes both example packages; no publication
was performed. Hermetic definition/config/runtime tests: **294 passed**.
Docs, lint, changed-file type checks and strict OpenSpec validation passed.
