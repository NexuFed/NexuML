# Direct native interaction foundation

Task 1.7 qualifies the headless, state-only path, not full policy playback.
The [Python example](../../../docs/src/content/docs/how-to/isaac-native-configuration.mdx#step-an-isaac-task-without-a-policy)
uses the same `EnvironmentDefinition`, `job_scope`, native runtime,
contract checks and close order as direct evaluation. The existing training
collector still uses `step` with no recording. `step_recorded` is optional:
the caller identifies the source and receives detached snapshots of the
contracted action and tensor submitted to the native environment. The host
monotonic issue time and native control clock are explicitly distinct. The
native trace verifies one control tick and does not invent per-sensor
acquisition timestamps or conflate submission with actuator feedback.

Unsupported asynchronous goals and out-of-contract tensor actions fail
without a native step. Headless Isaac play does not activate a viewer or
render target. ROS joint reach is simulation-only; the wall-clock and
playback requests tested here fail before creating a transport. Existing
sensor rate/freshness checks and the ROS transport's stale-sample rejection
remain in place. No physical hardware, asynchronous executor, grouped
action recording, or remote/display playback was qualified.

## Native state proof

Candidate based on `fcb4a81a150b0eb272be6708315ea3fa64e00813`, worktree
Python 3.12, Isaac Lab `70ab24fec3eabf51222817db0cdb5a4e7133bd7e` /
Kit 6.2.0.0, Torch 2.11.0+cu130, TensorDict 0.14.2, `cuda:0`
(RTX 4080 SUPER). Run with the prepared headless profile:

```bash
NEXUML_RUN_SIMULATOR_TESTS=1 .venv/bin/python -m pytest -q \
  tests/core/test_isaaclab_integration.py::test_native_scripted_and_policy_actions_share_the_headless_runtime
```

Three serial native Cartpole jobs used reset seed 23 and matched initial
state. The known script needed no policy and applied `+50 N` for normalized
`+0.5`, increasing cart velocity by `0.609563 m/s`. The independent actor
and scripted replay passed the same normalized `0.0602091849` input;
Isaac's action term observed `6.02091837 N` on both. Returned observations,
reward (`0.01663951`) and terminal flags matched. Each trace recorded native
step `0 → 1` at 60 Hz. The native context closed after every job.

Final run: **1 passed in 16.78 s**, with logs and raw JSON in
`/tmp/opencode/nexuml-decouple-foundation-final-UmIN3U/`. The receipts in the
test's output directory are `known.json` (`a93ae7390637b47aab639cd478373aface622f6f8f34331e50b40596b75e2013`),
`policy.json` (`24a11d7e56e8a45d6ef5d8882722ef4a03f538ca2f549f6b03b487fed5a32bd2`),
and `replay.json` (`c18433897d6a1e0bc47d696577651fe0960037e3fbcb19b484348d5e43a2c4f1`).

The same seed did not reproduce initial velocities across fresh startup and
an in-process reset after stepping: in this probe, fresh was `(0, 0)` and
repeated reset was `(-0.119108, 0.543395)` in `m/s` and `rad/s`. NexuML
returned the exact native `obs_buf` on that reset. This is not a promise of
native trajectory or full reset-state equivalence across different lifetimes.
The source comparison used fresh jobs with identical observed starting
states; it did not change the simulator or its reset code.

Focused runtime, sensor, ROS fake-transport, and evaluation regressions:
430 passed, 14 skipped, one system test deselected. The Isaac definition
and CARLA fake-runtime suites added 58 passed and one system test
deselected. User docs, lint, changed-file type checks and strict OpenSpec
validation passed.
