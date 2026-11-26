# Bounded shared-actor handoff

## Existing SDK

Preserve `local_fit` and `local_fit_manifest` schema 1 exactly. In that mode the
operator approves the dataset binding and the complete recipe and physical
contract. `base_version` is the digest of the supplied named tensors. Episode
splits, timing and unchanged frozen tensors are checked on each fit. The
existing consumer fixture is `tests/fixtures/so101_local_fit_contract.json`.
Do not relax schema 1 to make it usable by a different provider.

## Shared-actor schema for interactive fitting

The interactive mode uses the same public SDK and returns a full numeric update. It
distinguishes three identities:

- **Manifest:** ordered actor graph and implementation identity, policy input
  and action semantics, preprocessing and adapter ownership, and exact named
  tensor dtype, shape, and role. Define roles as `trainable_parameter`,
  `frozen_parameter` or `frozen_buffer`. An operator-approved trainable subset
  is part of the manifest. No running-state aggregation is supported; frozen
  buffers and parameters must remain bitwise equal to the approved base.
- **Base:** global version supplied by the caller and SHA-256 of the exact
  complete actor tensor state using the SDK's named tensor encoding. Global
  version is not a digest and neither is the manifest. Reject wrong versions,
  manifests, bases, names, dtypes, shapes, storage types, and nonfinite state
  before native allocation.
- **Local receipt:** opaque hash of the approved environment definition,
  provider asset/physics/implementation, algorithm, and effective recipe and
  budget. Return physical transitions, completed episodes, minibatch updates,
  and optimizer steps as distinct counts. The critic, optimizer, raw
  trajectories, and provider paths stay local. The manifest does not contain
  those values or assign federation weights.

Use the compiled **complete actor** (`CompiledPolicy`) as the state owner,
including adapter/processor state. Keep graph stage order and any named input
mapping order before hashing. A declared physical mapping must establish
units, axes, joint order, normalization, calibration, control rate, and
reset/temporal behavior; shape and tensor names alone do not establish that
two providers are compatible. Asset/source and critic-only fields remain
in local provenance rather than the shared actor interface. Report a missing
typed mapping as unsupported, not as an inferred compatible mapping.

The [SO-101 consumer fixture](../../../tests/fixtures/shared_actor_local_fit_contract.json)
enumerates the graph's actual state keys and contract roles. The SDK derives
the schema from the trusted recipe and explicitly supplied numeric state; the
example is not an approved version or base. The current local-fit suite passed
65/65, including the fixture and preflight rejection checks without allocation.

## Execution prerequisites

Load approved numeric state and freeze the approved subset before creating
the optimizer or the collector. Use the canonical local `NexuSession` and one
synchronous direct collector, with an exact whole-rollout frame budget within
the locally approved limit. Restore no critic or optimizer state; use fresh
trajectories and fresh learning state on both fits. Do not expose arbitrary
checkpoint loading, filesystem bindings, or evaluation jobs in a fit request.
Providers must explicitly accept simulated interactive fits. The default
rejects, including ROS 2. There is no physical online fitting mode.

Kit PhysX uses a non-returning process shutdown. Do not make it look like an
ordinary reusable in-process fit, or change its shutdown to force a return.
The established outer job scope can receive and persist the result before
native shutdown; the caller needs a fresh native job process per Kit fit. A
non-Kit provider can use a returning scope only after repeat-entry and
resource cleanup are verified.

The pinned CPU MuJoCo `InvertedPendulum-v5` and `Reacher-v5` did reset and step.
Neither was a compatible counterpart as configured: InvertedPendulum uses
four `float64` observations, a 40 ms control period and a `[-3, 3]` actuator
command with gear 100; the Isaac Cartpole reference uses `float32`, 16.67 ms
and normalized `[-1, 1]` over 100 N. Reacher has a different observation,
actuation and embodiment contract than SO-101. A matched reference needs
validated native configurations and mapping, not relabeled shapes. No
cross-provider or physical ROS 2 fit is qualified by this audit.

## Native SDK repeat-fit evidence

The same public `local_fit` call ran twice on the pinned SO-101 Kit PhysX
profile in separate native jobs. Both fits used 8 environments, 8 rollouts
of 128 physical transitions, one PPO epoch, and minibatches of 64. Each
returned **1,024 transitions, 8 completed episodes, 16 minibatch updates
and 32 optimizer steps**. Both initial critics were identical and fresh,
both optimizer states started empty, and the first rollout's stored behavior
probabilities matched the supplied actor before any update. The two approved
base digests differed, the manifest and local recipe matched, and only the
approved output bias changed. Each fit returned CPU tensors and a receipt
before its app shut down.

Run the native check in the pinned worktree environment:

```bash
NEXUML_RUN_SIMULATOR_TESTS=1 .venv/bin/python -m pytest -q \
  tests/core/test_isaaclab_integration.py::test_native_interactive_local_fit_restarts_from_two_approved_bases
```

Result: 1 passed in 22.16 s; full local receipts and logs:
`/tmp/opencode/nexuml-decouple-sdk-native-Gd3Pva/`. This tests the local fit
boundary, not a learning target, aggregation, or externally imported RSL-RL
weights.

## Paired simulator reference protocol

Use the existing native Isaac manager Cartpole and Gymnasium's CPU MuJoCo
InvertedPendulum with explicit native configuration. Map the same ordered
float32 cart position, pole angle, cart velocity, and pole angular velocity.
Validate the joint axis mapping with the measured response to both signs of
the same normalized cart effort (100 N per action unit). Match the decision
period at 60 Hz, the 120-step horizon, and failure at a pole angle outside
±0.2 rad. Keep native integration, dynamics and rewards identifiable; compare
raw returns only within each provider.

The fixture uses one numeric base (actor initialization seed 17), 1,024
physical transitions per local fit (eight environments, 128 transitions per
rollout, one epoch, minibatches of 64), and fit seeds 31 and 47 in each domain.
Fit both clients from the same approved base; equally average only approved
trainable tensors; evaluate base, both local-only policies and the candidate
on each domain at held-out seeds 211–214. Start a fresh local fit in each
domain from the exact candidate state. Record return, surviving steps,
termination/timeout, update norms and cosine agreement. The acceptance is
exact state/manifest identity, valid local fits and reported outcomes, not
learning improvement, dynamics equivalence, production aggregation or support
for every native task.

### Paired result

The opt-in paired test passed in **102.15 s**. Native Isaac/PhysX and
Gymnasium 1.3.0 / MuJoCo 3.12.0 (CPU) both used −50 and +50 N as requested,
with matching signed cart and pole velocity changes but different magnitudes.
Both started from one exact tensor digest and identical shared manifest. The
full app-specific recipe hashes differed. Only trainable tensors were
averaged with equal test weights. Both domains loaded that exact candidate
for the held-out evaluations and then ran a new fit from its digest.

| Evaluated actor | Isaac mean return / steps | CPU MuJoCo mean return / steps |
| --- | ---: | ---: |
| Common base | 0.157 / 12.75 | 26.00 / 27.00 |
| Isaac-local only | 1.926 / 116.75 | 82.75 / 83.75 |
| MuJoCo-local only | 0.182 / 14.25 | 21.75 / 22.75 |
| Equal-weight candidate | 0.409 / 28.00 | 32.25 / 33.25 |

The candidate improved over the common base in both domains in this small
run, but underperformed the Isaac-local actor on both. The MuJoCo-local fit
regressed against the base on its own held-out episodes. The two update L2
norms were 0.2285 (Isaac) and 0.1255 (MuJoCo), with cosine similarity 0.5823.
Both fits used the same PPO objective class, but retained distinct domain
rewards and dynamics. Four held-out episodes per condition do not establish
learning quality or justify promotion. The Isaac-local replay on Isaac
reached three timeouts; all other conditions had four native terminations.

All four fits performed 1,024 frames, 16 minibatch updates and 32 optimizer
steps. Completed episode counts were 56/42 on Isaac and 31/25 on MuJoCo for
rounds 1/2. Detailed receipts:
[local-fit-receipts.tar.gz](local-fit-receipts.tar.gz), SHA-256
`2687f0642b4f6a13f4b633d2ed1f1d607372d22b463e95b9b1915deefbd7382d`.
`pair/` contains the signed control probes, generated native MuJoCo model,
actual frame/episode counters, per-episode reports and tensor digests. The
`so101/` entries are the earlier native SDK proof; `source.txt` pins the paired
candidate source, which includes the singleton hash fix. Full paired logs:
`/tmp/opencode/nexuml-decouple-paired-cartpole-RHPv34/`.

### Same exported artifact in both simulators

A separate replay exported the exact test-only candidate tensor digest
`ceee4838d79b422bbd7187b589999fe9b920d158d863f81b114f9bca91ce8701`
from the Isaac declaration with `export_package`, then loaded the same
`pipeline.package` via `load_inference_package` in two fresh evaluators.
The held-out seeds 211–214 reproduced the in-memory candidate outcomes:
Isaac **0.409 / 28.00** and CPU MuJoCo **32.25 / 33.25** mean return /
steps. Both reports name package SHA-256
`c908b1f2bb2fd66d1ad768cb1a43549db7d0a9a7482665abd580850eba431710`.
The CPU evaluator retained the Isaac source asset and separate native
MuJoCo asset in its report. No further fit or promotion was performed.
The package, per-episode reports and logs are in
`/tmp/opencode/nexuml-decouple-paired-export-hm4h0F/`. This extends CPU
reference evidence, not the GPU MuJoCo acceptance in §5.

## Local consumer package

Use the [interactive SDK invocation and receipt guidance](../../../docs/src/content/docs/how-to/interactive-local-fit.mdx)
and the schema fixture above. The verification-only wheel pair is under
`/tmp/opencode/nexuml-decouple-candidate-aXt7RU/wheels/`; it is not published
and the `0.2.1` version label alone does not identify this candidate:

| Wheel | SHA-256 |
| --- | --- |
| `nexuml-0.2.1-py3-none-any.whl` | `fd836a73fd746238958c7a4d87d3a633c13afa4a46f4bbc7bb9fef837501dbba` |
| `nexuml_library-0.2.1-py3-none-any.whl` | `04e876f879e07d267b2819b4ba0571992c72d1376a2e23e95d452aad11d7a09f` |

Built offline and installed into a task-owned overlay on the prepared Python
3.12.16 profile; verified imports resolved to the installed wheels, then
passed 66 local-fit tests. Both installed-wheel native gates passed again in
122.91 s: the SO-101 two-base result and paired Cartpole two-round workflow
reproduced the update digests and held-out outcomes recorded above. Installed
proof: `/tmp/opencode/nexuml-decouple-installed-proof-kINOH2/`.
The wheel pair still requires the qualified
simulator/learner dependency profile for live RL. The actor inference package
and its dependencies are separate. Do not fetch stock `0.2.1` to reproduce
these interfaces.

| Profile | Bounded local-fit qualification |
| --- | --- |
| Dataset schema 1 | Existing supervised SDK, unchanged; installed-wheel regressions passed. |
| SO-101, headless PhysX | Public schema 2 API, two independent fits with different approved bases and a selected trainable subset. |
| Isaac Cartpole and CPU MuJoCo InvertedPendulum | Two rounds in the test fixture with validated declared mappings and distinct local provenance. No blanket opt-in for stock Gymnasium tasks. |
| Grouped actors, independent/distributed collectors, native checkpoint import | Not qualified. |
| Physical ROS 2 | No interactive support; real recorded experience and a matching embodiment contract are prerequisites for the offline proof. |

NexuFL owns authorization of the received manifest and base digest, local
recipe selection, federation, aggregation weights, and candidate promotion.
Only policy tensors and approved numeric receipts cross the local boundary.
For a real ROS 2 follow-up, require operator-approved episodes with the same
embodiment, measured joint zeros, units/order, action limits, control period,
reset and timestamp semantics as a qualified simulation mapping. Keep
hold-out splits by whole episode and use a common offline objective first.
No real episodes or matching physical mapping were supplied; the Cartpole
proof is not a physical robot mapping or authorization to actuate hardware.

Public metadata did not supply a direct substitute. The
[ROS Physical AI demo rosbags](https://github.com/ros-physical-ai/demos/blob/main/docs/demos/pretrained-demo.md)
were recorded in Gazebo. The separately published Apache-2.0
[real SO-101 ROS 2 pick-and-place recordings](https://huggingface.co/datasets/legalaspro/so101_pickplace_auto_v1/blob/5f6a8322afe3a65ac82388bbdae08f3924e68339/README.md)
are a possible future input, not the paired Cartpole contract or the
fixed-gripper SO-101 reaching recipe. They use six joint positions and
commands, including the gripper, at 50 Hz. A shared Isaac/MuJoCo/physical
task and verified calibration, action and timing mapping are still
required before fitting. The published dataset file list contains no
original bags or robot-specific calibration. Only metadata was inspected.
