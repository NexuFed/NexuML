# Spec Delta

## Purpose

Run, inspect and scale Isaac-backed NexuML experiments with explicit native-process ownership, secure remote access and truthful resource and completion evidence.

## ADDED Requirements

### Requirement: Process lifetime owns native runtime lifetime

Each simulator-owning process SHALL follow its pinned launch/import contract, own only its native resources and finalize checkpoints, exports and loggers before application termination. Discovery, inference-only loading and completed-budget resume SHALL allocate no simulator.

#### Scenario: A profile requires process-terminating fast close
- **WHEN** training fails or is cancelled
- **THEN** normal job cleanup runs first and the actual failure or signal-derived status is preserved rather than converted to success

### Requirement: Independent simulation roles do not share conflicting contexts

Training, rendered evaluation and independent simulations SHALL use validated separate process lifetimes where required. Scheduled evaluation SHALL either use an explicitly supported isolated execution path or fail before allocation; it SHALL NOT reset training state or silently ignore the requested schedule.

#### Scenario: Periodic evaluation is requested without a supported isolated path
- **WHEN** the run is validated
- **THEN** an actionable error directs the user to supported fresh-process replay without starting a second native context

### Requirement: Browser viewing works through a private SSH path

A pinned supported browser viewer SHALL inspect a small Newton replay through an SSH-forwarded loopback port without requiring a desktop display. Public sharing SHALL be disabled by default; interactive resets or state edits SHALL invalidate measured evaluation or be restricted to an explicitly labeled debug run.

#### Scenario: A developer connects over SSH
- **WHEN** the documented viewer and tunnel commands are used
- **THEN** the local browser displays the actual replay and disconnect/cancellation leaves no owned viewer or simulator process running

### Requirement: Isaac Sim streaming has independent network and encoder proof

WebRTC viewing SHALL verify NVENC, graphics passthrough, selected Lab experience and required signaling/media connectivity. Ordinary TCP SSH forwarding SHALL NOT be represented as forwarding UDP media. Unauthenticated endpoints SHALL remain private or receive an explicit authenticated protection layer.

#### Scenario: Signaling connects but the UDP path is blocked
- **WHEN** the streaming acceptance check runs
- **THEN** it reports incomplete connectivity rather than claiming UI support from an open TCP port

### Requirement: Distributed learning preserves native rank ownership

Multi-GPU and multi-node execution SHALL assign one owned native batch per rank, explicit devices/seeds, globally correct frame counts and supported synchronized updates. Checkpoint/log ownership and failure propagation SHALL be defined. Single-GPU tests SHALL NOT establish distributed support.

#### Scenario: One of two GPU ranks fails
- **WHEN** the distributed job unwinds
- **THEN** all owned ranks terminate, failure is retained and no partial rank-zero artifact is labeled a successful complete policy

### Requirement: Existing placement and tuning controls remain authoritative

HPC/cloud or Ray placement and parameter sweeps SHALL reuse existing execution, scenario and tuning conventions. Native handles SHALL be built at workers and never serialized. Unsupported GPU placement or population-based updates SHALL remain rejected until their synchronization/checkpoint semantics are verified.

#### Scenario: A remote execution request contains only a task definition
- **WHEN** the selected supported worker starts
- **THEN** it constructs native resources locally and records its dependency/task identities without receiving a live simulator from the parent

### Requirement: Performance is measured at the claimed layer

Reports SHALL distinguish startup, physics throughput, rendered throughput, end-to-end learning and policy latency with task/batch/device/resource identities. Compilation, CUDA graph or optimized sequence backends SHALL require correctness checks, explicit warm-up and measured benefit rather than automatic universal enablement.

#### Scenario: Faster physics does not improve training time
- **WHEN** a scaling result is published
- **THEN** physics and full-learning timings expose the bottleneck without claiming end-to-end speedup

### Requirement: Test tiers never manufacture completion

Fast CPU tests SHALL require no native installs, assets or hardware. Native GPU, graphics, ROS, multi-device and physical trials SHALL be opt-in with explicit prerequisites. Full-program claims SHALL require actual retained evidence for every required family; skips and prepared CI SHALL NOT count as passes.

#### Scenario: Only one GPU is available
- **WHEN** the distributed suite is not executed
- **THEN** its capability remains unverified with the missing resource stated, even if fake distributed tests pass

### Requirement: Verified milestone commits exclude execution evidence

Each verified implementation milestone SHALL produce a focused local commit of owned source, tests, user docs, configuration/locks and planning changes after its applicable checks pass. Execution evidence SHALL remain outside Git, even when embedded in planning/docs. Staged paths and content SHALL be inspected. Existing committed evidence/history SHALL remain intact; push, external review publication, merge and archive require separate direction.

#### Scenario: A milestone is ready to commit
- **WHEN** its applicable verification succeeds with unrelated baseline failures distinguished
- **THEN** only explicit milestone-owned files/hunks are staged, excluding reports, raw logs, generated run results/media, checkpoints, weights and unrelated work
- **AND** evidence remains locally reviewable or in an explicitly approved artifact location without automatic upload

#### Scenario: Evidence is mixed with planning changes
- **WHEN** staged content or a candidate file contains runtime evidence alongside authored planning or documentation
- **THEN** that evidence is excluded from the commit without discarding the user's local records or rewriting historical commits
- **AND** inability to isolate the changes or create the commit is reported as a blocker, not bypassed by a blanket stage
