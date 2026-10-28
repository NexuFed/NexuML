# Spec Delta

## Purpose

Add a maintained GPU MuJoCo learning path without confusing classic CPU MuJoCo, JAX-based MJX learning, Warp-native physics and NVIDIA Isaac/Newton integration.

## ADDED Requirements

### Requirement: A pinned MuJoCo Playground reference is discoverable and runnable

The library SHALL expose one small GPU classic-control reference, one articulated locomotion/manipulation reference and one bounded vision-based reference from an inspected MuJoCo Playground release, using upstream environments and assets rather than copying a second simulator library. Each SHALL declare actual CPU/GPU, JAX/Torch, renderer and license requirements.

#### Scenario: An example is discovered without Playground installed
- **WHEN** library discovery or static configuration runs
- **THEN** it constructs no JAX CUDA context, downloads no Menagerie assets and exposes a focused missing-extra message only on native execution

### Requirement: Physics implementation and training backend are independent

The selected MuJoCo Playground task SHALL identify MJX-JAX or MJWarp physics and the exact learning engine. JAX/Flax training SHALL NOT be mislabeled a NexuML Lightning/TorchRL learner. A native Torch/Warp environment adapter SHALL train through NexuML only when the upstream public API and numerical boundary are verified; other upstream trainers are explicit baselines. A JAX-native NexuML learner would require its own reviewed optimizer/serialization contract, not silent embedding.

#### Scenario: A Playground JAX PPO baseline runs
- **WHEN** the native upstream baseline is executed
- **THEN** reports identify it as an upstream comparison, with native checkpoint/framework identity and no claim of NexuML-owned actor updates

#### Scenario: NexuML-owned GPU training is reported
- **WHEN** one Playground/MJWarp task is marked integrated
- **THEN** a NexuML-session-controlled learning update, checkpoint/resume, complete export, clean inference and native replay have actually executed with matching action, observation and clock contracts

### Requirement: Native batch, reset and gradient semantics are proven

Adapters SHALL preserve environment/time axes, device residency, native final versus autoreset observations, correct true termination/truncation and valid on-/off-policy replay accounting. GPU physics and neural training timings SHALL be measured separately. Device transfer or differentiable physics assumptions SHALL be explicit.

#### Scenario: Some GPU environments finish before others
- **WHEN** native final transitions are collected
- **THEN** terminal observations, episode counters and value bootstrap agree with an independent numerical oracle and the next episode does not contaminate the previous one

### Requirement: GPU tasks have reproducible quality baselines

Frozen task/asset/physics versions, seeds, environment counts, limits, domain randomization and renderer inputs SHALL accompany comparisons. At least one maintained Playground reference SHALL compare its upstream quality baseline against a NexuML-owned compatible workflow on matching conditions. GPU vector throughput alone SHALL NOT count as learned-policy equivalence.

#### Scenario: Two policies use different control semantics
- **WHEN** their tasks have similar observation shapes but distinct action scales or frequencies
- **THEN** the report withholds direct quality comparison and explains the mismatch instead of relabeling results

### Requirement: Optional native profiles remain independently testable

A primary-stack-compatible Playground/MJWarp path SHALL be preferred. Required incompatible JAX/CUDA wheels SHALL use the explicit compatibility-lane rules from physical-ai-dependency-composition; no unapproved second training environment, hidden version override or silent fallback is allowed. CI smoke, GPU opt-in and vision/render opt-in SHALL be separately marked.

#### Scenario: An NVIDIA GPU is unavailable
- **WHEN** optional Playground native GPU tests are not executed
- **THEN** capability status remains unverified rather than passing via simulated TensorDict fixtures
