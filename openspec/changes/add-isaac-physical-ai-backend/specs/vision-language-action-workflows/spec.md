# Spec Delta

## Purpose

Use pretrained vision-language-action policies in NexuML for complete multimodal inference, adaptation and Isaac evaluation while preserving model-specific processors and action-head semantics.

## ADDED Requirements

### Requirement: Pretrained policy families have complete integration proofs

SmolVLA and one maintained GR00T release SHALL each have pinned model/code licenses, pretrained inference, task-specific supervised adaptation, export/reload and closed-loop evaluation on a compatible Isaac task. Model availability or a synthetic tiny-policy example SHALL NOT count as these real-model proofs.

#### Scenario: A pretrained family is reported as supported
- **WHEN** its capability evidence is inspected
- **THEN** actual model/input/action identities and its complete adaptation/evaluation round trip are available

### Requirement: Multimodal preprocessing survives packaging

VLA inference SHALL preserve multi-camera ordering, image preprocessing, proprioception, language tokenization, normalization and embodiment/action decoding. Required processor assets and versions SHALL be reconstructible without loading the upstream training runner.

#### Scenario: Camera names are reordered during deployment
- **WHEN** the incoming observation mapping differs from the recorded contract
- **THEN** execution applies an explicitly declared validated mapping or rejects it rather than swapping camera meaning silently

### Requirement: Action chunks are executed with explicit timing

VLA inference SHALL expose chunk horizon, replanning stride, temporal aggregation and reset behavior. Action rate and processor latency SHALL be measured against the target controller; pending actions SHALL NOT cross episode boundaries or mask missing observations.

#### Scenario: Inference is slower than the required command rate
- **WHEN** a deployment target is validated
- **THEN** the report identifies the mismatch and does not claim real-time readiness from average model throughput alone

### Requirement: Full and parameter-efficient adaptation remain reproducible

Supported full fine-tuning and parameter-efficient adaptation SHALL record trainable/frozen parameter groups, model/dataset revisions, precision and optimization settings. Exported adapter policies SHALL include or identify the exact base composition and preserve its reference behavior.

#### Scenario: A LoRA-style candidate is exported
- **WHEN** it is loaded in a clean inference environment
- **THEN** the exact base-plus-adapter policy reproduces the reference actions and its old/new-task regression results remain traceable

### Requirement: RL post-training matches the action model

At least one actual pretrained VLA SHALL support simulated RL post-training under NexuML's learning lifecycle. The objective SHALL match its token, continuous, diffusion or flow action distribution; Gaussian PPO log-probabilities SHALL NOT be invented for an incompatible head.

#### Scenario: A selected head cannot provide the required likelihood
- **WHEN** an incompatible policy-gradient objective is requested
- **THEN** configuration fails before optimization and the missing integration remains incomplete rather than using fabricated probabilities

### Requirement: VLA comparisons use equivalent tasks and inputs

Pretrained, supervised-adapted and RL-adapted policies SHALL be compared on frozen compatible tasks, sensor contracts and action semantics. Reports SHALL include success, forgetting, memory, latency tails and effective control rate rather than assuming all model families are interchangeable.

#### Scenario: ACT and a VLA use different camera or action contracts
- **WHEN** a comparison is requested
- **THEN** the report refuses a direct performance ranking until a validated equivalent task/interface is established

### Requirement: Runtime optimization requires independent parity

Quantized, compiled or exported VLA runtimes SHALL pass numerical and task-semantic parity against the reference policy, including processors and temporal state. Unsupported ONNX, TensorRT or ROS deployment conversions SHALL remain explicit, not successful merely because a file was generated.

#### Scenario: Conversion changes the action decoder
- **WHEN** the optimized policy is checked on representative sequences
- **THEN** failed parity blocks use of that target while the reference inference package remains usable

### Requirement: Pi0.5 has a resource-gated complete reference

A maintained pretrained π0.5 policy SHALL demonstrate inference, task-specific adaptation, resume, complete export/reload and compatible Isaac evaluation through NexuML. Processor/configuration, normalization statistics and action conventions SHALL be preserved. Model/backbone licenses and actual resource needs SHALL be verified; missing approved compute SHALL leave this reference incomplete rather than reduce it to a feasibility claim.

#### Scenario: Pi0.5 is adapted to a matching Isaac task
- **WHEN** approved pretrained assets and adequate resources are available
- **THEN** actual model updates, checkpoint restoration, clean-package action/processor parity and closed-loop task outcomes are retained
- **AND** comparison with other policies uses matched task/interfaces and reports old/new-task regression, memory and latency without presuming better performance

#### Scenario: Pi0.5 requires unavailable statistics or compute
- **WHEN** selected normalization lacks required training-split statistics or the run exceeds approved resources
- **THEN** execution reports the missing prerequisite without reading held-out statistics, changing normalization silently or starting an unapproved remote job
- **AND** neither package loading nor a tiny substitute model closes the complete-reference gate

### Requirement: Maintained foundation-model versions and processors are explicit

The GR00T reference SHALL target a currently verified maintained N1.7 integration, not silently reuse an obsolete N1.5 code/weight path. SmolVLA and the resource-gated pi0.5 remain complete workflows. Other models such as MolmoAct2 and Multitask DiT may enter only after versioned source/model/licensing/compute feasibility and matching action/embodiment reviews. Processor and tokenizer/normalizer identity SHALL survive complete policy packaging.

#### Scenario: Legacy GR00T N1.5 weights are selected for N1.7
- **WHEN** source, model or processor identity is incompatible
- **THEN** policy loading fails explicitly or selects an independently validated legacy path rather than silently changing model semantics
