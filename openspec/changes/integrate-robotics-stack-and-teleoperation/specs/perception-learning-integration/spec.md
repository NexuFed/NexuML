## ADDED Requirements

### Requirement: Perception trainability and policy semantics are explicit

NexuML SHALL support qualified external/frozen perception, jointly trainable PyTorch encoder/policy and hybrid native skills using existing pipeline/contracts. Model/processor identity, frozen/trainable/shared scope, frames, rates/freshness and deployed inputs SHALL be explicit. ROS/TensorRT inference SHALL not be assumed differentiable; optimized export requires parity.

#### Scenario: Frozen perception supplies object pose
- **WHEN** a compatible policy is trained
- **THEN** the pose is contracted and the perception state is not optimized by the policy learner.

### Requirement: Deployment perception uses available observations

Perception/skill workflows SHALL measure actual inputs, timing, dropped data and task outcomes against a native/classical baseline. Privileged observations SHALL not enter deployed actor inputs; independent supervised perception and policy learning reuse existing training interfaces.

#### Scenario: A vision policy is exercised in the full stack
- **WHEN** actual camera pipeline replaces privileged features
- **THEN** missing/stale inputs reject rather than using hidden ground truth.
