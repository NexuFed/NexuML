## ADDED Requirements

### Requirement: Ten exact Isaac examples are delivered and qualified

Every R01–R10 row SHALL be discoverable as a native Isaac learning reference with exact assets/task/control, real optimizer updates, resume/export/reload, held-out outcomes, play and cleanup. Reuse valid existing receipts. Asset-only, upstream-trained or scripted-only examples do not qualify; missing licensed assets/tasks remain blocked. Only selected matching alternate simulator references required.

#### Scenario: A Spot-with-arm or A2D asset loads
- **WHEN** no actual native learned task has been qualified
- **THEN** asset availability may be recorded but the required example remains incomplete; another robot cannot substitute.

### Requirement: LeKiwi is the primary mobile simulation reference

Exact LeKiwi V1 SHALL use maintained LeIsaac assets or validated official URDF conversion and explicit base/arm/gripper controls. Demonstrate human simulated teleop, validated episodes, actual learned manipulation, export/play and integrated native navigate/pick/transport/place with measured outcomes/classical baseline. Physical LeKiwi V1 SHALL be the second separately authorized physical learning milestone after SO-101. Physical evidence SHALL be separate from simulation evidence and require calibrated base/arm controls, recorded-input parity, independent stop/fallback, actual demonstrations and supervised task/correction capture with held-out local adaptation comparison.

#### Scenario: Physical hardware or approval is absent
- **WHEN** the matching simulation workflow is qualified
- **THEN** the physical milestone remains blocked without preventing independent software qualification.

#### Scenario: A simulated LeKiwi reaches an object
- **WHEN** the system reference invokes qualified navigation and learned manipulation
- **THEN** actual base and arm behavior, grasp/place outcomes and controller failures are observable.
