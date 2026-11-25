## ADDED Requirements

### Requirement: Native bounded goals are not fixed-step actions

Optional skill bindings SHALL reuse native Nav2, MoveIt 2 or compatible planner and G1 controller interfaces with typed goals, bounds, frames, acceptance/completion/failure, elapsed duration, deadline and cancellation. Variable-duration goals SHALL not masquerade as fixed-step RL without qualified semantics. Controller safety/balance remain native.

#### Scenario: A navigation goal exceeds its deadline
- **WHEN** a qualified request has not completed
- **THEN** the request is cancelled or safely handled, duration and failure are recorded and no unsupported transition is fabricated.
