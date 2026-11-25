## ADDED Requirements

### Requirement: Optional GPU buffer interop is measured

First qualify ordinary ROS messages; optional acceleration SHALL use maintained standard buffer/tensor mechanisms such as rosidl::Buffer when supported. Validate dtype/layout/device residency, ownership, synchronization, freshness, actual copies/latency/throughput and output parity; no universal/inter-node zero-copy claim. Unsupported acceleration SHALL be separately blocked or explicitly use normal messages.

#### Scenario: Compatible GPU ROS nodes exchange tensors
- **WHEN** the pinned backend and PyTorch bridge satisfy prerequisites
- **THEN** measured residency and parity establish supported acceleration or it remains unqualified without blocking ordinary ROS.
