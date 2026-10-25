# nexuml.execution.capacity

*module*

Kubernetes request accounting, not a replacement for the native scheduler.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/capacity.py#L1-L1)

## nexuml.execution.capacity.node_matches

*function*

```python
def node_matches(node: dict, pod_spec: dict, request: dict[str, float]) -> bool
```

Check observable node constraints and per-node allocatable fit.

**Returns**

- (`bool`) — Whether the node matches this bounded subset of scheduler constraints.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/capacity.py#L104-L148)

## nexuml.execution.capacity.node_resources

*function*

```python
def node_resources(nodes: list[dict], pods: list[dict] | None, pod_spec: dict) -> list[NodeResources]
```

Keep allocatable and estimated unallocated resources separate.

**Returns**

- (`list[NodeResources]`) — One safe resource row per visible node.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/capacity.py#L151-L191)

## nexuml.execution.capacity.pod_requests

*function*

```python
def pod_requests(spec: dict[str, Any]) -> dict[str, float]
```

Account for ordinary/init/sidecar containers, pod requests and overhead.

**Returns**

- (`dict[str, float]`) — Effective requests used for native scheduling, plus one pod slot.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/capacity.py#L43-L66)

## nexuml.execution.capacity.quantities

*function*

```python
def quantities(values: dict[str, Any]) -> dict[str, float]
```

Parse native Kubernetes quantities without confusing cores, bytes and devices.

**Returns**

- (`dict[str, float]`) — Named finite nonnegative scheduling quantities.

**Raises**

- `ValueError` — For negative or nonfinite resource requests.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/capacity.py#L8-L28)
