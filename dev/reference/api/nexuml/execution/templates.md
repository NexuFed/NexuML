# nexuml.execution.templates

*module*

Review and bind infrastructure-owned templates without synthesizing cluster policy.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/templates.py#L1-L1)

## nexuml.execution.templates.CONFIG_TOKEN

*attribute* · *module attribute*

```python
CONFIG_TOKEN = '{{NEXUML_CONFIG}}'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/templates.py#L20-L20)

## nexuml.execution.templates.authorized_path

*function*

```python
def authorized_path(value: str, root: Path | None = None) -> Path
```

Resolve an explicitly selected file without escaping an API's allowed root.

**Returns**

- (`Path`) — A canonical authorized path.

**Raises**

- `ValueError` — On traversal or symlink escape.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/templates.py#L23-L40)

## nexuml.execution.templates.pod_roles

*function*

```python
def pod_roles(manifest: dict, execution: Any) -> list[tuple[str, int, dict]]
```

Read every workload role without inventing an unseen controller footprint.

**Returns**

- (`list[tuple[str, int, dict]]`) — Named replica counts and existing pod specifications.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/templates.py#L43-L67)

## nexuml.execution.templates.template_plan

*function*

```python
def template_plan(scenario: Any, *, authorized_root: Path | None = None, invocation_id: str | None = None) -> dict[str, Any]
```

Capture the exact reviewed template and derived worker configuration.

**Returns**

- (`dict[str, Any]`) — A private frozen plan and a credential-free summary.

**Raises**

- `ValueError` — For missing handoff, unsafe inputs or unsupported launch topology.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/templates.py#L118-L348)
