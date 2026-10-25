# nexuml.core.provenance

*module*

Optional local provenance snapshots for scenario-file runs.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/provenance.py#L1-L1)

## nexuml.core.provenance.snapshot_scenario_file_run

*function*

```python
def snapshot_scenario_file_run(loaded: LoadedScenarioFile, artifact_dir: str | Path | None, *, project_root: str | Path | None = None, command: str, command_args: dict[str, Any] | None = None, search_space: dict[str, dict[str, Any]] | None = None, study: Any = None) -> Path | None
```

Snapshot source, built scenario YAML, metadata, git, and tuning summary.

**Returns**

- (`Path | None`) — Path to the snapshot directory, or ``None`` if *artifact_dir* is ``None``.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/provenance.py#L45-L107)
