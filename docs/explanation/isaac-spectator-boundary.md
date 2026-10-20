# Native spectator construction boundary

The Library spectator adapter reuses Isaac/Newton geometry and Viser transport.
It bypasses Isaac's constructor that temporarily reassigns the global server
class. A local native subclass supplies the private server instead; no vendor
files or process-global classes/functions are replaced.

## Restricted construction

- Exact installed Lab/Sim/Newton/Viser versions, source identities and bundled,
  noneditable frontend are checked before opening a listener.
- Bind only to `127.0.0.1` at the requested explicit port, normally 8080. Native
  public sharing, recording, streaming and port overrides are rejected. A port
  conflict fails before construction; reassignment fails the endpoint check.
- Construction requires the POSIX main thread and unowned `SIGALRM`. A temporary
  ten-second timer bounds synchronous constructor readiness; prior signal/timer
  state is restored, including cancellation during context entry.
- Partial viewer/server ownership is retained before constructors can fail.
  Shutdown checks the server and callback threads within one fifteen-second
  budget, reserving the pinned transport's synchronous one-second join. Failure
  to stop is an explicit job failure, never successful cleanup.

The constructor sidebar permits display pause only. Final native pause/reset
polls deny simulation handoff even if stale native flags are set. Camera and
display state are not permission to mutate physics, task configuration or actors.

## Scope of the gate

Run the bounded constructor/dispatch/failure tests explicitly in the prepared
environment:

```bash
CUDA_VISIBLE_DEVICES='' NEXUML_RUN_SYSTEM_TESTS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/core/test_isaaclab_viser.py -q
```

These are constructor-safety checks, not training, replay, sensor or browser
acceptance. Job activation, environment selection, markers, supported plots and
actual training/replay browsers remain separate required integration gates.
