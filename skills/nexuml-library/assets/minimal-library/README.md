# Offline external-library example

This tiny synthetic regression library demonstrates the typed NexuML 0.2.1
contracts. It is **not** a paper reproduction or a model-quality benchmark.
It uses installed NexuML only, without `nexuml_library` or a framework checkout.

Copy this directory to a fresh user-owned location when a starter is useful.
For an existing library, adapt only the relevant component pattern.
Select an environment with compatible NexuML and install this package editably:

```sh
uv pip install --python /path/to/project/python -e .
/path/to/project/python -m agent_demo.checks
/path/to/project/python -m nexuml.cli.main registry list scenarios --verbose
/path/to/project/python -m nexuml.cli.main resolve agent-demo-regression --output configs/baseline.yaml
/path/to/project/python -m nexuml.cli.main build configs/baseline.yaml
/path/to/project/python -m nexuml.cli.main train --scenario-file experiment.py --artifact-dir runs/smoke --max-epochs 1
```

Use the release-appropriate package index if NexuML is not on the default index.
The package declares its dependency; editable installation does not require a
separate `library add`. Do not add `src/agent_demo` under another module name.

The checks exercise parameter bounds, sample and label contracts, numerical
loss/gradient behavior, uneven-batch metric aggregation, fresh installed
discovery, portable YAML and compilation. Training uses 64 generated vectors,
CPU, one epoch, train/validation splits and **no final-test split**. Its validation
loss verifies execution, not statistical improvement. The consumer evaluator is
checked directly and disabled during this research smoke; merely declaring an
evaluator does not prove that a lifecycle phase has evaluated it.

All names, destinations and dimensions are illustrative. Preserve existing
user files, use fresh run folders and fit real datasets to their actual protocol.
