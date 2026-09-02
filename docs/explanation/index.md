# Concepts

Most ML projects do not become difficult because `torch.nn.Module` is difficult. They become difficult because dataset handling, model wiring, training, evaluation, logging, and export gradually turn into project-specific glue code.

A reusable project template helps with folders, but it still encourages copying implementations between projects. NexuML takes a different approach: reusable pieces live in libraries, while a scenario composes those pieces into an explicit experiment.

The design borrows the useful part of block-based systems such as Simulink: **reusable blocks, explicit inputs/outputs, clear data flow, and per-block configuration**, while keeping the implementation in normal Python/PyTorch code.

## The core concepts

- [Mental model](../learn/mental-model.md) — define → persist → materialize → run.
- [Architecture](architecture.md) — typed definitions, runtime construction, persistence, and ownership boundaries.
- [TensorDict data flow](tensordict.md) — the named tensor container connecting pipeline blocks.
- [Scenarios and configuration](../learn/scenarios.md) — how data, pipeline, training, evaluation, and execution are composed.
- [Components and discovery](../learn/decorators-and-discovery.md) — why Python uses concrete definitions while registry identities remain stable for discovery/YAML.
- [Coming from Lightning](../learn/from-lightning.md) — mapping from familiar Lightning concepts.
- [Pipeline diagrams](diagrams.md) — visualizing the compiled graph.
- [Library discovery](library-discovery.md) — built-in, installed, and local component libraries.

## A useful shorthand

If you remember only four things:

1. **ScenarioSpec describes the experiment.**
2. **Typed definitions describe reusable components.**
3. **TensorDict keys make the pipeline flow explicit.**
4. **Lightning runs the lifecycle after NexuML materializes the graph.**

For hands-on learning, use the [Tutorials](../tutorials.md) instead of reading every concept page first.
