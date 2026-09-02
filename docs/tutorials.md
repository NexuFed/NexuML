# Tutorials

The hands-on learning material lives in the separate [NexuMLTutorial repository](https://github.com/NexuFed/NexuMLTutorial). This keeps the framework documentation focused on concepts, task guides, and reference while the tutorial repository can contain complete runnable projects.

The tutorial repository is itself an external NexuML library. Its datasets, layers, evaluation code, and scenarios are tutorial-owned instead of importing finished implementations from `nexuml_library`. That is intentional: the goal is to teach how you build your own models with NexuML.

## Learning path

The current tutorial work is structured as:

| Stage | Topic | What it introduces |
| --- | --- | --- |
| 1 | MNIST: custom library basics | local library registration, dataset, model/loss/metric components, scenario composition, resolve/build/train |
| 2 | Speech Commands: native DALI audio | real file-backed audio, metadata-driven splits, native DALI loading |
| 3 | Swap only the encoder | pipeline composition by replacing a CNN with a Transformer while reusing the rest of the system |
| 4+ | Tutorial roadmap | tuning/tracking, preprocessing and dataset export, custom evaluation, checkpoints/transfer learning, export/inference, distributed execution |

The progression mirrors how NexuML is intended to be learned: first understand reusable blocks and explicit data flow, then add more sophisticated data and execution features only when you need them.

!!! warning "Use a tutorial revision compatible with your NexuML version"
    The current `feature/nex-210-audio-example` tutorial work predates the NexuML 0.2 typed-component configuration refactor. NexuML 0.2 rejects the legacy selector/parameter-bag syntax. Do not mix that older tutorial revision with a 0.2 installation until the tutorial branch has been migrated.

## Documentation vs tutorials

Use the **tutorials** when you want to learn by building a complete project. Use the **framework documentation** when you need an exact answer:

- [Guides](how-to/index.md) — how to perform a specific task;
- [Concepts](explanation/index.md) — why NexuML is structured this way;
- [Reference](reference/index.md) — exact configuration, CLI, backend, and API details.
