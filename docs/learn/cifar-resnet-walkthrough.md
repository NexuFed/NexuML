# CIFAR ResNet example

The old full CIFAR walkthrough duplicated the getting-started and tutorial material and accumulated operational details such as hard-coded checkpoint paths.

For the current workflow:

1. use [Your first scenario](../start/train-cifar-resnet.md) to discover, resolve, and build `cifar-resnet`;
2. read [Scenarios](scenarios.md) to understand typed scenario composition;
3. use the [Tutorials](../tutorials.md) when you want to build an external library/model yourself rather than inspect a finished base-library scenario.

The implementation itself remains the most accurate example of how the base recipe composes reusable helpers: `nexuml_library.scenarios.vision.cifar_resnet` in the generated [Python API](../reference/api/nexuml_library/scenarios/vision/cifar_resnet.md).
